"""Document parsing: PDF, DOCX, TXT → TextUnit stream with metadata."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from models import DocumentContent, TextUnit
from config import AppConfig, get_config

logger = logging.getLogger(__name__)

# Section-style headings (7.2 Title, CHAPTER 1, ALL CAPS) — not simple "1. Do thing" steps
_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+\.\d+(?:\.\d+)*\.?\s+\S.*"  # multi-level section numbers: 7.2 Foo
    r"|[A-Z][A-Z0-9\s\-/]{4,}$"  # ALL CAPS titles
    r"|(?:Chapter|Section|Appendix|Part)\s+[\w\.\-]+.*"
    r"|(?:Procedure|Process|Method)\s*:\s*.+"
    r")",
    re.IGNORECASE,
)
_MAJOR_SECTION_RE = re.compile(r"^\d+\.\d+\s+")
_PAGE_BREAK_RE = re.compile(r"^\s*(?:page\s+)?(\d+)\s*$", re.IGNORECASE)

_STEP_LINE_RE = re.compile(
    r"^\s*(?:\d+[\.\)]\s+|Step\s+\d+[:\.\s]+|[a-z][\.\)]\s+|[-•▪]\s+)",
    re.IGNORECASE,
)


class BaseDocumentParser(ABC):
    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, path: Path) -> DocumentContent:
        raise NotImplementedError


class TxtParser(BaseDocumentParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in {".txt", ".md", ".text"}

    def parse(self, path: Path) -> DocumentContent:
        text = path.read_text(encoding="utf-8", errors="replace")
        units: list[TextUnit] = []
        section: str | None = None
        paragraph = 0
        page = 1
        seen_major_section = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _MAJOR_SECTION_RE.match(line):
                if seen_major_section:
                    page += 1
                seen_major_section = True
            unit_type = _classify_line(line)
            if unit_type == "paragraph" and _looks_like_heading(line):
                section = line
                units.append(
                    TextUnit(
                        text=line,
                        page=1,
                        section=section,
                        paragraph=None,
                        unit_type="heading",
                    )
                )
                continue
            paragraph += 1
            units.append(
                TextUnit(
                    text=line,
                    page=1,
                    section=section,
                    paragraph=paragraph,
                    unit_type=unit_type,
                )
            )
        return DocumentContent(
            source_path=str(path.resolve()),
            source_name=path.name,
            mime_type="text/plain",
            units=units,
            page_count=page,
        )


class DocxParser(BaseDocumentParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in {".docx"}

    def parse(self, path: Path) -> DocumentContent:
        from docx import Document
        from docx.table import Table as DocxTable
        from docx.text.paragraph import Paragraph

        document = Document(str(path))
        units: list[TextUnit] = []
        section: str | None = None
        paragraph_idx = 0
        page = 1
        seen_major_section = False

        for block in _iter_block_items(document):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue
                page_match = _PAGE_BREAK_RE.match(text)
                if page_match:
                    page = max(page, int(page_match.group(1)))
                    continue
                style_name = (block.style.name or "") if block.style else ""
                is_heading = "Heading" in style_name or _looks_like_heading(text)
                if is_heading:
                    if _MAJOR_SECTION_RE.match(text):
                        if seen_major_section:
                            page += 1
                        seen_major_section = True
                    section = text
                    units.append(
                        TextUnit(
                            text=text,
                            page=page,
                            section=section,
                            paragraph=None,
                            unit_type="heading",
                            metadata={"style": style_name},
                        )
                    )
                else:
                    paragraph_idx += 1
                    units.append(
                        TextUnit(
                            text=text,
                            page=page,
                            section=section,
                            paragraph=paragraph_idx,
                            unit_type=_classify_line(text),
                            metadata={"style": style_name},
                        )
                    )
            elif isinstance(block, DocxTable):
                table_text = _table_to_text(block)
                if table_text:
                    paragraph_idx += 1
                    units.append(
                        TextUnit(
                            text=table_text,
                            page=page,
                            section=section,
                            paragraph=paragraph_idx,
                            unit_type="table",
                        )
                    )

        return DocumentContent(
            source_path=str(path.resolve()),
            source_name=path.name,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            units=units,
            page_count=page,
        )


class PdfParser(BaseDocumentParser):
    """Hybrid PDF parser: PyMuPDF for structure, pdfplumber for tables."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> DocumentContent:
        import fitz  # PyMuPDF

        units: list[TextUnit] = []
        section: str | None = None
        paragraph_idx = 0
        doc = fitz.open(str(path))
        body_sizes = _estimate_body_font_sizes(doc)
        median_size = sorted(body_sizes)[len(body_sizes) // 2] if body_sizes else 11.0
        heading_delta = self.config.parsing.heading_font_size_delta

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                lines_text: list[str] = []
                max_size = 0.0
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(span.get("text", "") for span in spans).strip()
                    if not line_text:
                        continue
                    lines_text.append(line_text)
                    for span in spans:
                        max_size = max(max_size, float(span.get("size", 0)))
                if not lines_text:
                    continue
                text = " ".join(lines_text).strip()
                if not text:
                    continue
                is_heading = (
                    max_size >= median_size + heading_delta
                    or _looks_like_heading(text)
                ) and len(text) >= self.config.parsing.min_heading_length
                if is_heading and len(text) < 200:
                    section = text
                    units.append(
                        TextUnit(
                            text=text,
                            page=page_number,
                            section=section,
                            paragraph=None,
                            unit_type="heading",
                            metadata={"font_size": max_size},
                        )
                    )
                else:
                    paragraph_idx += 1
                    units.append(
                        TextUnit(
                            text=text,
                            page=page_number,
                            section=section,
                            paragraph=paragraph_idx,
                            unit_type=_classify_line(text),
                            metadata={"font_size": max_size},
                        )
                    )

        if self.config.parsing.extract_tables:
            units.extend(self._extract_tables(path, existing_pages={u.page for u in units}))

        units.sort(key=lambda u: (u.page or 0, 0 if u.unit_type == "heading" else 1, u.paragraph or 0))
        page_count = len(doc)
        doc.close()
        return DocumentContent(
            source_path=str(path.resolve()),
            source_name=path.name,
            mime_type="application/pdf",
            units=units,
            page_count=page_count,
        )

    def _extract_tables(self, path: Path, existing_pages: set[int | None]) -> list[TextUnit]:
        tables: list[TextUnit] = []
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not available; skipping table extraction")
            return tables

        with pdfplumber.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                for table in page.extract_tables() or []:
                    rows = [
                        " | ".join((cell or "").strip() for cell in row)
                        for row in table
                        if row and any((cell or "").strip() for cell in row)
                    ]
                    if not rows:
                        continue
                    text = "\n".join(rows)
                    tables.append(
                        TextUnit(
                            text=text,
                            page=page_number,
                            section=None,
                            paragraph=None,
                            unit_type="table",
                            metadata={"source": "pdfplumber"},
                        )
                    )
        return tables


class DocumentParserService:
    """Facade selecting the appropriate parser for a file path."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._parsers: list[BaseDocumentParser] = [
            PdfParser(self.config),
            DocxParser(),
            TxtParser(),
        ]

    def parse(self, path: str | Path) -> DocumentContent:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        for parser in self._parsers:
            if parser.can_parse(file_path):
                logger.info("Parsing %s with %s", file_path.name, parser.__class__.__name__)
                content = parser.parse(file_path)
                logger.info(
                    "Parsed %s: %d units, %d pages",
                    file_path.name,
                    len(content.units),
                    content.page_count,
                )
                return content
        raise ValueError(f"Unsupported document type: {file_path.suffix}")


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 180 or len(stripped) < 3:
        return False
    # Numbered procedure steps are never headings
    if _STEP_LINE_RE.match(stripped) and not re.match(r"^\d+\.\d+", stripped):
        return False
    return bool(_HEADING_RE.match(stripped))


def _classify_line(text: str) -> str:
    stripped = text.strip()
    if _STEP_LINE_RE.match(stripped) and not re.match(r"^\d+\.\d+\s+\S", stripped):
        if re.match(r"^\s*[-*•▪]", stripped):
            return "list_item"
        return "step"
    if re.match(r"^\s*(\([a-z0-9]+\))\s+", stripped, re.IGNORECASE):
        return "list_item"
    return "paragraph"


def _estimate_body_font_sizes(doc: object) -> list[float]:
    sizes: list[float] = []
    for page_index in range(min(len(doc), 20)):  # type: ignore[arg-type]
        page = doc[page_index]  # type: ignore[index]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size", 0))
                    if size > 0:
                        sizes.append(size)
    return sizes or [11.0]


def _iter_block_items(parent: object):
    from docx.document import Document as DocxDocument
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph

    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        return

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _table_to_text(table: object) -> str:
    rows: list[str] = []
    for row in table.rows:  # type: ignore[attr-defined]
        cells = [(cell.text or "").strip() for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)
