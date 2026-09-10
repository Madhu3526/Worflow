"""Semantic chunking on headings, numbered steps, bullets, and tables only."""

from __future__ import annotations

import logging
import re

from config import AppConfig, get_config
from models import Chunk, DocumentContent, TextUnit

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Split documents on semantic boundaries — never by fixed character count."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._step_re = re.compile(
            self.config.chunking.numbered_step_pattern,
            re.IGNORECASE | re.MULTILINE,
        )

    def chunk(self, document: DocumentContent) -> list[Chunk]:
        chunks: list[Chunk] = []
        order = 0
        current_section = None

        for unit in document.units:
            if unit.unit_type == "heading":
                current_section = unit.text
                chunks.append(self._to_chunk(document, unit, order, current_section))
                order += 1
                continue

            if unit.unit_type == "table":
                if self.config.chunking.preserve_tables:
                    chunks.append(self._to_chunk(document, unit, order, current_section or unit.section))
                    order += 1
                continue

            # Split paragraphs that contain multiple numbered/bullet steps
            sub_units = self._split_on_semantic_boundaries(unit)
            for sub in sub_units:
                section = current_section or sub.section or unit.section
                chunks.append(self._to_chunk(document, sub, order, section))
                order += 1

        logger.info(
            "Chunked %s into %d semantic chunks",
            document.source_name,
            len(chunks),
        )
        return chunks

    def _split_on_semantic_boundaries(self, unit: TextUnit) -> list[TextUnit]:
        text = unit.text.strip()
        if not text:
            return []

        # If entire unit is already a step/list item, keep as one chunk
        if unit.unit_type in {"step", "list_item"}:
            return [unit]

        matches = list(self._step_re.finditer(text))
        if len(matches) <= 1 and "\n" not in text:
            return [unit]

        # Split at each numbered/bullet boundary when multiple exist
        parts: list[TextUnit] = []
        if matches and (len(matches) > 1 or matches[0].start() == 0):
            starts = [m.start() for m in matches]
            starts.append(len(text))
            for i in range(len(starts) - 1):
                segment = text[starts[i] : starts[i + 1]].strip()
                if not segment:
                    continue
                parts.append(
                    TextUnit(
                        text=segment,
                        page=unit.page,
                        section=unit.section,
                        paragraph=unit.paragraph,
                        unit_type="step" if self._step_re.match(segment) else unit.unit_type,
                        metadata=dict(unit.metadata),
                    )
                )
            if parts:
                return parts

        # Fall back: split on blank-line separated blocks (still semantic, not char count)
        blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
        if len(blocks) <= 1:
            return [unit]
        return [
            TextUnit(
                text=block,
                page=unit.page,
                section=unit.section,
                paragraph=unit.paragraph,
                unit_type=_infer_type(block, self._step_re),
                metadata=dict(unit.metadata),
            )
            for block in blocks
        ]

    def _to_chunk(
        self,
        document: DocumentContent,
        unit: TextUnit,
        order: int,
        section: str | None,
    ) -> Chunk:
        return Chunk(
            text=unit.text,
            page=unit.page,
            section=section,
            paragraph=unit.paragraph,
            unit_type=unit.unit_type,
            document_id=document.document_id,
            source_name=document.source_name,
            order_index=order,
            metadata=dict(unit.metadata),
        )


def _infer_type(text: str, step_re: re.Pattern[str]) -> str:
    if step_re.match(text):
        if re.match(r"^\s*[-*•▪]", text):
            return "list_item"
        return "step"
    return "paragraph"
