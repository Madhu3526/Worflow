"""Phase 1 tests: document parsing + semantic chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from chunking import SemanticChunker
from parser import DocumentParserService

FIXTURE = Path(__file__).parent / "fixtures" / "weld_repair.txt"


def test_parse_txt_extracts_units_with_metadata():
    parser = DocumentParserService()
    doc = parser.parse(FIXTURE)
    assert doc.source_name == "weld_repair.txt"
    assert doc.page_count == 2
    assert len(doc.units) > 5
    headings = [u for u in doc.units if u.unit_type == "heading"]
    assert headings
    assert any(u.section for u in doc.units)
    for unit in doc.units:
        assert unit.text
        assert unit.page is not None


def test_semantic_chunking_splits_on_steps_not_char_count():
    parser = DocumentParserService()
    chunker = SemanticChunker()
    doc = parser.parse(FIXTURE)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 5
    step_chunks = [c for c in chunks if c.unit_type in {"step", "list_item"}]
    assert step_chunks
    # Every chunk retains source metadata
    for chunk in chunks:
        assert chunk.document_id == doc.document_id
        assert chunk.source_name == doc.source_name
        assert chunk.text


def test_unsupported_extension_raises(tmp_path: Path):
    parser = DocumentParserService()
    bogus = tmp_path / "nope.xyz"
    bogus.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        parser.parse(bogus)
