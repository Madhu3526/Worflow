"""Orphan numbered lines like '6.' must not become workflow actions."""

from __future__ import annotations

from actions import ActionExtractor
from models import Chunk


def test_orphan_step_number_not_extracted():
    chunk = Chunk(
        text="6.\nOnce welding is complete, perform dye penetrant testing to confirm crack closure.",
        page=3,
        section="4.5  Engine Mount Bracket Weld Repair",
        document_id="doc_test",
        source_name="manual.docx",
        order_index=0,
        unit_type="paragraph",
    )
    result = ActionExtractor().extract([chunk])
    labels = [a.label().lower() for a in result.actions]
    assert not any(label == "perform 6" for label in labels)
    assert any("dye penetrant" in label for label in labels)
