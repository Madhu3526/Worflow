"""Phase 2 tests: coref, action extraction, sequence markers."""

from __future__ import annotations

from pathlib import Path

from actions import ActionExtractor
from chunking import SemanticChunker
from coref import CoreferenceResolver
from parser import DocumentParserService
from sequence_markers import SequenceMarkerExtractor

FIXTURE = Path(__file__).parent / "fixtures" / "weld_repair.txt"


def _pipeline_partial():
    doc = DocumentParserService().parse(FIXTURE)
    chunks = SemanticChunker().chunk(doc)
    chunks = CoreferenceResolver().resolve_chunks(chunks)
    actions = ActionExtractor().extract(chunks).actions
    edges = SequenceMarkerExtractor().extract(chunks, actions)
    return chunks, actions, edges


def test_coref_sets_resolved_text():
    doc = DocumentParserService().parse(FIXTURE)
    chunks = SemanticChunker().chunk(doc)
    resolver = CoreferenceResolver()
    resolved = resolver.resolve_chunks(chunks)
    assert all(c.resolved_text for c in resolved)
    sample = "Inspect the surface. Clean it thoroughly."
    out = resolver.resolve_text(sample)
    assert isinstance(out, str) and out


def test_action_extraction_finds_weld_steps():
    _, actions, _ = _pipeline_partial()
    assert len(actions) >= 5
    verbs = {a.action.lower() for a in actions}
    # Should capture several of the procedure verbs
    expected = {"inspect", "remove", "grind", "preheat", "apply", "perform"}
    assert len(verbs & expected) >= 3
    for action in actions:
        assert action.sentence
        assert action.document_id
        assert action.page is not None


def test_sequence_markers_emit_typed_edges():
    _, actions, edges = _pipeline_partial()
    assert edges
    relations = {e.relation.value for e in edges}
    assert relations & {"AFTER", "BEFORE", "THEN", "IF"}
    # Marker text should be present
    assert any(e.marker_text for e in edges)


def test_list_separators_split_into_separate_actions():
    from actions import ActionExtractor, split_list_separators
    from models import Chunk

    # Unit-level splitter
    parts = split_list_separators(
        "Inspect crack ↓ Remove damaged material + Grind surface → Preheat"
    )
    assert parts == [
        "Inspect crack",
        "Remove damaged material",
        "Grind surface",
        "Preheat",
    ]

    chunk = Chunk(
        text=(
            "Inspect crack ↓ Remove damaged material + Grind surface + Preheat component\n"
            "Apply weld passes → Ultrasonic testing → Stress relief"
        ),
        page=1,
        section="7.2 Major Weld Repair Procedure",
        document_id="doc_list",
        source_name="list_demo.txt",
        order_index=0,
        unit_type="paragraph",
    )
    chunk.resolved_text = chunk.text
    actions = ActionExtractor().extract([chunk]).actions
    labels = [a.label().lower() for a in actions]
    assert len(actions) >= 5
    # Must not remain a single fused action spanning the whole list
    assert not any("↓" in (a.sentence or "") or " + " in (a.sentence or "") for a in actions)
    joined = " | ".join(labels)
    assert "inspect" in joined
    assert "remove" in joined
    assert "grind" in joined
    assert "apply" in joined
    assert "ultrasonic" in joined
