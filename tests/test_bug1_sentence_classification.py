"""Bug 1: INSTRUCTION vs DESCRIPTION sentence classification."""

from __future__ import annotations

from actions.sentence_classifier import SentenceKind, classify_sentence


INSTRUCTIONS = [
    "4. Before welding, preheat the bracket to 120°C.",
    "Apply weld passes per the engine mount repair specification.",
]

DESCRIPTIONS = [
    "This procedure applies to hairline cracks found on the engine mount bracket during borescope inspection.",
    "This procedure differs from the fuselage skin repair in preheat temperature and inspection method.",
]


def test_instruction_sentences_classified_correctly():
    for sentence in INSTRUCTIONS:
        assert classify_sentence(sentence) == SentenceKind.INSTRUCTION, sentence


def test_description_sentences_classified_correctly():
    for sentence in DESCRIPTIONS:
        assert classify_sentence(sentence) == SentenceKind.DESCRIPTION, sentence


def test_description_sentences_not_extracted_as_actions():
    from actions import ActionExtractor
    from models import Chunk

    chunk = Chunk(
        text="\n".join(DESCRIPTIONS + INSTRUCTIONS),
        page=2,
        section="4.5  Engine Mount Bracket Weld Repair",
        document_id="doc_test",
        source_name="manual.txt",
        order_index=0,
        unit_type="paragraph",
    )
    chunk.resolved_text = chunk.text
    result = ActionExtractor().extract([chunk])
    assert result.context_by_chunk
    joined = " ".join(result.context_by_chunk.get(chunk.chunk_id, []))
    assert "applies to hairline cracks" in joined
    assert "differs from the fuselage skin repair" in joined
    action_sentences = {a.sentence for a in result.actions}
    assert not any("applies to hairline cracks" in s for s in action_sentences)
    assert not any("differs from the fuselage skin repair" in s for s in action_sentences)
    assert any("preheat the bracket" in a.sentence for a in result.actions)
