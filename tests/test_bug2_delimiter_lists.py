"""Bug 2: Delimiter-separated lists produce N separate action nodes."""

from __future__ import annotations

from actions import ActionExtractor, split_list_separators
from models import Chunk

PROMPT_FORMULA = (
    "Role + Context + Objective + Input + Tasks + Constraints + Output Format + Validation"
)


def test_split_list_separators_produces_eight_segments():
    parts = split_list_separators(PROMPT_FORMULA)
    assert len(parts) == 8
    assert parts[0] == "Role"
    assert parts[-1] == "Validation"


def test_prompt_formula_yields_eight_action_nodes():
    chunk = Chunk(
        text=PROMPT_FORMULA,
        page=1,
        section="One Formula to Remember",
        document_id="doc_prompt",
        source_name="prompt.txt",
        order_index=0,
        unit_type="paragraph",
    )
    chunk.resolved_text = chunk.text
    actions = ActionExtractor().extract([chunk]).actions
    assert len(actions) == 8
    labels = [a.label().lower() for a in actions]
    assert labels == [
        "role",
        "context",
        "objective",
        "input",
        "tasks",
        "constraints",
        "output format",
        "validation",
    ]
