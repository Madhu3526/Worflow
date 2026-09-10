"""Tests for causative action extraction and segmentation continuity fixes."""

from __future__ import annotations

from pathlib import Path

from actions import ActionExtractor
from config import get_config
from graph import WorkflowGraphBuilder
from models import Chunk
from segmentation import WorkflowSegmenter
from sequence_markers import SequenceMarkerExtractor


def test_causative_extraction_pattern():
    config = get_config()
    config.coref.resolve_text = False
    extractor = ActionExtractor(config)
    chunk = Chunk(
        chunk_id="chunk_test",
        text="Once welding is complete, allow the component to air cool for 30 minutes.",
        page=1,
        section="4.3  Fuselage Skin Crack — Field Repair Procedure",
        document_id="doc_test",
        source_name="manual.txt",
        order_index=0,
    )
    actions = extractor.extract([chunk]).actions
    assert len(actions) == 1
    action = actions[0]
    assert action.action == "cool"
    assert action.object == "component"
    assert action.metadata.get("modifier") == "air"
    assert action.metadata.get("duration") == "30 minutes"


def test_end_to_end_8_sentence_procedure():
    text = """4.3  Fuselage Skin Crack — Field Repair Procedure

1. Inspect the crack
The certified structures technician shall inspect the crack and mark its extent.

2. Remove damaged material
Remove damaged material from the affected area.

3. Grind surface
Grind the surrounding surface to bright metal.

4. Preheat
Before welding, preheat the component to the specified temperature.

5. Apply weld passes
Apply multiple weld passes according to the qualified WPS.

6. Once welding is complete, allow the component to air cool for 30 minutes.

7. Following cooling, perform ultrasonic testing to confirm weld integrity.

8. After ultrasonic testing passes, perform a stress relief heat treatment cycle."""

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks = []
    for i, line in enumerate(lines):
        chunks.append(Chunk(
            chunk_id=f"chunk_{i}",
            text=line,
            page=1,
            section="4.3  Fuselage Skin Crack — Field Repair Procedure",
            document_id="doc_test",
            source_name="manual.txt",
            order_index=i,
            unit_type="paragraph" if not line.startswith("4.3") else "heading",
        ))

    config = get_config()
    config.coref.resolve_text = False

    extractor = ActionExtractor(config)
    result = extractor.extract(chunks)

    marker_extractor = SequenceMarkerExtractor(config)
    edges = marker_extractor.extract(chunks, result.actions)

    segmenter = WorkflowSegmenter(config)
    groups = segmenter.segment(result.actions, edges)

    # We expect exactly 1 major group representing the entire procedure
    assert len(groups) == 1
    group = groups[0]

    builder = WorkflowGraphBuilder(config)
    workflow, graph = builder.build(group)

    # Assert exactly 8 nodes in the workflow (representing the 8 steps)
    assert len(workflow.nodes) == 8

    # Check correct order and actions
    expected_actions = ["inspect", "remove", "grind", "preheat", "apply", "cool", "perform", "perform"]
    for i, exp in enumerate(expected_actions):
        assert workflow.ordered_nodes[i].action == exp

    # Check monotonicity of ordering
    last_pos = (-1, -1, -1)
    action_lookup = {a.action_id: a for a in result.actions if a.action_id}
    for node in workflow.ordered_nodes:
        act = action_lookup.get(node.action_id or "")
        if act:
            pos = (act.page or 0, act.paragraph or 0, act.order_index)
            assert pos >= last_pos, f"Non-monotonic node ordering: {node.label()} pos={pos} last={last_pos}"
            last_pos = pos


def test_conditional_exception_handling():
    text = """4.3  Fuselage Skin Crack — Field Repair Procedure

1. Inspect the crack
The certified structures technician shall inspect the crack and mark its extent.

2. Remove damaged material
Remove damaged material from the affected area.

3. Grind surface
Grind the surrounding surface to bright metal.

4. Preheat
Before welding, preheat the component to the specified temperature.

5. Apply weld passes
Apply multiple weld passes according to the qualified WPS.

6. Once welding is complete, allow the component to air cool for 30 minutes.

7. Following cooling, perform ultrasonic testing to confirm weld integrity.

If ultrasonic testing indicates incomplete fusion, repeat steps 5–7 before proceeding to stress relief.

8. After ultrasonic testing passes, perform a stress relief heat treatment cycle."""

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks = []
    for i, line in enumerate(lines):
        chunks.append(Chunk(
            chunk_id=f"chunk_{i}",
            text=line,
            page=1,
            section="4.3  Fuselage Skin Crack — Field Repair Procedure",
            document_id="doc_test",
            source_name="manual.txt",
            order_index=i,
            unit_type="paragraph" if not line.startswith("4.3") else "heading",
        ))

    config = get_config()
    config.coref.resolve_text = False

    extractor = ActionExtractor(config)
    result = extractor.extract(chunks)

    # Verify that "indicate incomplete fusion" is NOT extracted as an action record
    for act in result.actions:
        assert "incomplete fusion" not in act.action
        assert "indicat" not in act.action

    marker_extractor = SequenceMarkerExtractor(config)
    edges = marker_extractor.extract(chunks, result.actions)

    segmenter = WorkflowSegmenter(config)
    groups = segmenter.segment(
        result.actions,
        edges,
        context_by_chunk=result.context_by_chunk,
        conditional_branches_by_chunk=result.conditional_branches_by_chunk
    )

    # We expect exactly 1 major group representing the entire procedure
    assert len(groups) == 1
    group = groups[0]

    # Verify the conditional branch is attached to the group
    assert len(group.conditional_branches) == 1
    branch = group.conditional_branches[0]
    assert branch.target_steps == [5, 6, 7]
    assert "incomplete fusion" in branch.sentence

    builder = WorkflowGraphBuilder(config)
    workflow, graph = builder.build(group)

    # Assert exactly 8 nodes in the workflow (not 9)
    assert len(workflow.nodes) == 8

    # Assert that the separate conditional_branches entry exists referencing steps 5-7
    assert len(workflow.conditional_branches) == 1
    assert workflow.conditional_branches[0].target_steps == [5, 6, 7]

    # Test monotonicity of ordering
    last_pos = (-1, -1, -1)
    action_lookup = {a.action_id: a for a in result.actions if a.action_id}
    for node in workflow.ordered_nodes:
        act = action_lookup.get(node.action_id or "")
        if act:
            pos = (act.page or 0, act.paragraph or 0, act.order_index)
            assert pos >= last_pos, f"Non-monotonic node ordering: {node.label()} pos={pos} last={last_pos}"
            last_pos = pos

