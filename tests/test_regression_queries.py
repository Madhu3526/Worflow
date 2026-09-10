"""Regression and verification tests for WORKFLOW_TO_WORKFLOW scoring visibility."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config import AppConfig, PathsConfig
from models import ComparisonType
from pipeline import WorkflowDiscoveryPipeline
from retrieval import QueryEngine
from similarity import Deduplicator, WorkflowSimilarityEngine
from workflow.sanitize import sanitize_workflow

AIRCRAFT = Path(__file__).parent / "fixtures" / "aircraft_manual.txt"
PROMPT = Path(__file__).parent / "fixtures" / "prompt_engineering.txt"


@pytest.fixture()
def pipeline(tmp_path: Path) -> WorkflowDiscoveryPipeline:
    paths = PathsConfig(
        data_dir=str(tmp_path / "data"),
        workflows_dir=str(tmp_path / "data" / "workflows"),
        indexes_dir=str(tmp_path / "data" / "indexes"),
        logs_dir=str(tmp_path / "logs"),
        meta_file=str(tmp_path / "data" / "workflows_meta.json"),
        faiss_index=str(tmp_path / "data" / "indexes" / "workflows.index"),
        faiss_meta=str(tmp_path / "data" / "indexes" / "faiss_meta.json"),
    )
    cfg = AppConfig(paths=paths)
    cfg.ensure_directories()
    return WorkflowDiscoveryPipeline(cfg)


def _process_both(pipeline: WorkflowDiscoveryPipeline) -> None:
    pipeline.process_file(AIRCRAFT)
    pipeline.process_file(PROMPT)


def _find_workflow(workflows, *needles: str):
    for wf in workflows:
        title = (wf.title_hint or "").lower()
        if any(n.lower() in title for n in needles):
            return wf
    raise AssertionError(f"No workflow matching {needles}")


def test_query_major_weld_repair_procedure(pipeline: WorkflowDiscoveryPipeline):
    _process_both(pipeline)
    engine = QueryEngine(
        pipeline.config,
        store=pipeline.store,
        index=pipeline.index,
    )
    answer = engine.answer("Where is the Major Weld Repair procedure?")

    assert answer.matched_workflow_id
    assert answer.candidates
    assert answer.scores is not None
    assert answer.scores.set_overlap > 0.0
    assert answer.scores.comparison_type == ComparisonType.QUERY_TO_WORKFLOW
    assert answer.scores.semantic_similarity is not None
    assert answer.confidence >= 0.55
    assert "semantic similarity" in answer.explanation
    assert "sequence alignment" not in answer.explanation

    top = answer.candidates[0].workflow
    assert "4.3" in (top.title_hint or "") or "Fuselage" in (top.title_hint or "")
    assert answer.matched_pages in {"1", "1–1"}
    assert len(top.ordered_nodes) == 8

    chain = answer.matched_chain.lower()
    expected_actions = ["inspect", "remove", "grind", "preheat", "apply", "cool", "perform"]
    for action in expected_actions:
        assert action in chain
    assert "differs from the fuselage skin repair" not in chain
    assert "applies to hairline cracks" not in chain
    assert "repeat steps" not in chain

    assert len(top.conditional_branches) == 1
    branch_text = top.conditional_branches[0].sentence.lower()
    assert "ultrasonic testing" in branch_text
    assert "repeat" in branch_text

    if len(answer.candidates) > 1:
        second = answer.candidates[1].workflow
        assert "4.5" in (second.title_hint or "") or second.workflow_id != top.workflow_id
        assert answer.candidates[0].scores.blended > answer.candidates[1].scores.blended


def test_query_professional_prompt_process(pipeline: WorkflowDiscoveryPipeline):
    _process_both(pipeline)
    engine = QueryEngine(
        pipeline.config,
        store=pipeline.store,
        index=pipeline.index,
    )
    answer = engine.answer("What is the process for building a professional prompt?")

    assert answer.matched_workflow_id
    assert answer.candidates

    matched = next(
        (
            c
            for c in answer.candidates
            if c.workflow.title_hint
            and (
                "Anatomy" in c.workflow.title_hint
                or "Universal Prompt Template" in c.workflow.title_hint
                or "Formula" in c.workflow.title_hint
            )
        ),
        answer.candidates[0],
    )
    wf = matched.workflow
    assert len(wf.nodes) >= 8

    formula_wf = next(
        (c.workflow for c in answer.candidates if c.workflow.title_hint and "Formula" in c.workflow.title_hint),
        None,
    )
    if formula_wf:
        assert len(formula_wf.nodes) == 8

    top3_titles = [c.workflow.title_hint or "" for c in answer.candidates[:3]]
    assert any("Universal Prompt Template" in t for t in top3_titles)


def test_workflow_to_workflow_compare_43_vs_45(pipeline: WorkflowDiscoveryPipeline):
    _process_both(pipeline)
    workflows = [sanitize_workflow(wf) for wf in pipeline.store.load_all()]
    wf43 = _find_workflow(workflows, "4.3", "Fuselage")
    wf45 = _find_workflow(workflows, "4.5", "Engine Mount")

    deduper = Deduplicator(
        pipeline.config,
        pipeline.index.embedder,
        WorkflowSimilarityEngine(pipeline.config, pipeline.index.embedder),
    )
    comparison = deduper.compare_pair(wf43, wf45)
    scores = comparison.graph_scores

    assert scores.comparison_type == ComparisonType.WORKFLOW_TO_WORKFLOW
    assert scores.sequence_alignment is not None
    assert scores.structural is not None
    assert scores.set_overlap >= 0.5
    assert scores.sequence_alignment >= 0.5
    assert scores.structural > 0.0
    assert 0.4 <= scores.blended < 0.9
    assert comparison.summary_cosine < comparison.dedup_threshold
    assert comparison.is_duplicate is False
    assert len(wf43.ordered_nodes) == 8
    assert len(wf45.ordered_nodes) >= 6


def test_dedup_logs_workflow_to_workflow_scores(
    pipeline: WorkflowDiscoveryPipeline,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="similarity")
    pipeline.process_file(AIRCRAFT)
    assert any("WORKFLOW_TO_WORKFLOW" in record.message for record in caplog.records)
