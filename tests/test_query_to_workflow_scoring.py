"""Query-to-workflow scoring must not penalize short NL questions with graph metrics."""

from __future__ import annotations

import pytest

from config import AppConfig, PathsConfig
from embeddings import EmbeddingService
from models import ComparisonType, Workflow, WorkflowNode
from similarity import WorkflowSimilarityEngine
from workflow import summarize


@pytest.fixture()
def engine(tmp_path):
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
    embedder = EmbeddingService(cfg)
    return WorkflowSimilarityEngine(cfg, embedder), cfg


def _query_workflow() -> Workflow:
    return Workflow(
        document="__query__",
        document_id="query",
        nodes=[
            WorkflowNode(node_id="q1", action="major", object="weld repair", order=0),
            WorkflowNode(node_id="q2", action="remove", object=None, order=1),
            WorkflowNode(node_id="q3", action="grind", object=None, order=2),
        ],
    )


def _candidate_workflow() -> Workflow:
    wf = Workflow(
        document="manual.docx",
        document_id="doc1",
        title_hint="4.3 Fuselage Skin Crack — Major Weld Repair",
        nodes=[
            WorkflowNode(node_id="n1", action="remove", object="damaged material", order=0),
            WorkflowNode(node_id="n2", action="grind", object="surface", order=1),
            WorkflowNode(node_id="n3", action="preheat", object="component", order=2),
            WorkflowNode(node_id="n4", action="apply", object="weld passes", order=3),
            WorkflowNode(node_id="n5", action="perform", object="ultrasonic testing", order=4),
            WorkflowNode(node_id="n6", action="perform", object="stress relief", order=5),
            WorkflowNode(node_id="n7", action="inspect", object="repair", order=6),
            WorkflowNode(node_id="n8", action="document", object="repair", order=7),
        ],
    )
    wf.summary = summarize(wf)
    return wf


def test_query_to_workflow_uses_semantic_and_set_overlap_only(engine):
    sim_engine, cfg = engine
    query = _query_workflow()
    candidate = _candidate_workflow()
    query_text = "Where is the Major Weld Repair procedure?"

    set_overlap = sim_engine.set_overlap(query.ordered_nodes, candidate.ordered_nodes)
    sequence = sim_engine.sequence_alignment(query.ordered_nodes, candidate.ordered_nodes)
    structural = sim_engine.structural_similarity(query, candidate)
    w = cfg.similarity
    old_blended = (
        w.set_overlap_weight * set_overlap
        + w.sequence_alignment_weight * sequence
        + w.structural_weight * structural
    )

    scores = sim_engine.score(
        query,
        candidate,
        comparison_type=ComparisonType.QUERY_TO_WORKFLOW,
        query_text=query_text,
    )

    assert scores.comparison_type == ComparisonType.QUERY_TO_WORKFLOW
    assert scores.sequence_alignment is None
    assert scores.structural is None
    assert scores.semantic_similarity is not None
    assert scores.set_overlap > 0.0

    expected = (
        w.query_semantic_weight * scores.semantic_similarity
        + w.query_set_overlap_weight * scores.set_overlap
    )
    assert scores.blended == pytest.approx(expected, abs=0.0001)
    assert scores.blended > old_blended
