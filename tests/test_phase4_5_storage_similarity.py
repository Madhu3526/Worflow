"""Phase 4–5 tests: storage, summaries, embeddings, similarity, evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import AppConfig, PathsConfig
from embeddings import EmbeddingService, FaissWorkflowIndex
from models import EvidenceSource, Workflow, WorkflowEdge, WorkflowNode
from retrieval import EvidenceRetriever, QueryEngine, format_explanation
from similarity import Deduplicator, WorkflowSimilarityEngine
from storage import WorkflowStore
from workflow import summarize


@pytest.fixture()
def tmp_config(tmp_path: Path) -> AppConfig:
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
    return cfg


def _sample_workflow(doc: str = "sample.txt", title: str = "Major Weld Repair") -> Workflow:
    nodes = [
        WorkflowNode(node_id="n1", action="inspect", object="crack", order=0, page=1, sentence="Inspect crack"),
        WorkflowNode(node_id="n2", action="remove", object="damaged material", order=1, page=1, sentence="Remove damaged material."),
        WorkflowNode(node_id="n3", action="grind", object="surface", order=2, page=2, sentence="Grind the surface."),
        WorkflowNode(node_id="n4", action="preheat", object="component", order=3, page=2, sentence="Preheat the component."),
        WorkflowNode(node_id="n5", action="apply", object="weld passes", order=4, page=2, sentence="Apply weld passes."),
        WorkflowNode(node_id="n6", action="perform", object="ultrasonic testing", order=5, page=3, sentence="Perform ultrasonic testing."),
        WorkflowNode(node_id="n7", action="perform", object="stress relief", order=6, page=3, sentence="Perform stress relief."),
    ]
    edges = [
        WorkflowEdge(source="n1", target="n2", relation="NEXT"),
        WorkflowEdge(source="n2", target="n3", relation="NEXT"),
        WorkflowEdge(source="n3", target="n4", relation="NEXT"),
        WorkflowEdge(source="n4", target="n5", relation="NEXT"),
        WorkflowEdge(source="n5", target="n6", relation="NEXT"),
        WorkflowEdge(source="n6", target="n7", relation="NEXT"),
    ]
    evidence = [
        EvidenceSource(document=doc, page=n.page, sentence=n.sentence or n.label())
        for n in nodes
    ]
    return Workflow(
        document=doc,
        document_id="doc_sample",
        pages=[1, 2, 3],
        sections=[title],
        nodes=nodes,
        edges=edges,
        evidence_sources=evidence,
        title_hint=title,
    )


def test_template_summary_is_deterministic():
    wf = _sample_workflow()
    s1 = summarize(wf)
    s2 = summarize(wf)
    assert s1 == s2
    assert "→" in s1
    assert "inspect" in s1.lower()


def test_storage_roundtrip(tmp_config: AppConfig):
    store = WorkflowStore(tmp_config)
    wf = _sample_workflow()
    wf.summary = summarize(wf)
    store.save(wf)
    loaded = store.load(wf.workflow_id)
    assert loaded is not None
    assert loaded.summary == wf.summary
    assert len(store.list_meta()) == 1


def test_faiss_index_and_search(tmp_config: AppConfig):
    embedder = EmbeddingService(tmp_config)
    index = FaissWorkflowIndex(tmp_config, embedder)
    wf = _sample_workflow()
    wf.summary = summarize(wf)
    count = index.rebuild([wf])
    assert count > 0
    hits = index.search("Major Weld Repair procedure ultrasonic testing", top_k=3)
    assert hits
    assert hits[0]["workflow_id"] == wf.workflow_id


def test_similarity_sub_scores(tmp_config: AppConfig):
    engine = WorkflowSimilarityEngine(tmp_config, EmbeddingService(tmp_config))
    left = _sample_workflow()
    right = _sample_workflow(title="Weld Repair Process")
    # Drop last node to create a partial match
    right.nodes = right.nodes[:-1]
    scores = engine.score(left, right)
    assert scores.comparison_type.value == "workflow_to_workflow"
    assert 0.0 <= scores.set_overlap <= 1.0
    assert scores.sequence_alignment is not None
    assert scores.structural is not None
    assert 0.0 <= scores.sequence_alignment <= 1.0
    assert 0.0 <= scores.structural <= 1.0
    assert 0.0 <= scores.blended <= 1.0
    # High overlap expected
    assert scores.set_overlap >= 0.5
    assert scores.blended >= 0.4


def test_dedup_flags_near_duplicates(tmp_config: AppConfig):
    embedder = EmbeddingService(tmp_config)
    deduper = Deduplicator(tmp_config, embedder)
    a = _sample_workflow()
    a.summary = summarize(a)
    b = _sample_workflow(doc="other.txt")
    b.summary = summarize(b)
    match = deduper.find_duplicate(b, [a])
    assert match is not None
    assert match[1] >= tmp_config.deduplication.cosine_threshold


def test_explainable_answer_format(tmp_config: AppConfig):
    store = WorkflowStore(tmp_config)
    wf = _sample_workflow()
    wf.summary = summarize(wf)
    store.save(wf)
    index = FaissWorkflowIndex(tmp_config, EmbeddingService(tmp_config))
    index.rebuild([wf])
    engine = QueryEngine(tmp_config, store=store, index=index)
    answer = engine.answer("Where is the Major Weld Repair procedure?")
    assert answer.matched_workflow_id == wf.workflow_id
    assert answer.scores is not None
    assert "Confidence:" in answer.explanation
    assert "set overlap" in answer.explanation
    assert "semantic similarity" in answer.explanation
    evidence = EvidenceRetriever(tmp_config).retrieve(wf)
    assert evidence
    text = format_explanation(
        "Where is the Major Weld Repair procedure?",
        wf,
        answer.scores,
        evidence,
    )
    assert "Matched Workflow:" in text
