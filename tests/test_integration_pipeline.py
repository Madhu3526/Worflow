"""Integration test: full pipeline on fixture document."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import AppConfig, PathsConfig
from pipeline import WorkflowDiscoveryPipeline
from retrieval import QueryEngine


FIXTURE = Path(__file__).parent / "fixtures" / "weld_repair.txt"


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


def test_full_pipeline_discovers_workflows(pipeline: WorkflowDiscoveryPipeline):
    result = pipeline.process_file(FIXTURE)
    assert result.chunks
    assert result.action_count >= 5
    assert result.workflows
    assert all(wf.summary for wf in result.workflows)
    assert all(wf.nodes for wf in result.workflows)
    assert result.indexed_vectors > 0

    engine = QueryEngine(
        pipeline.config,
        store=pipeline.store,
        index=pipeline.index,
    )
    answer = engine.answer("Where is the Major Weld Repair procedure?")
    assert answer.matched_workflow_id
    assert answer.confidence > 0
    assert answer.evidence
    assert answer.scores is not None
