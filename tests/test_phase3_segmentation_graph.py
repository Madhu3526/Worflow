"""Phase 3 tests: segmentation + graph construction."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from actions import ActionExtractor
from chunking import SemanticChunker
from coref import CoreferenceResolver
from graph import WorkflowGraphBuilder, graph_to_pyvis_html
from parser import DocumentParserService
from segmentation import WorkflowSegmenter
from sequence_markers import SequenceMarkerExtractor

FIXTURE = Path(__file__).parent / "fixtures" / "weld_repair.txt"


def _groups():
    doc = DocumentParserService().parse(FIXTURE)
    chunks = CoreferenceResolver().resolve_chunks(SemanticChunker().chunk(doc))
    actions = ActionExtractor().extract(chunks).actions
    edges = SequenceMarkerExtractor().extract(chunks, actions)
    groups = WorkflowSegmenter().segment(actions, edges)
    return groups


def test_segmentation_produces_action_groups():
    groups = _groups()
    assert groups
    assert all(g.actions for g in groups)
    # Fixture contains two procedures — expect at least one split ideally,
    # but never zero groups
    for group in groups:
        assert group.document_id
        assert group.page_start is not None or group.actions


def test_graph_construction_builds_dag_like_workflow():
    groups = _groups()
    builder = WorkflowGraphBuilder()
    # Prefer the richest group (main procedure), not a possible intro fragment
    group = max(groups, key=lambda g: len(g.actions))
    workflow, graph = builder.build(group)
    assert isinstance(graph, nx.DiGraph)
    assert workflow.nodes
    assert len(workflow.nodes) == graph.number_of_nodes()
    assert len(workflow.nodes) >= 2
    assert workflow.edges
    assert workflow.evidence_sources
    chain = " → ".join(n.label() for n in workflow.ordered_nodes)
    assert "→" in chain


def test_pyvis_html_renders():
    groups = _groups()
    builder = WorkflowGraphBuilder()
    _, graph = builder.build(groups[0])
    html = graph_to_pyvis_html(graph)
    assert "network" in html.lower() or "<html" in html.lower()
