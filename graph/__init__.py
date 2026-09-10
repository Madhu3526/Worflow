"""Workflow graph construction from ActionGroups + discourse edges."""

from __future__ import annotations

import logging
import re

import networkx as nx

from workflow.sanitize import looks_like_description
from config import AppConfig, get_config
from models import (
    ActionGroup,
    ActionRecord,
    DiscourseRelation,
    EvidenceSource,
    SequenceMarkerEdge,
    Workflow,
    WorkflowEdge,
    WorkflowNode,
)

logger = logging.getLogger(__name__)


class WorkflowGraphBuilder:
    """Build a directed NetworkX graph per ActionGroup."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def build(self, group: ActionGroup) -> tuple[Workflow, nx.DiGraph]:
        nodes = self._make_nodes(group.actions)
        node_by_action = {n.action_id: n for n in nodes if n.action_id}
        edges: list[WorkflowEdge] = []

        if self.config.graph.use_document_order:
            edges.extend(self._document_order_edges(nodes))

        if self.config.graph.use_numbered_steps:
            edges.extend(self._numbered_edges(group.actions, node_by_action))

        if self.config.graph.use_discourse_edges:
            edges.extend(self._discourse_edges(group.discourse_edges, nodes, node_by_action))

        edges = _dedupe_edges(edges)
        graph = self.to_networkx(nodes, edges)
        ordered_nodes = self._order_nodes(nodes, graph, group.actions)

        pages = sorted({n.page for n in ordered_nodes if n.page is not None})
        if not pages and group.pages:
            pages = list(group.pages)
        elif not pages and group.page_start is not None:
            pages = list(range(group.page_start, (group.page_end or group.page_start) + 1))

        sections: list[str] = []
        for n in ordered_nodes:
            if n.section and n.section not in sections:
                sections.append(n.section)

        evidence = [
            EvidenceSource(
                document=group.source_name,
                page=a.page,
                section=a.section,
                sentence=a.sentence,
                chunk_id=a.chunk_id,
                action_id=a.action_id,
            )
            for a in group.actions
        ]

        workflow = Workflow(
            document=group.source_name,
            document_id=group.document_id,
            pages=pages,
            sections=sections,
            nodes=ordered_nodes,
            edges=edges,
            evidence_sources=evidence,
            context_notes=list(group.context_notes),
            title_hint=group.title_hint,
            group_id=group.group_id,
            conditional_branches=list(group.conditional_branches),
            metadata={
                "boundary_confidence": group.boundary_confidence,
                "action_count": len(group.actions),
            },
        )
        logger.debug(
            "Built workflow graph: %d nodes, %d edges (%s)",
            len(ordered_nodes),
            len(edges),
            group.title_hint,
        )
        return workflow, graph

    def build_many(self, groups: list[ActionGroup]) -> list[tuple[Workflow, nx.DiGraph]]:
        return [self.build(group) for group in groups]

    def to_networkx(self, nodes: list[WorkflowNode], edges: list[WorkflowEdge]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in nodes:
            graph.add_node(
                node.node_id,
                label=node.label(),
                action=node.action,
                object=node.object,
                actor=node.actor,
                order=node.order,
                page=node.page,
            )
        for edge in edges:
            if edge.source in graph and edge.target in graph:
                graph.add_edge(
                    edge.source,
                    edge.target,
                    relation=edge.relation,
                    weight=edge.weight,
                    evidence=edge.evidence,
                )
        return graph

    def _make_nodes(self, actions: list[ActionRecord]) -> list[WorkflowNode]:
        prioritized = sorted(
            actions,
            key=lambda a: (
                0 if a.step_number is not None else 1,
                a.page or 0,
                a.step_number if a.step_number is not None else a.order_index,
                a.order_index,
            ),
        )
        nodes: list[WorkflowNode] = []
        seen_norm: set[str] = set()

        for action in prioritized:
            if action.sentence and looks_like_description(action.sentence):
                continue
            if looks_like_description(action.label()):
                continue
            # Prefer concise numbered-step labels over verbose paraphrases
            if action.step_number is None and any(
                other.step_number is not None and _labels_overlap(action.label(), other.label())
                for other in prioritized
            ):
                continue

            norm = _normalize_node_label(action.label())
            if not norm or norm in seen_norm:
                continue
            if any(_is_subsumed(norm, existing) for existing in seen_norm):
                continue

            # Drop previously kept labels that this one subsumes
            drop_ids = {
                n.node_id
                for n in nodes
                if _is_subsumed(_normalize_node_label(n.label()), norm)
            }
            if drop_ids:
                nodes = [n for n in nodes if n.node_id not in drop_ids]
                seen_norm = {_normalize_node_label(n.label()) for n in nodes}

            seen_norm.add(norm)
            nodes.append(
                WorkflowNode(
                    node_id=f"n_{action.action_id}",
                    actor=action.actor,
                    action=action.action,
                    object=action.object,
                    order=len(nodes),
                    page=action.page,
                    section=action.section,
                    sentence=action.sentence,
                    action_id=action.action_id,
                )
            )

        # Sort by step number when available, else original order
        step_lookup = {
            a.action_id: a.step_number
            for a in prioritized
            if a.step_number is not None
        }

        def sort_key(node: WorkflowNode) -> tuple[int, int]:
            step = step_lookup.get(node.action_id or "", None)
            if step is not None:
                return (step, node.page or 0)
            return (10_000 + node.order, node.page or 0)

        nodes.sort(key=sort_key)
        return [n.model_copy(update={"order": i}) for i, n in enumerate(nodes)]

    def _document_order_edges(self, nodes: list[WorkflowNode]) -> list[WorkflowEdge]:
        edges: list[WorkflowEdge] = []
        ordered = sorted(nodes, key=lambda n: n.order)
        for left, right in zip(ordered, ordered[1:]):
            edges.append(
                WorkflowEdge(
                    source=left.node_id,
                    target=right.node_id,
                    relation="NEXT",
                    weight=0.5,
                    evidence="document_order",
                )
            )
        return edges

    def _numbered_edges(
        self,
        actions: list[ActionRecord],
        node_by_action: dict[str, WorkflowNode],
    ) -> list[WorkflowEdge]:
        edges: list[WorkflowEdge] = []
        numbered = [a for a in actions if a.step_number is not None and a.action_id in node_by_action]
        numbered.sort(key=lambda a: (a.step_number or 0, a.order_index))
        for left, right in zip(numbered, numbered[1:]):
            src = node_by_action.get(left.action_id)
            tgt = node_by_action.get(right.action_id)
            if src and tgt:
                edges.append(
                    WorkflowEdge(
                        source=src.node_id,
                        target=tgt.node_id,
                        relation="NEXT",
                        weight=1.0,
                        evidence="numbered_step",
                    )
                )
        return edges

    def _discourse_edges(
        self,
        discourse: list[SequenceMarkerEdge],
        nodes: list[WorkflowNode],
        node_by_action: dict[str, WorkflowNode],
    ) -> list[WorkflowEdge]:
        edges: list[WorkflowEdge] = []
        for marker in discourse:
            src_node = node_by_action.get(marker.source_action_id or "")
            tgt_node = node_by_action.get(marker.target_action_id or "")
            if not src_node:
                src_node = _fuzzy_node(marker.source_phrase, nodes)
            if not tgt_node:
                tgt_node = _fuzzy_node(marker.target_phrase, nodes)
            if not src_node or not tgt_node or src_node.node_id == tgt_node.node_id:
                continue

            relation = marker.relation.value
            edges.append(
                WorkflowEdge(
                    source=src_node.node_id,
                    target=tgt_node.node_id,
                    relation=relation,
                    weight=1.0,
                    evidence=marker.marker_text,
                )
            )
        return edges

    def _order_nodes(
        self,
        nodes: list[WorkflowNode],
        graph: nx.DiGraph,
        actions: list[ActionRecord],
    ) -> list[WorkflowNode]:
        """Prefer document/step order; refine only with numbered-step skeleton."""
        if not nodes:
            return []

        action_lookup = {a.action_id: a for a in actions if a.action_id}

        def get_pos(node: WorkflowNode) -> tuple[int, int, int]:
            act = action_lookup.get(node.action_id or "")
            if act:
                return (act.page or 0, act.paragraph or 0, act.order_index)
            return (node.page or 0, 0, node.order)

        base = sorted(nodes, key=get_pos)
        try:
            strong = nx.DiGraph()
            strong.add_nodes_from(graph.nodes(data=True))
            for u, v, data in graph.edges(data=True):
                if data.get("evidence") == "numbered_step" and float(data.get("weight", 0)) >= 0.9:
                    strong.add_edge(u, v)

            # Use our stable_topological_sort to preserve source position for unconstrained nodes
            order_ids = stable_topological_sort(strong, nodes, action_lookup)
            by_id = {n.node_id: n for n in nodes}
            ordered: list[WorkflowNode] = []
            seen: set[str] = set()
            for node_id in order_ids:
                node = by_id.get(node_id)
                if node and node_id not in seen:
                    ordered.append(node.model_copy(update={"order": len(ordered)}))
                    seen.add(node_id)
            for node in base:
                if node.node_id not in seen:
                    ordered.append(node.model_copy(update={"order": len(ordered)}))
                    seen.add(node.node_id)

            # Debug print of each node's source position alongside its final position
            for idx, node in enumerate(ordered):
                pos = get_pos(node)
                logger.debug("Ordered Node %d: label='%s' (page=%d, para=%d, sent_idx=%d)",
                             idx, node.label(), pos[0], pos[1], pos[2])

            return ordered
        except nx.NetworkXException:
            pass

        ordered_base = [n.model_copy(update={"order": i}) for i, n in enumerate(base)]
        for idx, node in enumerate(ordered_base):
            pos = get_pos(node)
            logger.debug("Base Ordered Node %d: label='%s' (page=%d, para=%d, sent_idx=%d)",
                         idx, node.label(), pos[0], pos[1], pos[2])
        return ordered_base


def _normalize_node_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_subsumed(short: str, long: str) -> bool:
    if not short or not long or short == long:
        return False
    words_short = set(short.split())
    words_long = set(long.split())
    if words_short.issubset(words_long) and abs(len(long) - len(short)) < 24:
        return True
    return short in long and abs(len(long) - len(short)) < 24


def _labels_overlap(a: str, b: str) -> bool:
    na, nb = _normalize_node_label(a), _normalize_node_label(b)
    if not na or not nb:
        return False
    words_a = set(na.split())
    words_b = set(nb.split())
    if words_a.issubset(words_b) or words_b.issubset(words_a):
        return True
    return na == nb or na in nb or nb in na


def _fuzzy_node(phrase: str, nodes: list[WorkflowNode]) -> WorkflowNode | None:
    phrase_l = phrase.lower()
    best: WorkflowNode | None = None
    best_score = 0
    for node in nodes:
        score = 0
        if node.action.lower() in phrase_l:
            score += 2
        if node.object and node.object.lower() in phrase_l:
            score += 2
        label = node.label().lower()
        if label in phrase_l or phrase_l in label:
            score += 3
        if score > best_score:
            best_score = score
            best = node
    return best if best_score >= 3 else None


def _dedupe_edges(edges: list[WorkflowEdge]) -> list[WorkflowEdge]:
    best: dict[tuple[str, str, str], WorkflowEdge] = {}
    for edge in edges:
        key = (edge.source, edge.target, edge.relation)
        existing = best.get(key)
        if existing is None or edge.weight > existing.weight:
            best[key] = edge
    return list(best.values())


def graph_to_pyvis_html(graph: nx.DiGraph, height: str = "600px", width: str = "100%") -> str:
    """Render a NetworkX graph to pyvis HTML for Streamlit embedding."""
    from pyvis.network import Network

    net = Network(
        height=height, width=width, directed=True, notebook=False, cdn_resources="in_line"
    )
    net.barnes_hut()
    for node_id, data in graph.nodes(data=True):
        label = str(data.get("label") or node_id)
        title = f"{label}<br/>page={data.get('page')}<br/>order={data.get('order')}"
        net.add_node(node_id, label=label, title=title)
    for source, target, data in graph.edges(data=True):
        net.add_edge(
            source,
            target,
            title=str(data.get("relation", "")),
            label=str(data.get("relation", "")),
        )
    return net.generate_html()


def stable_topological_sort(
    graph: nx.DiGraph,
    nodes: list[WorkflowNode],
    action_lookup: dict[str, ActionRecord],
) -> list[str]:
    # Compute in-degrees
    in_degree = {u: 0 for u in graph.nodes}
    for u, v in graph.edges:
        in_degree[v] += 1

    # Helper to get source position key
    def get_pos(node_id: str) -> tuple[int, int, int]:
        node = next((n for n in nodes if n.node_id == node_id), None)
        if node:
            act = action_lookup.get(node.action_id or "")
            if act:
                return (act.page or 0, act.paragraph or 0, act.order_index)
            return (node.page or 0, 0, node.order)
        return (0, 0, 0)

    # Find all start nodes (in-degree 0)
    start_nodes = [u for u, deg in in_degree.items() if deg == 0]
    # Sort them by source position
    start_nodes.sort(key=get_pos)

    ordered_ids: list[str] = []
    while start_nodes:
        # Pop the node with the earliest source position
        curr = start_nodes.pop(0)
        ordered_ids.append(curr)

        # For each successor, decrement in-degree
        for succ in list(graph.successors(curr)):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                # Add to start_nodes and keep it sorted
                start_nodes.append(succ)
                start_nodes.sort(key=get_pos)

    # If there are cycle nodes (should not happen in DAG, but just in case),
    # append any remaining nodes in graph
    for u in graph.nodes:
        if u not in ordered_ids:
            ordered_ids.append(u)

    return ordered_ids


# Re-export for callers that import from graph (see workflow.sanitize for implementation).
from workflow.sanitize import is_spurious_workflow_node, sanitize_workflow  # noqa: E402,F401
