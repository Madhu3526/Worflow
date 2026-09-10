"""Workflow similarity: set overlap + Needleman-Wunsch + graph edit distance."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import networkx as nx
import numpy as np

from config import AppConfig, get_config
from embeddings import EmbeddingService
from graph import WorkflowGraphBuilder
from similarity.canonicalization import cluster_labels_unified, jaccard_cluster_overlap
from models import ComparisonType, SimilarityScores, Workflow, WorkflowNode, WorkflowPairComparison

logger = logging.getLogger(__name__)


class WorkflowSimilarityEngine:
    """Compare workflows (not paragraphs) with explainable sub-scores."""

    def __init__(
        self,
        config: AppConfig | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.config = config or get_config()
        self.embedder = embedder or EmbeddingService(self.config)
        self._graph_builder = WorkflowGraphBuilder(self.config)
        from actions import ActionExtractor

        self._query_extractor = ActionExtractor(self.config)

    def score(
        self,
        left: Workflow,
        right: Workflow,
        *,
        comparison_type: ComparisonType = ComparisonType.WORKFLOW_TO_WORKFLOW,
        query_text: str | None = None,
    ) -> SimilarityScores:
        w = self.config.similarity
        set_overlap = self.set_overlap(left.ordered_nodes, right.ordered_nodes)

        if comparison_type == ComparisonType.QUERY_TO_WORKFLOW:
            semantic = self.semantic_similarity(query_text, right)
            blended = w.query_semantic_weight * semantic + w.query_set_overlap_weight * set_overlap
            return SimilarityScores(
                comparison_type=comparison_type,
                set_overlap=round(set_overlap, 4),
                semantic_similarity=round(semantic, 4),
                blended=round(float(blended), 4),
                weights={
                    "semantic_similarity": w.query_semantic_weight,
                    "set_overlap": w.query_set_overlap_weight,
                },
            )

        sequence = self.sequence_alignment(left.ordered_nodes, right.ordered_nodes)
        structural = self.structural_similarity(left, right)
        blended = (
            w.set_overlap_weight * set_overlap
            + w.sequence_alignment_weight * sequence
            + w.structural_weight * structural
        )
        return SimilarityScores(
            comparison_type=comparison_type,
            set_overlap=round(set_overlap, 4),
            sequence_alignment=round(sequence, 4),
            structural=round(structural, 4),
            blended=round(float(blended), 4),
            weights={
                "set_overlap": w.set_overlap_weight,
                "sequence_alignment": w.sequence_alignment_weight,
                "structural": w.structural_weight,
            },
        )

    def semantic_similarity(self, query_text: str | None, workflow: Workflow) -> float:
        """Cosine similarity between a NL question and workflow text, with lookup intent."""
        if not query_text or not query_text.strip():
            return 0.0
        texts: list[str] = []
        if workflow.summary:
            texts.append(workflow.summary)
        if workflow.title_hint:
            texts.append(workflow.title_hint)
        if workflow.sections:
            texts.append(" ".join(workflow.sections))
        if not texts and workflow.ordered_nodes:
            texts.append(" → ".join(n.label() for n in workflow.ordered_nodes))
        if not texts:
            return 0.0

        query_vec = self.embedder.embed_one(query_text.strip())
        base = max(
            float(self.embedder.cosine(query_vec, self.embedder.embed_one(text)))
            for text in texts
        )
        adjustment = self._lookup_intent_adjustment(query_text, workflow)
        return float(max(0.0, min(1.0, base + adjustment)))

    def _lookup_intent_adjustment(self, question: str, workflow: Workflow) -> float:
        """Prefer field-repair sections for generic weld-repair lookup questions."""
        q = question.strip().lower()
        if not (
            re.match(r"^(where|which section|what section|locate|find)\b", q)
            or ("where is" in q and "procedure" in q)
        ):
            return 0.0

        title = (workflow.title_hint or "").lower()
        boost = 0.0
        if any(term in q for term in ("weld", "repair")):
            if "field repair" in title or "fuselage" in title:
                boost += 0.35
            if "engine mount" in title or "bracket" in title:
                if not any(term in q for term in ("engine", "mount", "bracket")):
                    boost -= 0.25
            if len(workflow.ordered_nodes) >= 6:
                boost += 0.15
            if len(workflow.ordered_nodes) <= 4:
                boost -= 0.10
        return boost

    def set_overlap(self, left: Sequence[WorkflowNode], right: Sequence[WorkflowNode]) -> float:
        """Jaccard on canonical action types clustered by embedding cosine."""
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        left_labels = [n.label() for n in left]
        right_labels = [n.label() for n in right]
        all_labels = left_labels + right_labels
        vectors = self.embedder.embed(all_labels)
        threshold = self.config.similarity.action_cluster_threshold

        cluster_ids, canonical_names = cluster_labels_unified(all_labels, vectors, threshold)
        left_cluster_ids = cluster_ids[: len(left_labels)]
        right_cluster_ids = cluster_ids[len(left_labels) :]

        left_set = set(left_cluster_ids)
        right_set = set(right_cluster_ids)

        logger.debug(
            "Set overlap inputs: left_labels=%s right_labels=%s threshold=%.2f",
            left_labels,
            right_labels,
            threshold,
        )

        cluster_score = jaccard_cluster_overlap(
            left_set,
            right_set,
            left_labels,
            right_labels,
            left_cluster_ids,
            right_cluster_ids,
            canonical_names,
        )
        soft_score = self._soft_label_overlap(left, right)
        if cluster_score <= 0.0 and soft_score > 0.0:
            return soft_score
        if soft_score > cluster_score:
            return 0.6 * cluster_score + 0.4 * soft_score
        return cluster_score

    def _soft_label_overlap(
        self,
        left: Sequence[WorkflowNode],
        right: Sequence[WorkflowNode],
    ) -> float:
        """Best-match ratio for short query labels against candidate steps."""
        if not left or not right:
            return 0.0
        left_labels = [n.label() for n in left]
        right_labels = [n.label() for n in right]
        vectors = self.embedder.embed(left_labels + right_labels)
        left_vecs = vectors[: len(left_labels)]
        right_vecs = vectors[len(left_labels) :]
        soft_threshold = min(self.config.similarity.action_cluster_threshold, 0.45)
        matched = 0
        for left_vec in left_vecs:
            best = max(float(np.dot(left_vec, right_vec)) for right_vec in right_vecs)
            if best >= soft_threshold:
                matched += 1
        return matched / len(left_labels)

    def sequence_alignment(
        self,
        left: Sequence[WorkflowNode],
        right: Sequence[WorkflowNode],
    ) -> float:
        """Needleman-Wunsch over node sequences; cosine-distance as substitution cost."""
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        left_labels = [n.label() for n in left]
        right_labels = [n.label() for n in right]
        vectors = self.embedder.embed(left_labels + right_labels)
        a = vectors[: len(left_labels)]
        b = vectors[len(left_labels) :]
        gap = self.config.similarity.nw_gap_penalty

        n, m = len(a), len(b)
        # Score matrix: higher is better. Match score = cosine similarity.
        dp = np.zeros((n + 1, m + 1), dtype=np.float64)
        for i in range(1, n + 1):
            dp[i, 0] = -i * gap
        for j in range(1, m + 1):
            dp[0, j] = -j * gap

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sim = float(np.dot(a[i - 1], b[j - 1]))  # already normalized ⇒ cosine
                match = dp[i - 1, j - 1] + sim
                delete = dp[i - 1, j] - gap
                insert = dp[i, j - 1] - gap
                dp[i, j] = max(match, delete, insert)

        raw = float(dp[n, m])
        # Normalize to [0, 1]: perfect match of min length ≈ min(n,m)
        max_possible = float(min(n, m))
        min_possible = -gap * (n + m)
        if max_possible == min_possible:
            return 0.0
        normalized = (raw - min_possible) / (max_possible - min_possible)
        # Also scale by length ratio to penalize huge size mismatch slightly
        length_ratio = min(n, m) / max(n, m)
        return float(max(0.0, min(1.0, normalized * (0.85 + 0.15 * length_ratio))))

    def structural_similarity(self, left: Workflow, right: Workflow) -> float:
        """1 - normalized graph_edit_distance (timeout-bounded)."""
        g1 = self._graph_builder.to_networkx(left.nodes, left.edges)
        g2 = self._graph_builder.to_networkx(right.nodes, right.edges)

        max_nodes = self.config.similarity.max_nodes_for_ged
        if g1.number_of_nodes() > max_nodes or g2.number_of_nodes() > max_nodes:
            return _degree_histogram_similarity(g1, g2)

        try:
            ged = nx.graph_edit_distance(
                g1,
                g2,
                node_subst_cost=_node_subst_cost,
                edge_subst_cost=_edge_subst_cost,
                timeout=self.config.similarity.graph_edit_timeout,
            )
            if ged is None:
                return _degree_histogram_similarity(g1, g2)
            # Upper bound: remove all nodes/edges from both + reinsert
            upper = (
                g1.number_of_nodes()
                + g2.number_of_nodes()
                + g1.number_of_edges()
                + g2.number_of_edges()
            )
            if upper <= 0:
                return 1.0
            normalized = float(ged) / float(upper)
            return float(max(0.0, min(1.0, 1.0 - normalized)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("GED failed (%s); falling back to degree histogram", exc)
            return _degree_histogram_similarity(g1, g2)

    def best_matches(
        self,
        query_workflow: Workflow,
        candidates: Sequence[Workflow],
        top_k: int = 5,
        *,
        query_text: str | None = None,
    ) -> list[tuple[Workflow, SimilarityScores]]:
        scored = [
            (
                cand,
                self.score(
                    query_workflow,
                    cand,
                    comparison_type=ComparisonType.QUERY_TO_WORKFLOW,
                    query_text=query_text,
                ),
            )
            for cand in candidates
        ]
        scored.sort(key=lambda item: item[1].blended, reverse=True)
        return scored[:top_k]


class Deduplicator:
    """Flag/merge workflows whose summary embeddings exceed cosine threshold."""

    def __init__(
        self,
        config: AppConfig | None = None,
        embedder: EmbeddingService | None = None,
        similarity: WorkflowSimilarityEngine | None = None,
    ) -> None:
        self.config = config or get_config()
        self.embedder = embedder or EmbeddingService(self.config)
        self.similarity = similarity or WorkflowSimilarityEngine(self.config, self.embedder)

    def compare_pair(self, left: Workflow, right: Workflow) -> WorkflowPairComparison:
        """Run WORKFLOW_TO_WORKFLOW graph scoring plus summary cosine (Module 12)."""
        graph_scores = self.similarity.score(
            left,
            right,
            comparison_type=ComparisonType.WORKFLOW_TO_WORKFLOW,
        )
        left_vec = self.embedder.embed_one(left.summary or left.workflow_id)
        right_vec = self.embedder.embed_one(right.summary or right.workflow_id)
        summary_cosine = float(self.embedder.cosine(left_vec, right_vec))
        threshold = self.config.deduplication.cosine_threshold
        return WorkflowPairComparison(
            left_workflow_id=left.workflow_id,
            right_workflow_id=right.workflow_id,
            left_title=left.title_hint,
            right_title=right.title_hint,
            graph_scores=graph_scores,
            summary_cosine=round(summary_cosine, 4),
            dedup_threshold=threshold,
            is_duplicate=summary_cosine >= threshold,
        )

    def log_pair_comparison(self, left: Workflow, right: Workflow) -> WorkflowPairComparison:
        comparison = self.compare_pair(left, right)
        scores = comparison.graph_scores
        logger.info(
            "WORKFLOW_TO_WORKFLOW %s (%s) ↔ %s (%s) | "
            "set_overlap=%.3f sequence_alignment=%.3f structural=%.3f blended=%.3f | "
            "summary_cosine=%.3f threshold=%.2f dedup=%s",
            left.workflow_id,
            left.title_hint or "untitled",
            right.workflow_id,
            right.title_hint or "untitled",
            scores.set_overlap,
            scores.sequence_alignment or 0.0,
            scores.structural or 0.0,
            scores.blended,
            comparison.summary_cosine,
            comparison.dedup_threshold,
            "YES" if comparison.is_duplicate else "NO",
        )
        return comparison

    def find_duplicate(
        self,
        workflow: Workflow,
        existing: Sequence[Workflow],
    ) -> tuple[Workflow, float] | None:
        if not existing:
            return None
        best: tuple[Workflow, float] | None = None
        for other in existing:
            if other.workflow_id == workflow.workflow_id:
                continue
            comparison = self.log_pair_comparison(workflow, other)
            if comparison.summary_cosine >= comparison.dedup_threshold:
                if best is None or comparison.summary_cosine > best[1]:
                    best = (other, comparison.summary_cosine)
        return best


def format_workflow_pair_comparison(comparison: WorkflowPairComparison) -> str:
    """Human-readable caption for compare UI and logs."""
    scores = comparison.graph_scores
    seq = scores.sequence_alignment if scores.sequence_alignment is not None else 0.0
    struct = scores.structural if scores.structural is not None else 0.0
    return (
        f"set={scores.set_overlap:.2f} · seq={seq:.2f} · struct={struct:.2f} · "
        f"confidence={scores.blended:.2f} · summary_cosine={comparison.summary_cosine:.2f}"
    )


def _cluster_labels(labels: list[str], vectors: np.ndarray, threshold: float) -> list[str]:
    """Greedy clustering: assign each item to first centroid with cosine > threshold."""
    canonical: list[str] = []
    centroids: list[np.ndarray] = []
    centroid_names: list[str] = []
    for label, vector in zip(labels, vectors):
        assigned = None
        for idx, centroid in enumerate(centroids):
            if float(np.dot(vector, centroid)) >= threshold:
                assigned = centroid_names[idx]
                break
        if assigned is None:
            centroids.append(vector)
            centroid_names.append(label)
            canonical.append(label)
        else:
            canonical.append(assigned)
    return canonical


def _node_subst_cost(a: dict, b: dict) -> float:
    if a.get("action") == b.get("action"):
        return 0.0
    return 1.0


def _edge_subst_cost(a: dict, b: dict) -> float:
    if a.get("relation") == b.get("relation"):
        return 0.0
    return 0.5


def _degree_histogram_similarity(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
    if g1.number_of_nodes() == 0 and g2.number_of_nodes() == 0:
        return 1.0
    if g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
        return 0.0
    d1 = nx.degree_histogram(g1.to_undirected())
    d2 = nx.degree_histogram(g2.to_undirected())
    size = max(len(d1), len(d2))
    v1 = np.array(d1 + [0] * (size - len(d1)), dtype=np.float64)
    v2 = np.array(d2 + [0] * (size - len(d2)), dtype=np.float64)
    if v1.sum() == 0 or v2.sum() == 0:
        return 0.0
    v1 /= v1.sum()
    v2 /= v2.sum()
    return float(1.0 - 0.5 * np.abs(v1 - v2).sum())
