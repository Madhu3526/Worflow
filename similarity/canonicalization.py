"""Canonical action clustering helpers for similarity scoring."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cluster_labels_unified(
    labels: list[str],
    vectors: np.ndarray,
    threshold: float,
) -> tuple[list[int], list[str]]:
    """Cluster all labels together; return cluster id per label and canonical names."""
    centroids: list[np.ndarray] = []
    canonical_names: list[str] = []
    cluster_ids: list[int] = []

    for label, vector in zip(labels, vectors):
        assigned: int | None = None
        for idx, centroid in enumerate(centroids):
            if float(np.dot(vector, centroid)) >= threshold:
                assigned = idx
                break
        if assigned is None:
            assigned = len(centroids)
            centroids.append(vector)
            canonical_names.append(label)
        cluster_ids.append(assigned)

    return cluster_ids, canonical_names


def jaccard_cluster_overlap(
    left_ids: set[int],
    right_ids: set[int],
    left_labels: list[str],
    right_labels: list[str],
    left_cluster_ids: list[int],
    right_cluster_ids: list[int],
    canonical_names: list[str],
) -> float:
    """Jaccard similarity on unified cluster IDs."""
    left_canonical = {
        canonical_names[cid] for cid in left_ids if cid < len(canonical_names)
    }
    right_canonical = {
        canonical_names[cid] for cid in right_ids if cid < len(canonical_names)
    }
    logger.debug(
        "Set overlap canonical sets: query=%s candidate=%s",
        sorted(left_canonical),
        sorted(right_canonical),
    )
    if not left_ids and not right_ids:
        return 1.0
    if not left_ids or not right_ids:
        return 0.0

    id_intersection = left_ids & right_ids
    id_union = left_ids | right_ids
    id_score = len(id_intersection) / len(id_union) if id_union else 0.0

    name_intersection = left_canonical & right_canonical
    name_union = left_canonical | right_canonical
    name_score = len(name_intersection) / len(name_union) if name_union else 0.0

    score = max(id_score, name_score)
    logger.debug(
        "Set overlap Jaccard: id_score=%.4f name_score=%.4f final=%.4f",
        id_score,
        name_score,
        score,
    )
    return score


def canonicalize_pair(
    label_a: str,
    label_b: str,
    embedder: object,
    threshold: float,
) -> bool:
    """Return True when two labels belong to the same canonical cluster."""
    vectors = embedder.embed([label_a, label_b])  # type: ignore[attr-defined]
    sim = float(np.dot(vectors[0], vectors[1]))
    return sim >= threshold
