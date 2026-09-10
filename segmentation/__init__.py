"""Workflow segmentation: structural + lexical + continuity boundary scoring."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Sequence

from config import AppConfig, get_config
from models import ActionGroup, ActionRecord, SequenceMarkerEdge, ConditionalBranch, coerce_conditional_branch

logger = logging.getLogger(__name__)


class WorkflowSegmenter:
    """Determine where one workflow ends and another begins."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._lexical_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in self.config.segmentation.lexical_patterns
        ]

    def segment(
        self,
        actions: Sequence[ActionRecord],
        discourse_edges: Sequence[SequenceMarkerEdge] | None = None,
        context_by_chunk: dict[str, list[str]] | None = None,
        conditional_branches_by_chunk: dict[str, list[ConditionalBranch]] | None = None,
    ) -> list[ActionGroup]:
        if not actions:
            return []

        ordered = sorted(actions, key=lambda a: (a.page or 0, a.order_index))
        edge_density = self._marker_density(ordered, discourse_edges or [])

        boundaries: list[tuple[int, float, dict[str, float]]] = []
        # Candidate split *before* index i
        for i in range(1, len(ordered)):
            scores = self._boundary_scores(ordered, i, edge_density)
            confidence = (
                self.config.segmentation.structural_weight * scores["structural"]
                + self.config.segmentation.lexical_weight * scores["lexical"]
                + self.config.segmentation.continuity_weight * scores["continuity"]
            )
            if confidence >= self.config.segmentation.boundary_threshold:
                boundaries.append((i, confidence, scores))

        if self.config.segmentation.merge_low_confidence:
            boundaries = self._filter_weak_adjacent(boundaries)

        groups = self._split_into_groups(
            ordered,
            boundaries,
            discourse_edges or [],
            context_by_chunk or {},
            conditional_branches_by_chunk or {},
        )
        groups = self._merge_small_groups(groups)
        logger.info(
            "Segmented %d actions into %d action groups (%d boundaries)",
            len(ordered),
            len(groups),
            len(boundaries),
        )
        return groups

    def _boundary_scores(
        self,
        actions: list[ActionRecord],
        index: int,
        edge_density: list[float],
    ) -> dict[str, float]:
        prev = actions[index - 1]
        curr = actions[index]

        structural = 0.0
        if (prev.section or "") != (curr.section or "") and (curr.section or prev.section):
            structural = 1.0
        elif prev.page is not None and curr.page is not None and curr.page - prev.page >= 2:
            structural = 0.6

        lexical = 0.0
        section_text = curr.section or ""
        sentence_text = curr.sentence or ""
        for pattern in self._lexical_patterns:
            if pattern.search(sentence_text):
                lexical = 1.0
                break
        # Also score when heading-like section just changed
        if (prev.section or "") != (curr.section or "") and section_text:
            for pattern in self._lexical_patterns:
                if pattern.search(section_text):
                    lexical = 1.0
                    break
            else:
                lexical = max(lexical, 0.35)

        continuity = 0.0
        # Check if the section contains numbered list items (numbered-list membership)
        in_numbered_list = False
        if section_text:
            in_numbered_list = any(a.section == section_text and a.step_number is not None for a in actions)

        if not in_numbered_list:
            window = self.config.segmentation.sequence_gap_window
            local = edge_density[max(0, index - window) : index + window]
            if local:
                avg = sum(local) / len(local)
                # Low density near the candidate boundary ⇒ topic break
                continuity = max(0.0, min(1.0, 1.0 - avg))

        return {
            "structural": structural,
            "lexical": lexical,
            "continuity": continuity,
        }

    def _marker_density(
        self,
        actions: list[ActionRecord],
        edges: Sequence[SequenceMarkerEdge],
    ) -> list[float]:
        if not actions:
            return []
        by_chunk: dict[str, int] = defaultdict(int)
        for edge in edges:
            by_chunk[edge.chunk_id] += 1
        densities: list[float] = []
        max_count = max(by_chunk.values()) if by_chunk else 1
        for action in actions:
            densities.append(by_chunk.get(action.chunk_id, 0) / max_count)
        return densities

    def _filter_weak_adjacent(
        self,
        boundaries: list[tuple[int, float, dict[str, float]]],
    ) -> list[tuple[int, float, dict[str, float]]]:
        if not boundaries:
            return []
        kept: list[tuple[int, float, dict[str, float]]] = []
        ceiling = self.config.segmentation.merge_confidence_ceiling
        for item in boundaries:
            index, confidence, scores = item
            if kept and index - kept[-1][0] <= 2:
                # Merge low-confidence adjacent splits: keep the stronger one
                if confidence <= ceiling and kept[-1][1] <= ceiling:
                    if confidence > kept[-1][1]:
                        kept[-1] = item
                    continue
                if confidence <= ceiling:
                    continue
            kept.append(item)
        return kept

    def _split_into_groups(
        self,
        actions: list[ActionRecord],
        boundaries: list[tuple[int, float, dict[str, float]]],
        edges: Sequence[SequenceMarkerEdge],
        context_by_chunk: dict[str, list[str]],
        conditional_branches_by_chunk: dict[str, list[ConditionalBranch]] | None = None,
    ) -> list[ActionGroup]:
        if conditional_branches_by_chunk is None:
            conditional_branches_by_chunk = {}
        split_indices = [0] + [b[0] for b in boundaries] + [len(actions)]
        confidences = {b[0]: b[1] for b in boundaries}
        groups: list[ActionGroup] = []

        for start, end in zip(split_indices, split_indices[1:]):
            group_actions = actions[start:end]
            if not group_actions:
                continue
            action_ids = {a.action_id for a in group_actions}
            group_edges = [
                e
                for e in edges
                if (e.source_action_id in action_ids or e.target_action_id in action_ids)
                or e.chunk_id in {a.chunk_id for a in group_actions}
            ]
            pages = sorted({a.page for a in group_actions if a.page is not None})
            sections = [a.section for a in group_actions if a.section]
            title = sections[0] if sections else None
            chunk_ids = {a.chunk_id for a in group_actions}
            group_context: list[str] = []
            seen_notes: set[str] = set()
            for cid in chunk_ids:
                for note in context_by_chunk.get(cid, []):
                    if note not in seen_notes:
                        group_context.append(note)
                        seen_notes.add(note)

            group_branches: list[ConditionalBranch] = []
            seen_branches: set[str] = set()
            for cid in chunk_ids:
                for branch in conditional_branches_by_chunk.get(cid, []):
                    normalized = coerce_conditional_branch(branch)
                    if normalized.sentence not in seen_branches:
                        group_branches.append(normalized)
                        seen_branches.add(normalized.sentence)
            for list_branches in conditional_branches_by_chunk.values():
                for branch in list_branches:
                    normalized = coerce_conditional_branch(branch)
                    if normalized.section == title and normalized.sentence not in seen_branches:
                        group_branches.append(normalized)
                        seen_branches.add(normalized.sentence)

            groups.append(
                ActionGroup(
                    document_id=group_actions[0].document_id,
                    source_name=group_actions[0].source_name,
                    section=title,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    pages=pages,
                    actions=group_actions,
                    discourse_edges=group_edges,
                    context_notes=group_context,
                    conditional_branches=group_branches,
                    boundary_confidence=confidences.get(start, 1.0),
                    title_hint=title,
                )
            )
        return groups

    def _merge_small_groups(
        self,
        groups: list[ActionGroup],
        min_actions: int = 2,
    ) -> list[ActionGroup]:
        """Merge singleton / tiny groups into neighbors to avoid fragment workflows."""
        if len(groups) <= 1:
            return groups
        merged: list[ActionGroup] = []
        for group in groups:
            if merged and len(group.actions) < min_actions:
                prev = merged[-1]
                prev.actions = list(prev.actions) + list(group.actions)
                prev.discourse_edges = list(prev.discourse_edges) + list(group.discourse_edges)
                pages = sorted({a.page for a in prev.actions if a.page is not None})
                prev.page_start = min(pages) if pages else prev.page_start
                prev.page_end = max(pages) if pages else prev.page_end
                prev.pages = pages
                if not prev.title_hint and group.title_hint:
                    prev.title_hint = group.title_hint
                    prev.section = group.section
                continue
            if (
                merged
                and len(merged[-1].actions) < min_actions
                and len(group.actions) >= min_actions
            ):
                # Absorb previous tiny intro into the current substantive group
                tiny = merged.pop()
                group.actions = list(tiny.actions) + list(group.actions)
                group.discourse_edges = list(tiny.discourse_edges) + list(group.discourse_edges)
                pages = sorted({a.page for a in group.actions if a.page is not None})
                group.page_start = min(pages) if pages else group.page_start
                group.page_end = max(pages) if pages else group.page_end
                group.pages = pages
            merged.append(group)

        # Final pass: if last group is tiny, fold into previous
        if len(merged) >= 2 and len(merged[-1].actions) < min_actions:
            last = merged.pop()
            prev = merged[-1]
            prev.actions = list(prev.actions) + list(last.actions)
            prev.discourse_edges = list(prev.discourse_edges) + list(last.discourse_edges)
            pages = sorted({a.page for a in prev.actions if a.page is not None})
            prev.page_start = min(pages) if pages else prev.page_start
            prev.page_end = max(pages) if pages else prev.page_end
            prev.pages = pages
        return merged
