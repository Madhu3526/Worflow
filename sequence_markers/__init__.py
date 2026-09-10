"""Sequence-marker extraction: discourse-level typed edges (own pass)."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from config import AppConfig, get_config
from models import ActionRecord, Chunk, DiscourseRelation, SequenceMarkerEdge

logger = logging.getLogger(__name__)


class SequenceMarkerExtractor:
    """Tag discourse transition words and emit typed edges for graph construction."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._patterns = self._compile_patterns()
        self._comma_patterns = self._compile_comma_patterns()

    def _compile_patterns(self) -> list[tuple[DiscourseRelation, re.Pattern[str], str]]:
        flags = re.IGNORECASE if self.config.sequence_markers.case_insensitive else 0
        compiled: list[tuple[DiscourseRelation, re.Pattern[str], str]] = []
        for relation_name, markers in self.config.sequence_markers.markers.items():
            relation = DiscourseRelation(relation_name)
            for marker in markers:
                escaped = re.escape(marker)
                pattern = re.compile(
                    rf"(?P<left>[^.。;]*?)\b{escaped}\b(?P<right>[^.。;]*)",
                    flags,
                )
                compiled.append((relation, pattern, marker))
        compiled.sort(key=lambda item: len(item[2]), reverse=True)
        return compiled

    def _compile_comma_patterns(self) -> list[tuple[DiscourseRelation, re.Pattern[str], str]]:
        """Patterns for sentence-initial markers: 'After X, Y' / 'Before X, Y'."""
        flags = re.IGNORECASE if self.config.sequence_markers.case_insensitive else 0
        compiled: list[tuple[DiscourseRelation, re.Pattern[str], str]] = []
        initial = {
            DiscourseRelation.AFTER: ["after", "once", "following", "upon", "subsequent to"],
            DiscourseRelation.BEFORE: ["before", "prior to", "preceding"],
            DiscourseRelation.THEN: ["then", "next", "afterward", "afterwards", "subsequently"],
            DiscourseRelation.IF: ["if", "when", "whenever", "provided that"],
        }
        for relation, markers in initial.items():
            for marker in markers:
                escaped = re.escape(marker)
                # Marker at start (optional leading words), clause, comma, consequence
                pattern = re.compile(
                    rf"^\s*{escaped}\s+(?P<a>.+?),\s*(?P<b>.+)$",
                    flags,
                )
                compiled.append((relation, pattern, marker))
        compiled.sort(key=lambda item: len(item[2]), reverse=True)
        return compiled

    def extract(
        self,
        chunks: Sequence[Chunk],
        actions: Sequence[ActionRecord] | None = None,
    ) -> list[SequenceMarkerEdge]:
        edges: list[SequenceMarkerEdge] = []
        actions_by_chunk: dict[str, list[ActionRecord]] = {}
        if actions:
            for action in actions:
                actions_by_chunk.setdefault(action.chunk_id, []).append(action)

        for chunk in chunks:
            text = chunk.working_text
            if not text.strip():
                continue
            chunk_actions = actions_by_chunk.get(chunk.chunk_id, [])
            for sentence in _split_sentences(text):
                edges.extend(
                    self._extract_from_sentence(sentence, chunk, chunk_actions)
                )

        if actions:
            edges.extend(self._numbered_step_edges(list(actions)))

        logger.info("Extracted %d sequence-marker edges", len(edges))
        return edges

    def _extract_from_sentence(
        self,
        sentence: str,
        chunk: Chunk,
        chunk_actions: list[ActionRecord],
    ) -> list[SequenceMarkerEdge]:
        results: list[SequenceMarkerEdge] = []

        # Prefer explicit "Marker X, Y" shapes
        for relation, pattern, marker in self._comma_patterns:
            match = pattern.match(sentence.strip())
            if not match:
                continue
            a = _clean_phrase(match.group("a"))
            b = _clean_phrase(match.group("b"))
            source, target = _orient_initial(relation, a, b)
            edge = self._make_edge(
                source, target, relation, marker, sentence, chunk, chunk_actions
            )
            if edge:
                results.append(edge)
            return results  # one discourse edge per sentence is enough

        seen_spans: set[tuple[int, int]] = set()
        for relation, pattern, marker in self._patterns:
            for match in pattern.finditer(sentence):
                span = (match.start(), match.end())
                if any(s[0] <= span[0] < s[1] for s in seen_spans):
                    continue
                left = _clean_phrase(match.group("left"))
                right = _clean_phrase(match.group("right"))
                source_phrase, target_phrase = _orient(relation, left, right)
                source_phrase, target_phrase = _fill_missing_sides(
                    source_phrase, target_phrase, sentence, chunk_actions, marker
                )
                edge = self._make_edge(
                    source_phrase,
                    target_phrase,
                    relation,
                    marker,
                    sentence,
                    chunk,
                    chunk_actions,
                )
                if edge:
                    results.append(edge)
                    seen_spans.add(span)
        return results

    def _make_edge(
        self,
        source_phrase: str | None,
        target_phrase: str | None,
        relation: DiscourseRelation,
        marker: str,
        sentence: str,
        chunk: Chunk,
        chunk_actions: list[ActionRecord],
    ) -> SequenceMarkerEdge | None:
        if not source_phrase or not target_phrase:
            return None
        if source_phrase.lower() == target_phrase.lower():
            return None
        return SequenceMarkerEdge(
            source_phrase=source_phrase,
            target_phrase=target_phrase,
            relation=relation,
            marker_text=marker,
            sentence=sentence,
            page=chunk.page,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_action_id=_match_action_id(source_phrase, chunk_actions),
            target_action_id=_match_action_id(target_phrase, chunk_actions),
        )

    def _numbered_step_edges(self, actions: list[ActionRecord]) -> list[SequenceMarkerEdge]:
        edges: list[SequenceMarkerEdge] = []
        numbered = [a for a in actions if a.step_number is not None]
        numbered.sort(key=lambda a: (a.page or 0, a.section or "", a.step_number or 0, a.order_index))
        for prev, curr in zip(numbered, numbered[1:]):
            if prev.section != curr.section:
                continue
            if (curr.step_number or 0) <= (prev.step_number or 0):
                continue
            edges.append(
                SequenceMarkerEdge(
                    source_phrase=prev.label(),
                    target_phrase=curr.label(),
                    relation=DiscourseRelation.THEN,
                    marker_text="numbered_step",
                    sentence=curr.sentence,
                    page=curr.page,
                    section=curr.section,
                    chunk_id=curr.chunk_id,
                    document_id=curr.document_id,
                    source_action_id=prev.action_id,
                    target_action_id=curr.action_id,
                    metadata={"derived_from": "step_number"},
                )
            )
        return edges


def _orient_initial(
    relation: DiscourseRelation,
    a: str,
    b: str,
) -> tuple[str | None, str | None]:
    """Map 'Marker A, B' into temporal (source → target)."""
    if relation == DiscourseRelation.AFTER:
        # After grinding, preheat → grind then preheat
        return a, b
    if relation == DiscourseRelation.BEFORE:
        # Before welding, preheat → preheat then welding
        return b, a
    if relation == DiscourseRelation.IF:
        return a, b
    # THEN / others: A then B if both present
    return a, b


def _orient(
    relation: DiscourseRelation,
    left: str,
    right: str,
) -> tuple[str | None, str | None]:
    if relation == DiscourseRelation.AFTER:
        return (left or None, right or None)
    if relation == DiscourseRelation.BEFORE:
        return (right or None, left or None) if right else (left or None, None)
    if relation in {DiscourseRelation.THEN, DiscourseRelation.IF}:
        return left or None, right or None
    return left or None, right or None


def _fill_missing_sides(
    source: str | None,
    target: str | None,
    sentence: str,
    actions: list[ActionRecord],
    marker: str,
) -> tuple[str | None, str | None]:
    if source and target:
        return source, target
    # Use the main verb phrase of the sentence as the consequence side
    main = None
    if actions:
        main = sorted(actions, key=lambda a: a.order_index)[-1].label()
    else:
        main = _strip_marker_prefix(sentence, marker)
    if source and not target:
        return source, main
    if target and not source:
        # For sentence-initial markers without a clear antecedent, use prior noun phrase in target
        # e.g. right="inspection, remove damaged..." already handled by comma patterns;
        # otherwise pair against the action itself only if we can split on comma
        if "," in (target or ""):
            left, right = [p.strip() for p in target.split(",", 1)]
            return left, right
        return main, target
    if not source and not target and len(actions) >= 2:
        ordered = sorted(actions, key=lambda a: a.order_index)
        return ordered[0].label(), ordered[-1].label()
    return source, target


def _strip_marker_prefix(sentence: str, marker: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(marker)}\b[,:]?\s*", re.IGNORECASE)
    return _clean_phrase(pattern.sub("", sentence))


def _match_action_id(phrase: str, actions: list[ActionRecord]) -> str | None:
    phrase_l = phrase.lower()
    best: ActionRecord | None = None
    best_score = 0
    for action in actions:
        label = action.label().lower()
        score = 0
        if action.action.lower() in phrase_l:
            score += 2
        if action.object and action.object.lower() in phrase_l:
            score += 2
        if label in phrase_l or phrase_l in label:
            score += 3
        if score > best_score:
            best_score = score
            best = action
    return best.action_id if best and best_score > 0 else None


def _clean_phrase(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip(" ,;:.-")
    cleaned = re.sub(r"^(and|or|but|,)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


_CONDITIONAL_RE = re.compile(
    r"^\s*if\b.*?\b(repeat|redo|re-perform|reperform|re-run|rerun|re-verify|reverify)\b",
    re.IGNORECASE,
)
_EXCEPTION_RE = re.compile(
    r"^\s*unless\b",
    re.IGNORECASE,
)


def is_conditional_sentence(text: str) -> bool:
    return bool(_CONDITIONAL_RE.match(text) or _EXCEPTION_RE.match(text))


def parse_conditional_sentence(sentence: str) -> ConditionalBranch:
    from models import ConditionalBranch
    text = sentence.strip()

    # 1. Target steps
    # Match range: "steps 5–7", "steps 5 to 7", "step 5-7"
    target_steps: list[int] = []
    range_match = re.search(r"\bsteps?\s+(\d+)\s*(?:-|–|—|to)\s*(\d+)", text, re.IGNORECASE)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        target_steps = list(range(start, end + 1))
    else:
        # Match individual steps: "step 5", "steps 5 and 6"
        matches = re.findall(r"\bsteps?\s+(\d+)", text, re.IGNORECASE)
        for m in matches:
            target_steps.append(int(m))

    # 2. Condition
    # e.g., "If ultrasonic testing indicates incomplete fusion, ..."
    condition = None
    if "," in text:
        parts = text.split(",", 1)
        first_clause = parts[0].strip()
        if re.match(r"^(if|unless)\b", first_clause, re.IGNORECASE):
            condition = re.sub(r"^(if|unless)\s+", "", first_clause, flags=re.IGNORECASE).strip()

    # 3. Action
    # e.g. "repeat", "redo", etc.
    action = None
    action_match = re.search(
        r"\b(repeat|redo|re-perform|reperform|re-run|rerun|re-verify|reverify)\b",
        text,
        re.IGNORECASE,
    )
    if action_match:
        action = action_match.group(1).lower()

    return ConditionalBranch(
        sentence=text,
        condition=condition,
        action=action,
        target_steps=target_steps,
    )
