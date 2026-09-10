"""Evidence retrieval and explainable answer formatting."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from workflow.sanitize import looks_like_description
from config import AppConfig, get_config
from embeddings import EmbeddingService, FaissWorkflowIndex
from models import (
    Chunk,
    ComparisonType,
    EvidenceSource,
    ExplainableAnswer,
    MatchResult,
    SimilarityScores,
    Workflow,
    WorkflowNode,
    build_match_result,
    coerce_workflow,
)
from similarity import WorkflowSimilarityEngine
from storage import WorkflowStore
from workflow.sanitize import sanitize_workflow

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Retrieve original supporting sentences for a matched workflow."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def retrieve(
        self,
        workflow: Workflow,
        query: str | None = None,
        embedder: EmbeddingService | None = None,
    ) -> list[EvidenceSource]:
        evidence = list(workflow.evidence_sources)
        if not evidence:
            evidence = [
                EvidenceSource(
                    document=workflow.document,
                    page=node.page,
                    section=node.section,
                    sentence=node.sentence or node.label(),
                    action_id=node.action_id,
                )
                for node in workflow.ordered_nodes
                if node.sentence or node.label()
            ]

        unique: list[EvidenceSource] = []
        seen: set[str] = set()
        for item in evidence:
            key = item.sentence.strip()
            if not key or looks_like_description(key):
                continue
            if key not in seen:
                unique.append(item)
                seen.add(key)

        max_n = self.config.retrieval.max_evidence_sentences
        if query and embedder and unique:
            return self._rank_by_query(unique, query, embedder)[:max_n]
        return unique[:max_n]

    def _rank_by_query(
        self,
        evidence: list[EvidenceSource],
        query: str,
        embedder: EmbeddingService,
    ) -> list[EvidenceSource]:
        texts = [e.sentence for e in evidence]
        q = embedder.embed_one(query)
        vectors = embedder.embed(texts)
        scored = [
            (float(embedder.cosine(q, vec)), item)
            for vec, item in zip(vectors, evidence)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]


class QueryEngine:
    """End-to-end: embed query → FAISS → similarity → evidence → explanation."""

    def __init__(
        self,
        config: AppConfig | None = None,
        store: WorkflowStore | None = None,
        index: FaissWorkflowIndex | None = None,
        similarity: WorkflowSimilarityEngine | None = None,
        evidence: EvidenceRetriever | None = None,
    ) -> None:
        self.config = config or get_config()
        self.store = store or WorkflowStore(self.config)
        self.index = index or FaissWorkflowIndex(self.config)
        self.similarity = similarity or WorkflowSimilarityEngine(
            self.config, self.index.embedder
        )
        self.evidence = evidence or EvidenceRetriever(self.config)

    def answer(self, question: str, top_k: int | None = None) -> ExplainableAnswer:
        top_k = top_k or self.config.retrieval.top_k
        hits = self.index.search(question, top_k=top_k)
        workflows = [sanitize_workflow(wf) for wf in self.store.load_all()]
        by_id = {wf.workflow_id: wf for wf in workflows}
        query_workflow = question_as_workflow(
            question, extractor=self.similarity._query_extractor
        )

        candidates: list[MatchResult] = []
        for hit in hits:
            workflow = by_id.get(hit["workflow_id"])
            if workflow is None:
                continue
            if float(hit["score"]) < self.config.retrieval.min_score:
                continue
            scores = self.similarity.score(
                query_workflow,
                workflow,
                comparison_type=ComparisonType.QUERY_TO_WORKFLOW,
                query_text=question,
            )
            evidence_list = self.evidence.retrieve(
                workflow, query=question, embedder=self.index.embedder
            )
            explanation = format_explanation(question, workflow, scores, evidence_list)
            candidates.append(
                build_match_result(
                    workflow=workflow,
                    scores=scores,
                    evidence=evidence_list,
                    faiss_score=float(hit["score"]),
                    explanation=explanation,
                )
            )

        if not candidates and workflows:
            for workflow in workflows:
                scores = self.similarity.score(
                    query_workflow,
                    workflow,
                    comparison_type=ComparisonType.QUERY_TO_WORKFLOW,
                    query_text=question,
                )
                evidence_list = self.evidence.retrieve(
                    workflow, query=question, embedder=self.index.embedder
                )
                candidates.append(
                    build_match_result(
                        workflow=workflow,
                        scores=scores,
                        evidence=evidence_list,
                        explanation=format_explanation(
                            question, workflow, scores, evidence_list
                        ),
                    )
                )

        candidates.sort(
            key=lambda c: (c.scores.blended, len(c.workflow.ordered_nodes)),
            reverse=True,
        )
        candidates = candidates[:top_k]

        if not candidates:
            return ExplainableAnswer(
                question=question,
                explanation="No matching workflows found in the local index.",
            )

        best = candidates[0]
        pages = best.workflow.pages
        page_str = (
            f"{pages[0]}–{pages[-1]}"
            if len(pages) > 1
            else (str(pages[0]) if pages else "n/a")
        )
        chain = " → ".join(n.label() for n in best.workflow.ordered_nodes)
        return ExplainableAnswer(
            question=question,
            matched_workflow_id=best.workflow.workflow_id,
            matched_chain=chain,
            matched_pages=page_str,
            evidence=[e.sentence for e in best.evidence],
            confidence=best.scores.blended,
            scores=best.scores,
            explanation=best.explanation,
            candidates=candidates,
        )


def _format_score_breakdown(scores: SimilarityScores) -> str:
    if scores.comparison_type == ComparisonType.QUERY_TO_WORKFLOW:
        semantic = scores.semantic_similarity if scores.semantic_similarity is not None else 0.0
        return (
            f"(semantic similarity: {semantic:.2f}, "
            f"set overlap: {scores.set_overlap:.2f})"
        )
    sequence = scores.sequence_alignment if scores.sequence_alignment is not None else 0.0
    structural = scores.structural if scores.structural is not None else 0.0
    return (
        f"(set overlap: {scores.set_overlap:.2f}, "
        f"sequence alignment: {sequence:.2f}, "
        f"structural: {structural:.2f})"
    )


def format_explanation(
    question: str,
    workflow: Workflow,
    scores: SimilarityScores,
    evidence: Sequence[EvidenceSource],
) -> str:
    chain = " → ".join(n.label() for n in workflow.ordered_nodes)
    pages = workflow.pages
    page_str = (
        f"{pages[0]}–{pages[-1]}"
        if len(pages) > 1
        else (str(pages[0]) if pages else "n/a")
    )
    evidence_lines = " / ".join(f'"{e.sentence}"' for e in evidence[:5])
    conf_pct = int(round(scores.blended * 100))

    rework = ""
    if workflow.conditional_branches:
        rework_list = []
        for cb in workflow.conditional_branches:
            rework_list.append(f"Rework path: {cb.sentence}")
        rework = "\n" + "\n".join(rework_list)

    return (
        f"Question: {question}\n"
        f"Matched Workflow: {chain}\n"
        f"Matched Pages: {page_str}\n"
        f"Evidence: {evidence_lines}\n"
        f"Confidence: {conf_pct}% {_format_score_breakdown(scores)}"
        f"{rework}"
    )


def question_as_workflow(
    question: str,
    extractor: object | None = None,
) -> Workflow:
    """Build a query workflow using the same action extraction pipeline as documents."""
    from actions import ActionExtractor, _is_spurious_action

    ext = extractor or ActionExtractor()
    lookup = _is_lookup_question(question)
    chunk = Chunk(
        text=question,
        resolved_text=question,
        page=1,
        document_id="query",
        source_name="__query__",
        order_index=0,
        unit_type="paragraph",
    )
    actions = []
    context_notes: list[str] = []
    if not lookup:
        result = ext.extract([chunk])  # type: ignore[attr-defined]
        actions = [
            a for a in result.actions
            if not _is_spurious_action(a)
        ]
        for notes in result.context_by_chunk.values():
            context_notes.extend(notes)

    built_nodes: list[WorkflowNode] = []
    seen: set[str] = set()
    for i, action in enumerate(actions):
        key = action.label().lower()
        if key in seen:
            continue
        seen.add(key)
        built_nodes.append(
            WorkflowNode(
                node_id=f"q_{i}",
                actor=action.actor,
                action=action.action,
                object=action.object,
                order=len(built_nodes),
                sentence=action.sentence,
                action_id=action.action_id,
            )
        )

    for phrase in _extract_query_phrases(question):
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        built_nodes.append(
            WorkflowNode(
                node_id=f"qp_{len(built_nodes)}",
                action=phrase,
                object=None,
                order=len(built_nodes),
                sentence=phrase,
            )
        )

    if not lookup:
        hint_nodes = _extract_query_action_hints(question, ext._nlp)  # type: ignore[attr-defined]
        for node in hint_nodes:
            key = node.label().lower()
            if key in seen:
                continue
            seen.add(key)
            node = node.model_copy(
                update={"order": len(built_nodes), "node_id": f"q_{len(built_nodes)}"}
            )
            built_nodes.append(node)

    if not built_nodes:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", question)
        stop = {
            "where", "is", "the", "a", "an", "of", "for", "to", "in", "on", "and",
            "or", "procedure", "process", "workflow", "how", "what", "which", "find",
            "locate", "show", "me", "about", "does", "do", "are", "major",
        }
        content = [t for t in tokens if t.lower() not in stop]
        for i, word in enumerate(content[:12]):
            built_nodes.append(
                WorkflowNode(
                    node_id=f"q_{i}",
                    action=word.lower(),
                    object=None,
                    order=i,
                    sentence=word,
                )
            )

    wf = Workflow(
        document="__query__",
        document_id="query",
        nodes=built_nodes,
        context_notes=context_notes,
        summary=question,
        title_hint=question,
    )
    wf.summary = question
    return wf


_PROCESS_HINTS = {
    "inspect",
    "remove",
    "grind",
    "preheat",
    "apply",
    "weld",
    "repair",
    "perform",
    "test",
    "testing",
    "clean",
    "extract",
    "install",
    "replace",
    "validate",
    "validation",
    "role",
    "context",
    "objective",
    "input",
    "tasks",
    "constraints",
    "output",
    "format",
}


def _extract_query_action_hints(question: str, nlp: object) -> list[WorkflowNode]:
    doc = nlp(question)
    nodes: list[WorkflowNode] = []
    seen: set[str] = set()
    for token in doc:
        lemma = token.lemma_.lower()  # type: ignore[attr-defined]
        text = token.text.lower()  # type: ignore[attr-defined]
        for hint in (lemma, text):
            if hint in _PROCESS_HINTS and hint not in seen:
                seen.add(hint)
                nodes.append(
                    WorkflowNode(
                        node_id=f"qh_{hint}",
                        action=hint,
                        object=None,
                        order=len(nodes),
                        sentence=hint,
                    )
                )
                break
    return nodes


def _title_overlap(question: str, workflow: Workflow) -> float:
    q_tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", question)}
    haystack = " ".join(
        filter(
            None,
            [workflow.title_hint or "", workflow.summary, " ".join(workflow.sections)],
        )
    ).lower()
    h_tokens = set(re.findall(r"[A-Za-z]+", haystack))
    if not q_tokens or not h_tokens:
        return 0.0
    token_score = len(q_tokens & h_tokens) / len(q_tokens)
    phrase_score = 0.0
    for phrase in _extract_query_phrases(question):
        if phrase.lower() in haystack:
            phrase_score = max(phrase_score, 1.0)
        elif all(part in h_tokens for part in phrase.lower().split()):
            phrase_score = max(phrase_score, 0.75)
    return max(token_score, phrase_score)


def _is_lookup_question(question: str) -> bool:
    q = question.strip().lower()
    return bool(
        re.match(r"^(where|which section|what section|locate|find)\b", q)
        or ("where is" in q and "procedure" in q)
    )


def _extract_query_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    q = question.strip()
    for match in re.finditer(
        r"\b(?:major\s+)?(?:weld\s+repair|field\s+repair|weld\s+passes)\b",
        q,
        re.IGNORECASE,
    ):
        phrase = re.sub(r"\s+", " ", match.group(0).strip().lower())
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    if not phrases:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", q)]
        stop = {
            "where", "is", "the", "a", "an", "of", "for", "to", "in", "on", "and",
            "or", "procedure", "process", "workflow", "how", "what", "which", "find",
            "locate", "show", "me", "about", "does", "do", "are",
        }
        content = [t for t in tokens if t not in stop]
        if len(content) >= 2:
            phrases.append(" ".join(content[:3]))
    return phrases


def _procedure_lookup_boost(question: str, workflow: Workflow) -> float:
    if not _is_lookup_question(question):
        return 0.0
    q = question.lower()
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
    return _clamp01(boost)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
