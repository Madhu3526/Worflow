"""Shared Pydantic domain models for the workflow discovery engine."""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

ModelT = TypeVar("ModelT", bound=BaseModel)


def coerce_model(model_cls: type[ModelT], value: ModelT | dict[str, Any]) -> ModelT:
    """Re-validate models through dict form to survive Streamlit module reload drift."""
    if isinstance(value, dict):
        return model_cls.model_validate(value)
    if hasattr(value, "model_dump"):
        return model_cls.model_validate(value.model_dump())
    raise TypeError(f"Cannot coerce {model_cls.__name__} from {type(value)!r}")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class TextUnit(BaseModel):
    """Atomic extracted unit from a document."""

    text: str
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    unit_type: str = "paragraph"  # paragraph | heading | table | list_item | step
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentContent(BaseModel):
    """Parsed document with ordered text units."""

    document_id: str = Field(default_factory=lambda: _new_id("doc"))
    source_path: str
    source_name: str
    mime_type: str
    units: list[TextUnit] = Field(default_factory=list)
    page_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Semantically bounded chunk retaining source metadata."""

    chunk_id: str = Field(default_factory=lambda: _new_id("chunk"))
    text: str
    resolved_text: str | None = None
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    unit_type: str = "paragraph"
    document_id: str
    source_name: str
    order_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def working_text(self) -> str:
        return self.resolved_text if self.resolved_text else self.text


class ActionRecord(BaseModel):
    """Extracted action with dependency-parse fields and source metadata."""

    action_id: str = Field(default_factory=lambda: _new_id("act"))
    actor: str | None = None
    action: str
    object: str | None = None
    sentence: str
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    document_id: str
    source_name: str
    chunk_id: str
    order_index: int = 0
    step_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def label(self) -> str:
        obj = f" {self.object}" if self.object else ""
        return f"{self.action}{obj}".strip()


class ConditionalBranch(BaseModel):
    """Parsed conditional loop or exception branch."""

    sentence: str
    condition: str | None = None
    action: str | None = None
    target_steps: list[int] = Field(default_factory=list)
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    chunk_id: str | None = None


def coerce_conditional_branch(branch: ConditionalBranch | dict[str, Any]) -> ConditionalBranch:
    """Re-validate so instances survive Streamlit hot-reload class-identity drift."""
    return coerce_model(ConditionalBranch, branch)


class DiscourseRelation(str, Enum):
    AFTER = "AFTER"
    BEFORE = "BEFORE"
    THEN = "THEN"
    IF = "IF"


class SequenceMarkerEdge(BaseModel):
    """Typed discourse edge between actions or action phrases."""

    edge_id: str = Field(default_factory=lambda: _new_id("edge"))
    source_phrase: str
    target_phrase: str
    relation: DiscourseRelation
    marker_text: str
    sentence: str
    page: int | None = None
    section: str | None = None
    chunk_id: str
    document_id: str
    source_action_id: str | None = None
    target_action_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionGroup(BaseModel):
    """Candidate workflow segment produced by workflow segmentation."""

    group_id: str = Field(default_factory=lambda: _new_id("group"))
    document_id: str
    source_name: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    pages: list[int] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    discourse_edges: list[SequenceMarkerEdge] = Field(default_factory=list)
    context_notes: list[str] = Field(default_factory=list)
    conditional_branches: list[ConditionalBranch] = Field(default_factory=list)
    boundary_confidence: float = 1.0
    title_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def page_range(self) -> tuple[int | None, int | None]:
        if self.pages:
            return self.pages[0], self.pages[-1]
        return self.page_start, self.page_end

    @field_validator("pages", mode="before")
    @classmethod
    def _uniq_sorted_pages(cls, value: Any) -> list[int]:
        if not value:
            return []
        return sorted({int(v) for v in value if v is not None})


class WorkflowNode(BaseModel):
    node_id: str
    actor: str | None = None
    action: str
    object: str | None = None
    order: int
    page: int | None = None
    section: str | None = None
    sentence: str | None = None
    action_id: str | None = None

    def label(self) -> str:
        obj = f" {self.object}" if self.object else ""
        return f"{self.action}{obj}".strip()


class WorkflowEdge(BaseModel):
    source: str
    target: str
    relation: str = "NEXT"
    weight: float = 1.0
    evidence: str | None = None


class EvidenceSource(BaseModel):
    document: str
    page: int | None = None
    section: str | None = None
    sentence: str
    chunk_id: str | None = None
    action_id: str | None = None


class Workflow(BaseModel):
    """Persisted workflow artifact."""

    workflow_id: str = Field(default_factory=lambda: _new_id("wf"))
    document: str
    document_id: str
    pages: list[int] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    context_notes: list[str] = Field(default_factory=list)
    conditional_branches: list[ConditionalBranch] = Field(default_factory=list)
    summary: str = ""
    title_hint: str | None = None
    group_id: str | None = None
    merged_from: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ordered_nodes(self) -> list[WorkflowNode]:
        return sorted(self.nodes, key=lambda n: n.order)

    @field_validator("pages", mode="before")
    @classmethod
    def _uniq_sorted_pages(cls, value: Any) -> list[int]:
        if not value:
            return []
        return sorted({int(v) for v in value if v is not None})


class ComparisonType(str, Enum):
    """Whether similarity compares a NL query to a workflow or two full graphs."""

    QUERY_TO_WORKFLOW = "query_to_workflow"
    WORKFLOW_TO_WORKFLOW = "workflow_to_workflow"


class SimilarityScores(BaseModel):
    comparison_type: ComparisonType = ComparisonType.WORKFLOW_TO_WORKFLOW
    set_overlap: float
    sequence_alignment: float | None = None
    structural: float | None = None
    semantic_similarity: float | None = None
    blended: float
    weights: dict[str, float] = Field(default_factory=dict)


class WorkflowPairComparison(BaseModel):
    """Graph + summary comparison for two persisted workflows (dedup / compare UI)."""

    left_workflow_id: str
    right_workflow_id: str
    left_title: str | None = None
    right_title: str | None = None
    graph_scores: SimilarityScores
    summary_cosine: float
    dedup_threshold: float
    is_duplicate: bool


class MatchResult(BaseModel):
    workflow: Workflow
    scores: SimilarityScores
    evidence: list[EvidenceSource] = Field(default_factory=list)
    faiss_score: float | None = None
    explanation: str = ""


class ExplainableAnswer(BaseModel):
    question: str
    matched_workflow_id: str | None = None
    matched_chain: str = ""
    matched_pages: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    scores: SimilarityScores | None = None
    explanation: str = ""
    candidates: list[MatchResult] = Field(default_factory=list)


def coerce_workflow(workflow: Workflow | dict[str, Any]) -> Workflow:
    return coerce_model(Workflow, workflow)


def coerce_evidence_source(source: EvidenceSource | dict[str, Any]) -> EvidenceSource:
    return coerce_model(EvidenceSource, source)


def coerce_similarity_scores(scores: SimilarityScores | dict[str, Any]) -> SimilarityScores:
    return coerce_model(SimilarityScores, scores)


def build_match_result(
    *,
    workflow: Workflow | dict[str, Any],
    scores: SimilarityScores | dict[str, Any],
    evidence: Sequence[EvidenceSource | dict[str, Any]],
    faiss_score: float = 0.0,
    explanation: str = "",
) -> MatchResult:
    """Construct MatchResult without stale-class isinstance checks."""
    return MatchResult.model_validate(
        {
            "workflow": coerce_workflow(workflow).model_dump(),
            "scores": coerce_similarity_scores(scores).model_dump(),
            "evidence": [coerce_evidence_source(item).model_dump() for item in evidence],
            "faiss_score": faiss_score,
            "explanation": explanation,
        }
    )


def coerce_match_result(result: MatchResult) -> MatchResult:
    return build_match_result(
        workflow=result.workflow,
        scores=result.scores,
        evidence=result.evidence,
        faiss_score=result.faiss_score,
        explanation=result.explanation,
    )


class CorpusStats(BaseModel):
    documents_processed: int = 0
    workflows_found: int = 0
    actions_extracted: int = 0
    avg_confidence: float = 0.0
    indexed_vectors: int = 0
