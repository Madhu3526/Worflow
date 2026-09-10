"""Template-based workflow summary generation (deterministic, no LLM)."""

from __future__ import annotations

from collections import Counter

from models import Workflow, WorkflowNode


def most_common_actor(nodes: list[WorkflowNode]) -> str:
    actors = [n.actor for n in nodes if n.actor]
    if not actors:
        return "Operator"
    return Counter(actors).most_common(1)[0][0]


def summarize(workflow: Workflow) -> str:
    """Deterministic template summary — this string is what gets embedded."""
    nodes = workflow.ordered_nodes
    if not nodes:
        title = workflow.title_hint or "Untitled workflow"
        return f"Workflow: {title}"
    actor = most_common_actor(nodes)
    chain = " → ".join(n.label() for n in nodes)
    if workflow.title_hint:
        return f"{workflow.title_hint}: {actor} performs: {chain}"
    return f"{actor} performs: {chain}"


def apply_summary(workflow: Workflow) -> Workflow:
    workflow.summary = summarize(workflow)
    return workflow
