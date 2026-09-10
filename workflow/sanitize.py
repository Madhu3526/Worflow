"""Runtime cleanup for stale or noisy workflow nodes (DOCX orphan steps, etc.)."""

from __future__ import annotations

import re

from models import Workflow, WorkflowNode, coerce_workflow

_SPURIOUS_NODE_LABEL_RE = re.compile(r"^perform \d+$", re.IGNORECASE)

# Lightweight description filter — kept local so Streamlit never depends on
# a hot-reloaded actions.sentence_classifier module at import time.
_DESCRIPTION_RE = re.compile(
    r"(?:"
    r"^\s*this\s+(?:procedure|section|process|method|document)\b"
    r"|^\s*the\s+following\b"
    r"|\b(?:differs?|different)\s+from\b"
    r"|\bcompared?\s+to\b"
    r"|\bapplies?\s+to\s+hairline\b"
    r"|^\s*(?:where|what|which|how)\b.*\?"
    r")",
    re.IGNORECASE,
)


def looks_like_description(text: str) -> bool:
    """Public helper for evidence/graph filters without importing actions."""
    cleaned = text.strip()
    if not cleaned:
        return True
    return bool(_DESCRIPTION_RE.search(cleaned))

def is_spurious_workflow_node(node: WorkflowNode) -> bool:
    """Drop orphan DOCX step numbers and other non-action nodes at read/query time."""
    label = node.label().strip()
    sentence = (node.sentence or "").strip()
    if _SPURIOUS_NODE_LABEL_RE.match(label):
        return True
    if re.match(r"^\d+\.\s*$", sentence):
        return True
    if looks_like_description(sentence or label):
        return True
    if re.match(r"^[A-Z0-9\s\(\)\-—]+$", sentence) and len(sentence) > 20:
        return True
    if node.action and node.action.lower() == "perform":
        obj = (node.object or "").strip()
        if re.fullmatch(r"\d+", obj):
            return True
        if obj.lower() == "testing passes":
            return True
    if node.action and node.action.lower() == "aircraft":
        return True
    if node.action and node.action.lower() == "complete" and "procedure below" in label.lower():
        return True
    return False


def sanitize_workflow(workflow: Workflow) -> Workflow:
    """Remove spurious nodes/edges so stale on-disk workflows still query cleanly."""
    kept = [n for n in workflow.ordered_nodes if not is_spurious_workflow_node(n)]
    if len(kept) == len(workflow.ordered_nodes):
        return coerce_workflow(workflow)
    kept_ids = {n.node_id for n in kept}
    edges = [e for e in workflow.edges if e.source in kept_ids and e.target in kept_ids]
    kept = [n.model_copy(update={"order": i}) for i, n in enumerate(kept)]
    evidence = [
        e
        for e in workflow.evidence_sources
        if e.sentence
        and not looks_like_description(e.sentence)
        and not re.match(r"^\d+\.\s*$", e.sentence.strip())
        and not _SPURIOUS_NODE_LABEL_RE.match(e.sentence.strip())
    ]
    cleaned = workflow.model_copy(update={"nodes": kept, "edges": edges, "evidence_sources": evidence})
    return coerce_workflow(cleaned)
