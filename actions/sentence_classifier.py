"""Sentence classification: INSTRUCTION vs DESCRIPTION (pre-pass for Module 4)."""

from __future__ import annotations

import re
from enum import Enum

_DESCRIPTION_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"this\s+(?:procedure|section|process|method|document)\b"
    r"|the\s+following\b"
    r"|note\s+that\b"
    r"|record\s+all\b"
    r"|(?:be\s+)?aware\s+that\b"
    r"|(?:it\s+is\s+)?(?:important|critical)\s+(?:to\s+note|that)\b"
    r")",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(?:"
    r"differs?\s+from"
    r"|different\s+from"
    r"|compared?\s+to"
    r"|in\s+contrast\s+to"
    r"|unlike\s+the"
    r"|as\s+opposed\s+to"
    r"|versus\b"
    r"|vs\.?\b"
    r")\b",
    re.IGNORECASE,
)

_NUMBERED_STEP_RE = re.compile(
    r"^\s*(?:\d+[\.\)\:]|\d+\.\d+\s+|step\s+\d+[:\.\s]+|[a-z][\.\)]\s+|[-•▪]\s+)",
    re.IGNORECASE,
)


class SentenceKind(str, Enum):
    INSTRUCTION = "INSTRUCTION"
    DESCRIPTION = "DESCRIPTION"
    CONDITIONAL = "CONDITIONAL"


def classify_sentence(sentence: str, sent: object | None = None) -> SentenceKind:
    """Classify a sentence before action extraction."""
    text = sentence.strip()
    if not text:
        return SentenceKind.DESCRIPTION

    from sequence_markers import is_conditional_sentence
    if is_conditional_sentence(text):
        return SentenceKind.CONDITIONAL

    if _NUMBERED_STEP_RE.match(text):
        return SentenceKind.INSTRUCTION

    if _DESCRIPTION_PREFIX_RE.match(text):
        return SentenceKind.DESCRIPTION

    if _COMPARISON_RE.search(text):
        return SentenceKind.DESCRIPTION

    if sent is not None and _is_imperative_instruction(sent):
        return SentenceKind.INSTRUCTION

    if _looks_imperative_text(text):
        return SentenceKind.INSTRUCTION

    if _looks_descriptive_prose(text, sent):
        return SentenceKind.DESCRIPTION

    return SentenceKind.INSTRUCTION


def is_description_sentence(sentence: str, sent: object | None = None) -> bool:
    """Return True when a sentence should never become a workflow action/node."""
    return classify_sentence(sentence, sent) == SentenceKind.DESCRIPTION


def _is_imperative_instruction(sent: object) -> bool:
    """ROOT verb with no explicit subject → instructional step."""
    tokens = [t for t in sent if not t.is_space and not t.is_punct]  # type: ignore[attr-defined]
    if not tokens:
        return False
    roots = [t for t in tokens if t.dep_ == "ROOT"]
    if not roots:
        return False
    root = roots[0]
    if root.pos_ not in {"VERB", "AUX"} and not str(root.tag_).startswith("VB"):
        return False
    has_subject = any(
        t.dep_ in {"nsubj", "nsubjpass", "csubj"}
        and t.lower_ not in {"there", "it"}
        for t in tokens
    )
    if has_subject:
        # "This procedure applies..." has nsubj — not imperative instruction
        for t in tokens:
            if t.dep_ in {"nsubj", "nsubjpass"} and t.lower_ in {
                "this",
                "that",
                "procedure",
                "section",
                "document",
            }:
                return False
        return False
    return True


def _looks_imperative_text(text: str) -> bool:
    """Heuristic for truncated imperative lines."""
    if _NUMBERED_STEP_RE.match(text):
        return True
    cleaned = _NUMBERED_STEP_RE.sub("", text).strip()
    if not cleaned:
        return False
    first = cleaned.split()[0].lower().strip(".,;:")
    imperative_verbs = {
        "apply",
        "remove",
        "inspect",
        "grind",
        "preheat",
        "perform",
        "clean",
        "install",
        "replace",
        "extract",
        "verify",
        "check",
        "use",
        "do",
        "ensure",
        "allow",
        "record",
    }
    # "Record all..." is explicitly DESCRIPTION in spec — exclude when followed by "all"
    if first == "record" and re.match(r"^record\s+all\b", cleaned, re.IGNORECASE):
        return False
    return first in imperative_verbs


def _looks_descriptive_prose(text: str, sent: object | None) -> bool:
    """Long explanatory sentences without clear imperative structure."""
    if len(text) > 120 and not _NUMBERED_STEP_RE.match(text):
        return True
    if sent is not None:
        roots = [t for t in sent if t.dep_ == "ROOT"]  # type: ignore[attr-defined]
        if roots and roots[0].lemma_.lower() in {
            "apply",
            "describe",
            "explain",
            "include",
            "contain",
            "differ",
            "relate",
            "cover",
            "refer",
        }:
            subjs = [t for t in sent if t.dep_ in {"nsubj", "nsubjpass"}]  # type: ignore[attr-defined]
            if subjs and subjs[0].lower_ in {"this", "that", "procedure", "section", "document"}:
                return True
    return False
