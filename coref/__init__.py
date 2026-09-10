"""Coreference resolution via fastcoref (CPU) with safe passthrough fallback."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from config import AppConfig, get_config
from models import Chunk

logger = logging.getLogger(__name__)


_FASTCOREF_MODEL = None
_FASTCOREF_AVAILABLE = None  # None: not tried yet, True/False: cached status


class CoreferenceResolver:
    """Resolve pronouns within each chunk before action extraction."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._model = None
        self._available = False
        if self.config.coref.resolve_text:
            self._try_load()

    def _try_load(self) -> None:
        global _FASTCOREF_MODEL, _FASTCOREF_AVAILABLE
        if _FASTCOREF_AVAILABLE is False:
            self._available = False
            self._model = None
            return
        if _FASTCOREF_AVAILABLE is True:
            self._available = True
            self._model = _FASTCOREF_MODEL
            return

        try:
            # fastcoref 2.x + recent transformers can be incompatible; probe carefully
            from fastcoref import FCoref

            model = FCoref(device=self.config.coref.device)
            # Force a tiny predict to surface load/runtime incompatibilities early
            _ = model.predict(texts=["The technician inspected the crack. It was deep."])
            _FASTCOREF_MODEL = model
            _FASTCOREF_AVAILABLE = True
            self._model = model
            self._available = True
            logger.info("Loaded fastcoref FCoref on %s", self.config.coref.device)
        except Exception as exc:  # noqa: BLE001 — optional dependency path
            _FASTCOREF_AVAILABLE = False
            self._model = None
            self._available = False
            if not self.config.coref.allow_passthrough:
                raise
            logger.warning(
                "fastcoref unavailable (%s); using heuristic coref passthrough",
                exc,
            )
            self._model = None
            self._available = False

    def resolve_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        resolved: list[Chunk] = []
        for chunk in chunks:
            if chunk.unit_type == "heading" or len(chunk.text.split()) < 3:
                chunk.resolved_text = chunk.text
                resolved.append(chunk)
                continue
            chunk.resolved_text = self.resolve_text(chunk.text)
            resolved.append(chunk)
        return list(resolved)

    def resolve_text(self, text: str) -> str:
        if not text.strip():
            return text
        if self._available and self._model is not None:
            return self._resolve_fastcoref(text)
        return self._resolve_heuristic(text)

    def _resolve_fastcoref(self, text: str) -> str:
        assert self._model is not None
        try:
            preds = self._model.predict(texts=[text])
            if not preds:
                return text
            pred = preds[0]
            # Prefer library helper when present
            if hasattr(pred, "get_resolved_text"):
                resolved = pred.get_resolved_text()
                if isinstance(resolved, str) and resolved.strip():
                    return resolved
            clusters = getattr(pred, "get_clusters", lambda **_: [])(as_strings=False)
            return _apply_clusters(text, clusters)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fastcoref predict failed: %s", exc)
            return self._resolve_heuristic(text)

    def _resolve_heuristic(self, text: str) -> str:
        """Lightweight offline pronoun substitution using nearest noun phrase.

        Not as accurate as neural coref, but deterministic and fully offline.
        """
        sentences = _split_sentences(text)
        if len(sentences) < 2:
            return text

        last_noun: str | None = None
        out: list[str] = []
        noun_re = re.compile(
            r"\b((?:the|a|an)\s+(?:[A-Za-z-]+\s+){0,3}[A-Za-z-]+|"
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        )
        pronoun_re = re.compile(r"\b(it|its|they|them|their|this|that)\b", re.IGNORECASE)

        for sentence in sentences:
            nouns = noun_re.findall(sentence)
            if nouns:
                # Prefer definite NPs as antecedents
                definite = [n for n in nouns if n.lower().startswith("the ")]
                last_noun = definite[-1] if definite else nouns[-1]

            if last_noun and pronoun_re.search(sentence):

                def repl(match: re.Match[str]) -> str:
                    token = match.group(0)
                    assert last_noun is not None
                    if token.lower() in {"its", "their"}:
                        return f"{last_noun}'s"
                    return last_noun

                sentence = pronoun_re.sub(repl, sentence, count=2)
            out.append(sentence)
        return " ".join(out)


def _apply_clusters(text: str, clusters: list) -> str:
    """Replace non-representative mentions with the first mention in each cluster."""
    replacements: list[tuple[int, int, str]] = []
    for cluster in clusters or []:
        if not cluster or len(cluster) < 2:
            continue
        # cluster items may be (start, end) char spans
        try:
            spans = [(int(s), int(e)) for s, e in cluster]
        except Exception:  # noqa: BLE001
            continue
        spans.sort(key=lambda x: x[0])
        rep_start, rep_end = spans[0]
        representative = text[rep_start:rep_end]
        for start, end in spans[1:]:
            mention = text[start:end]
            if mention.lower() in {
                "it",
                "its",
                "they",
                "them",
                "their",
                "this",
                "that",
                "he",
                "she",
                "him",
                "her",
            }:
                replacements.append((start, end, representative))

    if not replacements:
        return text
    replacements.sort(key=lambda x: x[0], reverse=True)
    chars = list(text)
    for start, end, replacement in replacements:
        chars[start:end] = list(replacement)
    return "".join(chars)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
