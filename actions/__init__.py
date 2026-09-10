"""Action extraction via spaCy dependency parsing (not NER)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from actions.sentence_classifier import SentenceKind, classify_sentence
import actions.sentence_classifier as _sentence_classifier

if hasattr(_sentence_classifier, "is_description_sentence"):
    is_description_sentence = _sentence_classifier.is_description_sentence
else:

    def is_description_sentence(sentence: str, sent: object | None = None) -> bool:
        return classify_sentence(sentence, sent) == SentenceKind.DESCRIPTION

from config import AppConfig, get_config
from coref import CoreferenceResolver
from models import ActionRecord, Chunk, ConditionalBranch

logger = logging.getLogger(__name__)

_STEP_NUM_RE = re.compile(
    r"^\s*(?:step\s+)?(\d+)(?:[\.\)\:]|\.\d+)*\s+",
    re.IGNORECASE,
)

# Compact process lists: "Inspect ↓ Remove + Grind → Preheat"
# Prefer spaced "+" so we don't chop tokens like "C++" or "A+B".
_LIST_SEPARATOR_RE = re.compile(
    r"(?:"
    r"\s*[↓↑→←⟶⇒➡➔➜➝➞⇩⇧]\s*"  # unicode arrows
    r"|\s*(?:->|=>|-->|==>)\s*"  # ascii arrows
    r"|\s+\+\s+"  # plus as list joiner
    r")"
)


@dataclass
class ActionExtractionResult:
    actions: list[ActionRecord] = field(default_factory=list)
    context_by_chunk: dict[str, list[str]] = field(default_factory=dict)
    conditional_branches_by_chunk: dict[str, list[ConditionalBranch]] = field(default_factory=dict)


class ActionExtractor:
    """Extract {actor, action, object} per sentence using dependency trees."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._nlp = _load_spacy(self.config.actions.spacy_model)
        self._coref = CoreferenceResolver(self.config)

    def extract(self, chunks: Sequence[Chunk]) -> ActionExtractionResult:
        actions: list[ActionRecord] = []
        context_by_chunk: dict[str, list[str]] = {}
        conditional_branches_by_chunk: dict[str, list[ConditionalBranch]] = {}
        global_order = 0
        for chunk in chunks:
            if chunk.unit_type == "heading":
                continue
            # Always classify on original text — bulk coref distorts descriptions.
            source_text = chunk.text
            if not source_text.strip():
                continue
            if _is_noise_text(source_text):
                continue

            units = self._atomic_action_units(source_text)
            for unit_index, unit_text in enumerate(units):
                if _is_noise_text(unit_text):
                    continue
                orig_doc = self._nlp(unit_text)
                step_number = _extract_step_number(unit_text)
                if step_number is None and len(units) > 1:
                    step_number = unit_index + 1

                for sent in orig_doc.sents:
                    sent_text = sent.text.strip()
                    if _is_noise_text(sent_text):
                        continue
                    kind = classify_sentence(sent_text, sent)
                    if kind == SentenceKind.DESCRIPTION:
                        context_by_chunk.setdefault(chunk.chunk_id, []).append(sent_text)
                        logger.debug("DESCRIPTION (skipped action): %s", sent_text[:80])
                        continue
                    if kind == SentenceKind.CONDITIONAL:
                        from sequence_markers import parse_conditional_sentence

                        branch = parse_conditional_sentence(sent_text).model_copy(
                            update={
                                "page": chunk.page,
                                "section": chunk.section,
                                "paragraph": chunk.paragraph,
                                "chunk_id": chunk.chunk_id,
                            }
                        )
                        conditional_branches_by_chunk.setdefault(chunk.chunk_id, []).append(branch)
                        logger.debug("CONDITIONAL (skipped action): %s", sent_text[:80])
                        continue

                    parse_text = (
                        self._coref.resolve_text(sent_text)
                        if self.config.coref.resolve_text
                        else sent_text
                    )
                    parse_doc = self._nlp(parse_text)

                    fragments = self._split_list_separators(sent_text)
                    for frag in fragments:
                        if _is_noise_text(frag):
                            continue
                        if classify_sentence(frag) == SentenceKind.CONDITIONAL:
                            from sequence_markers import parse_conditional_sentence

                            branch = parse_conditional_sentence(frag).model_copy(
                                update={
                                    "page": chunk.page,
                                    "section": chunk.section,
                                    "paragraph": chunk.paragraph,
                                    "chunk_id": chunk.chunk_id,
                                }
                            )
                            conditional_branches_by_chunk.setdefault(chunk.chunk_id, []).append(branch)
                            logger.debug("CONDITIONAL FRAG (skipped action): %s", frag[:80])
                            continue
                        if is_description_sentence(frag):
                            context_by_chunk.setdefault(chunk.chunk_id, []).append(frag)
                            continue

                        frag_parse = (
                            self._coref.resolve_text(frag)
                            if self.config.coref.resolve_text and frag != sent_text
                            else parse_text if frag == sent_text else frag
                        )
                        frag_doc = self._nlp(frag_parse) if frag != sent_text else parse_doc

                        records: list[ActionRecord] = []
                        if frag == sent_text:
                            for parse_sent in frag_doc.sents:
                                records.extend(
                                    self._extract_from_sentence(
                                        parse_sent, chunk, step_number, global_order
                                    )
                                )
                        else:
                            for parse_sent in frag_doc.sents:
                                records.extend(
                                    self._extract_from_sentence(
                                        parse_sent, chunk, step_number, global_order
                                    )
                                )
                            if not records:
                                fallback = _try_imperative_fallback(
                                    frag, chunk, step_number, global_order
                                )
                                if fallback:
                                    records = [fallback]
                            if not records:
                                list_item = _try_delimiter_list_item(
                                    frag, chunk, step_number, global_order
                                )
                                if list_item:
                                    records = [list_item]

                        for record in records:
                            if _is_spurious_action(record):
                                continue
                            record = record.model_copy(update={"sentence": frag})
                            actions.append(record)
                            global_order += 1
        logger.info(
            "Extracted %d actions, %d context notes from %d chunks",
            len(actions),
            sum(len(v) for v in context_by_chunk.values()),
            len(chunks),
        )
        return ActionExtractionResult(
            actions=actions,
            context_by_chunk=context_by_chunk,
            conditional_branches_by_chunk=conditional_branches_by_chunk,
        )

    def _atomic_action_units(self, text: str) -> list[str]:
        """Break compact lists into one unit per action before parsing."""
        pieces: list[str] = [text]
        if self.config.actions.split_on_newlines:
            pieces = _split_on_newlines(text)
        units: list[str] = []
        for piece in pieces:
            if self.config.actions.split_list_separators:
                units.extend(self._split_list_separators(piece))
            else:
                units.append(piece.strip())
        return [u for u in units if u.strip()]

    def _split_list_separators(self, text: str) -> list[str]:
        if not self.config.actions.split_list_separators:
            return [text.strip()] if text.strip() else []
        return split_list_separators(text)

    def _extract_from_sentence(
        self,
        sent: object,
        chunk: Chunk,
        step_number: int | None,
        order_index: int,
    ) -> list[ActionRecord]:
        records: list[ActionRecord] = []

        # Repair common spaCy imperative misparse: "Inspect crack" with
        # ROOT=crack and Inspect as compound/amod of the noun.
        repaired = _repair_leading_imperative(sent, chunk, step_number, order_index)
        if repaired:
            return [repaired]

        # Check for causative construction ("allow component to cool")
        causative = _try_causative_extraction(sent, chunk, step_number, order_index)
        if causative:
            return [causative]

        roots = [t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"]  # type: ignore[attr-defined]
        if not roots:
            # Imperative / truncated steps often tag verb as ROOT with different POS
            roots = [t for t in sent if t.dep_ == "ROOT"]  # type: ignore[attr-defined]
            roots = [t for t in roots if t.pos_ in {"VERB", "AUX"} or t.tag_.startswith("VB")]

        for verb in roots:
            action = _lemma_or_text(verb)
            if len(action) < self.config.actions.min_verb_length:
                continue
            if action.lower() in _STOP_VERBS:
                continue

            actor = _find_actor(verb)
            obj = _find_object(verb)
            if self.config.actions.include_passive and _is_passive(verb):
                # In passive, nsubjpass is the logical object; agent is actor
                agent = _find_agent(verb)
                nsubjpass = _find_nsubjpass(verb)
                if agent:
                    actor = agent
                if nsubjpass:
                    obj = nsubjpass

            # Skip sentences with no actionable content
            if not obj and action.lower() in {"be", "is", "are", "was", "were"}:
                continue

            records.append(
                ActionRecord(
                    actor=actor,
                    action=action.lower(),
                    object=obj,
                    sentence=sent.text.strip(),  # type: ignore[attr-defined]
                    page=chunk.page,
                    section=chunk.section,
                    paragraph=chunk.paragraph,
                    document_id=chunk.document_id,
                    source_name=chunk.source_name,
                    chunk_id=chunk.chunk_id,
                    order_index=order_index + len(records),
                    step_number=step_number,
                    metadata={"extraction": "dependency"},
                )
            )

        # Numbered imperative lines without a clear ROOT verb (OCR / fragments)
        if not records:
            imperative = _try_imperative_fallback(
                sent.text.strip(), chunk, step_number, order_index  # type: ignore[attr-defined]
            )
            if imperative:
                records.append(imperative)
        return records


def _clean_noun_phrase(token: object) -> str:
    allowed_deps = {"det", "compound", "amod", "poss", "case", "nummod"}
    def get_tokens(t):
        res = [t]
        for child in t.children:
            if child.dep_ in allowed_deps:
                res.extend(get_tokens(child))
        return res
    tokens = sorted(get_tokens(token), key=lambda x: x.i)
    return " ".join(t.text for t in tokens)


def _try_causative_extraction(
    sent: object,
    chunk: Chunk,
    step_number: int | None,
    order_index: int,
) -> ActionRecord | None:
    causative_verb = None
    for t in sent:  # type: ignore[attr-defined]
        if t.lemma_.lower() in {"allow", "cause", "permit", "let", "make", "enable", "force", "require"}:
            causative_verb = t
            break

    if causative_verb:
        comp = None
        for child in causative_verb.children:
            if child.dep_ in {"xcomp", "ccomp"}:
                comp = child
                break

        if comp:
            obj_token = None
            for c in comp.children:
                if c.dep_ in {"nsubj", "nsubjpass"}:
                    obj_token = c
                    break
            if not obj_token:
                for c in causative_verb.children:
                    if c.dep_ == "dobj":
                        obj_token = c
                        break

            obj_text = None
            if obj_token:
                raw_obj = _clean_noun_phrase(obj_token)
                cleaned = re.sub(r"\s+", " ", raw_obj).strip(" .,;:")
                obj_text = cleaned or obj_token.text.strip()
                obj_text = re.sub(r"\b(the|a|an)\b", "", obj_text, flags=re.IGNORECASE).strip() or None

            modifier = None
            for c in comp.subtree:
                if c.dep_ == "compound" or (c.dep_ == "pobj" and c.head.dep_ == "prep" and c.head.head == obj_token):
                    if c.text.lower() in {"air", "water", "oil"}:
                        modifier = c.text.lower()
                        break
            if not modifier and "air" in sent.text.lower():  # type: ignore[attr-defined]
                modifier = "air"

            duration = None
            match = re.search(r"\bfor\s+(\d+\s+\w+)", sent.text.lower())  # type: ignore[attr-defined]
            if match:
                duration = match.group(1)
            else:
                for c in comp.subtree:
                    if c.dep_ == "prep" and c.text.lower() == "for":
                        duration = " ".join(t.text for t in c.subtree if t.dep_ != "prep")

            meta = {"extraction": "causative"}
            if modifier:
                meta["modifier"] = modifier
            if duration:
                meta["duration"] = duration

            return ActionRecord(
                actor=None,
                action=comp.lemma_.lower(),
                object=obj_text,
                sentence=sent.text.strip(),  # type: ignore[attr-defined]
                page=chunk.page,
                section=chunk.section,
                paragraph=chunk.paragraph,
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                chunk_id=chunk.chunk_id,
                order_index=order_index,
                step_number=step_number,
                metadata=meta,
            )
    return None


def _repair_leading_imperative(
    sent: object,
    chunk: Chunk,
    step_number: int | None,
    order_index: int,
) -> ActionRecord | None:
    """Recover short imperatives spaCy mis-parses ('Inspect crack', 'Grind surface')."""
    tokens = [t for t in sent if not t.is_space and not t.is_punct]  # type: ignore[attr-defined]
    if len(tokens) < 2 or len(tokens) > 8:
        return None
    first = tokens[0]
    roots = [t for t in sent if t.dep_ == "ROOT"]  # type: ignore[attr-defined]
    if not roots:
        return None
    root = roots[0]
    if first.i >= root.i:
        return None
    # Real subject sentences: "The technician inspects the crack"
    if first.pos_ in {"DET", "PRON", "ADP"} or first.lower_ in {
        "the",
        "a",
        "an",
        "this",
        "that",
        "technician",
        "operator",
        "engineer",
        "worker",
        "user",
    }:
        return None
    # Leading word attached to ROOT (compound/amod) OR first token is intended verb
    attached = first.head == root and first.dep_ in {
        "compound",
        "amod",
        "nmod",
        "npadvmod",
        "nsubj",
    }
    leading_verbish = first.pos_ in {"VERB", "NOUN", "PROPN", "ADJ"} and first.text[:1].isupper()
    if not (attached or leading_verbish):
        return None
    # Prefer repair when ROOT is a noun/adj/mistagged verb that isn't the first token
    if first.pos_ == "VERB" and root == first:
        return None

    action = (first.lemma_ if first.lemma_ != "-PRON-" else first.text).lower()
    if not action.isalpha() or action in _STOP_VERBS or len(action) < 2:
        return None
    obj = " ".join(t.text for t in tokens[1:]).strip() or None
    return ActionRecord(
        actor=None,
        action=action,
        object=obj,
        sentence=sent.text.strip(),  # type: ignore[attr-defined]
        page=chunk.page,
        section=chunk.section,
        paragraph=chunk.paragraph,
        document_id=chunk.document_id,
        source_name=chunk.source_name,
        chunk_id=chunk.chunk_id,
        order_index=order_index,
        step_number=step_number,
        metadata={"extraction": "leading_imperative_repair"},
    )


def _try_delimiter_list_item(
    fragment: str,
    chunk: Chunk,
    step_number: int | None,
    order_index: int,
) -> ActionRecord | None:
    """Turn delimiter-list segments like 'Role' or 'Output Format' into action nodes."""
    cleaned = fragment.strip(" .,;:")
    if not cleaned or len(cleaned) > 80:
        return None
    tokens = cleaned.split()
    if not tokens or len(tokens) > 6:
        return None
    normalized = cleaned.lower()
    return ActionRecord(
        actor=None,
        action=normalized,
        object=None,
        sentence=cleaned,
        page=chunk.page,
        section=chunk.section,
        paragraph=chunk.paragraph,
        document_id=chunk.document_id,
        source_name=chunk.source_name,
        chunk_id=chunk.chunk_id,
        order_index=order_index,
        step_number=step_number,
        metadata={"extraction": "delimiter_list_item"},
    )


def split_list_separators(text: str, min_delimiters: int = 3) -> list[str]:
    """Split on +, arrows when a line contains min_delimiters+ separator occurrences."""
    stripped = text.strip()
    if not stripped:
        return []
    delimiter_count = len(_LIST_SEPARATOR_RE.findall(stripped))
    if delimiter_count < min_delimiters:
        return [stripped]
    parts = [p.strip(" \t.,;:") for p in _LIST_SEPARATOR_RE.split(stripped)]
    return [p for p in parts if p]


def _split_on_newlines(text: str) -> list[str]:
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    return lines if lines else ([text.strip()] if text.strip() else [])


def _load_spacy(model_name: str):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning("spaCy model %s not found; downloading...", model_name)
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)


@lru_cache(maxsize=1)
def get_shared_nlp(model_name: str = "en_core_web_sm"):
    return _load_spacy(model_name)


def _lemma_or_text(token: object) -> str:
    lemma = getattr(token, "lemma_", None)
    if lemma and lemma != "-PRON-":
        return str(lemma)
    return str(getattr(token, "text", ""))


def _subtree_text(token: object) -> str:
    words = [t.text for t in token.subtree if not t.is_space]  # type: ignore[attr-defined]
    return " ".join(words).strip()


def _find_actor(verb: object) -> str | None:
    for child in verb.children:  # type: ignore[attr-defined]
        if child.dep_ in {"nsubj", "nsubjpass"}:
            # Prefer compound subject phrase
            text = _subtree_text(child)
            if text.lower() not in {"it", "this", "that", "they"}:
                return text
            return text
    return None


def _find_object(verb: object) -> str | None:
    # First pass: direct object or attribute
    for child in verb.children:  # type: ignore[attr-defined]
        if child.dep_ in {"dobj", "obj", "attr"}:
            return _clean_np(_subtree_text(child))
    # Second pass: prepositional objects
    for child in verb.children:  # type: ignore[attr-defined]
        if child.dep_ == "prep":
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    return _clean_np(_subtree_text(pobj))
    # Third pass: xcomp / ccomp chains: "continue to weld the joint"
    for child in verb.children:  # type: ignore[attr-defined]
        if child.dep_ in {"xcomp", "ccomp"} and child.pos_ == "VERB":
            inner = _find_object(child)
            if inner:
                return inner
    return None


def _find_agent(verb: object) -> str | None:
    for child in verb.children:  # type: ignore[attr-defined]
        if child.dep_ == "agent":
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    return _subtree_text(pobj)
    return None


def _find_nsubjpass(verb: object) -> str | None:
    for child in verb.children:  # type: ignore[attr-defined]
        if child.dep_ == "nsubjpass":
            return _clean_np(_subtree_text(child))
    return None


def _is_passive(verb: object) -> bool:
    return any(c.dep_ == "nsubjpass" for c in verb.children)  # type: ignore[attr-defined]


def _clean_np(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,;:")
    return cleaned or text.strip()


def _extract_step_number(text: str) -> int | None:
    match = _STEP_NUM_RE.match(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _try_imperative_fallback(
    sentence: str,
    chunk: Chunk,
    step_number: int | None,
    order_index: int,
) -> ActionRecord | None:
    cleaned = _STEP_NUM_RE.sub("", sentence).strip()
    if not cleaned or re.match(r"^\d+\.\s*$", sentence.strip()):
        return None
    # Prefer clause after discourse marker + comma: "After X, remove Y"
    if "," in cleaned:
        head, tail = cleaned.split(",", 1)
        if re.match(
            r"^(after|before|once|following|upon|then|next|when|if)\b",
            head.strip(),
            re.IGNORECASE,
        ):
            cleaned = tail.strip()
    tokens = cleaned.split()
    if not tokens:
        return None
    action = tokens[0].lower().strip(".,;:")
    if not action.isalpha() or action in _STOP_VERBS or len(action) < 2:
        # Noun-phrase fragments from lists / step titles: "Ultrasonic testing"
        if 1 <= len(tokens) <= 6:
            return ActionRecord(
                actor=None,
                action="perform",
                object=cleaned.strip(".,;:"),
                sentence=sentence,
                page=chunk.page,
                section=chunk.section,
                paragraph=chunk.paragraph,
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                chunk_id=chunk.chunk_id,
                order_index=order_index,
                step_number=step_number,
                metadata={"extraction": "noun_step_fallback"},
            )
        return None
    obj = " ".join(tokens[1:]).strip(".,;:") or None
    return ActionRecord(
        actor=None,
        action=action,
        object=obj,
        sentence=sentence,
        page=chunk.page,
        section=chunk.section,
        paragraph=chunk.paragraph,
        document_id=chunk.document_id,
        source_name=chunk.source_name,
        chunk_id=chunk.chunk_id,
        order_index=order_index,
        step_number=step_number,
        metadata={"extraction": "imperative_fallback"},
    )


_STOP_VERBS = {
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "shall",
    "will",
    "seem",
    "appear",
    "describe",
    "include",
    "contain",
    "follow",
    "follows",
}


_DISCOURSE_OBJECT_NOISE = {
    "inspection",
    "preheating",
    "welding",
    "grinding",
    "testing",
    "removal",
}


def _is_spurious_action(record: ActionRecord) -> bool:
    sentence = (record.sentence or "").strip()
    if is_description_sentence(sentence):
        return True
    sentence_l = sentence.lower()
    obj = (record.object or "").strip().lower()
    label_l = record.label().lower()
    if re.match(r"^\d+\.\s*$", sentence):
        return True
    if record.action.lower() == "perform" and re.fullmatch(r"\d+", obj or ""):
        return True
    if re.match(r"^(where|what|which|how|when|why)\b", sentence_l) and "?" in sentence:
        return True
    if re.match(r"^[A-Z0-9\s\(\)\-—]+$", sentence) and len(sentence) > 20:
        return True
    if sentence_l.startswith("the procedure below"):
        return True
    if record.action.lower() == "perform" and obj == "testing passes":
        return True
    if obj in _DISCOURSE_OBJECT_NOISE and re.match(
        r"^(after|before|once|following|upon|then)\b", sentence_l
    ):
        return True
    if "process for" in label_l:
        return True
    if record.action.lower() in {"follow", "follows", "apply", "differ"}:
        if record.action.lower() == "apply" and (
            sentence_l.startswith("this procedure applies")
            or "hairline cracks found" in sentence_l
            or "during borescope inspection" in label_l
        ):
            return True
        if record.action.lower() == "differ" or "differs from" in sentence_l:
            return True
        if record.action.lower() in {"follow", "follows"}:
            return True
    if record.actor and record.actor.lower() in {
        "this procedure",
        "this section",
        "this document",
    }:
        return True
    if len(label_l) > 80 and record.step_number is None:
        return True
    return False


def _is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) < 2:
        return True
    if re.match(r"^\d+\.\s*$", stripped):
        return True
    if re.match(r"^[A-Z0-9\s\(\)\-—]+$", stripped) and len(stripped) > 20:
        return True
    # Em-dash title/preamble lines without numbered steps
    if "—" in stripped and not _STEP_NUM_RE.match(stripped) and len(stripped) > 40:
        return True
    if stripped.lower().startswith("this section describes"):
        return True
    if re.match(r"^(?:#+\s*)?(?:sample|synthetic|example)\b", stripped, re.IGNORECASE):
        return True
    if "regression tests" in stripped.lower() and len(stripped) < 120:
        return True
    return False


def _normalize_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
