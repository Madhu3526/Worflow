"""Sentence-Transformers embeddings + local FAISS index (CPU-only)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config import AppConfig, get_config
from models import Workflow

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embed text with all-MiniLM-L6-v2 on CPU."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model %s on %s",
                self.config.embeddings.model_name,
                self.config.embeddings.device,
            )
            self._model = SentenceTransformer(
                self.config.embeddings.model_name,
                device=self.config.embeddings.device,
            )
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.config.embeddings.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.embeddings.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-9)
        b_n = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_n, b_n))


class FaissWorkflowIndex:
    """FAISS index with parallel metadata JSON mapping vector IDs → workflow IDs."""

    def __init__(
        self,
        config: AppConfig | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.config = config or get_config()
        self.embedder = embedder or EmbeddingService(self.config)
        self.index_path = self.config.paths.resolve("faiss_index")
        self.meta_path = self.config.paths.resolve("faiss_meta")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._meta: dict[str, Any] = {"dim": 384, "entries": []}
        self._load()

    def _load(self) -> None:
        import faiss

        if self.meta_path.exists():
            self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        dim = int(self._meta.get("dim", 384))
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            logger.info("Loaded FAISS index with %d vectors", self._index.ntotal)
        else:
            self._index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
            self._meta = {"dim": dim, "entries": []}

    def save(self) -> None:
        import faiss

        assert self._index is not None
        faiss.write_index(self._index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        logger.info("Persisted FAISS index (%d vectors)", self._index.ntotal)

    def clear(self) -> None:
        import faiss

        dim = int(self._meta.get("dim", 384))
        self._index = faiss.IndexFlatIP(dim)
        self._meta = {"dim": dim, "entries": []}
        self.save()

    def rebuild(self, workflows: Sequence[Workflow]) -> int:
        """Re-index action texts, node labels, and workflow summaries."""
        import faiss

        self.clear()
        texts: list[str] = []
        entries: list[dict[str, Any]] = []

        for workflow in workflows:
            # Summary vector
            texts.append(workflow.summary or workflow.workflow_id)
            entries.append(
                {
                    "vector_id": len(entries),
                    "workflow_id": workflow.workflow_id,
                    "kind": "summary",
                    "text": workflow.summary,
                }
            )
            for node in workflow.ordered_nodes:
                label = node.label()
                texts.append(label)
                entries.append(
                    {
                        "vector_id": len(entries),
                        "workflow_id": workflow.workflow_id,
                        "kind": "node",
                        "text": label,
                        "node_id": node.node_id,
                    }
                )
                if node.sentence:
                    texts.append(node.sentence)
                    entries.append(
                        {
                            "vector_id": len(entries),
                            "workflow_id": workflow.workflow_id,
                            "kind": "action",
                            "text": node.sentence,
                            "node_id": node.node_id,
                        }
                    )

        if not texts:
            self.save()
            return 0

        vectors = self.embedder.embed(texts)
        dim = int(vectors.shape[1])
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self._meta = {"dim": dim, "entries": entries}
        self.save()
        return len(entries)

    def add_workflow(self, workflow: Workflow) -> None:
        payloads: list[tuple[str, dict[str, Any]]] = [
            (workflow.summary or workflow.workflow_id, {"kind": "summary", "text": workflow.summary}),
        ]
        for node in workflow.ordered_nodes:
            payloads.append((node.label(), {"kind": "node", "text": node.label(), "node_id": node.node_id}))
            if node.sentence:
                payloads.append(
                    (node.sentence, {"kind": "action", "text": node.sentence, "node_id": node.node_id})
                )

        texts = [t for t, _ in payloads]
        vectors = self.embedder.embed(texts)
        assert self._index is not None
        start = self._index.ntotal
        self._index.add(vectors)
        for offset, (_, meta) in enumerate(payloads):
            self._meta["entries"].append(
                {
                    "vector_id": start + offset,
                    "workflow_id": workflow.workflow_id,
                    **meta,
                }
            )
        self.save()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        assert self._index is not None
        if self._index.ntotal == 0:
            return []
        vector = self.embedder.embed_one(query).reshape(1, -1)
        scores, indices = self._index.search(vector, min(top_k * 3, self._index.ntotal))
        hits: list[dict[str, Any]] = []
        seen_workflows: set[str] = set()
        entries = self._meta.get("entries", [])
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(entries):
                continue
            entry = entries[idx]
            wf_id = entry["workflow_id"]
            if wf_id in seen_workflows:
                continue
            seen_workflows.add(wf_id)
            hits.append(
                {
                    "workflow_id": wf_id,
                    "score": float(score),
                    "kind": entry.get("kind"),
                    "text": entry.get("text"),
                    "vector_id": int(idx),
                }
            )
            if len(hits) >= top_k:
                break
        return hits

    @property
    def size(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)
