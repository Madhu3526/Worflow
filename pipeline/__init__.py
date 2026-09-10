"""End-to-end offline pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from actions import ActionExtractor
from chunking import SemanticChunker
from config import AppConfig, get_config, setup_logging
from coref import CoreferenceResolver
from embeddings import EmbeddingService, FaissWorkflowIndex
from graph import WorkflowGraphBuilder
from models import ActionGroup, Chunk, DocumentContent, Workflow
from parser import DocumentParserService
from segmentation import WorkflowSegmenter
from sequence_markers import SequenceMarkerExtractor
from similarity import Deduplicator, WorkflowSimilarityEngine
from storage import WorkflowStore
from workflow import apply_summary

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    document: DocumentContent
    chunks: list[Chunk]
    action_count: int
    groups: list[ActionGroup]
    workflows: list[Workflow]
    merged_count: int = 0
    indexed_vectors: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class WorkflowDiscoveryPipeline:
    """Runs Modules 1–12 for a single document."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        setup_logging(self.config)
        self.parser = DocumentParserService(self.config)
        self.chunker = SemanticChunker(self.config)
        self.coref = CoreferenceResolver(self.config)
        self.actions = ActionExtractor(self.config)
        self.markers = SequenceMarkerExtractor(self.config)
        self.segmenter = WorkflowSegmenter(self.config)
        self.graphs = WorkflowGraphBuilder(self.config)
        self.store = WorkflowStore(self.config)
        self.embedder = EmbeddingService(self.config)
        self.index = FaissWorkflowIndex(self.config, self.embedder)
        self.similarity = WorkflowSimilarityEngine(self.config, self.embedder)
        self.deduper = Deduplicator(self.config, self.embedder, self.similarity)

    def process_file(self, path: str | Path, reindex: bool = True) -> PipelineResult:
        path = Path(path)
        logger.info("=== Processing document: %s ===", path.name)

        document = self.parser.parse(path)
        chunks = self.chunker.chunk(document)
        # Coref runs per instruction sentence inside ActionExtractor (not bulk chunk rewrite)
        extraction = self.actions.extract(chunks)
        actions = extraction.actions
        discourse = self.markers.extract(chunks, actions)
        from models import coerce_conditional_branch

        branches_by_chunk = {
            chunk_id: [coerce_conditional_branch(b) for b in branches]
            for chunk_id, branches in extraction.conditional_branches_by_chunk.items()
        }
        groups = self.segmenter.segment(
            actions,
            discourse,
            context_by_chunk=extraction.context_by_chunk,
            conditional_branches_by_chunk=branches_by_chunk,
        )

        built = self.graphs.build_many(groups)
        workflows: list[Workflow] = []
        for workflow, _graph in built:
            apply_summary(workflow)
            workflows.append(workflow)

        existing = self.store.load_all()
        kept: list[Workflow] = []
        merged_count = 0
        for workflow in workflows:
            dup = self.deduper.find_duplicate(workflow, existing + kept)
            if dup and self.config.deduplication.auto_merge:
                primary, sim = dup
                logger.info(
                    "Duplicate of %s (cosine=%.3f) — merging evidence",
                    primary.workflow_id,
                    sim,
                )
                # If primary is already on disk, merge into it; else into in-memory kept
                if any(w.workflow_id == primary.workflow_id for w in existing):
                    merged = self.store.merge_workflows(primary, workflow)
                    # Replace in existing list
                    existing = [merged if w.workflow_id == merged.workflow_id else w for w in existing]
                else:
                    primary.evidence_sources.extend(workflow.evidence_sources)
                    primary.pages = sorted(set(primary.pages) | set(workflow.pages))
                    primary.merged_from.append(workflow.workflow_id)
                merged_count += 1
            else:
                kept.append(workflow)

        saved = self.store.save_many(kept)

        indexed = 0
        if reindex:
            all_workflows = self.store.load_all()
            indexed = self.index.rebuild(all_workflows)
            self.store.update_stats(indexed_vectors=indexed)

        result = PipelineResult(
            document=document,
            chunks=chunks,
            action_count=len(actions),
            groups=groups,
            workflows=kept,
            merged_count=merged_count,
            indexed_vectors=indexed,
            details={
                "saved_paths": [str(p) for p in saved],
                "discourse_edges": len(discourse),
                "units": len(document.units),
            },
        )
        logger.info(
            "Done %s: %d chunks, %d actions, %d workflows (%d merged)",
            path.name,
            len(chunks),
            len(actions),
            len(kept),
            merged_count,
        )
        return result

    def reindex_all(self) -> int:
        workflows = self.store.load_all()
        count = self.index.rebuild(workflows)
        self.store.update_stats(indexed_vectors=count)
        self.store.rebuild_meta_from_disk()
        return count

    def repair_corpus(self) -> tuple[int, int]:
        """Sanitize persisted workflows and rebuild the FAISS index."""
        sanitized = self.store.sanitize_all_workflows()
        indexed = self.reindex_all()
        return sanitized, indexed
