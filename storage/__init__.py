"""Local file-based workflow persistence (JSON + meta index). No database."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import AppConfig, get_config
from models import CorpusStats, Workflow, coerce_workflow

logger = logging.getLogger(__name__)


class WorkflowStore:
    """Persist workflows as JSON files and maintain workflows_meta.json."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.workflows_dir = self.config.paths.resolve("workflows_dir")
        self.meta_path = self.config.paths.resolve("meta_file")
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            self._write_meta({"workflows": [], "stats": CorpusStats().model_dump()})

    def save(self, workflow: Workflow) -> Path:
        path = self.workflows_dir / f"{workflow.workflow_id}.json"
        path.write_text(workflow.model_dump_json(indent=2), encoding="utf-8")
        self._upsert_meta_entry(workflow)
        logger.info("Saved workflow %s → %s", workflow.workflow_id, path.name)
        return path

    def save_many(self, workflows: list[Workflow]) -> list[Path]:
        return [self.save(wf) for wf in workflows]

    def load(self, workflow_id: str) -> Workflow | None:
        path = self.workflows_dir / f"{workflow_id}.json"
        if not path.exists():
            return None
        return coerce_workflow(Workflow.model_validate_json(path.read_text(encoding="utf-8")))

    def load_all(self) -> list[Workflow]:
        workflows: list[Workflow] = []
        for path in sorted(self.workflows_dir.glob("wf_*.json")):
            try:
                workflows.append(
                    coerce_workflow(Workflow.model_validate_json(path.read_text(encoding="utf-8")))
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s: %s", path.name, exc)
        return workflows

    def delete(self, workflow_id: str) -> bool:
        path = self.workflows_dir / f"{workflow_id}.json"
        existed = path.exists()
        if existed:
            path.unlink()
        meta = self._read_meta()
        meta["workflows"] = [w for w in meta.get("workflows", []) if w.get("workflow_id") != workflow_id]
        self._write_meta(meta)
        return existed

    def clear_all(self) -> int:
        """Remove all persisted workflows and reset corpus metadata."""
        removed = 0
        for path in self.workflows_dir.glob("wf_*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        self._write_meta({"workflows": [], "stats": CorpusStats().model_dump()})
        logger.info("Cleared %d workflow files from %s", removed, self.workflows_dir)
        return removed

    def list_meta(self) -> list[dict[str, Any]]:
        return list(self._read_meta().get("workflows", []))

    def get_stats(self) -> CorpusStats:
        raw = self._read_meta().get("stats") or {}
        return CorpusStats.model_validate(raw)

    def update_stats(self, **kwargs: Any) -> CorpusStats:
        meta = self._read_meta()
        stats = CorpusStats.model_validate(meta.get("stats") or {})
        data = stats.model_dump()
        data.update(kwargs)
        updated = CorpusStats.model_validate(data)
        meta["stats"] = updated.model_dump()
        self._write_meta(meta)
        return updated

    def rebuild_meta_from_disk(self) -> None:
        workflows = self.load_all()
        entries = [_meta_entry(wf) for wf in workflows]
        docs = {wf.document_id for wf in workflows}
        stats = CorpusStats(
            documents_processed=len(docs),
            workflows_found=len(workflows),
            actions_extracted=sum(len(wf.nodes) for wf in workflows),
            avg_confidence=0.0,
            indexed_vectors=0,
        )
        self._write_meta({"workflows": entries, "stats": stats.model_dump()})

    def sanitize_all_workflows(self) -> int:
        """Rewrite all workflows on disk after runtime node cleanup."""
        from workflow import apply_summary
        from workflow.sanitize import sanitize_workflow

        updated = 0
        for wf in self.load_all():
            clean = apply_summary(sanitize_workflow(wf))
            if (
                len(clean.nodes) != len(wf.nodes)
                or clean.summary != wf.summary
                or clean.evidence_sources != wf.evidence_sources
            ):
                updated += 1
            self.save(clean)
        if updated:
            self.rebuild_meta_from_disk()
        return updated

    def merge_workflows(self, primary: Workflow, duplicate: Workflow) -> Workflow:
        """Combine duplicate under one workflowId with multiple evidence_sources."""
        existing_sentences = {e.sentence for e in primary.evidence_sources}
        for evidence in duplicate.evidence_sources:
            if evidence.sentence not in existing_sentences:
                primary.evidence_sources.append(evidence)
                existing_sentences.add(evidence.sentence)

        primary.pages = sorted(set(primary.pages) | set(duplicate.pages))
        for section in duplicate.sections:
            if section not in primary.sections:
                primary.sections.append(section)
        primary.merged_from = list(set(primary.merged_from + [duplicate.workflow_id] + duplicate.merged_from))
        primary.metadata["merged_documents"] = list(
            {
                *(primary.metadata.get("merged_documents") or [primary.document]),
                duplicate.document,
            }
        )
        self.save(primary)
        self.delete(duplicate.workflow_id)
        logger.info("Merged %s into %s", duplicate.workflow_id, primary.workflow_id)
        return primary

    def _upsert_meta_entry(self, workflow: Workflow) -> None:
        meta = self._read_meta()
        entries = [w for w in meta.get("workflows", []) if w.get("workflow_id") != workflow.workflow_id]
        entries.append(_meta_entry(workflow))
        meta["workflows"] = entries
        stats = CorpusStats.model_validate(meta.get("stats") or {})
        docs = {e.get("document_id") for e in entries}
        stats.documents_processed = len(docs)
        stats.workflows_found = len(entries)
        stats.actions_extracted = sum(int(e.get("node_count", 0)) for e in entries)
        meta["stats"] = stats.model_dump()
        self._write_meta(meta)

    def _read_meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {"workflows": [], "stats": CorpusStats().model_dump()}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _meta_entry(workflow: Workflow) -> dict[str, Any]:
    pages = workflow.pages
    page_str = f"{pages[0]}–{pages[-1]}" if len(pages) > 1 else (str(pages[0]) if pages else "")
    return {
        "workflow_id": workflow.workflow_id,
        "document": workflow.document,
        "document_id": workflow.document_id,
        "title_hint": workflow.title_hint,
        "summary": workflow.summary,
        "pages": pages,
        "page_range": page_str,
        "sections": workflow.sections,
        "node_count": len(workflow.nodes),
        "edge_count": len(workflow.edges),
        "merged_from": workflow.merged_from,
    }
