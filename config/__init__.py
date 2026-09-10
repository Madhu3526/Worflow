"""Configuration loading for the workflow discovery engine."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    data_dir: str = "data"
    workflows_dir: str = "data/workflows"
    indexes_dir: str = "data/indexes"
    logs_dir: str = "logs"
    meta_file: str = "data/workflows_meta.json"
    faiss_index: str = "data/indexes/workflows.index"
    faiss_meta: str = "data/indexes/faiss_meta.json"

    def resolve(self, key: str) -> Path:
        relative = Path(getattr(self, key))
        path = relative if relative.is_absolute() else PROJECT_ROOT / relative
        return path


class ParsingConfig(BaseModel):
    extract_tables: bool = True
    heading_font_size_delta: float = 2.0
    min_heading_length: int = 3


class ChunkingConfig(BaseModel):
    numbered_step_pattern: str = (
        r"^\s*(\d+[\.\)]\s+|\d+\.\d+\s+|Step\s+\d+[:\.\s]+|[a-z][\.\)]\s+|[-•▪]\s+)"
    )
    preserve_tables: bool = True


class CorefConfig(BaseModel):
    device: str = "cpu"
    resolve_text: bool = True
    allow_passthrough: bool = True


class ActionsConfig(BaseModel):
    min_verb_length: int = 2
    include_passive: bool = True
    spacy_model: str = "en_core_web_sm"
    split_list_separators: bool = True
    split_on_newlines: bool = True


class SequenceMarkersConfig(BaseModel):
    markers: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "AFTER": ["after", "once", "following", "upon", "subsequent to", "thereafter"],
            "BEFORE": ["before", "prior to", "preceding"],
            "THEN": ["then", "next", "afterward", "afterwards", "subsequently"],
            "IF": ["if", "when", "whenever", "provided that"],
        }
    )
    case_insensitive: bool = True


class SegmentationConfig(BaseModel):
    structural_weight: float = 0.35
    lexical_weight: float = 0.40
    continuity_weight: float = 0.25
    boundary_threshold: float = 0.45
    merge_low_confidence: bool = True
    merge_confidence_ceiling: float = 0.55
    sequence_gap_window: int = 5
    lexical_patterns: list[str] = Field(default_factory=list)


class GraphConfig(BaseModel):
    use_numbered_steps: bool = True
    use_document_order: bool = True
    use_discourse_edges: bool = True


class SimilarityConfig(BaseModel):
    set_overlap_weight: float = 0.4
    sequence_alignment_weight: float = 0.4
    structural_weight: float = 0.2
    query_semantic_weight: float = 0.6
    query_set_overlap_weight: float = 0.4
    action_cluster_threshold: float = 0.7
    nw_gap_penalty: float = 0.35
    graph_edit_timeout: float = 2.0
    max_nodes_for_ged: int = 25


class DeduplicationConfig(BaseModel):
    cosine_threshold: float = 0.9
    auto_merge: bool = True


class EmbeddingsConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 5
    min_score: float = 0.25
    max_evidence_sentences: int = 8


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class AppConfig(BaseSettings):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    coref: CorefConfig = Field(default_factory=CorefConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    sequence_markers: SequenceMarkersConfig = Field(default_factory=SequenceMarkersConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def ensure_directories(self) -> None:
        for key in ("data_dir", "workflows_dir", "indexes_dir", "logs_dir"):
            self.paths.resolve(key).mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> AppConfig:
    path = Path(config_path) if config_path else Path(__file__).parent / "settings.yaml"
    raw = _load_yaml(path)
    config = AppConfig(**raw)
    config.ensure_directories()
    return config


def setup_logging(config: AppConfig | None = None) -> None:
    cfg = config or get_config()
    log_dir = cfg.paths.resolve("logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format=cfg.logging.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "engine.log", encoding="utf-8"),
        ],
    )


__all__ = ["AppConfig", "PROJECT_ROOT", "get_config", "setup_logging"]
