"""Configuration management for mogu-memory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_db_path() -> Path:
    return Path(os.environ.get("MOGU_DB_PATH", Path.home() / ".mogu-memory" / "memories.db"))


@dataclass
class Config:
    """Application configuration."""

    # Database
    db_path: Path = field(default_factory=_default_db_path)

    # Embedding model
    embedding_model: str = "cl-nagoya/ruri-v3-310m"
    embedding_dim: int = 768

    # Search parameters
    rrf_k: int = 60
    decay_half_life_days: float = 30.0
    default_top_k: int = 5

    # Chunking
    max_chunk_chars: int = 4000

    def ensure_db_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
