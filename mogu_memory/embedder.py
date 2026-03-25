"""Embedding generation using Ruri v3-310m."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mogu_memory.config import Config

if TYPE_CHECKING:
    import numpy as np


class Embedder:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            import logging
            import os
            import warnings

            # Suppress noisy warnings before importing heavy libraries
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TQDM_DISABLE", "1")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            warnings.filterwarnings("ignore", message=".*reference_compile.*")

            from sentence_transformers import SentenceTransformer

            # Must set after import — importing resets log levels
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Ruri v3 expects "検索クエリ: " or "検索文書: " prefix
        vec = self.model.encode(f"検索文書: {text}", normalize_embeddings=True)
        return vec.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        vec = self.model.encode(f"検索クエリ: {query}", normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str], is_query: bool = False, batch_size: int = 1) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        prefix = "検索クエリ: " if is_query else "検索文書: "
        # Truncate to 1000 chars to avoid OOM on long tool results
        prefixed = [f"{prefix}{t[:1000]}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return [v.tolist() for v in vecs]
