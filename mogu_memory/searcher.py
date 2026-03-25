"""Hybrid search with RRF score fusion and time decay."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from mogu_memory.config import Config
from mogu_memory.db import MemoryDB
from mogu_memory.embedder import Embedder


class Searcher:
    """Hybrid search combining FTS5 keyword search and vector similarity."""

    def __init__(
        self,
        db: MemoryDB | None = None,
        embedder: Embedder | None = None,
        config: Config | None = None,
    ) -> None:
        self.config = config or Config()
        self.db = db or MemoryDB(self.config)
        self.embedder = embedder or Embedder(self.config)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search with RRF fusion and time decay.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            project_path: Optional filter by project path.

        Returns:
            List of memory dicts with 'final_score' field, sorted by score descending.
        """
        top_k = top_k or self.config.default_top_k
        k = self.config.rrf_k

        # Run both searches
        fts_results = self.db.search_fts(query)
        query_embedding = self.embedder.embed_query(query)
        vec_results = self.db.search_vec(query_embedding)

        # Build RRF scores
        scores: dict[int, float] = {}
        memory_data: dict[int, dict[str, Any]] = {}

        for rank, result in enumerate(fts_results):
            mid = result["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
            memory_data[mid] = result

        for rank, result in enumerate(vec_results):
            mid = result["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
            memory_data[mid] = result

        # Apply time decay and optional project filter
        now = datetime.now(timezone.utc)
        results = []

        for mid, rrf_score in scores.items():
            data = memory_data[mid]

            # Project filter
            if project_path and data.get("project_path") != project_path:
                continue

            # Time decay: 0.5 ^ (days_elapsed / half_life)
            created_str = data.get("created_at", "")
            try:
                created = datetime.fromisoformat(created_str).replace(tzinfo=timezone.utc)
                days_elapsed = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                days_elapsed = 0.0

            decay = math.pow(0.5, days_elapsed / self.config.decay_half_life_days)
            final_score = rrf_score * decay

            results.append({
                "id": mid,
                "session_id": data.get("session_id", ""),
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "project_path": data.get("project_path"),
                "created_at": created_str,
                "rrf_score": rrf_score,
                "decay": decay,
                "final_score": final_score,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]
