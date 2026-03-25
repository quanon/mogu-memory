# CLAUDE.md

## Project Overview

mogu-memory is a long-term memory system for Claude Code sessions.
It stores conversation transcripts as Q&A chunks with hybrid search (FTS5 + vector).

## Tech Stack

- **Language**: Python 3.10+
- **Database**: SQLite + FTS5 (trigram) + sqlite-vec
- **Embedding**: Ruri v3-310m (sentence-transformers, 310-dim, local CPU)
- **CLI**: Click
- **Testing**: pytest

## Project Structure

```text
mogu_memory/
├── cli.py        # CLI entry point (save / search / stats)
├── chunker.py    # Transcript → Q&A chunk splitting (rule-based)
├── embedder.py   # Ruri v3-310m embedding generation
├── db.py         # SQLite + FTS5 + sqlite-vec operations
├── searcher.py   # Hybrid search + RRF fusion + time decay
└── config.py     # Configuration management
```

## Development

```bash
# Setup
uv sync --dev

# Run tests (fast, no model required)
uv run pytest tests/test_chunker.py tests/test_db.py tests/test_searcher.py

# Run all tests including slow embedding tests
uv run pytest -m slow
```

## CLI Usage

```bash
# Save a transcript
mogu-memory save --session-id <id> --transcript <path.jsonl> --project <path>

# Search memories
mogu-memory search "query" --top-k 5 --project <path>

# Show stats
mogu-memory stats
```

## Key Design Decisions

- Ruri v3 requires prefixes: `検索文書: ` for documents, `検索クエリ: ` for queries
- FTS5 uses trigram tokenizer for Japanese/English mixed text
- sqlite-vec KNN queries require `k = ?` constraint (not `LIMIT`)
- RRF fusion with k=60, time decay with 30-day half-life
- Chunks are Q&A pairs from Human/Assistant turns
- DB default path: `~/.mogu-memory/memories.db` (override via `MOGU_DB_PATH` env var)

## Code Conventions

- Write code comments and docstrings in English
- Use `from __future__ import annotations` for modern type hints
- Lazy-load heavy dependencies (sentence-transformers) to keep CLI responsive
