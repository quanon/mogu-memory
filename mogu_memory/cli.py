"""CLI entry point for mogu-memory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from mogu_memory.config import Config


@click.group()
@click.option("--db-path", type=click.Path(), default=None, help="Path to SQLite database file.")
@click.pass_context
def cli(ctx: click.Context, db_path: str | None) -> None:
    """mogu-memory: Long-term memory system for Claude Code sessions."""
    config = Config()
    if db_path:
        config.db_path = Path(db_path)
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.option("--session-id", required=True, help="Session identifier.")
@click.option("--transcript", required=True, type=click.Path(exists=True), help="Path to transcript JSONL file.")
@click.option("--project", default=None, help="Project path for context.")
@click.pass_context
def save(ctx: click.Context, session_id: str, transcript: str, project: str | None) -> None:
    """Save a conversation transcript to memory."""
    config = ctx.obj["config"]

    from mogu_memory.chunker import chunk_transcript
    from mogu_memory.db import MemoryDB
    from mogu_memory.embedder import Embedder

    click.echo(f"Chunking transcript: {transcript}")
    chunks = chunk_transcript(transcript, config.max_chunk_chars)

    if not chunks:
        click.echo("No chunks extracted. Nothing to save.")
        return

    click.echo(f"Extracted {len(chunks)} chunks. Generating embeddings...")

    embedder = Embedder(config)
    texts = [c.raw_text for c in chunks]
    embeddings = embedder.embed_batch(texts)

    click.echo("Saving to database...")
    db = MemoryDB(config)

    # Delete existing data for this session (upsert)
    deleted = db.delete_session(session_id)
    if deleted:
        click.echo(f"Replaced {deleted} existing memories for session {session_id}.")

    for chunk, embedding in zip(chunks, embeddings):
        db.save_memory(
            session_id=session_id,
            chunk_index=chunk.chunk_index,
            question=chunk.question,
            answer=chunk.answer,
            raw_text=chunk.raw_text,
            embedding=embedding,
            project_path=project,
        )

    db.close()
    click.echo(f"Saved {len(chunks)} memories for session {session_id}.")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=None, type=int, help="Number of results to return.")
@click.option("--project", default=None, help="Filter by project path.")
@click.option("--json-output", is_flag=True, help="Output as JSON.")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int | None, project: str | None, json_output: bool) -> None:
    """Search memories for a query."""
    if not query.strip():
        click.echo("No results found.")
        return

    config = ctx.obj["config"]

    if not config.db_path.exists():
        click.echo("No results found.")
        return

    from mogu_memory.db import MemoryDB

    db = MemoryDB(config)
    stats = db.get_stats()
    if stats["total_memories"] == 0:
        db.close()
        click.echo("No results found.")
        return

    from mogu_memory.searcher import Searcher

    searcher = Searcher(db=db, config=config)
    results = searcher.search(query, top_k=top_k, project_path=project)

    if not results:
        click.echo("No results found.")
        return

    if json_output:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n--- Result {i} (score: {r['final_score']:.4f}) ---")
        click.echo(f"Session: {r['session_id']}")
        click.echo(f"Project: {r.get('project_path', 'N/A')}")
        click.echo(f"Date: {r.get('created_at', 'N/A')}")
        click.echo(f"Q: {r['question'][:200]}")
        click.echo(f"A: {r['answer'][:500]}")


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics."""
    config = ctx.obj["config"]

    from mogu_memory.db import MemoryDB

    db = MemoryDB(config)
    s = db.get_stats()
    db.close()

    click.echo(f"Total memories: {s['total_memories']}")
    click.echo(f"Total sessions: {s['total_sessions']}")
    click.echo("\nBy project:")
    for p in s["projects"]:
        path = p["project_path"] or "(no project)"
        click.echo(f"  {path}: {p['cnt']} memories")


if __name__ == "__main__":
    cli()
