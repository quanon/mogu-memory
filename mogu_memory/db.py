"""SQLite + FTS5 + sqlite-vec database operations."""

from __future__ import annotations

import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from mogu_memory.config import Config


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes, dim: int) -> list[float]:
    """Deserialize binary blob back to a list of floats."""
    return list(struct.unpack(f"{dim}f", blob))


class MemoryDB:
    """Database interface for memories storage."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.config.ensure_db_dir()
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.config.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables(conn)
        return conn

    def _init_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                project_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, chunk_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                question,
                answer,
                raw_text,
                content='memories',
                content_rowid='id',
                tokenize='trigram'
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, question, answer, raw_text)
                VALUES (new.id, new.question, new.answer, new.raw_text);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, question, answer, raw_text)
                VALUES ('delete', old.id, old.question, old.answer, old.raw_text);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, question, answer, raw_text)
                VALUES ('delete', old.id, old.question, old.answer, old.raw_text);
                INSERT INTO memories_fts(rowid, question, answer, raw_text)
                VALUES (new.id, new.question, new.answer, new.raw_text);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                memory_id INTEGER PRIMARY KEY,
                embedding float[{self.config.embedding_dim}]
            );
        """)

    def save_memory(
        self,
        session_id: str,
        chunk_index: int,
        question: str,
        answer: str,
        raw_text: str,
        embedding: list[float],
        project_path: str | None = None,
    ) -> int:
        """Save a single memory chunk. Returns the memory id."""
        cur = self.conn.execute(
            """INSERT INTO memories (session_id, chunk_index, question, answer, raw_text, project_path)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, chunk_index) DO UPDATE SET
                   question=excluded.question,
                   answer=excluded.answer,
                   raw_text=excluded.raw_text,
                   project_path=excluded.project_path,
                   created_at=CURRENT_TIMESTAMP""",
            (session_id, chunk_index, question, answer, raw_text, project_path),
        )
        memory_id = cur.lastrowid

        # Upsert vector
        self.conn.execute(
            "DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,)
        )
        self.conn.execute(
            "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
            (memory_id, _serialize_f32(embedding)),
        )
        self.conn.commit()
        return memory_id

    def delete_session(self, session_id: str) -> int:
        """Delete all memories for a session. Returns count of deleted rows."""
        # Get IDs to delete from vec table
        rows = self.conn.execute(
            "SELECT id FROM memories WHERE session_id = ?", (session_id,)
        ).fetchall()
        ids = [r["id"] for r in rows]

        if ids:
            placeholders = ",".join("?" * len(ids))
            self.conn.execute(
                f"DELETE FROM memories_vec WHERE memory_id IN ({placeholders})", ids
            )
            self.conn.execute(
                "DELETE FROM memories WHERE session_id = ?", (session_id,)
            )
            self.conn.commit()
        return len(ids)

    def search_fts(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Full-text search using FTS5 trigram tokenizer."""
        # Escape double quotes in query for FTS5
        escaped = query.replace('"', '""')
        rows = self.conn.execute(
            """SELECT m.id, m.session_id, m.question, m.answer, m.raw_text,
                      m.project_path, m.created_at,
                      rank AS score
               FROM memories_fts fts
               JOIN memories m ON m.id = fts.rowid
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (f'"{escaped}"', limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_vec(self, embedding: list[float], limit: int = 50) -> list[dict[str, Any]]:
        """Vector similarity search using sqlite-vec."""
        rows = self.conn.execute(
            """SELECT v.memory_id AS id, v.distance AS score,
                      m.session_id, m.question, m.answer, m.raw_text,
                      m.project_path, m.created_at
               FROM memories_vec v
               JOIN memories m ON m.id = v.memory_id
               WHERE embedding MATCH ?
                 AND k = ?
               ORDER BY distance""",
            (_serialize_f32(embedding), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        total = self.conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()["cnt"]
        sessions = self.conn.execute(
            "SELECT COUNT(DISTINCT session_id) AS cnt FROM memories"
        ).fetchone()["cnt"]
        projects = self.conn.execute(
            """SELECT project_path, COUNT(*) AS cnt
               FROM memories
               GROUP BY project_path
               ORDER BY cnt DESC"""
        ).fetchall()
        return {
            "total_memories": total,
            "total_sessions": sessions,
            "projects": [dict(r) for r in projects],
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
