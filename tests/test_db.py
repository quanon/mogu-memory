"""Tests for database module."""

import tempfile
from pathlib import Path

import pytest

from mogu_memory.config import Config
from mogu_memory.db import MemoryDB


@pytest.fixture
def db(tmp_path):
    config = Config(db_path=tmp_path / "test.db", embedding_dim=4)
    database = MemoryDB(config)
    yield database
    database.close()


def _dummy_embedding(dim: int = 4) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4][:dim]


def test_save_and_stats(db):
    db.save_memory(
        session_id="s1",
        chunk_index=0,
        question="What is Python?",
        answer="A programming language.",
        raw_text="Q: What is Python?\nA: A programming language.",
        embedding=_dummy_embedding(),
    )
    stats = db.get_stats()
    assert stats["total_memories"] == 1
    assert stats["total_sessions"] == 1


def test_save_multiple(db):
    for i in range(3):
        db.save_memory(
            session_id="s1",
            chunk_index=i,
            question=f"Question {i}",
            answer=f"Answer {i}",
            raw_text=f"Q: Question {i}\nA: Answer {i}",
            embedding=_dummy_embedding(),
        )
    stats = db.get_stats()
    assert stats["total_memories"] == 3
    assert stats["total_sessions"] == 1


def test_upsert_same_session_chunk(db):
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="Q1", answer="A1", raw_text="Q1 A1",
        embedding=_dummy_embedding(),
    )
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="Q1 updated", answer="A1 updated", raw_text="Q1u A1u",
        embedding=_dummy_embedding(),
    )
    stats = db.get_stats()
    assert stats["total_memories"] == 1


def test_delete_session(db):
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="Q", answer="A", raw_text="QA",
        embedding=_dummy_embedding(),
    )
    deleted = db.delete_session("s1")
    assert deleted == 1
    assert db.get_stats()["total_memories"] == 0


def test_fts_search(db):
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="How to use decorators in Python?",
        answer="Decorators wrap functions to extend their behavior.",
        raw_text="Q: decorators\nA: wrap functions",
        embedding=_dummy_embedding(),
    )
    results = db.search_fts("decorator")
    assert len(results) >= 1
    assert "decorator" in results[0]["question"].lower()


def test_vec_search(db):
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="Q", answer="A", raw_text="QA",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    results = db.search_vec([1.0, 0.0, 0.0, 0.0])
    assert len(results) >= 1


def test_project_in_stats(db):
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="Q", answer="A", raw_text="QA",
        embedding=_dummy_embedding(),
        project_path="/home/user/project",
    )
    stats = db.get_stats()
    assert stats["projects"][0]["project_path"] == "/home/user/project"
