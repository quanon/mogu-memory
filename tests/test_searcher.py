"""Tests for searcher module."""

import pytest

from mogu_memory.config import Config
from mogu_memory.db import MemoryDB
from mogu_memory.searcher import Searcher


@pytest.fixture
def config(tmp_path):
    return Config(db_path=tmp_path / "test.db", embedding_dim=4)


@pytest.fixture
def db(config):
    database = MemoryDB(config)
    yield database
    database.close()


def _dummy_embedding(val: float = 0.5) -> list[float]:
    return [val, val, val, val]


def test_search_returns_results(db, config):
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="How to use Python decorators?",
        answer="Use @decorator syntax above a function definition.",
        raw_text="Q: decorators\nA: @decorator syntax",
        embedding=_dummy_embedding(1.0),
    )

    class MockEmbedder:
        def embed_query(self, query):
            return _dummy_embedding(1.0)

    searcher = Searcher(db=db, embedder=MockEmbedder(), config=config)
    results = searcher.search("decorator")
    assert len(results) >= 1
    assert results[0]["final_score"] > 0


def test_search_empty_db(db, config):
    class MockEmbedder:
        def embed_query(self, query):
            return _dummy_embedding()

    searcher = Searcher(db=db, embedder=MockEmbedder(), config=config)
    results = searcher.search("anything")
    assert results == []


def test_search_project_filter(db, config):
    db.save_memory(
        session_id="s1", chunk_index=0,
        question="Q1", answer="A1", raw_text="Q1 A1",
        embedding=_dummy_embedding(1.0),
        project_path="/project/a",
    )
    db.save_memory(
        session_id="s2", chunk_index=0,
        question="Q2", answer="A2", raw_text="Q2 A2",
        embedding=_dummy_embedding(0.9),
        project_path="/project/b",
    )

    class MockEmbedder:
        def embed_query(self, query):
            return _dummy_embedding(1.0)

    searcher = Searcher(db=db, embedder=MockEmbedder(), config=config)
    results = searcher.search("Q", project_path="/project/a")
    assert all(r["project_path"] == "/project/a" for r in results)
