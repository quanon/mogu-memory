"""Tests for embedder module (uses actual model - slow on first run)."""

import pytest

from mogu_memory.config import Config
from mogu_memory.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder to avoid reloading the model per test."""
    return Embedder()


@pytest.mark.slow
def test_embed_single(embedder):
    vec = embedder.embed("Pythonでデコレータの使い方")
    assert len(vec) == 310
    assert all(isinstance(v, float) for v in vec)


@pytest.mark.slow
def test_embed_query(embedder):
    vec = embedder.embed_query("デコレータとは何ですか")
    assert len(vec) == 310


@pytest.mark.slow
def test_embed_batch(embedder):
    texts = ["テスト1", "テスト2", "テスト3"]
    vecs = embedder.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 310 for v in vecs)


@pytest.mark.slow
def test_embedding_normalized(embedder):
    """Embeddings should be L2-normalized (unit vectors)."""
    vec = embedder.embed("テスト文章")
    norm = sum(v**2 for v in vec) ** 0.5
    assert abs(norm - 1.0) < 0.01
