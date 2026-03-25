"""Tests for chunker module."""

import json
import tempfile
from pathlib import Path

from mogu_memory.chunker import Chunk, chunk_messages, chunk_transcript


def test_chunk_simple_conversation():
    messages = [
        {"role": "human", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    chunks = chunk_messages(messages)
    assert len(chunks) == 1
    assert chunks[0].question == "What is Python?"
    assert chunks[0].answer == "Python is a programming language."
    assert chunks[0].chunk_index == 0


def test_chunk_multiple_turns():
    messages = [
        {"role": "human", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
        {"role": "human", "content": "Explain decorators"},
        {"role": "assistant", "content": "Decorators wrap functions to extend behavior."},
    ]
    chunks = chunk_messages(messages)
    assert len(chunks) == 2


def test_chunk_skips_trivial():
    messages = [
        {"role": "human", "content": "ok"},
        {"role": "assistant", "content": "ok"},
    ]
    chunks = chunk_messages(messages)
    assert len(chunks) == 0


def test_chunk_handles_content_blocks():
    messages = [
        {"role": "human", "content": "Read a file"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll read that file for you."},
                {"type": "tool_use", "name": "Read", "input": {"path": "/tmp/x.py"}},
            ],
        },
    ]
    chunks = chunk_messages(messages)
    assert len(chunks) == 1
    assert "Read" in chunks[0].answer


def test_chunk_long_text_split():
    long_answer = "x" * 5000
    messages = [
        {"role": "human", "content": "Generate a long response"},
        {"role": "assistant", "content": long_answer},
    ]
    chunks = chunk_messages(messages, max_chunk_chars=2000)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.raw_text) <= 2000


def test_chunk_transcript_from_file():
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "What is 2+2?"},
            "uuid": "1",
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "4"},
            "uuid": "2",
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")
        path = f.name

    chunks = chunk_transcript(path)
    assert len(chunks) == 1
    assert "2+2" in chunks[0].question

    Path(path).unlink()


def test_chunk_consecutive_human_messages():
    messages = [
        {"role": "human", "content": "First question"},
        {"role": "human", "content": "Actually, let me rephrase"},
        {"role": "assistant", "content": "Here's the answer."},
    ]
    chunks = chunk_messages(messages)
    assert len(chunks) == 1
    assert "rephrase" in chunks[0].question


def test_chunk_user_role_alias():
    """Claude Code transcripts may use 'user' instead of 'human'."""
    messages = [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]
    chunks = chunk_messages(messages)
    assert len(chunks) == 1


def test_chunk_transcript_nested_format():
    """Real Claude Code transcript uses nested format with type/message fields."""
    lines = [
        {"type": "file-history-snapshot", "messageId": "abc"},
        {
            "type": "user",
            "message": {"role": "user", "content": "Explain decorators"},
            "isMeta": False,
            "uuid": "1",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Decorators wrap functions."}],
            },
            "uuid": "2",
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")
        path = f.name

    chunks = chunk_transcript(path)
    assert len(chunks) == 1
    assert "decorator" in chunks[0].question.lower()
    assert "wrap" in chunks[0].answer.lower()

    Path(path).unlink()


def test_chunk_transcript_skips_meta():
    """Meta messages (isMeta=True) should be skipped."""
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "system caveat"},
            "isMeta": True,
            "uuid": "1",
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "What is Python?"},
            "isMeta": False,
            "uuid": "2",
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "A programming language."},
            "uuid": "3",
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")
        path = f.name

    chunks = chunk_transcript(path)
    assert len(chunks) == 1
    assert "Python" in chunks[0].question

    Path(path).unlink()
