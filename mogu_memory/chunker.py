"""Conversation log chunking into Q&A format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    """A single Q&A chunk extracted from a conversation."""

    question: str
    answer: str
    raw_text: str
    chunk_index: int


def _is_tool_result_only(msg: dict[str, Any]) -> bool:
    """Check if a message consists only of tool_result blocks (no human text)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return False
    if not isinstance(content, list):
        return False
    return all(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


def _is_tool_use_only(msg: dict[str, Any]) -> bool:
    """Check if a message consists only of tool_use blocks (no text response)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return False
    if not isinstance(content, list):
        return False
    has_text = any(
        isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
        for b in content
    )
    return not has_text


def _extract_text_from_message(msg: dict[str, Any]) -> str:
    """Extract human-readable text from a message, ignoring tool blocks."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content

    # Only extract text blocks, skip tool_use/tool_result
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
    return "\n".join(parts)


def _is_noise_message(text: str) -> bool:
    """Check if a message is noise (slash commands, tool results only, etc.)."""
    stripped = text.strip()
    if not stripped:
        return True
    # Slash commands: <command-name>/foo</command-name>
    if "<command-name>" in stripped:
        return True
    # Tool result only messages (no human-readable content)
    if stripped.startswith("[Result]") and "\n" not in stripped:
        return True
    # Local command caveats / stdout
    if "<local-command-caveat>" in stripped:
        return True
    if "<local-command-stdout>" in stripped:
        return True
    # Single-word permission responses (Yes, No, y, n, etc.)
    if stripped.lower() in ("yes", "no", "y", "n", "ok"):
        return True
    return False


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text that exceeds max_chars into smaller pieces at paragraph boundaries."""
    max_chars = max(max_chars, 200)  # Guard against zero/negative values
    if len(text) <= max_chars:
        return [text]

    pieces = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        pieces.append(current.strip())

    # If any piece is still too long, force-split at max_chars
    result = []
    for piece in pieces:
        while len(piece) > max_chars:
            result.append(piece[:max_chars])
            piece = piece[max_chars:]
        if piece:
            result.append(piece)

    return result


def _parse_transcript_line(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single line from a Claude Code transcript JSONL.

    Claude Code transcripts use a nested format:
        {"type": "user"|"assistant", "message": {"role": "...", "content": "..."}, ...}
    Non-message lines (file-history-snapshot, system, etc.) are skipped.
    """
    line_type = obj.get("type", "")

    # Only process user and assistant messages
    if line_type not in ("user", "assistant"):
        return None

    # Skip meta/system messages (e.g. tool results from local commands)
    if obj.get("isMeta"):
        return None

    message = obj.get("message")
    if message and isinstance(message, dict):
        return message

    # Fallback: if "role" exists at top level (flat format)
    if "role" in obj:
        return obj

    return None


def chunk_transcript(transcript_path: str | Path, max_chunk_chars: int = 4000) -> list[Chunk]:
    """Parse a Claude Code transcript (JSONL) and chunk into Q&A pairs.

    Supports the nested Claude Code transcript format:
        {"type": "user", "message": {"role": "user", "content": "..."}, ...}
    """
    path = Path(transcript_path)
    messages = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = _parse_transcript_line(obj)
            if msg:
                messages.append(msg)

    return chunk_messages(messages, max_chunk_chars)


def chunk_messages(messages: list[dict[str, Any]], max_chunk_chars: int = 4000) -> list[Chunk]:
    """Chunk a list of messages into Q&A pairs."""
    # Group into (human, assistant) pairs
    pairs: list[tuple[str, str]] = []
    current_human = ""
    current_assistant = ""

    for msg in messages:
        role = msg.get("role", "")

        # Skip tool_result-only user messages (automated responses to tool calls)
        if role in ("human", "user") and _is_tool_result_only(msg):
            continue

        # Skip tool_use-only assistant messages (intermediate tool calls)
        if role == "assistant" and _is_tool_use_only(msg):
            continue

        text = _extract_text_from_message(msg)

        # Skip slash commands and system-generated messages
        if _is_noise_message(text):
            continue

        if role == "human" or role == "user":
            # If we have a pending pair, save it
            if current_human and current_assistant:
                pairs.append((current_human.strip(), current_assistant.strip()))
                current_assistant = ""
            if current_human and not current_assistant:
                # Consecutive human messages: merge
                current_human += "\n" + text
            else:
                current_human = text
        elif role == "assistant":
            current_assistant += "\n" + text if current_assistant else text

    # Don't forget the last pair
    if current_human and current_assistant:
        pairs.append((current_human.strip(), current_assistant.strip()))

    # Convert pairs to chunks, splitting long ones
    chunks: list[Chunk] = []
    idx = 0

    for question, answer in pairs:
        # Skip empty or trivial exchanges
        if len(question.strip()) < 5 and len(answer.strip()) < 5:
            continue

        raw_text = f"Q: {question}\nA: {answer}"

        if len(raw_text) <= max_chunk_chars:
            chunks.append(Chunk(
                question=question,
                answer=answer,
                raw_text=raw_text,
                chunk_index=idx,
            ))
            idx += 1
        else:
            # Split answer into pieces, keep question as context
            # Reserve space for "Q: " + question + " (continued NN/NN)" + "\nA: "
            overhead = len(question) + 50  # 50 chars for prefix/suffix/continued label
            answer_pieces = _split_long_text(answer, max_chunk_chars - overhead)
            for i, piece in enumerate(answer_pieces):
                q = question if i == 0 else f"{question} (continued {i + 1}/{len(answer_pieces)})"
                chunks.append(Chunk(
                    question=q,
                    answer=piece,
                    raw_text=f"Q: {q}\nA: {piece}",
                    chunk_index=idx,
                ))
                idx += 1

    return chunks
