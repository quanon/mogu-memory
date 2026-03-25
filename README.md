# mogu-memory

<p align="center">
  <img src="assets/logo.png" alt="mogu-memory" width="320">
</p>

Long-term memory system for [Claude Code](https://claude.ai/code) sessions.

Automatically saves conversation transcripts as Q&A chunks and provides hybrid search (FTS5 full-text + vector similarity) so Claude can recall context from past sessions.

## How it works

1. **Save**: When a Claude Code session ends, a hook script chunks the transcript into Q&A pairs, generates embeddings with [Ruri v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m), and stores everything in a local SQLite database.
2. **Search**: A custom slash command (`/mogu-memory-search`) lets Claude search past sessions using hybrid ranking (BM25 + cosine similarity + time decay).

## Tech stack

- **Python 3.14+** / **uv**
- **SQLite** + FTS5 (trigram tokenizer) + [sqlite-vec](https://github.com/asg017/sqlite-vec)
- **Ruri v3-310m**: Japanese/English bilingual embedding model (310-dim, runs on CPU)
- **Click**: CLI framework

## Setup

```bash
git clone <repo-url>
cd mogu-memory
uv sync
```

## Integration with Claude Code

### 1. Auto-save hook (SessionEnd)

Create `~/.claude/hooks/mogu-memory-save.sh`:

```bash
#!/bin/bash
# Save Claude Code session transcript to mogu-memory on SessionEnd
set -euo pipefail

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id')
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path')
CWD=$(echo "$INPUT" | jq -r '.cwd')

if [ ! -f "$TRANSCRIPT" ]; then
  exit 0
fi

MOGU="$HOME/workspace/mogu-memory"
LOG="$HOME/.mogu-memory/save.log"

# Run in background to avoid hook timeout (model loading is slow)
nohup uv run --project "$MOGU" mogu-memory save \
  --session-id "$SESSION_ID" \
  --transcript "$TRANSCRIPT" \
  --project "$CWD" \
  >> "$LOG" 2>&1 &
```

```bash
chmod +x ~/.claude/hooks/mogu-memory-save.sh
```

Register the hook in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/mogu-memory-save.sh"
          }
        ]
      }
    ]
  }
}
```

### 2. Search skill (`/mogu-memory-search`)

Create `~/.claude/skills/mogu-memory-search/SKILL.md`:

````markdown
---
name: mogu-memory-search
description: Search past Claude Code session memories
argument-hint: <search query>
---

Search mogu-memory for past conversation context related to the query.

Run this command to search:

```bash
$HOME/workspace/mogu-memory/.venv/bin/mogu-memory search "$ARGUMENTS" --top-k 5 --json-output
```

Display the results to the user in a readable format. If results are found, summarize the key points from past sessions that are relevant to the query.
````

Now you can type `/mogu-memory-search <query>` in any Claude Code session to recall past context.

## CLI reference

```bash
# Save a transcript
mogu-memory save --session-id <id> --transcript <path.jsonl> --project <path>

# Search memories
mogu-memory search "query" --top-k 5 --project <path>

# Show database stats
mogu-memory stats

# Delete all memories
mogu-memory reset        # with confirmation prompt
mogu-memory reset -y     # skip confirmation
```

## Data storage

- Database: `~/.mogu-memory/memories.db` (override with `MOGU_DB_PATH` env var)
- Save log: `~/.mogu-memory/save.log`

## Development

```bash
uv sync --dev

# Run fast tests (no model required)
uv run pytest tests/test_chunker.py tests/test_db.py tests/test_searcher.py

# Run all tests including embedding tests
uv run pytest -m slow
```
