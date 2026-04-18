# Adaptive Memory for Open WebUI

Give your Open WebUI assistant persistent memory across conversations.

This function remembers useful things you tell it, brings back relevant memories in later chats, and can optionally mirror those memories to Mem0.

## What It Does

Once enabled, the function works in the background to:
- Save important facts, preferences, goals, and relationships from your conversations
- Recall relevant memories when you start a new chat or continue an old one
- Avoid saving obvious duplicates
- Keep memory growth under control with pruning and summarization
- Optionally mirror memory changes to Mem0

Example:
- You say: "I prefer Python over JavaScript."
- Later, when you ask for coding help, that preference can be recalled automatically.

## How It Works

At a high level:
1. You chat normally.
2. The function looks at your recent user messages and asks an LLM to identify memories worth saving.
3. Those memories are stored in Open WebUI's memory system.
4. On future messages, the function finds the most relevant saved memories and injects them back into context.

Everything is designed to stay mostly automatic once configured.

## Installation

1. Upload `adaptive_memory_v4.0.py` in Open WebUI Functions.
2. Open the function settings and configure the valves you want.
3. Enable the function for the model or models you use.

## Recommended Setup

### For Most People

Use:
- `embedding_source = auto`
- `embedding_provider_type = local`
- `embedding_model_name = all-MiniLM-L6-v2`
- your normal Ollama or OpenAI-compatible model for memory extraction

This gives you a solid default setup with local embeddings when available.

### If You Want Mem0 Mirroring

Enable:
- `enable_mem0_sync = true`

Then configure:
- `mem0_api_base_url`
- `mem0_api_key`
- `mem0_app_id`

If you want all mirrored memories to go to one Mem0 user ID, set:
- `mem0_user_id_override = jefe`

## Important Settings

### Memory Quality

Useful valves:
- `recent_messages_n`: how much recent user context to use during extraction
- `related_memories_n`: how many relevant memories to inject
- `relevance_threshold`: how strict retrieval should be
- `deduplicate_memories`: whether to skip near-duplicates
- `use_embeddings_for_deduplication`: usually best left on

### Memory Size Control

Useful valves:
- `max_total_memories`: max memories per user before pruning begins
- `pruning_strategy`: `fifo` or `least_relevant`
- `enable_summarization_task`: whether to summarize older memories
- `summarization_interval`: how often summarization runs

### Chat Experience

Useful valves:
- `show_memories`: whether recalled memories are injected into prompt context
- `show_status`: whether chat status messages are shown
- `memory_format`: `bullet`, `paragraph`, or `numbered`

### Mem0

Useful valves:
- `enable_mem0_sync`
- `mem0_user_id_template`
- `mem0_user_id_override`
- `mem0_timeout_seconds`

## Mem0 User ID Behavior

If Mem0 mirroring is enabled, the function decides which Mem0 user ID to use in this order:
1. Per-user `mem0_user_id_override`
2. Global `mem0_user_id_override`
3. Previously stored Mem0 user mapping
4. `mem0_user_id_template`

In practice, this means:
- If you set the global override to `jefe`, new mirrored memories will use `jefe`
- A per-user override can still beat the global one
- Older cached mappings no longer win over the global override

## What Stays In Open WebUI

Open WebUI remains the local source of truth for the actual memories.

This function adds:
- embedding cache persistence
- optional Mem0 sync state
- retrieval, deduplication, summarization, and cleanup logic

## Background Features

The function currently supports:
- Memory summarization
- Error counter logging
- Orphaned vector cleanup

It also has valves for some future-facing background settings, but not every valve in the schema currently has a matching running task in this file.

## Files It Maintains

Besides Open WebUI's own memory records, the function may create private sidecar files under `DATA_DIR/cache`:
- `embeddings.sqlite`
- `mem0_sync.sqlite`

These are internal support files used for embedding persistence and Mem0 mapping state.

## Requirements

Needed:
- Open WebUI
- `numpy`
- `aiohttp`
- `pydantic`
- `pytz`

Optional:
- `sentence-transformers` for local embeddings
- `prometheus-client` for metrics

## Notes

- The function stores memories in Open WebUI's memory system.
- It keeps Open WebUI's vector DB in sync when possible, but its own retrieval path is based on the stored memories plus embeddings it manages.
- If you change embedding models or providers, previously cached embeddings may no longer match and will be regenerated over time.
- Summarization only deletes source memories after the new summary memory is successfully saved.

## Credit

This project is a fork of [gramanoid's owui-adaptive-memory](https://github.com/gramanoid/owui-adaptive-memory).

## License

MIT
