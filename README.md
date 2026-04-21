# Adaptive Memory for Open WebUI

Give Open WebUI persistent, user-specific memory with semantic recall, deduplication, pruning, summarization, persistent embedding cache, and optional Mem0 mirroring.

## Overview

This function watches user messages, extracts durable facts and preferences, stores them in Open WebUI memory, and injects the most relevant memories back into later chats.

It is designed to stay mostly automatic once configured:
- Extract likely long-term memories from recent user messages
- Save them into Open WebUI memory
- Avoid obvious duplicates
- Recall relevant memories with embeddings
- Keep vector storage and embedding cache in sync
- Prune or summarize older memories
- Optionally queue CRUD for scheduled Mem0 mirroring
- Reconcile deleted Mem0 records back into Open WebUI

## Current Behavior

At a high level:
1. The user sends a message.
2. `inlet()` loads the user's existing Open WebUI memories.
3. If Mem0 sync is enabled, mapped Mem0 records are checked and locally deleted if the upstream Mem0 memory is gone.
4. Relevant memories are selected with embeddings and injected into the prompt.
5. After the assistant responds, `outlet()` asks the configured LLM to propose memory operations.
6. Valid `NEW`, `UPDATE`, and `DELETE` operations are applied locally.
7. When Mem0 is enabled, local CRUD is either mirrored inline or queued for the next scheduled Mem0 sync cycle, depending on `mem0_sync_strategy`.

Open WebUI remains the primary local store. Mem0 is optional and best-effort.

## Key Features

- Semantic recall using embeddings
- Persistent embedding cache in SQLite
- In-memory LRU embedding cache
- Duplicate detection with embeddings or text similarity fallback
- Memory pruning with `fifo` or `least_relevant`
- Background memory summarization
- Background orphaned vector cleanup
- Optional Mem0 mirroring for create, update, and delete
- Scheduled background Mem0 batch syncing with persistent queueing
- Best-effort Mem0 delete reconciliation back into Open WebUI
- Per-user or global Mem0 user ID overrides
- Router-aware memory creation path for vector sync compatibility

## Installation

1. Upload `adaptive_memory_v4.0.py` to Open WebUI Functions.
2. Enable the function for the model or models you want.
3. Configure the valves you care about.

## Recommended Setup

### Default

Use:
- `embedding_source = auto`
- `embedding_provider_type = local`
- `embedding_model_name = all-MiniLM-L6-v2`
- your normal Ollama or OpenAI-compatible model for memory extraction

That gives you Open WebUI embeddings when available, with plugin embeddings as a fallback.

### Mem0 Mirroring

Enable:
- `enable_mem0_sync = true`

Then set:
- `mem0_api_base_url`
- `mem0_api_key`
- `mem0_app_id`

Optional:
- `mem0_user_id_template`
- `mem0_user_id_override`
- `mem0_infer_on_create`
- per-user `mem0_user_id_override`

## Mem0 Sync Model

When Mem0 sync is enabled, the function does four separate things:

1. Local creates, updates, and deletes are always applied in Open WebUI first.
2. If `mem0_sync_strategy = background`, Mem0 work is written into a persistent SQLite queue and processed later on the configured interval.
3. If `mem0_sync_strategy = inline`, Mem0 work is attempted immediately during the request path.
4. During later inbound requests, missing Mem0 records are reconciled back into Open WebUI by deleting the mapped local memory.

In background mode, queued jobs are coalesced by `(user_id, memory_id)`, so repeated updates collapse into the latest state before the next Mem0 sync cycle runs.

That last piece matters: this is no longer just one-way mirroring. If a mapped memory is deleted upstream in Mem0, the local Open WebUI copy is cleaned up on a later reconciliation pass.

`mem0_infer_on_create` is disabled by default. When you turn it on, mirrored create requests let Mem0 infer facts from the provided message, which can improve Mem0-side deduplication and conflict resolution. This only affects create ingestion; direct update/delete calls still use explicit memory IDs.

Reconciliation is best-effort and intentionally conservative:
- expected Mem0 `404` responses are treated as normal for reconciliation
- unexpected HTTP failures do not delete local memories
- stale mappings for already-missing local memories are pruned from `mem0_sync.sqlite`

### Recommended Mem0 Mode

For most setups, use:
- `mem0_sync_strategy = background`
- `mem0_sync_batch_interval_seconds = 7200`

That keeps Mem0 latency out of the chat path and makes syncing behave more like the summarization task.

## Mem0 User ID Resolution

When Mem0 mirroring is enabled, the Mem0 user/entity ID is chosen in this order:

1. Per-user `mem0_user_id_override`
2. Matching targeted entry in global `mem0_user_id_override`
3. Previously stored Mem0 user mapping
4. `mem0_user_id_template`

Examples:
- Set global override to `xxxxxxxx:jefe` to route only Open WebUI user `xxxxxxxx` to Mem0 user `jefe`.
- Set multiple targeted mappings with commas, semicolons, or new lines, for example `user_a:jefe, user_b:ana`.
- Set a per-user override to route only one Open WebUI user to a custom Mem0 entity.

Plain global values like `jefe` are ignored. This valve is now only for explicit per-user mappings, so unmatched users fall back to their stored mapping or the normal template flow.

## Important Valves

### Memory Extraction and Recall

- `recent_messages_n`
- `related_memories_n`
- `relevance_threshold`
- `vector_similarity_threshold`
- `show_memories`
- `memory_format`

### Deduplication

- `deduplicate_memories`
- `use_embeddings_for_deduplication`
- `embedding_similarity_threshold`
- `similarity_threshold`
- `enable_short_preference_shortcut`
- `short_preference_no_dedupe_length`
- `preference_keywords_no_dedupe`

### Memory Quality Filters

- `min_memory_length`
- `min_confidence_threshold`
- `filter_trivia`
- `blacklist_topics`
- `whitelist_keywords`
- `allowed_memory_banks`
- `default_memory_bank`

### Size Control

- `max_total_memories`
- `pruning_strategy`
- `enable_summarization_task`
- `summarization_interval`
- `summarization_min_cluster_size`
- `summarization_similarity_threshold`
- `summarization_max_cluster_size`
- `summarization_min_memory_age_days`

### Mem0

- `enable_mem0_sync`
- `mem0_api_base_url`
- `mem0_api_key`
- `mem0_app_id`
- `mem0_timeout_seconds`
- `mem0_sync_strategy`
- `mem0_sync_batch_size`
- `mem0_sync_batch_interval_seconds`
- `mem0_sync_retry_delay_seconds`
- `mem0_reconcile_cooldown_seconds`
- `mem0_user_id_template`
- `mem0_user_id_override`
- `mem0_infer_on_create`
- `log_user_id_on_memory_save`

### Background Tasks

- `enable_summarization_task`
- `enable_error_logging_task`
- `enable_vector_cleanup_task`
- `summarization_interval`
- `error_logging_interval`
- `vector_cleanup_interval`

## Background Tasks That Actually Run

Current active background loops:
- Memory summarization
- Mem0 background sync
- Error counter logging
- Orphaned vector cleanup
- Rogue task scavenging on startup

There are a few task-related valves in the schema that look future-facing, but not every one currently has an implemented loop in this file.

## Storage and Sidecar Files

Open WebUI still stores the actual memory rows.

This function may also maintain sidecar files under `DATA_DIR/cache`:
- `embeddings.sqlite`: persistent embedding cache
- `mem0_sync.sqlite`: Mem0 memory mappings, user mappings, and queued background sync jobs

Legacy embedding JSON cache files may also appear if SQLite persistence fails and the code falls back.

## Notes on Vectors and Embeddings

- The function tries to keep Open WebUI's vector DB in sync with local memory CRUD.
- Retrieval uses the stored memories plus embeddings managed by this function.
- If embedding provider settings change, cached embeddings may become incompatible and will be regenerated over time.
- A background vector cleanup task removes orphaned vectors when possible.

## Notes on Summarization

- Summarization only considers eligible non-summary memories.
- Clusters are formed with embedding similarity.
- A new summary memory is saved first.
- Source memories are deleted only after the summary save succeeds.
- Source memory deletion also cleans vectors, persistent embeddings, and Mem0 mirror state.

## Notes on Logging

Recent cleanup changed two logging behaviors:
- expected Mem0 reconciliation `404`s are no longer treated as warning-level failures
- duplicate emission from logger propagation has been disabled so each record should log once

## Requirements

Required:
- Open WebUI
- `numpy`
- `aiohttp`
- `pydantic`
- `pytz`

Optional:
- `sentence-transformers` for local embeddings
- `prometheus-client` for metrics

## Caveats

- Mem0 sync is best-effort, not transactional.
- In `background` mode, Mem0 can lag behind local memory by up to `mem0_sync_batch_interval_seconds`.
- Reconciliation runs during inbound requests, not as a separate continuous Mem0 polling loop.
- If Mem0 is unavailable, local memory remains intact.
- Some logging and configuration fields are broader than the currently implemented feature set.

## Credit

This project is a fork of [gramanoid's owui-adaptive-memory](https://github.com/gramanoid/owui-adaptive-memory).

## License

MIT
