# Adaptive Memory for Open WebUI

Give Open WebUI persistent, user-specific memory with semantic recall, deduplication, pruning, summarization, persistent embedding cache, and optional Mem0 mirroring.

## Overview

This function watches user messages, extracts durable facts and preferences, stores them in Open WebUI memory, and injects the most relevant memories back into later chats.

This README reflects the current `adaptive_memory_v4.0.py` implementation. Some older valves are still present for compatibility with saved Open WebUI settings, but only the implemented behavior is described as active below.

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
6. Valid `NEW` operations are applied locally. `UPDATE` and `DELETE` operations must also pass a conservative current-message intent gate.
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
- Scheduled background Mem0 batch syncing with persistent queueing and durable job claiming
- Best-effort Mem0 delete reconciliation back into Open WebUI
- Per-user or global Mem0 user ID overrides
- Router-aware memory creation path for vector sync compatibility
- Conservative intent gating for LLM-proposed memory updates/deletes
- Heuristic filtering for obvious secrets and credentials
- Safe production breadcrumbs with hashed identifiers and reason codes

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
- `llm_provider_type = ollama` with `llm_api_endpoint_url = http://host.docker.internal:11434/api/chat`, or an OpenAI-compatible endpoint with `llm_api_key`
- your normal model in `llm_model_name` for memory extraction

That gives you Open WebUI embeddings when available, with plugin embeddings as a fallback. You do not need to change the embedding valves if Open WebUI already has a working embedding function and the local fallback is acceptable.

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

Background workers atomically claim ready jobs in SQLite before processing. Claimed rows are marked `processing` with `claimed_at`, `claimed_by`, `status`, `attempt_count`, and `last_error` metadata. Successful jobs are deleted from the queue. Failed jobs are returned to `queued` with a retry time. If a worker dies mid-job, stale `processing` claims become claimable again after `mem0_sync_claim_timeout_seconds`.

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
- `mem0_sync_claim_timeout_seconds = 300`

That keeps Mem0 latency out of the chat path and makes syncing behave more like the summarization task.

## Memory Mutation Safety

The LLM can propose `NEW`, `UPDATE`, and `DELETE` operations, but destructive or mutating operations are gated by the current user message before execution.

- `DELETE` is allowed only when the current user message clearly asks to forget, delete, remove, erase, or stop remembering something.
- `UPDATE` is allowed only when the current user message clearly asks to update, correct, change, replace, revise, or otherwise correct stale information.
- Instructions found only inside recalled memories, quoted text, or prompt-injection text are not enough.
- Ambiguous messages default to no destructive action.

This is intentionally conservative. A vague message may save a new corrected fact instead of updating an old one.

## Prompt And Privacy Boundaries

Recalled memories are injected as untrusted factual context, not instructions. This reduces the blast radius of stored prompt-injection text, but it does not make memory content harmless. Treat saved memories as user-controlled data.

The function also rejects obvious high-risk secrets before storage, including common API key labels, bearer tokens, passwords, private-key blocks, database URLs with credentials, SSN-like values, and credit-card-like values that pass a Luhn check. This is heuristic filtering, not full DLP. It will miss some sensitive data and may reject a small number of benign strings that look like credentials.

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

### Embeddings

- `embedding_source`
- `embedding_provider_type`
- `embedding_model_name`
- `embedding_api_url`
- `embedding_api_key`

### LLM Provider

- `llm_provider_type`
- `llm_model_name`
- `llm_api_endpoint_url`
- `llm_api_key`
- `max_retries`
- `retry_delay`

### Memory Extraction

- `recent_messages_n`
- `memory_identification_prompt`
- `enable_json_stripping`
- `enable_fallback_regex`
- `enable_short_preference_shortcut`
- `short_preference_no_dedupe_length`
- `preference_keywords_no_dedupe`

### Recall and Injection

- `related_memories_n`
- `relevance_threshold`
- `vector_similarity_threshold`
- `show_memories`
- `max_injected_memory_length`
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
- `enable_identity_memories`
- `enable_behavior_memories`
- `enable_preference_memories`
- `enable_goal_memories`
- `enable_relationship_memories`
- `enable_possession_memories`

### Size Control

- `max_total_memories`
- `pruning_strategy`
- `enable_summarization_task`
- `summarization_interval`
- `summarization_strategy`
- `summarization_min_cluster_size`
- `summarization_similarity_threshold`
- `summarization_max_cluster_size`
- `summarization_min_memory_age_days`
- `summarization_memory_prompt`

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
- `mem0_sync_claim_timeout_seconds`
- `mem0_reconcile_cooldown_seconds`
- `mem0_user_id_template`
- `mem0_user_id_override`
- `mem0_infer_on_create`

### Background Tasks and Logging

- `enable_summarization_task`
- `enable_error_logging_task`
- `enable_vector_cleanup_task`
- `summarization_interval`
- `error_logging_interval`
- `vector_cleanup_interval`
- `enable_debug_logging`
- `debug_error_counter_logs`
- `log_user_id_on_memory_save`

### User Valves

- `enabled`
- `show_status`
- `mem0_user_id_override`
- `timezone`

The active user-level controls are `enabled`, `show_status`, and `mem0_user_id_override`. User `timezone` is still present in the schema but is not currently used by the processing path.

## Background Tasks That Actually Run

Current active background loops:
- Memory summarization
- Mem0 background sync
- Error counter logging
- Orphaned vector cleanup
- Rogue task scavenging on startup

Compatibility valves that are present in the schema but not currently wired into active behavior:
- `enable_date_update_task`
- `date_update_interval`
- `enable_model_discovery_task`
- `model_discovery_interval`
- `save_relevance_threshold`
- `memory_threshold`
- `llm_skip_relevance_threshold`
- `top_n_memories`
- `cache_ttl_seconds`
- `use_llm_for_relevance`
- `memory_relevance_prompt`
- `memory_merge_prompt`
- `enable_error_counter_guard`
- `error_guard_threshold`
- `error_guard_window_seconds`
- global `show_status`
- global `timezone`
- user `timezone`

## Storage and Sidecar Files

Open WebUI still stores the actual memory rows.

This function may also maintain sidecar files under `DATA_DIR/cache`:
- `embeddings.sqlite`: persistent embedding cache
- `mem0_sync.sqlite`: Mem0 memory mappings, user mappings, and queued background sync jobs

Legacy embedding JSON cache files from older versions may also exist; the function reads and migrates them into SQLite when possible.

## Notes on Vectors and Embeddings

- The function tries to keep Open WebUI's vector DB in sync with local memory CRUD.
- Retrieval uses the stored memories plus embeddings managed by this function.
- If the embedding model or provider type changes, persistent cache entries are treated as incompatible and regenerated over time.
- If the embedding API URL or key changes for the same model/provider type, the plugin provider is refreshed and the in-memory embedding cache is cleared; existing persistent embeddings for that same model/provider type may still be reused.
- A background vector cleanup task removes orphaned vectors when possible.

## Notes on Summarization

- Summarization considers eligible memories, including prior summaries, so newer related facts can be folded into an existing summary later.
- Clusters are formed as connected groups. `embeddings` uses embedding similarity, `tags` uses shared tags plus memory bank, and `hybrid` uses embedding similarity with a small threshold relaxation when tags and bank also match.
- A new summary memory is saved first.
- Source memories are deleted only after the summary save succeeds.
- Source memory deletion also cleans vectors, persistent embeddings, and Mem0 mirror state.

## Notes on Logging

The function now emits safe, sparse, key/value-style breadcrumbs around the memory lifecycle:

- Open WebUI entry points: `owui_entry_started`, `owui_entry_skipped`, `owui_entry_completed`
- extraction and operation decisions: `memory_extraction_completed`, `memory_operation_decision`, `memory_operations_completed`
- intent gating: `memory_intent_gate_decision`
- privacy filtering: `memory_candidate_blocked`
- retrieval/injection: `memory_retrieval_completed`, `memory_injection_completed`
- storage/vector/embedding work: `memory_create_succeeded`, `memory_update_succeeded`, `memory_delete_succeeded`, `embedding_cache_*`
- Mem0 queue work: `mem0_queue_job_queued`, `mem0_queue_claim_succeeded`, `mem0_queue_retry_scheduled`, `mem0_queue_batch_completed`
- external calls: `external_request_attempted`, `external_request_succeeded`, `external_request_failed`, `llm_request_*`

Identifiers in logs are hashed: `user_hash`, `session_hash`, `memory_hash`, `job_hash`, and worker hashes are stable enough for debugging but do not expose raw IDs.

The logs intentionally never include:
- raw user messages
- raw memory contents
- raw prompts or completions
- raw LLM responses
- raw Mem0 request payloads
- API keys, bearer tokens, passwords, private keys, database URLs with credentials, SSNs, or credit-card-like values

Common reason codes:
- `explicit_create_candidate`
- `explicit_update_intent`
- `explicit_delete_intent`
- `blocked_missing_update_intent`
- `blocked_missing_delete_intent`
- `blocked_sensitive_content`
- `blocked_empty_input`
- `blocked_malformed_llm_response`
- `blocked_prompt_injection_risk`
- `retrieval_no_memories`
- `storage_unavailable`
- `user_context_missing`

Set `enable_debug_logging = true` to enable DEBUG-level safe breadcrumbs such as cache hits/misses and queue claim internals. Set `debug_error_counter_logs = true` if you also want periodic error counter snapshots. Leave both disabled for quieter normal operation.

Privacy limitations still apply. Secret detection and log redaction are heuristic safeguards, not full DLP. A determined or unusual secret format may evade detection, so upstream policy and operator discipline still matter.

Recent cleanup also changed these logging behaviors:
- expected Mem0 reconciliation `404`s are no longer treated as warning-level failures
- duplicate emission from logger propagation has been disabled so each record should log once
- Mem0 failure response bodies are summarized, redacted, and logged only as size/hash metadata

## Tests

Run the local validation suite with:

```bash
python -m py_compile adaptive_memory_v4.0.py tests\adaptive_memory_loader.py tests\test_adaptive_memory_helpers.py tests\test_extract_memory_id.py
python -m unittest discover -s tests
git diff --check
```

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
- In multi-worker deployments, SQLite row claiming prevents the same queued job from being processed concurrently by cooperating workers that share the same `mem0_sync.sqlite` file.
- Reconciliation runs during inbound requests, not as a separate continuous Mem0 polling loop.
- If Mem0 is unavailable, local memory remains intact.
- Secret filtering is heuristic, not a replacement for upstream privacy controls or careful operator policy.
- Some compatibility valves remain in the schema so existing Open WebUI saved settings do not break, even though they do not currently change behavior.

## Credit

This project is a fork of [gramanoid's owui-adaptive-memory](https://github.com/gramanoid/owui-adaptive-memory).

## License

MIT
