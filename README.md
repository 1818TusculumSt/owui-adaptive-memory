# Adaptive Memory for Open WebUI

Persistent, user-specific memory with semantic recall, deduplication, pruning, summarization, and optional Mem0 mirroring.

## Quick Start

1. Upload `adaptive_memory_v4.0.py` to Open WebUI Functions
2. Enable the function for your models
3. Set these minimum valves:

| Valve | Value |
|-------|-------|
| `llm_provider_type` | `ollama` (or `openai_compatible`) |
| `llm_api_endpoint_url` | `http://host.docker.internal:11434/api/chat` |
| `llm_model_name` | `llama3:latest` (or your model) |
| `embedding_source` | `auto` |

The function will use Open WebUI's embedding engine when available, falling back to the bundled `all-MiniLM-L6-v2` model. Everything else works on defaults.

## How It Works

```
User message → inlet()
  ├─ Load existing memories
  ├─ Reconcile Mem0 (if enabled)
  ├─ Select relevant memories via embeddings + optional LLM scoring
  └─ Inject context into system prompt
       ↓
    LLM responds
       ↓
  outlet()
  ├─ Ask configured LLM to propose memory operations (NEW / UPDATE / DELETE)
  ├─ Gate UPDATE/DELETE against user's current message intent
  ├─ Deduplicate, filter secrets, apply quality checks
  ├─ Save to Open WebUI, sync vectors, mirror to Mem0 (if configured)
  └─ Emit status if user valve `show_status` is on
```

## Valve Reference

### Essential

| Valve | Default | Description |
|-------|---------|-------------|
| `llm_provider_type` | `ollama` | `ollama` or `openai_compatible` |
| `llm_model_name` | `llama3:latest` | Model used for memory extraction and relevance scoring |
| `llm_api_endpoint_url` | `http://host.docker.internal:11434/api/chat` | API endpoint |
| `llm_api_key` | `None` | API key (required for `openai_compatible`) |
| `embedding_source` | `auto` | `auto` / `owui` / `plugin` |
| `embedding_provider_type` | `local` | `local` or `openai_compatible` (plugin fallback) |
| `embedding_model_name` | `all-MiniLM-L6-v2` | Plugin-side embedding model |
| `embedding_api_url` | `None` | Embedding API endpoint (plugin fallback) |
| `embedding_api_key` | `None` | Embedding API key (plugin fallback) |

### Memory Extraction

| Valve | Default | Description |
|-------|---------|-------------|
| `recent_messages_n` | `5` | Recent user messages included in the extraction prompt |
| `memory_identification_prompt` | *(long prompt)* | System prompt for the extraction LLM |
| `enable_json_stripping` | `True` | Strip markdown fences and extra text from LLM JSON response |
| `enable_fallback_regex` | `True` | Fallback regex extraction if JSON parsing fails |
| `enable_short_preference_shortcut` | `True` | Bypass the LLM for short preference statements (e.g. "I like X") |
| `short_preference_no_dedupe_length` | `100` | Max chars for a short preference to skip dedup |
| `preference_keywords_no_dedupe` | `favorite,love,...` | Keywords that trigger the short-preference shortcut |

### Quality Filters

| Valve | Default | Description |
|-------|---------|-------------|
| `min_memory_length` | `8` | Minimum content length to store |
| `min_confidence_threshold` | `0.5` | Minimum LLM confidence (0-1) to store |
| `filter_trivia` | `True` | Reject short, low-information statements |
| `blacklist_topics` | `None` | Comma-separated topics to reject |
| `whitelist_keywords` | `None` | Keywords that bypass sensitive-content filtering |
| `allowed_memory_banks` | `General,Personal,Work` | Valid memory banks |
| `default_memory_bank` | `General` | Default bank when LLM doesn't specify |
| `enable_identity_memories` | `True` | Allow `identity` tag memories |
| `enable_behavior_memories` | `True` | Allow `behavior` tag memories |
| `enable_preference_memories` | `True` | Allow `preference` tag memories |
| `enable_goal_memories` | `True` | Allow `goal` tag memories |
| `enable_relationship_memories` | `True` | Allow `relationship` tag memories |
| `enable_possession_memories` | `True` | Allow `possession` tag memories |

### Retrieval & Injection

| Valve | Default | Description |
|-------|---------|-------------|
| `related_memories_n` | `5` | Max relevant memories retrieved |
| `relevance_threshold` | `0.60` | Minimum relevance score (0-1) for injection |
| `vector_similarity_threshold` | `0.20` | Minimum cosine similarity for initial vector candidate filter |
| `use_llm_for_relevance` | `True` | Use an additional LLM call for relevance scoring |
| `top_n_memories` | `5` | Max candidates sent to the LLM relevance scorer |
| `llm_skip_relevance_threshold` | `0.93` | If all vector scores exceed this, skip the LLM relevance call |
| `memory_relevance_prompt` | *(prompt)* | System prompt for the relevance-scoring LLM |
| `max_injected_memory_length` | `300` | Truncate injected memory text to this length |
| `memory_format` | `bullet` | `bullet` or `numbered` |
| `show_memories` | `True` | Show memory context in the injected prompt label |

### Deduplication

| Valve | Default | Description |
|-------|---------|-------------|
| `deduplicate_memories` | `True` | Enable deduplication before saving |
| `use_embeddings_for_deduplication` | `True` | Use embeddings for semantic duplicate detection |
| `embedding_similarity_threshold` | `0.75` | Cosine similarity threshold for embedding dedup |
| `similarity_threshold` | `0.95` | Text sequence similarity threshold (fallback) |

### Size Control

| Valve | Default | Description |
|-------|---------|-------------|
| `max_total_memories` | `200` | Maximum memories per user |
| `pruning_strategy` | `fifo` | `fifo` or `least_relevant` |
| `enable_summarization_task` | `True` | Run background memory summarization |
| `summarization_interval` | `7200` | Seconds between summarization runs |
| `summarization_strategy` | `hybrid` | `embeddings` / `tags` / `hybrid` clustering |
| `summarization_min_cluster_size` | `2` | Min related memories to form a cluster |
| `summarization_max_cluster_size` | `8` | Max memories per cluster |
| `summarization_similarity_threshold` | `0.7` | Similarity threshold for clustering |
| `summarization_min_memory_age_days` | `7` | Min age before a memory is eligible for summarization |
| `summarization_memory_prompt` | *(prompt)* | System prompt for the summarization LLM |

### Mem0 Mirroring *(optional)*

| Valve | Default | Description |
|-------|---------|-------------|
| `enable_mem0_sync` | `False` | Enable Mem0 mirroring |
| `mem0_api_base_url` | `https://api.mem0.ai` | Mem0 API base URL |
| `mem0_api_key` | `None` | Mem0 API key |
| `mem0_app_id` | `openwebui-adaptive-memory` | Mem0 app namespace |
| `mem0_timeout_seconds` | `30` | Mem0 API timeout |
| `mem0_sync_strategy` | `background` | `background` (queue + batch) or `inline` |
| `mem0_sync_batch_size` | `10` | Max jobs per background batch |
| `mem0_sync_batch_interval_seconds` | `7200` | Seconds between sync cycles |
| `mem0_sync_retry_delay_seconds` | `15` | Delay before retrying a failed job |
| `mem0_sync_claim_timeout_seconds` | `300` | Stale claim timeout for multi-worker safety |
| `mem0_sync_max_retries` | `20` | Max retries before permanently dropping a job (0 = unlimited) |
| `mem0_reconcile_cooldown_seconds` | `30` | Min seconds between Mem0 reconciliation checks per user |
| `mem0_user_id_template` | `owui:{user_id}` | Template mapping OWUI users to Mem0 user IDs |
| `mem0_user_id_override` | `""` | Per-user mapping table (`owui_id:mem0_id, ...`) |
| `mem0_infer_on_create` | `False` | Pass `infer=true` on Mem0 create requests |

Mem0 mirroring is **best-effort, not transactional**. Local Open WebUI memory is always the primary store. In `background` mode, Mem0 can lag by up to `mem0_sync_batch_interval_seconds`. If Mem0 is unavailable, local memory remains intact.

**Mem0 user ID resolution order:**
1. Per-user valve `mem0_user_id_override`
2. Matching entry in global `mem0_user_id_override` (e.g. `user_a:jefe`)
3. Previously stored Mem0 user mapping
4. `mem0_user_id_template`

### Background Tasks & Logging

| Valve | Default | Description |
|-------|---------|-------------|
| `enable_error_logging_task` | `True` | Periodic error counter snapshots |
| `error_logging_interval` | `1800` | Seconds between error counter logs |
| `enable_vector_cleanup_task` | `True` | Clean up orphaned vectors |
| `vector_cleanup_interval` | `7200` | Seconds between vector cleanup runs |
| `enable_debug_logging` | `False` | Enable DEBUG-level safe breadcrumbs |
| `debug_error_counter_logs` | `False` | Include error counter snapshots in debug output |
| `log_user_id_on_memory_save` | `False` | Log plain user IDs on memory save (off by default) |

### User Valves *(per-user overrides)*

| Valve | Default | Description |
|-------|---------|-------------|
| `enabled` | `True` | Enable memory for this user |
| `show_status` | `True` | Show memory-saved status message after each response |
| `mem0_user_id_override` | `""` | Per-user Mem0 user ID |

## Mutation Safety

The LLM can propose `NEW`, `UPDATE`, and `DELETE` operations. Destructive operations are gated:

- **DELETE** — only allowed when the user's *current message* explicitly asks to forget, delete, remove, or stop remembering
- **UPDATE** — only allowed when the user's *current message* explicitly asks to correct, change, replace, or revise

Instructions buried in recalled memory text, quoted text, or prompt-injection attempts are ignored. Ambiguous messages default to no destructive action. This is intentionally conservative.

## Privacy Protections

- **Secret filtering**: Blocks API keys, bearer tokens, passwords, private keys, DB URLs with credentials, SSNs, and Luhn-validated credit card numbers before storage. Heuristic, not full DLP.
- **Safe logging**: All logged identifiers are hashed. Raw user messages, memory contents, prompts, completions, and API keys are never logged.
- **Memory injection**: Recalled memories are injected as untrusted factual context, not instructions, reducing prompt-injection blast radius.

## Sidecar Files

Located under `DATA_DIR/cache`:
- `embeddings.sqlite` — persistent embedding cache
- `mem0_sync.sqlite` — Mem0 memory mappings, user mappings, and queued background sync jobs

Legacy JSON cache files from older versions are read and migrated to SQLite on access.

## Inactive Valves

These valves exist in the schema to preserve saved Open WebUI settings but are not currently wired to behavior:

`enable_date_update_task`, `date_update_interval`, `enable_model_discovery_task`, `model_discovery_interval`, `save_relevance_threshold`, `memory_threshold`, `cache_ttl_seconds`, `memory_merge_prompt`, `enable_error_counter_guard`, `error_guard_threshold`, `error_guard_window_seconds`, global `show_status`, global `timezone`, user `timezone`.

## Requirements

- **Required**: Open WebUI, `numpy`, `aiohttp`, `pydantic`, `pytz`
- **Optional**: `sentence-transformers` (local embeddings), `prometheus-client` (metrics)

## Tests

```bash
python -m py_compile adaptive_memory_v4.0.py tests/adaptive_memory_loader.py tests/test_adaptive_memory_helpers.py tests/test_extract_memory_id.py tests/test_extract_message_text.py
python -m unittest discover -s tests
git diff --check
```

## License

MIT. Forked from [gramanoid/owui-adaptive-memory](https://github.com/gramanoid/owui-adaptive-memory).
