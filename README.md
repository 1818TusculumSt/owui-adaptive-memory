# 🧠 Adaptive Memory for Open WebUI

> Persistent, user-specific memory with semantic recall, deduplication, pruning, summarization, multi-signal relevance scoring, contradiction detection, and optional Mem0 mirroring.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## 🚀 Quick Start

1. Upload `adaptive_memory_v4.0.py` to Open WebUI Functions
2. Enable the function for your models
3. Set these minimum valves:

| Valve | Value |
|-------|-------|
| `llm_provider_type` | `ollama` (or `openai_compatible`) |
| `llm_api_endpoint_url` | `http://host.docker.internal:11434/api/chat` |
| `llm_model_name` | `llama3:latest` (or your model) |
| `embedding_source` | `auto` |

> 💡 The function uses Open WebUI's embedding engine when available, falling back to the bundled `all-MiniLM-L6-v2` model. Everything else works on defaults.

## 🔄 How It Works

```
User message → inlet()
  ├─ Load existing memories
  ├─ Reconcile Mem0 (if enabled)
  ├─ Select relevant memories via vector search + multi-signal boost (v5)
  │   ├─ Recency/importance/access weight boost
  │   └─ Optional one-hop neighbor retrieval (semantically adjacent memories; off by default)
  ├─ Track access stats (throttled DB writes)
  ├─ Inject context into last user message (stable prefix for prompt caching)
  └─ Emit rich status (high-importance count)
       ↓
    LLM responds
       ↓
  outlet()
  ├─ Ask configured LLM to propose memory operations (NEW / UPDATE / DELETE)
  │   └─ LLM now provides importance (1-5) and stability (stable/fluid/transient)
  ├─ Run extraction quality gate (reject general knowledge, downgrade transient; 30+ transient patterns)
  ├─ Gate UPDATE/DELETE against user's current message intent
  ├─ Check contradiction against near-match memories (auto-promote NEW → UPDATE)
  ├─ Deduplicate, filter secrets, apply quality checks
  ├─ Update session context summary for next turn
  ├─ Save to Open WebUI, sync vectors, mirror to Mem0 (if configured)
  └─ Emit status if user valve show_status is on

Background:
  ├─ Summarization loop → decay-score sorted clusters, metadata inheritance
  └─ Stale detection loop → mark/summarize/delete old low-importance memories

Slash commands (inlet):
  ├─ /memories    → list all memories with importance stars
  ├─ /forget kw   → delete a specific memory by keyword
  └─ /remember ... → save a new memory directly
```

## 📊 Importance & Stability Scoring (v4.3.0)

The system uses a **three-layer** approach to assign importance (1–5) and stability (stable/fluid/transient) to every memory:

### Layer 1: LLM Extraction
The extraction prompt now includes **11 examples** covering all 5 importance levels, distribution guidance (~60% should be 2–3), and negative examples showing what NOT to over-score. The LLM provides initial importance/stability with every memory operation.

### Layer 2: Content-Based Lexical Signals
Before tag floors apply, the system analyzes the memory content for semantic signals:

| Signal Type | Examples | Effect |
|-------------|----------|--------|
| **Boost** | "always", "never", "love", "hate", "favorite", "my name is", "i work as", family terms | +1 importance |
| **Demote** | "today", "yesterday", "right now", "maybe", "had for dinner", "debugging", "shopping for" | −1 importance |

This ensures a pizza order scores **2** instead of 3, a name scores **5** instead of 4, and a passing location mention scores **1** instead of 4.

### Layer 3: Softened Tag Floors
Tag floors (e.g. `identity` → minimum 4, `relationship` → minimum 4) are no longer hard overrides:
- If the LLM's score is **within 1 point** of the tag floor, the LLM is trusted
- Only if the gap is **≥2 points** does the system bump the score up (to floor−1)
- If the LLM didn't provide a score at all, the hard floor still applies (backward compatible)

The same approach applies to stability: only upgrades if the LLM is 2+ levels below the tag floor.

**Result:** The old clustering at importance 3–4 and stability fluid–stable is gone. Memories now meaningfully differentiate across all 5 importance levels and all 3 stability classes.

## ⚙️ Valve Reference

### 🔌 Essential

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

### 🧲 Memory Extraction

| Valve | Default | Description |
|-------|---------|-------------|
| `recent_messages_n` | `5` | Recent user messages included in the extraction prompt |
| `memory_identification_prompt` | *(long prompt)* | System prompt for the extraction LLM — requires `importance` (1-5) and `stability` (`stable`/`fluid`/`transient`). v4.3.0: 11 examples across all levels, distribution guidance, negative examples. |
| `enable_json_stripping` | `True` | Strip markdown fences and extra text from LLM JSON response |
| `enable_fallback_regex` | `True` | Fallback regex extraction if JSON parsing fails |
| `enable_short_preference_shortcut` | `True` | Bypass the LLM for short preference statements (e.g. "I like X") |
| `short_preference_no_dedupe_length` | `100` | Max chars for a short preference to skip dedup |
| `preference_keywords_no_dedupe` | `favorite,love,...` | Keywords that trigger the short-preference shortcut |

### 🎯 Quality Filters

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

### 🏆 Multi-Signal Memory (Phases 0-5)

| Valve | Default | Description |
|-------|---------|-------------|
| `enable_importance_scoring` | `True` | Enable importance scoring (1-5) during memory extraction |
| `enable_stability_decay` | `True` | Enable stability-based differential decay for relevance and pruning |
| `enable_access_tracking` | `True` | Track memory access counts and last-accessed timestamps |
| `recency_boost_weight` | `0.10` | Weight for recency in relevance scoring (vector similarity is ~70%) |
| `importance_weight` | `0.15` | Weight for importance in relevance scoring |
| `access_boost_weight` | `0.05` | Weight for access frequency in relevance scoring |
| `access_update_interval` | `5` | Only persist access stat updates every N retrievals per memory |
| `enable_contradiction_detection` | `True` | Detect contradictions between new and existing memories; auto-promote NEW to UPDATE |
| `contradiction_similarity_threshold` | `0.65` | Min cosine similarity before contradiction check is attempted |
| `enable_conversation_context` | `True` | Include a brief conversation context summary in the extraction prompt |
| `enable_neighbor_retrieval` | `False` | Pull in semantically adjacent memories even if they didn't match the query directly |
| `neighbor_hop_similarity` | `0.80` | Cosine similarity threshold for a memory to be considered a neighbor |
| `neighbor_penalty` | `0.7` | Score multiplier (0-1) applied to neighbor memories |
| `max_neighbors_per_memory` | `2` | Max neighbor memories to pull in per selected memory |
| `enable_stale_detection_task` | `True` | Background task that detects stale, low-importance memories |
| `stale_detection_interval` | `86400` | Seconds between stale memory detection runs |
| `stale_threshold_days` | `90` | Days since last access before a memory is considered stale |
| `stale_action` | `summarize` | Action on stale memories: `log` / `summarize` / `delete` |
| `enable_memory_acknowledgment` | `True` | Instruct the LLM to naturally acknowledge relevant memories |
| `enable_memory_commands` | `True` | Enable `/memories`, `/forget`, and `/remember` slash commands |
| `enable_extraction_quality_gate` | `True` | Run a rule-based quality filter on extracted memories before saving |
| `retrieval_scoring_version` | `v5` | `v4` = original vector-only, `v5` = multi-signal with recency/importance/access |

### 🔍 Retrieval & Injection

| Valve | Default | Description |
|-------|---------|-------------|
| `related_memories_n` | `5` | Max relevant memories retrieved |
| `relevance_threshold` | `0.60` | Minimum relevance score (0-1) for injection |
| `vector_similarity_threshold` | `0.20` | Minimum cosine similarity for initial vector candidate filter |
| `use_llm_for_relevance` | `True` | Use an additional LLM call for relevance scoring |
| `top_n_memories` | `5` | Max candidates sent to the LLM relevance scorer |
| `llm_skip_relevance_threshold` | `0.93` | If all vector scores exceed this, skip the LLM relevance call |
| `memory_relevance_prompt` | *(prompt)* | System prompt for the relevance-scoring LLM — now includes importance/recency/stability/access metadata |
| `max_injected_memory_length` | `300` | Truncate injected memory text to this length |
| `memory_format` | `bullet` | `bullet` / `paragraph` / `numbered` |
| `show_memories` | `True` | Show memory context in the injected prompt label |
| `inject_memories_into_user_message` | `True` | Inject memories into the last user message instead of the system prompt — preserves a stable prefix for prompt caching (DeepSeek, OpenCode, etc.) |
| `deterministic_memory_ordering` | `True` | Sort recalled memories by ID before injection — same selection produces identical prompt text, improving cache hit rates |

### 🗂️ Deduplication

| Valve | Default | Description |
|-------|---------|-------------|
| `deduplicate_memories` | `True` | Enable deduplication before saving |
| `use_embeddings_for_deduplication` | `True` | Use embeddings for semantic duplicate detection |
| `embedding_similarity_threshold` | `0.75` | Cosine similarity threshold for embedding dedup |
| `similarity_threshold` | `0.95` | Text sequence similarity threshold (fallback) |

### 📦 Size Control

| Valve | Default | Description |
|-------|---------|-------------|
| `max_total_memories` | `200` | Maximum memories per user |
| `pruning_strategy` | `fifo` | `fifo` / `least_relevant` / `tiered_decay` (importance & stability-aware) |
| `enable_summarization_task` | `True` | Run background memory summarization |
| `summarization_interval` | `7200` | Seconds between summarization runs |
| `summarization_strategy` | `hybrid` | `embeddings` / `tags` / `hybrid` clustering |
| `summarization_min_cluster_size` | `2` | Min related memories to form a cluster |
| `summarization_max_cluster_size` | `8` | Max memories per cluster |
| `summarization_similarity_threshold` | `0.7` | Similarity threshold for clustering |
| `summarization_min_memory_age_days` | `7` | Min age before a memory is eligible for summarization |
| `summarization_memory_prompt` | *(prompt)* | System prompt for the summarization LLM |

### ☁️ Mem0 Mirroring *(optional)*

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
| `mem0_sync_max_retries` | `20` | Max retries before permanently dropping a job (`0` = unlimited) |
| `mem0_reconcile_cooldown_seconds` | `30` | Min seconds between Mem0 reconciliation checks per user |
| `mem0_user_id_template` | `owui:{user_id}` | Template mapping OWUI users to Mem0 user IDs |
| `mem0_user_id_override` | `""` | Per-user mapping table (`owui_id:mem0_id, ...`) |
| `mem0_infer_on_create` | `False` | Pass `infer=true` on Mem0 create requests |

> ⚠️ Mem0 mirroring is **best-effort, not transactional**. Local Open WebUI memory is always the primary store. In `background` mode, Mem0 can lag by up to `mem0_sync_batch_interval_seconds`. If Mem0 is unavailable, local memory remains intact.

**👤 Mem0 user ID resolution order:**
1. Per-user valve `mem0_user_id_override`
2. Matching entry in global `mem0_user_id_override` (e.g. `user_a:jefe`)
3. Previously stored Mem0 user mapping
4. `mem0_user_id_template`

### 📡 Background Tasks & Logging

| Valve | Default | Description |
|-------|---------|-------------|
| `enable_error_logging_task` | `True` | Periodic error counter snapshots |
| `enable_vector_cleanup_task` | `True` | Clean up orphaned vectors |
| `error_logging_interval` | `1800` | Seconds between error counter logs |
| `vector_cleanup_interval` | `7200` | Seconds between vector cleanup runs |
| `enable_debug_logging` | `False` | Enable DEBUG-level safe breadcrumbs |
| `debug_error_counter_logs` | `False` | Include error counter snapshots in debug output |
| `log_user_id_on_memory_save` | `False` | Log plain user IDs on memory save (off by default) |

### 👤 User Valves *(per-user overrides)*

| Valve | Default | Description |
|-------|---------|-------------|
| `enabled` | `True` | Enable memory for this user |
| `show_status` | `True` | Show memory-saved status message after each response (now includes high-importance count) |
| `mem0_user_id_override` | `""` | Per-user Mem0 user ID |

## 📐 Storage Format

Each memory is stored as a single text field. All metadata is packed and unpacked via regex for backward compatibility:

```
[Tags: identity, behavior] User is a software engineer [Memory Bank: Work] [Confidence: 0.95]
```

With multi-signal features enabled, memories include additional fields:

```
[Tags: identity, behavior] User is a software engineer [Memory Bank: Work] [Confidence: 0.95] [Importance: 5] [Stability: stable] [LastAccessed: 2025-01-15] [AccessCount: 12]
```

| Field | Range | Description |
|-------|-------|-------------|
| `Importance` | 1-5 | 5=core identity, 4=strong preference, 3=moderate, 2=situational, 1=trivia. Boosted/demoted by content signals; tag floors are soft gap-based. See *Importance & Stability Scoring* below. |
| `Stability` | stable/fluid/transient | `stable`=years, `fluid`=months, `transient`=days/weeks. Tag floors only upgrade if the LLM is 2+ levels below. |
| `LastAccessed` | ISO date | Updated on retrieval (throttled per `access_update_interval`) |
| `AccessCount` | int | How many times the memory has been retrieved |

> All new fields are optional — old-format memories parse correctly with defaults. The `migrate_memory_to_new_format()` helper lazily upgrades old memories when they're touched.

## 🔍 Relevance Scoring: v4 vs v5

| Aspect | v4 (original) | v5 (multi-signal) |
|--------|---------------|-------------------|
| Vector similarity | 100% weight | ~70% weight |
| Recency (age) | ignored | 10% weight, decay rate modulated by stability class |
| Importance | ignored | 15% weight |
| Access frequency | ignored | 5% weight |
| Decay | uniform | stable=0%, fluid=0.003/day, transient=0.015/day |
| Importance modulation | none | importance modulates decay via ×`(importance−3)×0.25`: 5=halved, 1=+50%, 3=no change |

Set `retrieval_scoring_version="v4"` to revert to original behavior.

## 🔄 Contradiction Detection

When a new memory is similar (but not identical) to an existing one, the system checks for contradiction:

```
Existing: "User prefers light mode"
New: "I now prefer dark mode"
→ Contradiction detected → auto-promoted to UPDATE (replaces old memory)

Existing: "User likes coffee"
New: "I also like tea"
→ No contradiction → saved as NEW
```

The check uses a separate LLM call with a focused contradiction-detection prompt. Controlled by `enable_contradiction_detection` and `contradiction_similarity_threshold`.

## 🗂️ Pruning Strategies

| Strategy | Behavior |
|----------|----------|
| `fifo` | Removes oldest memories first |
| `least_relevant` | Scores by `confidence - (age × 0.01)`, removes lowest scores |
| `tiered_decay` | Scores by `confidence - (age × decay_rate) + (access × 0.02) + (importance × 0.05)`. Decay rate is modulated by importance: `decay × (1 − (importance−3) × 0.25)`. Stable/important/frequently-accessed memories are preserved; transient/old/rarely-accessed memories are pruned first |

## 💬 Slash Commands

When `enable_memory_commands` is `True`, users can manage memories directly in chat:

| Command | Example | Behavior |
|---------|---------|----------|
| `/memories` | `/memories` | Lists up to 20 memories with importance stars, age, and memory bank |
| `/forget` | `/forget Kubernetes` | Deletes memories matching the keyword |
| `/remember` | `/remember I prefer dark mode` | Saves a direct memory (skips LLM extraction) |

These commands are intercepted in the inlet pipeline and do not reach the LLM.

## 🛡️ Mutation Safety

The LLM can propose `NEW`, `UPDATE`, and `DELETE` operations. Destructive operations are gated:

- **🗑️ DELETE** — only allowed when the user's *current message* explicitly asks to forget, delete, remove, or stop remembering
- **✏️ UPDATE** — only allowed when the user's *current message* explicitly asks to correct, change, replace, or revise, or when contradiction detection auto-promotes

> 🔒 Instructions buried in recalled memory text, quoted text, or prompt-injection attempts are ignored. Ambiguous messages default to no destructive action. This is intentionally conservative.

## 🔐 Privacy Protections

| Guard | Detail |
|-------|--------|
| 🤫 **Secret filtering** | Blocks API keys, bearer tokens, passwords, private keys, DB URLs with credentials, SSNs, and Luhn-validated credit card numbers before storage. Heuristic, not full DLP. |
| 🔏 **Safe logging** | All logged identifiers are hashed. Raw user messages, memory contents, prompts, completions, and API keys are never logged. |
| 🧱 **Memory injection** | Recalled memories are injected as untrusted factual context, not instructions, reducing prompt-injection blast radius. |
| ✅ **Extraction quality gate** | Rule-based filter rejects general knowledge statements and downgrades transient content before saving. Detects 30+ transient patterns (temporal, task, consumption). |

## 📁 Sidecar Files

Located under `DATA_DIR/cache`:

| File | Purpose |
|------|---------|
| `embeddings.sqlite` | Persistent embedding cache |
| `mem0_sync.sqlite` | Mem0 memory mappings, user mappings, and queued background sync jobs |

> 📜 Legacy JSON cache files from older versions are read and migrated to SQLite on access.

## 💤 Inactive Valves

> These valves exist in the schema to preserve saved Open WebUI settings but are not currently wired to behavior.

`enable_date_update_task` `·` `date_update_interval` `·` `enable_model_discovery_task` `·` `model_discovery_interval` `·` `save_relevance_threshold` `·` `memory_threshold` `·` `cache_ttl_seconds` `·` `memory_merge_prompt` `·` `enable_error_counter_guard` `·` `error_guard_threshold` `·` `error_guard_window_seconds` `·` global `show_status` `·` global `timezone` `·` user `timezone`

## 📋 Requirements

- **Required**: Open WebUI, `numpy`, `aiohttp`, `pydantic`, `pytz`
- **Optional**: `sentence-transformers` (local embeddings), `prometheus-client` (metrics)

## 🧪 Tests

```bash
python -m py_compile adaptive_memory_v4.0.py tests/adaptive_memory_loader.py \
    tests/test_adaptive_memory_helpers.py tests/test_extract_memory_id.py \
    tests/test_extract_message_text.py
python -m unittest discover -s tests
git diff --check
```

Current: **122 tests** (up from 70).

## 📄 License

MIT · Forked from [alackmann/owui-adaptive-memory](https://github.com/alackmann/owui-adaptive-memory)
