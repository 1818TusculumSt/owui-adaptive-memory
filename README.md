# Adaptive Memory for Open WebUI 🧠

Give your AI persistent memory across conversations. It remembers your preferences, facts about you, and past discussions automatically.

## ✨ What This Does

This plugin makes your AI remember things about you between chats. Tell it once that you prefer Python over JavaScript, and it'll remember for future conversations. No manual management needed.

**How it works:**
1. You chat normally with your AI
2. The plugin extracts important facts about you from the conversation
3. Those facts get stored and retrieved automatically in future chats
4. Your AI has context about you without you repeating yourself

## 🙏 Credit Where It's Due

This is a fork of [gramanoid's owui-adaptive-memory](https://github.com/gramanoid/owui-adaptive-memory). His original plugin proved the concept works and laid the foundation.

**Why fork it?**

The original worked but the code was difficult to follow and had bugs that made it unreliable. I wanted:
- **Cleaner, more elegant code** that's easier to understand and modify
- **Shorter, more maintainable architecture** without unnecessary complexity
- **Clear, understandable memory processes** so you can actually see what's happening

Plus I fixed the production issues:
- Memory deletions left orphaned embeddings in the vector database
- Summarization created memory leaks
- Background tasks duplicated themselves after plugin reloads
- No UPDATE operation support
- Lock management issues

I added proper vector database synchronization, background task lifecycle management, comprehensive error handling, and a persistent embedding cache to reduce API calls.

Recent improvements:
- Better extraction and recall quality by embedding the actual memory text instead of the metadata wrapper
- Safer deduplication, pruning, and summarization behavior
- Cleaner memory injection with configurable formatting
- Persistent embedding cache migrated to a private SQLite sidecar with lazy import from legacy JSON cache files
- Optional Mem0 mirroring for memory create, update, and delete operations
- Configurable Mem0 user routing with global and per-user override support
- Safer Mem0 API handling with the documented `/v1/memories/` endpoint and fallback to `/v1/memories`

**I actively maintain and use this function.**

## 📦 Installation

1. Download `adaptive_memory_v4.0.py`
2. In Open WebUI: **Functions** → **+** → Upload the file
3. Configure the settings (called "valves" in OWUI)
4. Enable it for your models

## ⚙️ Configuration

The important settings:

**Embedding Model:**
- Use `local` with `all-MiniLM-L6-v2` for offline/free operation
- Or use `openai_compatible` with any API endpoint

**LLM Model:**
- Point to your Ollama instance or any OpenAI-compatible API
- This is what extracts memories from conversations

**Memory Settings:**
- `max_total_memories`: How many memories to keep per user (default: 200)
- `summarization_interval`: How often to consolidate old memories (default: 2 hours)
- Lower `summarization_similarity_threshold` to group more memories together (0.5-0.7 recommended)

**Optional Mem0 Mirroring:**
- `enable_mem0_sync`: Mirror local memory CRUD operations to Mem0
- `mem0_api_base_url`: Base URL for the Mem0 API
- `mem0_api_key`: Required when Mem0 mirroring is enabled
- `mem0_app_id`: App namespace used for mirrored memories
- `mem0_user_id_template`: Default mapping from Open WebUI user id to Mem0 user id
- `mem0_user_id_override`: Global override shown in the main valve UI; when set, it forces all mirrored memories to use that exact Mem0 user id

**Mem0 user id precedence:**
1. Per-user `mem0_user_id_override`
2. Global `mem0_user_id_override`
3. Stored Mem0 user mapping from previous syncs
4. `mem0_user_id_template`

## 🔒 Open WebUI Compatibility

This function keeps Open WebUI's own memory system intact:
- Open WebUI's database remains the source of truth for memories
- Open WebUI's vector database remains the source of truth for memory search
- The plugin's persistent sidecar databases are private to this function and exist only to avoid regenerating embeddings or losing Mem0 linkage state unnecessarily

The current sidecar backends are:
- `DATA_DIR/cache/embeddings.sqlite`
- `DATA_DIR/cache/mem0_sync.sqlite`

Legacy cache migration:
- Older per-user JSON cache files are imported automatically on first access
- JSON fallback is still kept for safety if SQLite is unavailable
- No Open WebUI schema changes are required

Mem0 sync state:
- Open WebUI memories stay authoritative even when Mem0 mirroring is enabled
- The plugin stores Open WebUI memory id -> Mem0 memory id mappings privately so updates/deletes stay aligned
- The plugin also stores Open WebUI user id -> resolved Mem0 user id mappings so background activity stays consistent

## 🔁 Mem0 Mirroring Notes

When enabled, the plugin mirrors local memory lifecycle events to Mem0 on a best-effort basis:
- New memories are mirrored to Mem0 after local save succeeds
- Updated memories are pushed to the existing mirrored Mem0 record
- Deleted, pruned, and summarized-away memories attempt to delete the mirrored Mem0 record too

API behavior:
- The plugin uses Mem0's documented create route `/v1/memories/`
- If a proxy or deployment rejects that path with `404` or `405`, it retries with `/v1/memories`
- Request logging is summarized for debugging so failures are easier to trace without dumping full payloads

## 💬 How to Use It

Just chat. That's it.

The plugin works silently in the background:
- Extracts facts about you from conversations
- Retrieves relevant memories when needed
- Shows status messages when saving/loading memories (can be disabled)

Want to see what it remembers? Check **Settings** → **Personalization** → **Memories** in Open WebUI.

If you change embedding models/providers and want a clean re-embed of Open WebUI's native memory vectors, use Open WebUI's own memory reset/rebuild flow rather than editing the plugin cache by hand.

## 📋 Requirements

Comes with Open WebUI:
- `numpy`, `aiohttp`, `pydantic`

Optional (improves functionality):
- `sentence-transformers` - For local embeddings (otherwise uses API)
- `prometheus-client` - For metrics (gracefully skips if unavailable)

## 📄 License

MIT License - Use it however you want.

## 🐛 Issues?

Open an issue on this repo. I actively maintain and use this function.
