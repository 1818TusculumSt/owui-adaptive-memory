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

I added proper vector database synchronization, background task lifecycle management, comprehensive error handling, and persistent embedding cache to reduce API calls.

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

## 💬 How to Use It

Just chat. That's it.

The plugin works silently in the background:
- Extracts facts about you from conversations
- Retrieves relevant memories when needed
- Shows status messages when saving/loading memories (can be disabled)

Want to see what it remembers? Check **Settings** → **Personalization** → **Memories** in Open WebUI.

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
