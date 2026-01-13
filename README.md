# Adaptive Memory v4.0 🧠

> **Intelligent, persistent memory for your LLMs**  
> Transform conversations into lasting knowledge with enterprise-grade memory management for Open WebUI.

## What is Adaptive Memory?

Adaptive Memory is a sophisticated plugin that gives Large Language Models persistent, personalized memory across conversations. It automatically extracts, categorizes, and retrieves user-specific information—creating natural, context-aware interactions that remember what matters.

## ✨ Key Features

### 🎯 **Intelligent Memory Extraction**
Automatically identifies and stores facts, preferences, relationships, and goals from conversations using LLM-powered analysis with confidence scoring.

### 🏗️ **Modular Architecture**
Built on a clean, pipeline-based design for reliability and extensibility:
- **EmbeddingManager**: Flexible embedding generation with local and API provider support
- **MemoryPipeline**: Core memory identification, retrieval, and processing logic
- **TaskManager**: Robust background task lifecycle with ghost task detection
- **ErrorManager**: Centralized error tracking and reporting
- **JSONParser**: Multi-strategy parsing with fallback mechanisms

### ⚡ **Advanced Background Processing**
- **Automatic Summarization**: Intelligently clusters and consolidates older memories (configurable interval, default 2 hours)
- **Semantic Deduplication**: Prevents duplicate memories using embedding-based similarity
- **Task Health Monitoring**: Built-in scavenger system detects and eliminates rogue tasks

### 🎨 **Smart Categorization**
Organizes memories with:
- **Tags**: identity, preference, behavior, relationship, goal, possession
- **Memory Banks**: Personal, Work, General contexts for focused retrieval

### 🔍 **Vector-Based Retrieval**
Efficient semantic search using cosine similarity with configurable thresholds and LRU caching for performance.

### 📊 **Enterprise Monitoring**
- **Prometheus Metrics**: Full instrumentation for embedding requests, retrieval latency, and error tracking
- **Real-time Status**: Live notifications during memory operations
- **Comprehensive Logging**: Timestamped, versioned logging throughout the pipeline

### 🔌 **Flexible Integration**
- **Embedding Providers**: Local SentenceTransformer models or OpenAI-compatible APIs
- **LLM Support**: Ollama and OpenAI-compatible endpoints with customizable configurations
- **Persistent Caching**: File-based embedding cache with automatic model compatibility validation

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Adaptive Memory v4.0                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌──────────────────┐                │
│  │ EmbeddingManager│  │  MemoryPipeline  │                │
│  ├─────────────────┤  ├──────────────────┤                │
│  │ • LRU Cache     │  │ • Identification │                │
│  │ • Persistence   │  │ • Retrieval      │                │
│  │ • Providers     │  │ • Processing     │                │
│  └────────┬────────┘  └────────┬─────────┘                │
│           │                    │                           │
│  ┌────────┴────────────────────┴─────────┐                │
│  │         TaskManager                   │                │
│  ├───────────────────────────────────────┤                │
│  │ • Background Summarization            │                │
│  │ • Ghost Task Detection                │                │
│  │ • Lifecycle Management                │                │
│  └───────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Background Tasks

| Task | Purpose | Default Interval |
|------|---------|-----------------|
| **Memory Summarization** | Clusters and consolidates related older memories | 2 hours |
| **Error Logging** | Reports error counters for monitoring | 30 minutes |
| **Date Updates** | Maintains current temporal context | 1 hour |

## 🎛️ Configuration

All settings are configurable via Open WebUI valves:

### Embedding Settings
- **Provider Type**: `local` or `openai_compatible`
- **Model Name**: Choose your embedding model
- **API Configuration**: URL and API key for remote providers

### LLM Settings
- **Provider**: Ollama or OpenAI-compatible APIs
- **Model Selection**: Configure analysis and summarization models
- **Endpoints**: Custom API URLs

### Memory Management
- **Confidence Threshold**: Minimum confidence for memory extraction (default: 0.7)
- **Similarity Threshold**: Vector similarity cutoff for retrieval (default: 0.6)
- **Max Related Memories**: Number of memories to inject per prompt (default: 5)
- **Task Intervals**: Customize background processing schedules

## 📦 Installation

1. Download `adaptive_memory_v4.0.py`
2. Navigate to Open WebUI → Functions
3. Upload the plugin file
4. Configure valves according to your setup
5. Enable the function for desired models

## 🔧 Requirements

**Core Dependencies** (included with Open WebUI):
- `numpy`, `aiohttp`, `pydantic`

**Optional Dependencies**:
- `sentence-transformers` - For local embedding models (falls back to API provider if not installed)
- `prometheus-client` - For metrics instrumentation (gracefully disabled if not available)

## 💡 How It Works

1. **Extraction**: User messages are analyzed by an LLM to identify memorable information
2. **Filtering**: Multi-layered pipeline focuses on user-specific facts, not general knowledge
3. **Storage**: Memories are categorized, tagged, and stored with vector embeddings
4. **Retrieval**: Semantic search finds relevant memories for each conversation
5. **Injection**: Top-N memories are added to the system prompt for context
6. **Maintenance**: Background tasks consolidate and optimize memory over time

## 🛠️ Recent Improvements (v4.0.1)

✅ **Fixed**: Lock management now uses regular dict instead of WeakValueDictionary to prevent premature garbage collection  
✅ **Enhanced**: Explicit lock cleanup prevents unbounded memory growth  
✅ **Improved**: Background task scavenger eliminates ghost tasks  
✅ **Added**: Comprehensive task lifecycle management

## 🤝 Contributing

This is a fork of the original OpenWebUI Adaptive Memory plugin, evolved with enterprise-grade features and architectural improvements. Contributions, issues, and feature requests are welcome!

## 📄 License

Follow the original Open WebUI licensing terms.

---

**Made with ❤️ for the Open WebUI community**
