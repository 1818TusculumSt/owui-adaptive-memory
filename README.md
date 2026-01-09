**OVERVIEW**

Adaptive Memory is a sophisticated plugin that provides persistent, personalized memory capabilities for Large Language Models (LLMs) within OpenWebUI. It enables LLMs to remember key information about users across separate conversations, creating a more natural and personalized experience. The system dynamically extracts, filters, stores, and retrieves user-specific information from conversations, then intelligently injects relevant memories into future LLM prompts.

**KEY FEATURES**

* **Intelligent Memory Extraction**: Automatically identifies facts, preferences, relationships, and goals from user messages using LLM-powered analysis
* **Smart Categorization**: Categorizes memories with appropriate tags (identity, preference, behavior, relationship, goal, possession) and memory banks (Personal, Work, General)
* **Advanced Filtering**: Multi-layered pipeline that focuses on user-specific information while filtering out general knowledge or trivia
* **Robust Processing**: JSON parsing with multiple fallback mechanisms for reliable memory extraction
* **Semantic Deduplication**: Smart deduplication using embedding-based similarity to prevent duplicate memories
* **Vector-Based Retrieval**: Efficient memory retrieval using cosine similarity with configurable thresholds
* **Background Summarization**: Automatic clustering and summarization of related older memories to prevent clutter and maintain relevance
* **Flexible Embedding Support**: Choice between local SentenceTransformer models or OpenAI-compatible API endpoints
* **Configurable LLM Integration**: Support for both Ollama and OpenAI-compatible APIs with customizable endpoints
* **Real-time Status Updates**: Live status notifications during memory processing and background operations
* **Comprehensive Monitoring**: Prometheus metrics instrumentation for performance tracking
* **Memory Bank Organization**: Categorize memories into Personal, Work, General contexts for focused retrieval
* **Background Task Management**: Automated maintenance tasks with configurable intervals

**ARCHITECTURE (v4.0)**

The system is built with a modular, pipeline-based architecture:

* **EmbeddingManager**: Handles embedding generation with support for local and API-based providers
* **MemoryPipeline**: Core logic for memory identification, retrieval, and processing
* **TaskManager**: Manages background tasks for summarization and maintenance
* **ErrorManager**: Centralized error tracking and reporting
* **JSONParser**: Robust JSON extraction with multiple fallback strategies

**BACKGROUND TASKS**

* **Memory Summarization**: Automatically clusters and summarizes related memories (default: every 2 hours)
* **Error Logging**: Periodic logging of error counters for monitoring (default: every 30 minutes)
* **Date Updates**: Keeps temporal context current (default: every hour)

**MAJOR CHANGES (v4.0)**

* **Complete Architecture Refactor**: Modular pipeline-based design for better maintainability and extensibility
* **Enhanced Background Processing**: Automatic memory summarization with clustering algorithms
* **Task Management System**: Robust background task system with proper lifecycle management
* **Improved Embedding Support**: Flexible provider system with local and API options
* **Comprehensive Logging**: Detailed logging throughout the pipeline for debugging and monitoring
* **Memory Confidence Scoring**: Confidence-based filtering for higher quality memories
* **Prometheus Metrics**: Full instrumentation for performance monitoring
* **Status Notifications**: Real-time updates during memory operations
* **Error Resilience**: Improved error handling and recovery mechanisms

**KNOWN ISSUES**

* **Valve UI Persistence**: OpenWebUI valve changes may not persist properly in some installations. If valve settings don't take effect, try restarting OpenWebUI completely or temporarily modify default values in the code.
