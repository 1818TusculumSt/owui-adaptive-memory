import json
import copy  # Add deepcopy import
import traceback
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, Set
import logging
import re
import asyncio
import pytz
import difflib
from difflib import SequenceMatcher
import time
import os  # Added for local embedding model discovery
import hashlib  # For message de-duplication
import random  # For jitter in background task loops

# ----------------------------
# Metrics & Monitoring Imports
# ----------------------------
try:
    from prometheus_client import Counter, Histogram, REGISTRY, generate_latest  # type: ignore
except ImportError:
    # Fallback: define dummy Counter/Histogram if prometheus_client not installed
    class _NoOpMetric:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass

    Counter = Histogram = _NoOpMetric

# Define Prometheus metrics (or no-op if client missing)
EMBEDDING_REQUESTS = Counter('adaptive_memory_embedding_requests_total', 'Total number of embedding requests', ['provider'])
EMBEDDING_ERRORS = Counter('adaptive_memory_embedding_errors_total', 'Total number of embedding errors', ['provider'])
EMBEDDING_LATENCY = Histogram('adaptive_memory_embedding_latency_seconds', 'Latency of embedding generation', ['provider'])

RETRIEVAL_REQUESTS = Counter('adaptive_memory_retrieval_requests_total', 'Total number of get_relevant_memories calls', [])
RETRIEVAL_ERRORS = Counter('adaptive_memory_retrieval_errors_total', 'Total number of retrieval errors', [])
RETRIEVAL_LATENCY = Histogram('adaptive_memory_retrieval_latency_seconds', 'Latency of get_relevant_memories execution', [])

# Embedding model imports
from sentence_transformers import SentenceTransformer
import numpy as np

import aiohttp
from aiohttp import ClientError, ClientSession
from fastapi.requests import Request
from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, model_validator, field_validator, validator

# Updated imports for OpenWebUI 0.5+
from open_webui.routers.memories import (
    add_memory,
    AddMemoryForm,
    query_memory,
    QueryMemoryForm,
    delete_memory_by_id,
    Memories,
)
from open_webui.models.users import Users
from open_webui.main import app as webui_app
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT  # For persistent embedding storage

# Set up logging
logger = logging.getLogger("openwebui.plugins.adaptive_memory")
handler = logging.StreamHandler()


class JsonFormatter(logging.Formatter):
    def format(self, record):
        import json as _json

        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return _json.dumps(log_record)


formatter = JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False # Prevent duplicate logs if root logger has handlers
# Do not override root logger level; respect GLOBAL_LOG_LEVEL or root config


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = []
    memory_bank: Optional[str] = None  # NEW – bank assignment
    confidence: Optional[float] = None


class Filter:
    # Class-level singleton attributes to avoid missing attribute errors
    _embedding_model = None # Keep the underlying attribute name
    _memory_embeddings = {}
    _relevance_cache = {}

    # Class-level locks for message deduplication (shared across all instances)
    _message_locks = {}  # Dict of message_hash -> asyncio.Lock
    _locks_lock = asyncio.Lock()  # Lock to protect the _message_locks dict itself

    @property
    def _local_embedding_model(self): # RENAMED from embedding_model
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use the model name from valves for local loading too
                local_model_name = self.valves.embedding_model_name if self.valves.embedding_provider_type == 'local' else 'all-MiniLM-L6-v2'
                logger.info(f"Loading local embedding model: {local_model_name}")
                self._embedding_model = SentenceTransformer(local_model_name)
            except Exception as e:
                logger.error(f"Failed to load local SentenceTransformer model: {e}")
                self._embedding_model = None
        return self._embedding_model

    @property
    def memory_embeddings(self):
        """Return the class-level shared embedding cache"""
        return Filter._memory_embeddings

    @property
    def relevance_cache(self):
        """Return the class-level shared relevance cache"""
        return Filter._relevance_cache

    class Valves(BaseModel):
        """Configuration valves for the filter"""

        # ------ Begin Embedding Model Configuration ------ ADDED
        embedding_provider_type: Literal["local", "openai_compatible"] = Field(
            default="local",
            description="Type of embedding provider ('local' for SentenceTransformer or 'openai_compatible' for API)",
        )
        embedding_model_name: str = Field(
            default="all-MiniLM-L6-v2",  # Default to the local model
            description="Name of the embedding model to use (e.g., 'all-MiniLM-L6-v2', 'text-embedding-3-small')",
        )
        embedding_api_url: Optional[str] = Field(
            default=None,
            description="API endpoint URL for the embedding provider (required if type is 'openai_compatible')",
        )
        embedding_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the embedding provider (required if type is 'openai_compatible')",
        )
        # ------ End Embedding Model Configuration ------

        # ------ Begin Background Task Management Configuration ------
        enable_summarization_task: bool = Field(
            default=True,
            description="Enable or disable the background memory summarization task"
        )
        summarization_interval: int = Field(
            default=7200,  # 2 hours performance setting
            description="Interval in seconds between memory summarization runs"
        )
        
        enable_error_logging_task: bool = Field(
            default=True,
            description="Enable or disable the background error counter logging task"
        )
        error_logging_interval: int = Field(
            default=1800,  # 30 minutes performance setting
            description="Interval in seconds between error counter log entries"
        )
        
        enable_date_update_task: bool = Field(
            default=True,
            description="Enable or disable the background date update task"
        )
        date_update_interval: int = Field(
            default=3600,  # 1 hour performance setting
            description="Interval in seconds between date information updates"
        )
        
        enable_model_discovery_task: bool = Field(
            default=True,
            description="Enable or disable the background model discovery task"
        )
        model_discovery_interval: int = Field(
            default=7200,  # 2 hours performance setting
            description="Interval in seconds between model discovery runs"
        )
        # ------ End Background Task Management Configuration ------
        
        # ------ Begin Summarization Configuration ------
        summarization_min_cluster_size: int = Field(
            default=3,
            description="Minimum number of memories in a cluster for summarization"
        )
        summarization_similarity_threshold: float = Field(
            default=0.7,
            description="Threshold for considering memories related when using embedding similarity"
        )
        summarization_max_cluster_size: int = Field(
            default=8,
            description="Maximum memories to include in one summarization batch"
        )
        summarization_min_memory_age_days: int = Field(
            default=7,
            description="Minimum age in days for memories to be considered for summarization"
        )
        summarization_strategy: Literal["embeddings", "tags", "hybrid"] = Field(
            default="hybrid",
            description="Strategy for clustering memories: 'embeddings' (semantic similarity), 'tags' (shared tags), or 'hybrid' (combination)"
        )
        summarization_memory_prompt: str = Field(
            default="""You are a memory summarization assistant. Your task is to combine related memories about a user into a concise, comprehensive summary.

Given a set of related memories about a user, create a single paragraph that:
1. Captures all key information from the individual memories
2. Resolves any contradictions (prefer newer information)
3. Maintains specific details when important
4. Removes redundancy
5. Presents the information in a clear, concise format

Focus on preserving the user's:
- Explicit preferences
- Identity details
- Goals and aspirations
- Relationships
- Possessions
- Behavioral patterns

Your summary should be factual, concise, and maintain the same tone as the original memories.
Produce a single paragraph summary of approximately 50-100 words that effectively condenses the information.

Example:
Individual memories:
- "User likes to drink coffee in the morning"
- "User prefers dark roast coffee"
- "User mentioned drinking 2-3 cups of coffee daily"

Good summary:
"User is a coffee enthusiast who drinks 2-3 cups daily, particularly enjoying dark roast varieties in the morning."

Analyze the following related memories and provide a concise summary.""",
            description="System prompt for summarizing clusters of related memories"
        )
        # ------ End Summarization Configuration ------
        
        # ------ Begin Filtering & Saving Configuration ------
        enable_json_stripping: bool = Field(
            default=True,
            description="Attempt to strip non-JSON text before/after the main JSON object/array from LLM responses."
        )
        enable_fallback_regex: bool = Field(
            default=True,  # Enable for performance fallback
            description="If primary JSON parsing fails, attempt a simple regex fallback to extract at least one memory."
        )
        enable_short_preference_shortcut: bool = Field(
            default=True,
            description="If JSON parsing fails for a short message containing preference keywords, directly save the message content."
        )
        # --- NEW: Deduplication bypass for short preference statements ---
        short_preference_no_dedupe_length: int = Field(
            default=100,  # Allow longer short-preference statements to bypass deduplication
            description="If a NEW memory's content length is below this threshold and contains preference keywords, skip deduplication checks to avoid false positives."
        )
        preference_keywords_no_dedupe: str = Field(
            default="favorite,love,like,prefer,enjoy",
            description="Comma-separated keywords indicating user preferences that, when present in a short statement, trigger deduplication bypass."
        )

        # Blacklist topics (comma-separated substrings) - NOW OPTIONAL
        blacklist_topics: Optional[str] = Field(
            default=None,  # Default to None instead of empty string or default list
            description="Optional: Comma-separated list of topics to ignore during memory extraction",
        )

        # Enable trivia filtering
        filter_trivia: bool = Field(
            default=True,
            description="Enable filtering of trivia/general knowledge memories after extraction",
        )

        # Whitelist keywords (comma-separated substrings) - NOW OPTIONAL
        whitelist_keywords: Optional[str] = Field(
            default=None,  # Default to None
            description="Optional: Comma-separated keywords that force-save a memory even if blacklisted",
        )

        # Maximum total memories per user
        max_total_memories: int = Field(
            default=200,
            description="Maximum number of memories per user; prune oldest beyond this",
        )

        pruning_strategy: Literal["fifo", "least_relevant"] = Field(
            default="fifo",
            description="Strategy for pruning memories when max_total_memories is exceeded: 'fifo' (oldest first) or 'least_relevant' (lowest relevance to current message first).",
        )

        # Minimum memory length
        min_memory_length: int = Field(
            default=8, # Lowered default from 10
            description="Minimum length of memory content to be saved",
        )

        # --- NEW: Confidence Score Threshold ---
        min_confidence_threshold: float = Field(
            default=0.5,  # Default minimum confidence score to save a memory
            description="Minimum confidence score (0-1) required for an extracted memory to be saved. Scores below this are discarded."
        )
        # --- END NEW ---

        # Number of recent user messages to include in extraction context
        recent_messages_n: int = Field(
            default=5,
            description="Number of recent user messages to include in extraction prompt context",
        )

        # Relevance threshold for saving memories
        save_relevance_threshold: float = Field(
            default=0.8,
            description="Minimum relevance score (based on relevance calculation method) to save a memory",
        )

        # Max length of injected memory content (characters)
        max_injected_memory_length: int = Field(
            default=300,
            description="Maximum length of each injected memory snippet",
        )

        # --- Generic LLM Provider Configuration ---
        llm_provider_type: Literal["ollama", "openai_compatible"] = Field(
            default="ollama",
            description="Type of LLM provider ('ollama' or 'openai_compatible')",
        )
        llm_model_name: str = Field(
            default="llama3:latest",  # Default sensible for Ollama
            description="Name of the LLM model to use (e.g., 'llama3:latest', 'gpt-4o')",
        )
        llm_api_endpoint_url: str = Field(
            # Change default to use host.docker.internal for accessing Ollama on host
            default="http://host.docker.internal:11434/api/chat",
            description="API endpoint URL for the LLM provider (e.g., 'http://host.docker.internal:11434/api/chat', 'https://api.openai.com/v1/chat/completions')",
        )
        llm_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the LLM provider (required if type is 'openai_compatible')",
        )
        # --- End Generic LLM Provider Configuration ---

        # Memory processing settings
        related_memories_n: int = Field(
            default=5,
            description="Number of related memories to consider",
        )
        relevance_threshold: float = Field(
            default=0.60, # Lowered default further for better relevance
            description="Minimum relevance score (0-1) for memories to be considered relevant for injection after scoring"
        )
        memory_threshold: float = Field(
            default=0.6,
            description="Threshold for similarity when comparing memories (0-1)",
        )

        # Upgrade plan configs
        vector_similarity_threshold: float = Field(
            default=0.60,  # Lowered default further for better relevance
            description="Minimum cosine similarity for initial vector filtering (0-1)"
        )
        # NEW: If vector similarities are confidently high, skip the expensive LLM relevance call even
        #       when `use_llm_for_relevance` is True. This reduces overall LLM usage (Improvement #5).
        llm_skip_relevance_threshold: float = Field(
            default=0.93, # Slightly higher to reduce frequency of LLM calls (performance tuning)
            description="If *all* vector-filtered memories have similarity >= this threshold, treat the vector score as final relevance and skip the additional LLM call."
        )
        top_n_memories: int = Field(
            default=3, # Performance setting
            description="Number of top similar memories to pass to LLM",
        )
        cache_ttl_seconds: int = Field(
            default=86400,
            description="Cache time-to-live in seconds (default 24 hours)",
        )

        # --- Relevance Calculation Configuration ---
        use_llm_for_relevance: bool = Field(
            default=False, # Performance setting: rely on vector similarity
            description="Use LLM call for final relevance scoring (if False, relies solely on vector similarity + relevance_threshold)",
        )
        # --- End Relevance Calculation Configuration ---

        # Deduplicate identical memories
        deduplicate_memories: bool = Field(
            default=True,
            description="Prevent storing duplicate or very similar memories",
        )

        use_embeddings_for_deduplication: bool = Field(
            default=True,
            description="Use embedding-based similarity for more accurate semantic duplicate detection (if False, uses text-based similarity)",
        )

        # NEW: Dedicated threshold for embedding-based duplicate detection (higher because embeddings are tighter)
        embedding_similarity_threshold: float = Field(
            default=0.97,
            description="Threshold (0-1) for considering two memories duplicates when using embedding similarity."
        )

        similarity_threshold: float = Field(
            default=0.95,  # Tighten duplicate detection to minimise false positives
            description="Threshold for detecting similar memories (0-1) using text or embeddings"
        )

        # Time settings
        timezone: str = Field(
            default="Asia/Dubai",
            description="Timezone for date/time processing (e.g., 'America/New_York', 'Europe/London')",
        )

        # UI settings
        show_status: bool = Field(
            default=True, description="Show memory operations status in chat"
        )
        show_memories: bool = Field(
            default=True, description="Show relevant memories in context"
        )
        memory_format: Literal["bullet", "paragraph", "numbered"] = Field(
            default="bullet", description="Format for displaying memories in context"
        )

        # Memory categories
        enable_identity_memories: bool = Field(
            default=True,
            description="Enable collecting Basic Identity information (age, gender, location, etc.)",
        )
        enable_behavior_memories: bool = Field(
            default=True,
            description="Enable collecting Behavior information (interests, habits, etc.)",
        )
        enable_preference_memories: bool = Field(
            default=True,
            description="Enable collecting Preference information (likes, dislikes, etc.)",
        )
        enable_goal_memories: bool = Field(
            default=True,
            description="Enable collecting Goal information (aspirations, targets, etc.)",
        )
        enable_relationship_memories: bool = Field(
            default=True,
            description="Enable collecting Relationship information (friends, family, etc.)",
        )
        enable_possession_memories: bool = Field(
            default=True,
            description="Enable collecting Possession information (things owned or desired)",
        )

        # Error handling
        max_retries: int = Field(
            default=2, description="Maximum number of retries for API calls"
        )

        retry_delay: float = Field(
            default=1.0, description="Delay between retries (seconds)"
        )

        # System prompts
        memory_identification_prompt: str = Field(
            default='''You are an automated JSON data extraction system. Your ONLY function is to identify user-specific, persistent facts, preferences, goals, relationships, or interests from the user's messages and output them STRICTLY as a JSON array of operations.

**ABSOLUTE OUTPUT REQUIREMENT: FAILURE TO COMPLY WILL BREAK THE SYSTEM.**
1.  Your **ENTIRE** response **MUST** be **ONLY** a valid JSON array starting with `[` and ending with `]`. 
2.  **NO EXTRA TEXT**: Do **NOT** include **ANY** text, explanations, greetings, apologies, notes, or markdown formatting (like ```json) before or after the JSON array. 
3.  **ARRAY ALWAYS**: Even if you find only one memory, it **MUST** be enclosed in an array: `[{"operation": ...}]`. Do **NOT** output a single JSON object `{...}`.
4.  **EMPTY ARRAY**: If NO relevant user-specific memories are found, output **ONLY** an empty JSON array: `[]`.

**JSON OBJECT STRUCTURE (Each element in the array):**
*   Each element **MUST** be a JSON object: `{"operation": "NEW", "content": "...", "tags": ["..."], "memory_bank": "...", "confidence": float}`
*   **confidence**: You **MUST** include a confidence score (float between 0.0 and 1.0) indicating certainty that the extracted text is a persistent user fact/preference. High confidence (0.8-1.0) for direct statements, lower (0.5-0.7) for inferences or less certain preferences.
*   **memory_bank**: You **MUST** include a `memory_bank` field, choosing from: "General", "Personal", "Work". Default to "General" if unsure.
*   **tags**: You **MUST** include a `tags` field with a list of relevant tags from: ["identity", "behavior", "preference", "goal", "relationship", "possession"].

**INFORMATION TO EXTRACT (User-Specific ONLY):**
*   **Explicit Preferences/Statements:** User states "I love X", "My favorite is Y", "I enjoy Z". Extract these verbatim with high confidence.
*   **Identity:** Name, location, age, profession, etc. (high confidence)
*   **Goals:** Aspirations, plans (medium/high confidence depending on certainty).
*   **Relationships:** Mentions of family, friends, colleagues (high confidence).
*   **Possessions:** Things owned or desired (medium/high confidence).
*   **Behaviors/Interests:** Topics the user discusses or asks about (implying interest - medium confidence).

**RULES (Reiteration - Critical):**
+1. **JSON ARRAY ONLY**: `[`...`]` - Nothing else!
+2. **CONFIDENCE REQUIRED**: Every object needs a `"confidence": float` field.
+3. **MEMORY BANK REQUIRED**: Every object needs a `"memory_bank": "..."` field.
+4. **TAGS REQUIRED**: Every object needs a `"tags": [...]` field.
+5. **USER INFO ONLY**: Discard trivia, questions *to* the AI, temporary thoughts.

**FAILURE EXAMPLES (DO NOT DO THIS):**
*   `Okay, here is the JSON: [...]` <-- INVALID (extra text)
*   ` ```json
[{"operation": ...}]
``` ` <-- INVALID (markdown)
*   `{"memories": [...]}` <-- INVALID (not an array)
*   `{"operation": ...}` <-- INVALID (not in an array)
*   `[{"operation": ..., "content": ..., "tags": [...]}]` <-- INVALID (missing confidence/bank)

**GOOD EXAMPLE OUTPUT (Strictly adhere to this):**
```
[
  {
    "operation": "NEW",
    "content": "User has been a software engineer for 8 years",
    "tags": ["identity", "behavior"],
    "memory_bank": "Work",
    "confidence": 0.95
  },
  {
    "operation": "NEW",
    "content": "User has a cat named Whiskers",
    "tags": ["relationship", "possession"],
    "memory_bank": "Personal",
    "confidence": 0.9
  },
  {
    "operation": "NEW",
    "content": "User prefers working remotely",
    "tags": ["preference", "behavior"],
    "memory_bank": "Work",
    "confidence": 0.7
  },
  {
    "operation": "NEW",
    "content": "User's favorite book might be The Hitchhiker's Guide to the Galaxy",
    "tags": ["preference"],
    "memory_bank": "Personal",
    "confidence": 0.6
  }
]
```

Analyze the following user message(s) and provide **ONLY** the JSON array output. Double-check your response starts with `[` and ends with `]` and contains **NO** other text whatsoever.''', # Use triple single quotes for multiline string
            description="System prompt for memory identification (Emphasizing strict JSON array output and required fields)",
        )

        memory_relevance_prompt: str = Field(
            default="""You are a memory retrieval assistant. Your task is to determine which memories are relevant to the current context of a conversation.

IMPORTANT: **Do NOT mark general knowledge, trivia, or unrelated facts as relevant.** Only user-specific, persistent information should be rated highly.

Given the current user message and a set of memories, rate each memory's relevance on a scale from 0 to 1, where:
- 0 means completely irrelevant
- 1 means highly relevant and directly applicable

Consider:
- Explicit mentions in the user message
- Implicit connections to the user's personal info, preferences, goals, or relationships
- Potential usefulness for answering questions **about the user**
- Recency and importance of the memory

Examples:
- "User likes coffee" → likely relevant if coffee is mentioned
- "World War II started in 1939" → **irrelevant trivia, rate near 0**
- "User's friend is named Sarah" → relevant if friend is mentioned

Return your analysis as a JSON array with each memory's content, ID, and relevance score.
Example: [{"memory": "User likes coffee", "id": "123", "relevance": 0.8}]

Your output must be valid JSON only. No additional text.""",
            description="System prompt for memory relevance assessment",
        )

        memory_merge_prompt: str = Field(
            default="""You are a memory consolidation assistant. When given sets of memories, you merge similar or related memories while preserving all important information.

IMPORTANT: **Do NOT merge general knowledge, trivia, or unrelated facts.** Only merge user-specific, persistent information.

Rules for merging:
1. If two memories contradict, keep the newer information
2. Combine complementary information into a single comprehensive memory
3. Maintain the most specific details when merging
4. If two memories are distinct enough, keep them separate
5. Remove duplicate memories

Return your result as a JSON array of strings, with each string being a merged memory.
Your output must be valid JSON only. No additional text.""",
            description="System prompt for merging memories",
        )

        @field_validator(
            'summarization_interval', 'error_logging_interval', 'date_update_interval',
            'model_discovery_interval', 'max_total_memories', 'min_memory_length',
            'recent_messages_n', 'related_memories_n', 'top_n_memories',
            'cache_ttl_seconds', 'max_retries', 'max_injected_memory_length',
            'summarization_min_cluster_size', 'summarization_max_cluster_size', # Added
            'summarization_min_memory_age_days', # Added
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{info.field_name} must be a non-negative integer")
            return v

        @field_validator(
            'save_relevance_threshold', 'relevance_threshold', 'memory_threshold',
            'vector_similarity_threshold', 'similarity_threshold',
            'summarization_similarity_threshold',
            'llm_skip_relevance_threshold',  # New field included
            'embedding_similarity_threshold',  # Validate new embedding threshold as 0-1
            'min_confidence_threshold',  # NEW: Validate confidence threshold as 0-1
            check_fields=False
        )
        def check_threshold_float(cls, v, info):
            """Ensure threshold values are between 0.0 and 1.0"""
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
                )
            # Special documentation for similarity_threshold since it now has two usage contexts
            if info.field_name == 'similarity_threshold':
                logger.debug(
                    f"Set similarity_threshold to {v} - this threshold is used for both text-based and embedding-based deduplication based on the 'use_embeddings_for_deduplication' setting."
                )
            return v

        @field_validator('retry_delay')
        def check_non_negative_float(cls, v, info):
            if not isinstance(v, float) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return v
        
        @field_validator('timezone')
        def check_valid_timezone(cls, v):
            try:
                pytz.timezone(v)
            except pytz.exceptions.UnknownTimeZoneError:
                raise ValueError(f"Invalid timezone string: {v}")
            except Exception as e:
                 raise ValueError(f"Error validating timezone '{v}': {e}")
            return v

        # Keep existing model validator for LLM config
        @model_validator(mode="after")
        def check_llm_config(self):
            if self.llm_provider_type == "openai_compatible" and not self.llm_api_key:
                raise ValueError(
                    "API Key (llm_api_key) is required when llm_provider_type is 'openai_compatible'"
                )

            # Basic URL validation for Ollama default
            if self.llm_provider_type == "ollama":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "Ollama API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )
                # Could add more specific Ollama URL checks if needed

            # Basic URL validation for OpenAI compatible
            if self.llm_provider_type == "openai_compatible":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "OpenAI Compatible API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )

            return self

        # --- End Pydantic Validators for Valves ---

        # Control verbosity of error counter logging. When True, counters are logged at DEBUG level; when False, they are suppressed.
        debug_error_counter_logs: bool = Field(
            default=False,
            description="Emit detailed error counter logs at DEBUG level (set to True for troubleshooting).",
        )

        # ------ End Filtering & Saving Configuration ------

        # ------ Begin Memory Bank Configuration ------
        allowed_memory_banks: List[str] = Field(
            default=["General", "Personal", "Work"],
            description="List of allowed memory bank names for categorization."
        )
        default_memory_bank: str = Field(
            default="General",
            description="Default memory bank assigned when LLM omits or supplies an invalid bank."
        )
        # ------ End Memory Bank Configuration ------

        # ------ Begin Error Handling & Guarding Configuration (single authoritative block) ------
        enable_error_counter_guard: bool = Field(
            default=True,
            description="Enable guard to temporarily disable LLM/embedding features if specific error rates spike."
        )
        error_guard_threshold: int = Field(
            default=5,
            description="Number of errors within the window required to activate the guard."
        )
        error_guard_window_seconds: int = Field(
            default=600,  # 10 minutes
            description="Rolling time-window (in seconds) over which errors are counted for guarding logic."
        )
        # ------ End Error Handling & Guarding Configuration ------

        @field_validator(
            'allowed_memory_banks', # Add validation for this field
            check_fields=False # Run even if other validation fails
        )
        def check_allowed_memory_banks(cls, v):
            if not isinstance(v, list) or not v or v == ['']:
                logger.warning(f"Invalid 'allowed_memory_banks' loaded: {v}. Falling back to default.")
                # Return the default defined in the model itself
                return cls.model_fields['allowed_memory_banks'].default
            # Ensure all items are strings and non-empty after stripping
            cleaned_list = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned_list:
                logger.warning(f"Empty list after cleaning 'allowed_memory_banks': {v}. Falling back to default.")
                return cls.model_fields['allowed_memory_banks'].default
            return cleaned_list # Return the cleaned list

        # --- NEW Validator for Embedding Config ---
        @model_validator(mode="after")
        def check_embedding_config(self):
            if self.embedding_provider_type == "openai_compatible":
                if not self.embedding_api_key:
                    raise ValueError(
                        "API Key (embedding_api_key) is required when embedding_provider_type is 'openai_compatible'"
                    )
                if not self.embedding_api_url or not self.embedding_api_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "A valid API URL (embedding_api_url) starting with http:// or https:// is required when embedding_provider_type is 'openai_compatible'"
                    )
            elif self.embedding_provider_type == "local":
                # Optionally add checks for local model availability if needed
                pass
            return self
        # --- End Pydantic Validators for Valves ---

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable or disable the memory function"
        )
        show_status: bool = Field(
            default=True, description="Show memory processing status updates"
        )
        timezone: str = Field(
            default="",
            description="User's timezone (overrides global setting if provided)",
        )

    def __init__(self):
        """Initialize filter and schedule background tasks"""
        try:
            # Log instance creation for debugging multiple execution issue
            import uuid
            self._instance_id = str(uuid.uuid4())[:8]
            logger.info(f"Filter instance {self._instance_id} initializing (PID: {os.getpid()})")
            
            # Force re-initialization of valves using the current class definition
            default_valves_instance = self.Valves() # Create instance to access defaults easily
            self.config: Dict[str, Any] = {}

            # --- Attempt to load valves from config during init ---
            loaded_config_valves_dict = {}
            try:
                # Check if config attribute exists and has valves
                if hasattr(self, "config") and isinstance(self.config, dict) and "valves" in self.config and isinstance(self.config["valves"], dict):
                    loaded_config_valves_dict = self.config["valves"]
                    logger.info("Found 'valves' key in self.config during __init__.")
                else:
                    logger.info("self.config did not exist or lacked a 'valves' key during __init__, will use Pydantic defaults.")

                # Initialize self.valves using loaded config OR defaults if load failed/missing
                # Pydantic will automatically use defaults for missing keys in loaded_config_valves_dict
                self.valves = self.Valves(**loaded_config_valves_dict)
                logger.info("Successfully initialized self.valves using loaded config and/or Pydantic defaults.")

                # --- Explicit Check for allowed_memory_banks Post-Initialization ---
                # Re-validate allowed_memory_banks specifically, as it seems prone to loading issues
                if not isinstance(self.valves.allowed_memory_banks, list) or not self.valves.allowed_memory_banks or self.valves.allowed_memory_banks == ['']:
                    logger.warning(f"Post-init check found invalid allowed_memory_banks: {self.valves.allowed_memory_banks}. Resetting to default.")
                    self.valves.allowed_memory_banks = default_valves_instance.allowed_memory_banks # Use default from instance

            except Exception as e:
                logger.error(f"Error initializing/loading valves from self.config during __init__ (using defaults): {e}\n{traceback.format_exc()}")
                # Ensure self.valves is still a valid Valves instance using defaults on error
                self.valves = default_valves_instance
            # --- End valve loading attempt ---

            self.stored_memories = None
            self._error_message = None # Stores the reason for the last failure (e.g., json_parse_error)
            self._aiohttp_session = None

            # --- Added initialisations to prevent AttributeError ---
            # Track already-processed user messages to avoid duplicate extraction
            self._processed_messages: Set[str] = set()
            # Simple metrics counter dictionary
            self.metrics: Dict[str, int] = {"llm_call_count": 0}
            # Hold last processed body for confirmation tagging
            self._last_body: Dict[str, Any] = {}

            # Background tasks tracking
            self._background_tasks = set()

            # Error counters
            self.error_counters = {
                "embedding_errors": 0,
                "llm_call_errors": 0,
                "json_parse_errors": 0,
                "memory_crud_errors": 0,
            }
            
            # Log configuration for deduplication, helpful for testing and validation
            logger.debug(f"Memory deduplication settings:")
            try:
                logger.debug(f"  - deduplicate_memories: {self.valves.deduplicate_memories}")
                logger.debug(f"  - use_embeddings_for_deduplication: {self.valves.use_embeddings_for_deduplication}")
                logger.debug(f"  - similarity_threshold: {self.valves.similarity_threshold}")
            except Exception as e:
                logger.error(f"Error logging dedupe settings: {e}")

            # Model discovery results
            self.available_ollama_models = []
            self.available_openai_models = []
            # NEW: store locally available SentenceTransformer models
            self.available_local_embedding_models = []

            # Add current date awareness for prompts
            self.current_date = datetime.now()
            try:
                self.date_info = self._update_date_info()
            except Exception as e:
                 logger.error(f"Error updating date info: {e}")
                 self.date_info = {}

            # Note: _memory_embeddings and _relevance_cache are properties, no init needed.

            # Error counter tracking for guard mechanism
            from collections import deque
            self.error_timestamps = {
                "json_parse_errors": deque(),
                # Add other error types here if needed for guarding
            }
            self._guard_active = False
            self._guard_activated_at = 0

            # Initialize duplicate counters (used in process_memories)
            self._duplicate_skipped = 0
            self._duplicate_refreshed = 0

            # ------------------------------------------------------------
            # Guard/feature-flag initialisation
            # ------------------------------------------------------------
            self._llm_feature_guard_active: bool = False
            self._embedding_feature_guard_active: bool = False

            # Track that background tasks are not yet re-initialised via inlet()
            self._background_tasks_started: bool = False

            # Initialize background tasks immediately on startup
            try:
                self._initialize_background_tasks()
            except Exception as e:
                logger.error(f"Error in _initialize_background_tasks call: {e}")
            
        except Exception as e:
            logger.error(f"CRITICAL INIT FAILURE in Adaptive Memory: {e}\n{traceback.format_exc()}")
            # Attempt to set safe defaults so basic chat *might* work even if memory fails
            if not hasattr(self, 'valves'):
                 self.valves = self.Valves()

    def _initialize_background_tasks(self):
        """Initialize background tasks if they are not already running."""
        logger.info(f"Initializing background tasks. Summarization enabled: {self.valves.enable_summarization_task}")

        # Error logging task
        if self.valves.enable_error_logging_task:
            if not hasattr(self, "_error_log_task") or self._error_log_task is None or self._error_log_task.done():
                self._error_log_task = asyncio.create_task(self._log_error_counters_loop())
                self._background_tasks.add(self._error_log_task)
                self._error_log_task.add_done_callback(self._background_tasks.discard)
                logger.info("Started error logging background task")

        # Summarization task
        if self.valves.enable_summarization_task:
             if not hasattr(self, "_summarization_task") or self._summarization_task is None or self._summarization_task.done():
                self._summarization_task = asyncio.create_task(
                    self._summarize_old_memories_loop()
                )
                self._background_tasks.add(self._summarization_task)
                self._summarization_task.add_done_callback(self._background_tasks.discard)
                logger.info("Started memory summarization background task")

        # Date update task
        if self.valves.enable_date_update_task:
             if not hasattr(self, "_date_update_task") or self._date_update_task is None or (hasattr(self._date_update_task, 'done') and self._date_update_task.done()):
                self._date_update_task = self._schedule_date_update()
                logger.info("Scheduled date update background task")
        
        # Model discovery task
        if self.valves.enable_model_discovery_task:
             if not hasattr(self, "_model_discovery_task") or self._model_discovery_task is None or (hasattr(self._model_discovery_task, 'done') and self._model_discovery_task.done()):
                self._model_discovery_task = self._schedule_model_discovery()
                logger.info("Scheduled model discovery background task")

    async def _calculate_memory_age_days(self, memory: Dict[str, Any]) -> float:
        """Calculate age of a memory in days."""
        created_at = memory.get("created_at")
        if not created_at or not isinstance(created_at, datetime):
            return float("inf")  # Treat memories without valid dates as infinitely old

        # Ensure created_at is timezone-aware (assume UTC if not)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        # Get current time, also timezone-aware
        now_utc = datetime.now(timezone.utc)

        delta = now_utc - created_at
        return delta.total_seconds() / (24 * 3600)

    async def _find_memory_clusters(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find clusters of related memories based on configured strategy."""
        clusters = []
        processed_ids = set()
        strategy = self.valves.summarization_strategy
        threshold = self.valves.summarization_similarity_threshold
        min_age_days = self.valves.summarization_min_memory_age_days

        # --- Filter by Age First ---
        eligible_memories = []
        for mem in memories:
            age = await self._calculate_memory_age_days(mem)
            if age >= min_age_days:
                eligible_memories.append(mem)
            else:
                processed_ids.add(mem.get("id")) # Mark young memories as processed
        
        logger.debug(f"Summarization: Found {len(eligible_memories)} memories older than {min_age_days} days.")

        if not eligible_memories:
            return []

        # --- Embedding Clustering --- (Only if strategy is 'embeddings' or 'hybrid')
        embedding_clusters = []
        if strategy in ["embeddings", "hybrid"]:
            logger.debug(f"Clustering eligible memories using embeddings (threshold: {threshold})...")
            # Ensure all eligible memories have embeddings
            for mem in eligible_memories:
                mem_id = mem.get("id")
                if mem_id not in self.memory_embeddings:
                    try:
                        mem_text = mem.get("memory", "")
                        if mem_text:
                            mem_emb = await self._get_embedding(mem_text)
                            if mem_emb is not None:
                                self.memory_embeddings[mem_id] = mem_emb
                        else:
                             # Mark as None if no text to prevent repeated attempts
                             self.memory_embeddings[mem_id] = None
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for memory {mem_id} during clustering: {e}")
                        self.memory_embeddings[mem_id] = None # Mark as failed
            
            # Simple greedy clustering based on similarity
            temp_eligible = eligible_memories[:] # Work with a copy
            while temp_eligible:
                current_mem = temp_eligible.pop(0)
                current_id = current_mem.get("id")
                if current_id in processed_ids:
                    continue
                
                current_emb = self.memory_embeddings.get(current_id)
                if current_emb is None:
                    processed_ids.add(current_id)
                    continue # Skip if no embedding
                    
                cluster = [current_mem]
                processed_ids.add(current_id)
                
                remaining_after_pop = []
                for other_mem in temp_eligible:
                    other_id = other_mem.get("id")
                    if other_id in processed_ids:
                        continue
                        
                    other_emb = self.memory_embeddings.get(other_id)
                    if other_emb is None:
                         remaining_after_pop.append(other_mem)
                         continue # Skip if no embedding
                         
                    # Calculate similarity
                    try:
                        # Check for dimension mismatch before dot product
                        if current_emb.shape[0] != other_emb.shape[0]:
                            logger.warning(
                                f"Dimension mismatch for memories {current_id} ({current_emb.shape[0]}) and {other_id} ({other_emb.shape[0]})"
                            )

                            # Re-embed the memory with smaller dimension (likely the old one)
                            if other_emb.shape[0] < current_emb.shape[0]:
                                # Re-embed other_mem to match current_emb dimension
                                other_text = other_mem.get("memory", "")
                                if other_text:
                                    new_emb = await self._re_embed_memory_if_needed(
                                        other_id, other_text, other_emb.shape[0], current_emb.shape[0]
                                    )
                                    if new_emb is not None:
                                        other_emb = new_emb  # Use re-embedded version
                                        self.memory_embeddings[other_id] = new_emb
                                    else:
                                        remaining_after_pop.append(other_mem)
                                        continue
                                else:
                                    remaining_after_pop.append(other_mem)
                                    continue
                            else:
                                # Current memory has smaller dimension, skip clustering for now
                                # (current memory will be re-embedded when accessed later)
                                remaining_after_pop.append(other_mem)
                                continue

                        similarity = float(np.dot(current_emb, other_emb))
                        if similarity >= threshold:
                            cluster.append(other_mem)
                            processed_ids.add(other_id)
                        else:
                           remaining_after_pop.append(other_mem) # Keep for next iteration
                    except Exception as e:
                       logger.warning(f"Error comparing embeddings for {current_id} and {other_id}: {e}")
                       remaining_after_pop.append(other_mem)
                
                temp_eligible = remaining_after_pop # Update list for next outer loop iteration
                
                if len(cluster) >= self.valves.summarization_min_cluster_size:
                    embedding_clusters.append(cluster)
                    logger.debug(f"Found embedding cluster of size {len(cluster)} starting with ID {current_id}")
            logger.debug(f"Identified {len(embedding_clusters)} potential clusters via embeddings.")
            # If strategy is only embeddings, return now
            if strategy == "embeddings":
                 return embedding_clusters
        
        # --- Tag Clustering --- (Only if strategy is 'tags' or 'hybrid')
        tag_clusters = []
        if strategy in ["tags", "hybrid"]:
            logger.debug(f"Clustering eligible memories using tags...")
            from collections import defaultdict
            tag_map = defaultdict(list)
            
            # Group memories by tag
            for mem in eligible_memories:
                mem_id = mem.get("id")
                # Skip if already clustered by embeddings in hybrid mode
                if strategy == "hybrid" and mem_id in processed_ids:
                     continue
                     
                content = mem.get("memory", "")
                tags_match = re.match(r"\[Tags: (.*?)\]", content)
                if tags_match:
                    tags = [tag.strip() for tag in tags_match.group(1).split(",")]
                    for tag in tags:
                        tag_map[tag].append(mem)
            
            # Create clusters from tag groups
            cluster_candidates = list(tag_map.values())
            for candidate in cluster_candidates:
                # Filter out already processed IDs (important for hybrid)
                current_cluster = [mem for mem in candidate if mem.get("id") not in processed_ids]
                if len(current_cluster) >= self.valves.summarization_min_cluster_size:
                    tag_clusters.append(current_cluster)
                    # Mark these IDs as processed for hybrid mode
                    for mem in current_cluster:
                        processed_ids.add(mem.get("id"))
                    logger.debug(f"Found tag cluster of size {len(current_cluster)} based on tags: {[t for t,mems in tag_map.items() if candidate[0] in mems]}")
            logger.debug(f"Identified {len(tag_clusters)} potential clusters via tags.")
            if strategy == "tags":
                 return tag_clusters
        
        # --- Hybrid Strategy: Combine and return --- 
        if strategy == "hybrid":
             # Simply concatenate the lists of clusters found by each method
             logger.debug(f"Combining {len(embedding_clusters)} embedding clusters and {len(tag_clusters)} tag clusters for hybrid strategy.")
             all_clusters = embedding_clusters + tag_clusters
             return all_clusters
        
        # Should not be reached if strategy is valid, but return empty list as fallback
        return []

    async def _summarize_old_memories_loop(self):
        """Periodically summarize old memories into concise summaries"""
        try:
            while True:
                # Use configurable interval with small random jitter to prevent thundering herd
                # Re-check interval every 5 seconds to allow responsive updates
                jitter = random.uniform(0.9, 1.1)
                
                # Dynamic sleep loop
                elapsed = 0
                while elapsed < (self.valves.summarization_interval * jitter):
                    await asyncio.sleep(5)
                    elapsed += 5
                    # If interval setting was drastically reduced, break early
                    if elapsed > (self.valves.summarization_interval * jitter):
                        break
                logger.info("Starting periodic memory summarization run...")
                
                try:
                    # Fetch all users to process memories for each
                    users = []
                    try:
                        users = Users.get_users()
                    except Exception as e:
                        logger.warning(f"Failed to retrieve users list: {e}")

                    if not users:
                        logger.info("No users found for memory summarization.")
                        continue

                    for user_obj in users:
                        user_id = user_obj.id
                        try:
                            # Get all memories for the user
                            all_user_memories = await self._get_formatted_memories(user_id)
                            if len(all_user_memories) < self.valves.summarization_min_cluster_size:
                                # logger.debug(f"Summarization skipped: Not enough memories for user '{user_id}'.") # Reduce noise
                                continue
                                
                            logger.debug(f"Retrieved {len(all_user_memories)} total memories for user '{user_id}' for summarization.")
                            
                            # Find clusters of related, old memories
                            memory_clusters = await self._find_memory_clusters(all_user_memories)
                            
                            if not memory_clusters:
                                logger.debug(f"No eligible memory clusters found for user '{user_id}' for summarization.")
                                continue
                            
                            logger.info(f"Found {len(memory_clusters)} memory clusters to potentially summarize for user '{user_id}'.")
                            
                            # Process each cluster
                            summarized_count = 0
                            deleted_count = 0
                            for cluster in memory_clusters:
                                # Ensure cluster still meets minimum size after potential filtering in _find_memory_clusters
                                if len(cluster) < self.valves.summarization_min_cluster_size:
                                    continue
                                
                                # Limit cluster size for the LLM call
                                cluster_to_summarize = cluster[:self.valves.summarization_max_cluster_size]
                                logger.debug(f"Attempting to summarize cluster of size {len(cluster_to_summarize)} (max: {self.valves.summarization_max_cluster_size}).")

                                # Extract memory texts for the LLM prompt
                                mem_texts = [m.get("memory", "") for m in cluster_to_summarize]
                                # Sort by date to help LLM resolve contradictions potentially
                                cluster_to_summarize.sort(key=lambda m: m.get("created_at", datetime.min.replace(tzinfo=timezone.utc)))
                                combined_text = "\n- ".join([m.get("memory", "") for m in cluster_to_summarize])

                                # Use the new configurable summarization prompt
                                system_prompt = self.valves.summarization_memory_prompt
                                user_prompt = f"Related memories to summarize:\n- {combined_text}"

                                logger.debug(f"Calling LLM to summarize cluster. System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")
                                summary = await self.query_llm_with_retry(system_prompt, user_prompt)

                                if summary and not summary.startswith("Error:"):                            
                                    # Format summary with tags (e.g., from the first memory in cluster? Or generate new ones?)
                                    # For simplicity, let's try inheriting tags from the *first* memory in the sorted cluster
                                    first_mem_content = cluster_to_summarize[0].get("memory", "")
                                    tags = []
                                    tags_match = re.match(r"\[Tags: (.*?)\]", first_mem_content)
                                    if tags_match:
                                        tags = [tag.strip() for tag in tags_match.group(1).split(",")]
                                    
                                    # Add a specific "summarized" tag
                                    if "summarized" not in tags:
                                        tags.append("summarized")
                                        
                                    formatted_summary = f"[Tags: {', '.join(tags)}] {summary.strip()}"
                                    
                                    logger.info(f"Generated summary for cluster: {formatted_summary[:100]}...")
                                    
                                    # Save summary as new memory
                                    try:
                                        new_mem_op = MemoryOperation(operation="NEW", content=formatted_summary, tags=tags)
                                        await self._execute_memory_operation(new_mem_op, user_obj)
                                        summarized_count += 1
                                    except Exception as add_err:
                                        logger.error(f"Failed to save summary memory: {add_err}")
                                        continue # Skip deleting originals if saving summary fails
                                    
                                    # Delete original memories in the summarized cluster
                                    for mem_to_delete in cluster_to_summarize:
                                        try:
                                            delete_op = MemoryOperation(operation="DELETE", id=mem_to_delete["id"])
                                            await self._execute_memory_operation(delete_op, user_obj)
                                            deleted_count += 1
                                        except Exception as del_err:
                                            logger.warning(f"Failed to delete old memory {mem_to_delete.get('id')} during summarization: {del_err}")
                                            # Continue deleting others even if one fails
                                    logger.debug(f"Deleted {deleted_count} original memories after summarization.")
                                else:
                                    logger.warning(f"LLM failed to generate summary for cluster starting with ID {cluster_to_summarize[0].get('id')}. Response: {summary}")

                            if summarized_count > 0:
                                logger.info(f"Successfully generated {summarized_count} summaries and deleted {deleted_count} original memories for user '{user_id}'.")
                            else:
                                logger.info(f"No summaries were generated in this run for user '{user_id}'.")
                        except Exception as e:
                            logger.error(f"Error processing memories for user {user_id}: {e}")
                            continue
                        
                except Exception as e:
                    logger.error(f"Error in summarization loop for a user: {e}\n{traceback.format_exc()}")
                    # Continue loop even if one user fails
        except asyncio.CancelledError:
            logger.info("Memory summarization task cancelled.")
        except Exception as e:
            logger.error(f"Fatal error in summarization task loop: {e}\n{traceback.format_exc()}")

    def _update_date_info(self):
        """Update the date information dictionary with current time"""
        return {
            "iso_date": self.current_date.strftime("%Y-%m-%d"),
            "year": self.current_date.year,
            "month": self.current_date.strftime("%B"),
            "day": self.current_date.day,
            "weekday": self.current_date.strftime("%A"),
            "hour": self.current_date.hour,
            "minute": self.current_date.minute,
            "iso_time": self.current_date.strftime("%H:%M:%S"),
        }

    async def _log_error_counters_loop(self):
        """Periodically log error counters"""
        try:
            while True:
                # Use configurable interval with small random jitter
                jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                interval = self.valves.error_logging_interval * jitter
                await asyncio.sleep(interval)
                
                # Determine logging behaviour based on valve settings
                if self.valves.debug_error_counter_logs:
                    # Verbose debug logging – every interval
                    logger.debug(f"Error counters: {self.error_counters}")
                else:
                    # Only log when at least one counter is non-zero to reduce clutter
                    if any(count > 0 for count in self.error_counters.values()):
                        logger.info(f"Error counters (non-zero): {self.error_counters}")

                # Point 8: Error Counter Guard Logic
                if self.valves.enable_error_counter_guard:
                    now = time.time()
                    window = self.valves.error_guard_window_seconds
                    threshold = self.valves.error_guard_threshold

                    # Check JSON parse errors
                    error_type = "json_parse_errors"
                    # Record current count as a timestamp
                    current_count = self.error_counters[error_type]
                    # --- NOTE: This simple approach assumes the counter *increases* to track new errors.
                    # If the counter could be reset externally, a more robust timestamp queue is needed.
                    # For simplicity, assuming monotonically increasing count for now.
                    # A better approach: Store timestamp of each error occurrence.
                    # Let's refine this: Add timestamp whenever the error counter increments.
                    # We need to modify where the counter is incremented.

                    # --- Revised approach: Use a deque to store timestamps of recent errors --- 
                    timestamps = self.error_timestamps[error_type]
                    
                    # Remove old timestamps outside the window
                    while timestamps and timestamps[0] < now - window:
                        timestamps.popleft()

                    # Check if the count within the window exceeds the threshold
                    if len(timestamps) >= threshold:
                        if not self._guard_active:
                            logger.warning(f"Guard Activated: {error_type} count ({len(timestamps)}) reached threshold ({threshold}) in window ({window}s). Temporarily disabling LLM relevance and embedding dedupe.")
                            self._guard_active = True
                            self._guard_activated_at = now
                            # Temporarily disable features
                            self._original_use_llm_relevance = self.valves.use_llm_for_relevance
                            self._original_use_embedding_dedupe = self.valves.use_embeddings_for_deduplication
                            self.valves.use_llm_for_relevance = False
                            self.valves.use_embeddings_for_deduplication = False
                        elif self._guard_active:
                            # Deactivate guard if error rate drops below threshold (with hysteresis?)
                            # For simplicity, deactivate immediately when below threshold.
                            logger.info(f"Guard Deactivated: {error_type} count ({len(timestamps)}) below threshold ({threshold}). Re-enabling LLM relevance and embedding dedupe.")
                            self._guard_active = False
                            # Restore original settings
                            if hasattr(self, '_original_use_llm_relevance'):
                                self.valves.use_llm_for_relevance = self._original_use_llm_relevance
                            if hasattr(self, '_original_use_embedding_dedupe'):
                                self.valves.use_embeddings_for_deduplication = self._original_use_embedding_dedupe
        except asyncio.CancelledError:
            logger.debug("Error counter logging task cancelled")
        except Exception as e:
            logger.error(
                f"Error in error counter logging task: {e}\n{traceback.format_exc()}"
            )

    def _schedule_date_update(self):
        """Schedule a regular update of the date information"""

        async def update_date_loop():
            try:
                while True:
                    # Use configurable interval with small random jitter
                    jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                    interval = self.valves.date_update_interval * jitter
                    await asyncio.sleep(interval)
                    
                    self.current_date = self.get_formatted_datetime()
                    self.date_info = self._update_date_info()
                    logger.debug(f"Updated date information: {self.date_info}")
            except asyncio.CancelledError:
                logger.debug("Date update task cancelled")
            except Exception as e:
                logger.error(f"Error in date update task: {e}")

        # Start the update loop in the background
        task = asyncio.create_task(update_date_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _schedule_model_discovery(self):
        """Schedule a regular update of available models"""

        async def discover_models_loop():
            try:
                while True:
                    try:
                        # Discover models
                        await self._discover_models()
                        
                        # Use configurable interval with small random jitter
                        jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                        interval = self.valves.model_discovery_interval * jitter
                        await asyncio.sleep(interval)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error in model discovery: {e}")
                        # On error, retry sooner (1/6 of normal interval)
                        await asyncio.sleep(self.valves.model_discovery_interval / 6)
            except asyncio.CancelledError:
                logger.debug("Model discovery task cancelled")

        # Start the discovery loop in the background
        task = asyncio.create_task(discover_models_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _discover_models(self):
        """Discover available models from open_webui.configured providers"""
        logger.debug("Starting model discovery")

        # Create a session if needed
        session = await self._get_aiohttp_session()

        # Discover Ollama models
        try:
            ollama_url = "http://host.docker.internal:11434/api/tags"
            async with session.get(ollama_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "models" in data:
                        self.available_ollama_models = [
                            model["name"] for model in data["models"]
                        ]
                    logger.debug(
                        f"Discovered {len(self.available_ollama_models)} Ollama models"
                    )
        except Exception as e:
            logger.warning(f"Error discovering Ollama models: {e}")
            self.available_ollama_models = []

        # ------------------------------------------------------------
        # NEW: Discover local SentenceTransformer embedding models
        # ------------------------------------------------------------
        try:
            cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
            discovered_models = []
            if os.path.isdir(cache_dir):
                for entry in os.listdir(cache_dir):
                    entry_path = os.path.join(cache_dir, entry)
                    # Heuristic: model folders usually contain config.json or modules.json
                    if os.path.isdir(entry_path) and any(
                        os.path.isfile(os.path.join(entry_path, fname)) for fname in ("config.json", "modules.json")
                    ):
                        discovered_models.append(entry)

            # Provide sensible default if nothing discovered
            if not discovered_models:
                discovered_models = ["all-MiniLM-L6-v2"]

            self.available_local_embedding_models = discovered_models
            logger.debug(
                f"Discovered {len(discovered_models)} local SentenceTransformer models: {discovered_models}"
            )
        except Exception as e:
            logger.warning(f"Error discovering local embedding models: {e}")
            self.available_local_embedding_models = []

    def get_formatted_datetime(self, user_timezone=None):
        """
        Get properly formatted datetime with timezone awareness

        Args:
            user_timezone: Optional timezone string to override the default

        Returns:
            Timezone-aware datetime object
        """
        timezone_str = user_timezone or self.valves.timezone or "UTC"

        # Normalize common aliases
        alias_map = {
            "UAE/Dubai": "Asia/Dubai",
            "GMT+4": "Asia/Dubai",
            "GMT +4": "Asia/Dubai",
            "Dubai": "Asia/Dubai",
            "EST": "America/New_York",
            "PST": "America/Los_Angeles",
            "CST": "America/Chicago",
            "IST": "Asia/Kolkata",
            "BST": "Europe/London",
            "GMT": "Etc/GMT",
            "UTC": "UTC",
        }
        tz_key = timezone_str.strip()
        timezone_str = alias_map.get(tz_key, timezone_str)

        try:
            utc_now = datetime.utcnow()
            local_tz = pytz.timezone(timezone_str)
            local_now = utc_now.replace(tzinfo=pytz.utc).astimezone(local_tz)
            return local_now
        except pytz.exceptions.UnknownTimeZoneError:
            logger.warning(
                f"Invalid timezone: {timezone_str}, falling back to default 'Asia/Dubai'."
            )
            try:
                local_tz = pytz.timezone("Asia/Dubai")
                local_now = (
                    datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(local_tz)
                )
                return local_now
            except Exception:
                logger.warning("Fallback timezone also invalid, using UTC")
                return datetime.utcnow().replace(tzinfo=pytz.utc)

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
            )
        return self._aiohttp_session

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Intercepts incoming messages, extracts memories, injects relevant ones.

        Handles chat commands: /memory list, /memory forget [id], /memory edit [id] [new content],
        /memory summarize [topic/tag], /note [content], /memory mark_important [id],
        /memory unmark_important [id], /memory list_banks, /memory assign_bank [id] [bank]
        """
        logger.debug(
            f"Inlet received body keys: {list(body.keys())} for user: {__user__.get('id', 'N/A') if __user__ else 'N/A'}"
        )

        # Ensure user info is present
        if not __user__ or not __user__.get("id"):
            logger.warning("Inlet: User info or ID missing, skipping processing.")
            return body
        user_id = __user__["id"]

        # --- Extract Message for De-duplication ---
        # Extract message content BEFORE doing any heavy processing
        final_message = None
        if body.get("stream") is False and body.get("messages"):
            final_message = body["messages"][-1].get("content")
        elif body.get("stream") is True and body.get("done", False):
            final_message = body.get("message", {}).get("content")
        if final_message is None and body.get("messages"):
            final_message = body["messages"][-1].get("content")

        # --- MESSAGE DE-DUPLICATION - ACQUIRE LOCK IMMEDIATELY ---
        # Prevent multiple worker processes from processing the same message
        # This MUST happen before any heavy processing (valve loading, background tasks, etc.)
        message_hash = None
        if final_message:
            message_hash = self._create_message_hash(user_id, final_message)

            if not await self._try_acquire_message_lock(message_hash):
                logger.info(
                    f"Inlet: Message {message_hash[:8]}... already being processed by another instance, "
                    f"skipping duplicate inlet execution"
                )
                return body  # Early return - another instance is handling this

            logger.info(f"Inlet: Acquired lock for message {message_hash[:8]}..., proceeding with processing")

        # --- Initialization & Valve Loading ---
        # Load valves early, handle potential errors
        try:
            # NOTE: Do NOT reload self.valves here! Open WebUI sets it directly with user configuration.
            # Reloading from self.config (which is empty) would overwrite user settings with defaults.

            # Load user-specific valves (may override some per-user settings)
            user_valves = self._get_user_valves(__user__)

            if not user_valves.enabled:
                logger.debug(f"Memory plugin disabled for user {user_id}. Skipping.")
                return body # Return early if disabled

            # Respect per-user setting for status visibility, ensuring it's set after loading
            show_status = self.valves.show_status and user_valves.show_status
        except Exception as e:
            logger.error(f"Failed to load valves for user {user_id}: {e}")
            # Attempt to inform the UI, but ignore secondary errors to
            # avoid masking the original stack-trace
            try:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                        "type": "error",
                        "content": f"Error loading memory configuration: {e}",
                    },
                )
            except Exception:
                pass
            # Prevent processing when config is invalid
            return body

        # --- Background Task Initialization (Ensure runs once) ---
        # Use hasattr for a simple check if tasks have been started
        if not hasattr(self, "_background_tasks_started"):
            self._initialize_background_tasks()
            self._background_tasks_started = True


        # --- Check for Guard Conditions ---
        if self._llm_feature_guard_active:
            logger.warning("LLM feature guard active. Skipping LLM-dependent memory operations.")
        if self._embedding_feature_guard_active:
            logger.warning("Embedding feature guard active. Skipping embedding-dependent memory operations.")

        # --- Command Handling ---
        # Check if the final message is a command before processing memories
        if final_message and final_message.strip().startswith("/"):
            command_parts = final_message.strip().split()
            command = command_parts[0].lower()

            # --- /memory list_banks Command --- NEW
            if command == "/memory" and len(command_parts) >= 2 and command_parts[1].lower() == "list_banks":
                logger.info(f"Handling command: /memory list_banks for user {user_id}")
                try:
                    allowed_banks = self.valves.allowed_memory_banks
                    default_bank = self.valves.default_memory_bank
                    bank_list_str = "\n".join([f"- {bank} {'(Default)' if bank == default_bank else ''}" for bank in allowed_banks])
                    response_msg = f"**Available Memory Banks:**\n{bank_list_str}"
                    await self._safe_emit(__event_emitter__, {"type": "info", "content": response_msg})
                    body["messages"] = [] # Prevent LLM call
                    body["prompt"] = "Command executed." # Placeholder for UI
                    body["bypass_prompt_processing"] = True # Signal to skip further processing
                    return body
                except Exception as e:
                    logger.error(f"Error handling /memory list_banks: {e}")
                    await self._safe_emit(__event_emitter__, {"type": "error", "content": "Failed to list memory banks."})
                    # Allow fall through maybe? Or block? Let's block.
                    body["messages"] = []
                    body["prompt"] = "Error executing command." # Placeholder for UI
                    body["bypass_prompt_processing"] = True
                    return body

            # --- /memory assign_bank Command --- NEW
            elif command == "/memory" and len(command_parts) >= 4 and command_parts[1].lower() == "assign_bank":
                logger.info(f"Handling command: /memory assign_bank for user {user_id}")
                try:
                    memory_id = command_parts[2]
                    target_bank = command_parts[3]

                    if target_bank not in self.valves.allowed_memory_banks:
                        allowed_banks_str = ", ".join(self.valves.allowed_memory_banks)
                        await self._safe_emit(__event_emitter__, {"type": "error", "content": f"Invalid bank '{target_bank}'. Allowed banks: {allowed_banks_str}"})
                    else:
                        # 1. Query the specific memory
                        # Note: query_memory might return multiple if content matches, need filtering by ID
                        query_result = await query_memory(
                            user_id=user_id,
                            form_data=QueryMemoryForm(query=memory_id, k=1000) # Query broadly first
                        )
                        target_memory = None
                        if query_result and query_result.memories:
                            for mem in query_result.memories:
                                if mem.id == memory_id:
                                    target_memory = mem
                                    break

                        if not target_memory:
                            await self._safe_emit(__event_emitter__, {"type": "error", "content": f"Memory with ID '{memory_id}' not found."})
                        else:
                            # 2. Check if bank is already correct
                            current_bank = target_memory.metadata.get("memory_bank", self.valves.default_memory_bank)
                            if current_bank == target_bank:
                                await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Memory '{memory_id}' is already in bank '{target_bank}'."})
                            else:
                                # 3. Update the memory (delete + add with modified metadata)
                                new_metadata = target_memory.metadata.copy()
                                new_metadata["memory_bank"] = target_bank
                                new_metadata["timestamp"] = datetime.now(timezone.utc).isoformat() # Update timestamp
                                new_metadata["source"] = "adaptive_memory_v3_assign_bank_cmd"

                                await delete_memory_by_id(user_id=user_id, memory_id=memory_id)
                                await add_memory(
                                    user_id=user_id,
                                    form_data=AddMemoryForm(
                                        content=target_memory.content,
                                        metadata=new_metadata
                                    )
                                )
                                await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Successfully assigned memory '{memory_id}' to bank '{target_bank}'."})
                                self._increment_error_counter("memory_bank_assigned_cmd")

                except IndexError:
                     await self._safe_emit(__event_emitter__, {"type": "error", "content": "Usage: /memory assign_bank [memory_id] [bank_name]"})
                except Exception as e:
                    logger.error(f"Error handling /memory assign_bank: {e}\n{traceback.format_exc()}")
                    await self._safe_emit(__event_emitter__, {"type": "error", "content": f"Failed to assign memory bank: {e}"})
                    self._increment_error_counter("assign_bank_cmd_error")

                # Always bypass LLM after handling command
                body["messages"] = []
                body["prompt"] = "Command executed." # Placeholder
                body["bypass_prompt_processing"] = True
                return body

            # --- Other /memory commands (Placeholder/Example - Adapt as needed) ---
            elif command == "/memory":
                # Example: Check for /memory list, /memory forget, etc.
                # Implement logic similar to assign_bank: parse args, call OWUI functions, emit status
                # Remember to add command handlers here based on other implemented features
                logger.info(f"Handling generic /memory command stub for user {user_id}: {final_message}")
                await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Memory command '{final_message}' received (implementation pending)."})
                body["messages"] = []
                body["prompt"] = "Memory command received." # Placeholder
                body["bypass_prompt_processing"] = True
                return body

            # --- /note command (Placeholder/Example) ---
            elif command == "/note":
                 logger.info(f"Handling /note command stub for user {user_id}: {final_message}")
                 # Implement logic for Feature 6 (Scratchpad)
                 await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Note command '{final_message}' received (implementation pending)."})
                 body["messages"] = []
                 body["prompt"] = "Note command received." # Placeholder
                 body["bypass_prompt_processing"] = True
                 return body

        # --- Memory Injection --- #
        if self.valves.show_memories and not self._embedding_feature_guard_active: # Guard embedding-dependent retrieval
            try:
                logger.debug(f"Retrieving relevant memories for user {user_id}")
                # Use user-specific timezone for relevance calculation context
                relevant_memories = await self.get_relevant_memories(
                    current_message=final_message if final_message else "",
                    user_id=user_id,
                    user_timezone=user_valves.timezone # Use user-specific timezone
                )
                if relevant_memories:
                    logger.info(
                        f"Injecting {len(relevant_memories)} relevant memories for user {user_id}"
                    )
                    # --- Emit Status: Injecting Memories --- ADDED
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Injecting {len(relevant_memories)} memories into context...",
                                    "done": False, # Still part of the pre-processing
                                },
                            },
                    )
                    self._inject_memories_into_context(body, relevant_memories)
                else:
                    logger.debug(f"No relevant memories found for user {user_id}")
            except Exception as e:
                logger.error(
                    f"Error retrieving/injecting memories: {e}\n{traceback.format_exc()}"
                )
                await self._safe_emit(
                    __event_emitter__,
                    {"type": "error", "content": "Error retrieving relevant memories."},
                )

        # Release the message lock before returning
        if message_hash:
            self._release_message_lock(message_hash)
            logger.debug(f"Inlet: Released lock for message {message_hash[:8]}... before returning")

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Process LLM response, extract memories, and update the response"""
        # logger.debug("****** OUTLET FUNCTION CALLED ******") # REMOVED

        # Log function entry with user info for debugging 5x execution issue
        debug_user_id = __user__.get("id") if __user__ else "NO_USER"
        logger.info(f"Outlet called for user_id: {debug_user_id}")

        # DEFENSIVE: Make a deep copy of the body to avoid dictionary changed size during iteration
        # This was a source of many subtle bugs
        body_copy = copy.deepcopy(body)

        # Skip processing if user is not authenticated
        if not __user__:
            logger.warning("No user information available - skipping memory processing")
            return body_copy

        # Get user's ID for memory storage
        user_id = __user__.get("id")
        if not user_id:
            logger.warning("User object contains no ID - skipping memory processing")
            return body_copy

        # --- Extract Message for De-duplication ---
        # Extract message content BEFORE doing any heavy processing
        last_user_message_content = None
        messages_copy = copy.deepcopy(body_copy.get("messages", []))
        if messages_copy:
            # Find the actual last user message in the history included in the body
            for msg in reversed(messages_copy):
                if msg.get("role") == "user" and msg.get("content"):
                    last_user_message_content = msg.get("content")
                    break

        # --- MESSAGE DE-DUPLICATION - ACQUIRE LOCK IMMEDIATELY ---
        # Prevent multiple worker processes from processing the same message
        # This MUST happen before any heavy processing (valve loading, background tasks, etc.)
        message_hash = None
        if last_user_message_content:
            message_hash = self._create_message_hash(user_id, last_user_message_content)

            if not await self._try_acquire_message_lock(message_hash):
                logger.info(
                    f"Outlet: Message {message_hash[:8]}... already being processed by another instance, "
                    f"skipping duplicate outlet execution"
                )
                return body_copy  # Early return - another instance is handling this

            logger.info(f"Outlet: Acquired lock for message {message_hash[:8]}..., proceeding with processing")

        # Check if user has enabled memory function
        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            logger.info(f"Memory function is disabled for user {user_id}")
            # Release lock before returning
            if message_hash:
                self._release_message_lock(message_hash)
            return body_copy

        # Get user's timezone if set
        user_timezone = user_valves.timezone or self.valves.timezone

        # --- BEGIN MEMORY PROCESSING IN OUTLET ---
        # Process the *last user message* for memory extraction *after* the LLM response
        message_history_for_context = []

        try:
            # Get up to N messages *before* the last user message for context
            if last_user_message_content:
                user_msg_index = -1
                for i, msg in enumerate(messages_copy):
                    if msg.get("role") == "user" and msg.get("content") == last_user_message_content:
                        user_msg_index = i
                        break
                if user_msg_index != -1:
                    start_index = max(0, user_msg_index - self.valves.recent_messages_n)
                    message_history_for_context = messages_copy[start_index:user_msg_index]

                logger.info(f"Starting memory processing in outlet for user message: {last_user_message_content[:60]}...")
                # Use asyncio.create_task for non-blocking processing
                # Reload valves inside _process_user_memories ensures latest config
                memory_task = asyncio.create_task(
                    self._process_user_memories(
                        user_message=last_user_message_content,
                        user_id=user_id,
                        event_emitter=__event_emitter__,
                        show_status=user_valves.show_status, # Still show status if user wants
                        user_timezone=user_timezone,
                        recent_chat_history=message_history_for_context,
                    )
                )
                # Optional: Add callback or handle task completion if needed, but allow it to run in background
                # memory_task.add_done_callback(lambda t: logger.info(f"Outlet memory task finished: {t.result()}"))
            else:
                 logger.warning("Could not find last user message in outlet body to process for memories.")

        except Exception as e:
            logger.error(f"Error initiating memory processing in outlet: {e}\n{traceback.format_exc()}")
            # Release lock on error
            if message_hash:
                self._release_message_lock(message_hash)
        # --- END MEMORY PROCESSING IN OUTLET ---

        # Add confirmation message if memories were processed
        try:
            if user_valves.show_status:
                await self._add_confirmation_message(body_copy)
        except Exception as e:
            logger.error(f"Error adding confirmation message: {e}")

        # Release the message lock before returning
        if message_hash:
            self._release_message_lock(message_hash)
            logger.debug(f"Released lock for message {message_hash[:8]}... before returning")

        # Return the modified response
        return body_copy

    async def _safe_emit(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        data: Dict[str, Any],
    ) -> None:
        """Safely emit an event, handling missing emitter"""
        if not event_emitter:
            logger.debug("Event emitter not available")
            return

        try:
            await event_emitter(data)
        except Exception as e:
            logger.error(f"Error in event emitter: {e}")

    def _get_user_valves(self, __user__: dict) -> UserValves:
        """Extract and validate user valves settings"""
        if not __user__:
            logger.warning("No user information provided")
            return self.UserValves()

        # Access the valves attribute directly from the UserModel object
        user_valves_data = getattr(
            __user__, "valves", {}
        )  # Use getattr for safe access

        # Ensure we have a dictionary to work with
        if not isinstance(user_valves_data, dict):
            logger.warning(
                f"User valves attribute is not a dictionary (type: {type(user_valves_data)}), using defaults."
            )
            user_valves_data = {}

        try:
            # Validate and return the UserValves model
            return self.UserValves(**user_valves_data)
        except Exception as e:
            # Default to enabled if validation/extraction fails
            logger.error(
                f"Could not determine user valves settings from data {user_valves_data}: {e}"
            )
            return self.UserValves()  # Return default UserValves on error

    def _create_message_hash(self, user_id: str, message_content: str) -> str:
        """
        Create a unique hash for a message to use as a de-duplication key.
        Uses user_id + message_content + time bucket (5 second intervals) to allow
        the same message to be processed again after a short delay.
        """
        # Round timestamp to 5-second buckets to allow reprocessing after a delay
        time_bucket = int(time.time() / 5) * 5

        # Create hash from user_id, message, and time bucket
        hash_input = f"{user_id}:{message_content}:{time_bucket}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]  # Use first 16 chars

    async def _try_acquire_message_lock(self, message_hash: str) -> bool:
        """
        Try to acquire an exclusive lock for processing this message.
        Returns True if lock was acquired, False if another instance is processing it.
        Uses class-level asyncio locks shared across all Filter instances.
        """
        try:
            # Protect access to the _message_locks dict itself
            async with self._locks_lock:
                # Get or create a lock for this specific message
                if message_hash not in self._message_locks:
                    self._message_locks[message_hash] = asyncio.Lock()
                message_lock = self._message_locks[message_hash]

            # Try to acquire the lock atomically using acquire() with immediate check
            # This prevents race condition where multiple instances check locked() before any acquires
            await asyncio.wait_for(
                message_lock.acquire(),
                timeout=0.001  # 1ms timeout - effectively non-blocking
            )
            # If we reach here, lock was successfully acquired
            logger.debug(f"Acquired lock for message {message_hash[:8]}...")
            return True

        except asyncio.TimeoutError:
            # Lock is held by another instance
            logger.debug(f"Message {message_hash[:8]}... is being processed by another instance (timeout)")
            return False
        except Exception as e:
            logger.error(f"Error acquiring message lock: {e}")
            # On error, allow processing to continue (fail-open)
            return True

    def _release_message_lock(self, message_hash: str) -> None:
        """Release the lock for a message after processing is complete."""
        try:
            if message_hash in self._message_locks:
                lock = self._message_locks[message_hash]
                if lock.locked():
                    lock.release()
                    logger.debug(f"Released lock for message {message_hash[:8]}...")
        except Exception as e:
            logger.warning(f"Error releasing message lock: {e}")

    async def _get_formatted_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user and format them for processing"""
        memories_list = []
        embeddings_loaded = 0
        embeddings_missing = 0

        try:
            # Get memories using Memories.get_memories_by_user_id
            user_memories = Memories.get_memories_by_user_id(user_id=str(user_id))

            if user_memories:
                for memory in user_memories:
                    # Safely extract attributes with fallbacks
                    memory_id = str(getattr(memory, "id", "unknown"))
                    memory_content = getattr(memory, "content", "")
                    created_at = getattr(memory, "created_at", None)
                    updated_at = getattr(memory, "updated_at", None)

                    # Load embedding from database if available
                    embedding_data = getattr(memory, "embedding", None)
                    if embedding_data is not None:
                        try:
                            # Convert embedding to numpy array if it's a list/string
                            if isinstance(embedding_data, str):
                                # Deserialize JSON string
                                embedding_array = np.array(json.loads(embedding_data), dtype=np.float32)
                            elif isinstance(embedding_data, (list, tuple)):
                                # Already a list, convert to numpy
                                embedding_array = np.array(embedding_data, dtype=np.float32)
                            elif isinstance(embedding_data, np.ndarray):
                                # Already numpy array
                                embedding_array = embedding_data.astype(np.float32)
                            else:
                                # Unknown format, skip
                                logger.warning(f"Unknown embedding format for memory {memory_id}: {type(embedding_data)}")
                                embedding_array = None

                            if embedding_array is not None and len(embedding_array) > 0:
                                # Cache the embedding
                                self.memory_embeddings[memory_id] = embedding_array
                                embeddings_loaded += 1
                                logger.debug(f"Loaded embedding from DB for memory {memory_id}, dim: {len(embedding_array)}")
                            else:
                                embeddings_missing += 1
                        except Exception as e:
                            logger.warning(f"Failed to deserialize embedding for memory {memory_id}: {e}")
                            embeddings_missing += 1
                    else:
                        embeddings_missing += 1

                    memories_list.append(
                        {
                            "id": memory_id,
                            "memory": memory_content,
                            "created_at": created_at,
                            "updated_at": updated_at,
                        }
                    )

            logger.info(
                f"Retrieved {len(memories_list)} memories for user {user_id} "
                f"(embeddings: {embeddings_loaded} loaded, {embeddings_missing} missing)"
            )
            return memories_list

        except Exception as e:
            logger.error(
                f"Error getting formatted memories: {e}\n{traceback.format_exc()}"
            )
            return []

    def _inject_memories_into_context(
        self, body: Dict[str, Any], memories: List[Dict[str, Any]]
    ) -> None:
        """Inject relevant memories into the system context"""
        if not memories:
            # Suppress fallback injection when no relevant memories
            return

        # Sort memories by relevance if available
        sorted_memories = sorted(
            memories, key=lambda x: x.get("relevance", 0), reverse=True
        )

        # Format memories based on user preference
        memory_context = self._format_memories_for_context(
            sorted_memories, self.valves.memory_format
        )

        # Prepend instruction to avoid LLM meta-comments
        instruction = (
            "Here is background info about the user. "
            "Do NOT mention this info explicitly unless relevant to the user's query. "
            "Do NOT explain what you remember or don't remember. "
            "Do NOT summarize or list what you know or don't know about the user. "
            "Do NOT say 'I have not remembered any specific information' or similar. "
            "Do NOT explain your instructions, context, or memory management. "
            "Do NOT mention tags, dates, or internal processes. "
            "Only answer the user's question directly.\n\n"
        )
        memory_context = instruction + memory_context

        # Log injected memories for debugging
        logger.debug(f"Injected memories:\n{memory_context[:500]}...")

        # Add to system message or create a new one if none exists
        if "messages" in body:
            system_message_exists = False
            for message in body["messages"]:
                if message["role"] == "system":
                    message["content"] += f"\n\n{memory_context}"
                    system_message_exists = True
                    break

            if not system_message_exists:
                body["messages"].insert(
                    0, {"role": "system", "content": memory_context}
                )

    def _format_memories_for_context(
        self, memories: List[Dict[str, Any]], format_type: str
    ) -> str:
        """Format memories for context injection based on format preference"""
        if not memories:
            return ""

        max_len = getattr(self.valves, "max_injected_memory_length", 300)

        # Start with header
        memory_context = "I recall the following about you:\n"

        # Extract tags and add each memory according to specified format
        if format_type == "bullet":
            for mem in memories:
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", mem["memory"])
                if tags_match:
                    tags = tags_match.group(1)
                    content = tags_match.group(2)[:max_len]
                    memory_context += f"- {content} (tags: {tags})\n"
                else:
                    content = mem["memory"][:max_len]
                    memory_context += f"- {content}\n"

        elif format_type == "numbered":
            for i, mem in enumerate(memories, 1):
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", mem["memory"])
                if tags_match:
                    tags = tags_match.group(1)
                    content = tags_match.group(2)[:max_len]
                    memory_context += f"{i}. {content} (tags: {tags})\n"
                else:
                    content = mem["memory"][:max_len]
                    memory_context += f"{i}. {content}\n"

        else:  # paragraph format
            memories_text = []
            for mem in memories:
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", mem["memory"])
                if tags_match:
                    content = tags_match.group(2)[:max_len]
                    memories_text.append(content)
                else:
                    content = mem["memory"][:max_len]
                    memories_text.append(content)

            memory_context += f"{'. '.join(memories_text)}.\n"

        return memory_context

    async def _process_user_memories(
        self,
        user_message: str,
        user_id: str,
        event_emitter: Optional[
            Callable[[Any], Awaitable[None]]
        ] = None,  # Renamed for clarity
        show_status: bool = True,
        user_timezone: str = None,
        recent_chat_history: Optional[
            List[Dict[str, Any]]
        ] = None,  # Added this argument
    ) -> List[Dict[str, Any]]:
        """Process user message to extract and store memories

        Returns:
            List of stored memory operations
        """
        # Start timer
        start_time = time.perf_counter()

        # Reset error message
        self._error_message = None

        # Emit "processing memories" status if enabled
        if show_status:
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": "📝 Extracting potential new memories from your message…",
                        "done": False,
                    },
                },
            )

        # Debug logging for function entry
        logger.debug(
            f"Starting _process_user_memories for user {user_id} with message: {user_message[:50]}..."
        )

        # Get user valves
        user_valves = None
        try:
            user = Users.get_user_by_id(user_id)
            user_valves = self._get_user_valves(user)

            # Debug logging for user valves
            logger.debug(
                f"Retrieved user valves with memory enabled: {user_valves.enabled}"
            )

            if not user_valves.enabled:
                logger.info(f"Memory function disabled for user: {user_id}")
                if show_status:
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "⏸️ Adaptive Memory is disabled in your settings – skipping memory save.",
                                "done": True,
                            },
                        },
                    )
                return []
        except Exception as e:
            logger.error(f"Error getting user valves: {e}")
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ Unable to access memory settings – aborting memory save process.",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for memory identification start
        logger.debug(f"Starting memory identification for message: {user_message[:60]}...")

        # Step 1: Use LLM to identify memories in the message
        memories = []
        parse_error_occurred = False # Track if parsing failed
        try:
            # Get user's existing memories for context (optional - can also be None)
            existing_memories = None
            # If the LLM needs context of existing memories:
            try:
                existing_memories = await self._get_formatted_memories(user_id)
                logger.debug(
                    f"Retrieved {len(existing_memories)} existing memories for context"
                )
            except Exception as e:
                logger.warning(f"Could not get existing memories (continuing): {e}")

            # Process message to extract memory operations
            memories = await self.identify_memories(
                user_message,
                existing_memories=existing_memories,
                user_timezone=user_timezone,
            )

            # Debug logging after memory identification
            logger.debug(
                f"Memory identification complete. Found {len(memories)} potential memories"
            )

        except Exception as e:
            self.error_counters["llm_call_errors"] += 1
            logger.error(f"Error identifying memories: {e}\n{traceback.format_exc()}")
            self._error_message = f"llm_error: {str(e)[:50]}..." # Point 6: More specific error
            parse_error_occurred = True # Indicate identification failed
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": f"⚠️ Memory error: {str(e)}",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for filtering
        logger.debug("Starting memory filtering step...")

        # Step 2: Filter memories (apply blacklist/whitelist/trivia filtering)
        filtered_memories = []
        if memories:
            # Apply filters based on valves
            try:
                # Get filter configuration valves
                min_length = self.valves.min_memory_length
                blacklist = self.valves.blacklist_topics
                whitelist = self.valves.whitelist_keywords
                filter_trivia = self.valves.filter_trivia

                logger.debug(
                    f"Using filters: min_length={min_length}, blacklist={blacklist}, whitelist={whitelist}, filter_trivia={filter_trivia}"
                )

                # Default trivia patterns (common knowledge patterns)
                trivia_patterns = [
                    r"\b(when|what|who|where|how)\s+(is|was|were|are|do|does|did)\b",  # Common knowledge questions
                    r"\b(fact|facts)\b",  # Explicit facts
                    r"\b(in the year|in \d{4})\b",  # Historical dates
                    r"\b(country|countries|capital|continent|ocean|sea|river|mountain|planet)\b",  # Geographic/scientific
                    r"\b(population|inventor|invented|discovered|founder|founded|created|author|written|directed)\b",  # Attribution/creation
                ]

                # Known meta-request phrases
                meta_request_phrases = [
                    "remember this",
                    "make a note",
                    "don't forget",
                    "keep in mind",
                    "save this",
                    "add this to",
                    "log this",
                    "put this in",
                ]

                # Process each memory with filtering
                for memory in memories:
                    # Validate operation
                    if not self._validate_memory_operation(memory):
                        logger.debug(f"Invalid memory operation: {str(memory)}")
                        continue

                    # Extract content for filtering
                    content = memory.get("content", "").strip()

                    # Apply minimum length filter
                    if len(content) < min_length:
                        logger.debug(
                            f"Memory too short ({len(content)} < {min_length}): {content}"
                        )
                        continue

                    # Check if it's a meta-request
                    is_meta_request = False
                    for phrase in meta_request_phrases:
                        if phrase.lower() in content.lower():
                            is_meta_request = True
                            logger.debug(f"Meta-request detected: {content}")
                            break

                    if is_meta_request:
                        continue

                    # Check blacklist (if configured)
                    if blacklist:
                        is_blacklisted = False
                        for topic in blacklist.split(","):
                            topic = topic.strip().lower()
                            if topic and topic in content.lower():
                                # Check whitelist override
                                is_whitelisted = False
                                if whitelist:
                                    for keyword in whitelist.split(","):
                                        keyword = keyword.strip().lower()
                                        if keyword and keyword in content.lower():
                                            is_whitelisted = True
                                            logger.debug(
                                                f"Whitelisted term '{keyword}' found in blacklisted content"
                                            )
                                            break

                                if not is_whitelisted:
                                    is_blacklisted = True
                                    logger.debug(
                                        f"Blacklisted topic '{topic}' found: {content}"
                                    )
                                    break

                        if is_blacklisted:
                            continue

                    # Check trivia patterns (if enabled)
                    if filter_trivia:
                        is_trivia = False
                        for pattern in trivia_patterns:
                            if re.search(pattern, content.lower()):
                                logger.debug(
                                    f"Trivia pattern '{pattern}' matched: {content}"
                                )
                                is_trivia = True
                                break

                        if is_trivia:
                            continue

                    # Memory passed all filters
                    filtered_memories.append(memory)
                    logger.debug(f"Memory passed all filters: {content}")

                logger.info(
                    f"Filtered memories: {len(filtered_memories)}/{len(memories)} passed"
                )
            except Exception as e:
                logger.error(f"Error filtering memories: {e}\n{traceback.format_exc()}")
                filtered_memories = (
                    memories  # On error, attempt to process all memories
                )

        # --- NEW: Confidence Score Filtering ---
        memories_passing_confidence = []
        low_confidence_discarded = 0
        min_conf = self.valves.min_confidence_threshold
        logger.debug(f"Applying confidence filter (threshold: {min_conf})...")
        for mem in filtered_memories:
            confidence_score = float(mem.get("confidence", 0.0)) # Ensure float comparison
            if confidence_score >= min_conf:
                memories_passing_confidence.append(mem)
            else:
                low_confidence_discarded += 1
                logger.debug(f"Discarding memory due to low confidence ({confidence_score:.2f} < {min_conf}): {str(mem.get('content', ''))[:50]}...")
        
        # Emit status message if any memories were discarded due to low confidence
        if low_confidence_discarded > 0 and show_status:
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": f"ℹ️ Discarded {low_confidence_discarded} potential memories due to low confidence (< {min_conf}).",
                        "done": False, # Indicate processing is ongoing
                    },
                },
            )
        
        # Use the confidence-filtered list for subsequent processing
        filtered_memories = memories_passing_confidence
        # --- END NEW ---

        # Debug logging after filtering
        logger.debug(f"After filtering: {len(filtered_memories)} memories remain")

        # If no memories to process after filtering, log and return
        if not filtered_memories: # Check if the list is empty
            # --- Check for JSON Parse Error --- NEW
            if self._error_message == "json_parse_error" and show_status:
                 await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ LLM response invalid - memory extraction failed.",
                            "done": True, # Mark as done even on error
                        },
                    },
                )
                 return [] # Exit after emitting error status
            # --- END JSON Parse Error Check ---

            # Point 5: Immediate-Save Shortcut for short preferences on parse error
            if (
                self.valves.enable_short_preference_shortcut
                and parse_error_occurred
                and len(user_message) <= 60
                and any(keyword in user_message.lower() for keyword in ["favorite", "love", "like", "enjoy"])
            ):
                logger.info("JSON parse failed, but applying short preference shortcut.")
                try:
                    shortcut_op = MemoryOperation(
                        operation="NEW",
                        content=user_message.strip(), # Save the raw message content
                        tags=["preference"] # Assume preference tag
                    )
                    await self._execute_memory_operation(shortcut_op, user) # Directly execute
                    logger.info(f"Successfully saved memory via shortcut: {user_message[:50]}...")
                    # Set a specific status message for this case
                    self._error_message = None # Clear parse error flag
                    # Since we bypassed normal processing, we need a result list for status reporting
                    saved_operations_list = [shortcut_op.model_dump()] # Use model_dump() for Pydantic v2+
                    # Skip the rest of the processing steps as we forced a save
                except Exception as shortcut_err:
                    logger.error(f"Error during short preference shortcut save: {shortcut_err}")
                    self._error_message = "shortcut_save_error"
                    saved_operations_list = [] # Indicate save failed
            else:
                # Normal case: No memories identified or filtered out, and no shortcut applied
                logger.info("No valid memories to process after filtering/identification.")
                if show_status and not self._error_message:
                    # Determine reason for no save
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"ℹ️ Memory save skipped – {final_status_reason.replace('_', ' ')}."
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": status_desc,
                                "done": True,
                            },
                        },
                    )
                return [] # Return empty list as nothing was saved through normal path
        else:
           # We have filtered_memories, proceed with normal processing
           pass # Continue to Step 3

        # Step 3: Get current memories and handle max_total_memories limit
        try:
            current_memories_data = await self._get_formatted_memories(user_id)
            logger.debug(
                f"Retrieved {len(current_memories_data)} existing memories from database"
            )

            # If we'd exceed the maximum memories per user, apply pruning
            max_memories = self.valves.max_total_memories
            current_count = len(current_memories_data)
            new_count = len(filtered_memories) # Only count NEW operations towards limit for pruning decision
            
            if current_count + new_count > max_memories:
                to_remove = current_count + new_count - max_memories
                logger.info(
                    f"Memory limit ({max_memories}) would be exceeded. Need to prune {to_remove} memories."
                )
                
                memories_to_prune_ids = []
                
                # Choose pruning strategy based on valve
                strategy = self.valves.pruning_strategy
                logger.info(f"Applying pruning strategy: {strategy}")
                
                if strategy == "least_relevant":
                    try:
                        # Calculate relevance for all existing memories against the current user message
                        memories_with_relevance = []
                        # Re-use logic similar to get_relevant_memories but for *all* memories
                        
                        user_embedding = None
                        try:
                            user_embedding = await self._get_embedding(user_message)
                        except Exception as e:
                            logger.warning(f"Could not encode user message for relevance pruning: {e}")

                        # Determine if we can use vectors or need LLM fallback (respecting valve)
                        can_use_vectors = user_embedding is not None
                        needs_llm = self.valves.use_llm_for_relevance

                        # --- Calculate Scores --- 
                        if not needs_llm and can_use_vectors:
                             # Vector-only relevance calculation
                            for mem_data in current_memories_data:
                                mem_id = mem_data.get("id")
                                mem_emb = self.memory_embeddings.get(mem_id)
                                # Ensure embedding exists or try to compute it
                                if mem_emb is None:
                                    try:
                                        mem_text = mem_data.get("memory") or ""
                                        if mem_text:
                                            mem_emb = await self._get_embedding(mem_text)
                                            if mem_emb is not None:
                                                self.memory_embeddings[mem_id] = mem_emb # Cache it
                                    except Exception as e:
                                        logger.warning(f"Failed to compute embedding for existing memory {mem_id}: {e}")
                                        mem_emb = None # Mark as failed
                                
                                if mem_emb is not None:
                                    # Check for dimension mismatch before dot product
                                    if user_embedding.shape[0] != mem_emb.shape[0]:
                                        logger.warning(
                                            f"Dimension mismatch in relevance scoring: user={user_embedding.shape[0]}, "
                                            f"memory={mem_emb.shape[0]}. Attempting to re-embed memory."
                                        )

                                        # Try to re-embed the memory with current provider
                                        mem_text = mem_data.get("memory", "")
                                        if mem_text:
                                            new_mem_emb = await self._re_embed_memory_if_needed(
                                                mem_id, mem_text, mem_emb.shape[0], user_embedding.shape[0]
                                            )
                                            if new_mem_emb is not None:
                                                # Successfully re-embedded, calculate relevance
                                                sim_score = float(np.dot(user_embedding, new_mem_emb))
                                                memories_with_relevance.append({"id": mem_id, "relevance": sim_score})
                                            else:
                                                # Re-embedding failed, assign low relevance
                                                logger.warning(f"Re-embedding failed for memory {mem_id}, assigning low relevance")
                                                memories_with_relevance.append({"id": mem_id, "relevance": 0.0})
                                        else:
                                            # No memory text to re-embed, assign low relevance
                                            memories_with_relevance.append({"id": mem_id, "relevance": 0.0})
                                    else:
                                        sim_score = float(np.dot(user_embedding, mem_emb))
                                        memories_with_relevance.append({"id": mem_id, "relevance": sim_score})
                                else:
                                    # Assign low relevance if embedding fails
                                    memories_with_relevance.append({"id": mem_id, "relevance": 0.0})
                        elif needs_llm:
                            # LLM-based relevance calculation (simplified, no caching needed here)
                            # Prepare memories for LLM prompt
                            memory_strings_for_llm = [
                                f"ID: {mem['id']}, CONTENT: {mem['memory']}" 
                                for mem in current_memories_data
                            ]
                            system_prompt = self.valves.memory_relevance_prompt
                            llm_user_prompt = f"""Current user message: "{user_message}"

Available memories:
{json.dumps(memory_strings_for_llm)}

Rate the relevance of EACH memory to the current user message."""
                            
                            try:
                                llm_response_text = await self.query_llm_with_retry(system_prompt, llm_user_prompt)
                                llm_relevance_results = self._extract_and_parse_json(llm_response_text)
                                
                                if isinstance(llm_relevance_results, list):
                                    # Map results back to IDs
                                    llm_scores = {item.get("id"): item.get("relevance", 0.0) for item in llm_relevance_results if isinstance(item, dict)}
                                    for mem_data in current_memories_data:
                                        mem_id = mem_data.get("id")
                                        score = llm_scores.get(mem_id, 0.0) # Default to 0 if LLM missed it
                                        memories_with_relevance.append({"id": mem_id, "relevance": score})
                                else:
                                    logger.warning("LLM relevance check for pruning failed to return valid list. Pruning might default to FIFO.")
                                    # Fallback: assign 0 relevance to all, effectively making it FIFO-like for this run
                                    memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]
                            except Exception as llm_err:
                                logger.error(f"Error during LLM relevance check for pruning: {llm_err}")
                                memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]
                        else: # Cannot use vectors and LLM not enabled - default to FIFO-like
                             logger.warning("Cannot determine relevance for pruning (no embeddings/LLM). Pruning will be FIFO-like.")
                             memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]

                        # --- Sort and Select for Pruning ---                     
                        # Sort by relevance ascending (lowest first)
                        memories_with_relevance.sort(key=lambda x: x.get("relevance", 0.0))
                        
                        # Select the IDs of the least relevant memories to remove (take the first `to_remove` items after sorting)
                        memories_to_prune_ids = [mem["id"] for mem in memories_with_relevance[:to_remove]]
                        logger.info(f"Identified {len(memories_to_prune_ids)} least relevant memories for pruning.")
                        
                    except Exception as relevance_err:
                        logger.error(f"Error calculating relevance for pruning, falling back to FIFO: {relevance_err}")
                        # Fallback to FIFO on any error during relevance calculation
                        strategy = "fifo"
                        
                # Default or fallback FIFO strategy
                if strategy == "fifo":
                    # Sort by timestamp ascending (oldest first)
                    # Make sure timestamp exists, fallback to a very old date if not
                    default_date = datetime.min.replace(tzinfo=timezone.utc)
                    sorted_memories = sorted(
                        current_memories_data, 
                        key=lambda x: x.get("created_at", default_date)
                    )
                    memories_to_prune_ids = [mem["id"] for mem in sorted_memories[:to_remove]]
                    logger.info(f"Identified {len(memories_to_prune_ids)} oldest memories (FIFO) for pruning.")

                # Execute pruning if IDs were identified
                if memories_to_prune_ids:
                    pruned_count = 0
                    for memory_id_to_delete in memories_to_prune_ids:
                        try:
                            delete_op = MemoryOperation(operation="DELETE", id=memory_id_to_delete)
                            await self._execute_memory_operation(delete_op, user)
                            pruned_count += 1
                        except Exception as e:
                            logger.error(f"Error pruning memory {memory_id_to_delete}: {e}")
                    logger.info(f"Successfully pruned {pruned_count} memories.")
                else:
                    logger.warning("Pruning needed but no memory IDs identified for deletion.")
                    
        except Exception as e:
            logger.error(
                f"Error handling max_total_memories: {e}\n{traceback.format_exc()}"
            )
            # Continue processing the new memories even if pruning failed

        # Debug logging before processing operations
        logger.debug("Beginning to process memory operations...")

        # Step 4: Process the filtered memories
        processing_error: Optional[Exception] = None
        try:
            # process_memories now returns the list of successfully executed operations
            logger.debug(f"Calling process_memories with {len(filtered_memories)} items: {str(filtered_memories)}") # Log the exact list being passed
            saved_operations_list = await self.process_memories(
                filtered_memories, user_id
            )
            logger.debug(
                f"Memory saving attempt complete, returned {len(saved_operations_list)} successfully saved operations."
            )
        except Exception as e:
            processing_error = e
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            self._error_message = f"processing_error: {str(e)[:50]}..." # Point 6: More specific error

        # Debug confirmation logs
        if saved_operations_list:
            logger.info(
                f"Successfully processed and saved {len(saved_operations_list)} memories"
            )
        elif processing_error:
            logger.warning(
                f"Memory processing failed due to an error: {processing_error}"
            )
        else:
            logger.warning(
                "Memory processing finished, but no memories were saved (potentially due to duplicates or errors during save).)"
            )

        # Emit completion status
        if show_status:
            elapsed_time = time.perf_counter() - start_time
            # Base the status on the actual saved operations list
            saved_count = len(saved_operations_list)  # Directly use length of result
            if saved_count > 0:
                # Check if it was the shortcut save
                if any(op.get("content") == user_message.strip() for op in saved_operations_list):
                     status_desc = f"✅ Saved 1 memory via shortcut ({elapsed_time:.2f}s)"
                else:
                    plural = "memory" if saved_count == 1 else "memories"
                    status_desc = f"✅ Added {saved_count} new {plural} to your memory bank ({elapsed_time:.2f}s)"
            else:
                # Build smarter status based on duplicate counters
                if getattr(self, "_duplicate_refreshed", 0):
                    status_desc = f"✅ Memory refreshed (duplicate confirmed) ({elapsed_time:.2f}s)"
                elif getattr(self, "_duplicate_skipped", 0):
                    status_desc = f"✅ Preference already saved – duplicate ignored ({elapsed_time:.2f}s)"
                else:
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"⚠️ Memory save skipped – {final_status_reason.replace('_', ' ')} ({elapsed_time:.2f}s)"
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": status_desc,
                        "done": True,
                    },
                },
            )

        # Return the list of operations that were actually saved
        return saved_operations_list

    async def identify_memories(
        self,
        input_text: str,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        user_timezone: str = None,
    ) -> List[Dict[str, Any]]:
        """Identify potential memories from text using LLM"""
        logger.debug(
            f"Starting memory identification from input text: {input_text[:50]}..."
        )

        # Remove <details> blocks that may interfere with processing
        input_text = re.sub(r"<details>.*?</details>", "", input_text, flags=re.DOTALL)

        # Clean up and prepare the input
        clean_input = input_text.strip()
        logger.debug(f"Cleaned input text length: {len(clean_input)}")

        # Prepare the system prompt
        try:
            # Get the base prompt template
            memory_prompt = self.valves.memory_identification_prompt

            # Add datetime context
            now_str = self.get_formatted_datetime(user_timezone)
            datetime_context = f"Current datetime: {now_str}"

            # Add memory categories context based on enabled flags
            categories = []
            if self.valves.enable_identity_memories:
                categories.append("identity")
            if self.valves.enable_behavior_memories:
                categories.append("behavior")
            if self.valves.enable_preference_memories:
                categories.append("preference")
            if self.valves.enable_goal_memories:
                categories.append("goal")
            if self.valves.enable_relationship_memories:
                categories.append("relationship")
            if self.valves.enable_possession_memories:
                categories.append("possession")

            categories_str = ", ".join(categories)

            # Add existing memories context if provided
            existing_memories_str = ""
            if existing_memories and len(existing_memories) > 0:
                existing_memories_str = "Existing memories:\n"
                for i, mem in enumerate(
                    existing_memories[:5]
                ):  # Limit to 5 recent memories
                    existing_memories_str += f"- {mem.get('content', 'Unknown')}\n"

            # Combine all context
            context = f"{datetime_context}\nEnabled categories: {categories_str}\n{existing_memories_str}"

            # Log the components of the prompt
            logger.debug(f"Memory identification context: {context}")

            # Create the final system prompt with context
            system_prompt = f"{memory_prompt}\n\nCONTEXT:\n{context}"

            logger.debug(
                f"Final memory identification system prompt length: {len(system_prompt)}"
            )
        except Exception as e:
            logger.error(f"Error building memory identification prompt: {e}")
            system_prompt = self.valves.memory_identification_prompt

        # Call LLM to identify memories
        start_time = time.time()
        logger.debug(
            f"Calling LLM for memory identification with provider: {self.valves.llm_provider_type}, model: {self.valves.llm_model_name}"
        )

        try:
            # Construct the user prompt with few-shot examples
            user_prompt = f"""Analyze the following user message and extract relevant memories:
>>> USER MESSAGE START <<<
+{clean_input}
>>> USER MESSAGE END <<<

--- EXAMPLES OF DESIRED OUTPUT FORMAT ---
Example 1 Input: "I really love pizza, especially pepperoni."
Example 1 Output: [{{"operation": "NEW", "content": "User loves pizza, especially pepperoni", "tags": ["preference"], "confidence": 0.85}}]

Example 2 Input: "What's the weather like today?"
Example 2 Output: []

Example 3 Input: "My sister Jane is visiting next week. I should buy her flowers."
Example 3 Output: [{{"operation": "NEW", "content": "User has a sister named Jane", "tags": ["relationship"], "confidence": 0.9}}, {{"operation": "NEW", "content": "User's sister Jane is visiting next week", "tags": ["relationship"], "confidence": 0.95}}]
--- END EXAMPLES ---

Produce ONLY the JSON array output for the user message above, adhering strictly to the format requirements outlined in the system prompt.
"""
            # Note: Doubled curly braces {{ }} are used to escape them within the f-string for the JSON examples.

            # Log the user prompt structure for debugging
            logger.debug(
                f"User prompt structure with few-shot examples:\n{user_prompt[:500]}..."
            )  # Log first 500 chars

            # Call LLM with the modified prompts
            llm_response = await self.query_llm_with_retry(
                system_prompt, user_prompt
            )  # Pass the new user_prompt
            elapsed = time.time() - start_time
            logger.debug(
                f"LLM memory identification completed in {elapsed:.2f}s, response length: {len(llm_response)}"
            )
            logger.debug(f"LLM raw response for memory identification: {llm_response}")

            # --- Handle LLM Errors --- #
            if llm_response.startswith("Error:"):
                self.error_counters["llm_call_errors"] += 1
                if "LLM_CONNECTION_FAILED" in llm_response:
                    logger.error(f"LLM Connection Error during identification: {llm_response}")
                    self._error_message = "llm_connection_error"
                else:
                    logger.error(f"LLM Error during identification: {llm_response}")
                    self._error_message = "llm_error"
                return [] # Return empty list on LLM error

            # Parse the response (assumes JSON format)
            result = self._extract_and_parse_json(llm_response)
            logger.debug(
                f"Parsed result type: {type(result)}, content: {str(result)[:500]}"
            )

            # Check if we got a dict instead of a list (common LLM error)
            if isinstance(result, dict):
                logger.warning(
                    "LLM returned a JSON object instead of an array. Attempting conversion."
                )
                result = self._convert_dict_to_memory_operations(result)
                logger.debug(f"Converted dict to {len(result)} memory operations")

            # Check for empty result
            if not result:
                logger.warning("No memory operations identified by LLM")
                return []

            # Validate operations format
            valid_operations = []
            invalid_count = 0

            if isinstance(result, list):
                for op in result:
                    if self._validate_memory_operation(op):
                        valid_operations.append(op)
                    else:
                        invalid_count += 1

                logger.debug(
                    f"Identified {len(valid_operations)} valid memory operations, {invalid_count} invalid"
                )
                return valid_operations
            else:
                logger.error(
                    f"LLM returned invalid format (neither list nor dict): {type(result)}"
                )
                self._error_message = (
                    "LLM returned invalid format. Expected JSON array."
                )
                return []

        except Exception as e:
            logger.error(
                f"Error in memory identification: {e}\n{traceback.format_exc()}"
            )
            self.error_counters["llm_call_errors"] += 1
            self._error_message = f"Memory identification error: {str(e)}"
            return []

    def _validate_memory_operation(self, op: Dict[str, Any]) -> bool:
        """Validate memory operation format and required fields"""
        if not isinstance(op, dict):
            logger.warning(f"Invalid memory operation format (not a dict): {op}")
            return False

        # Check if operation field exists, if not try to infer it
        if "operation" not in op:
            # Look for typical patterns to guess the operation type
            if any(k.lower() == "operation" for k in op.keys()):
                # Operation may be under a different case
                for k, v in op.items():
                    if k.lower() == "operation" and isinstance(v, str):
                        op["operation"] = v
                        break

            # Look for operation in original format but in wrong place
            elif isinstance(op, dict) and any(
                v in ["NEW", "UPDATE", "DELETE"] for v in op.values()
            ):
                for k, v in op.items():
                    if v in ["NEW", "UPDATE", "DELETE"]:
                        op["operation"] = v
                        # Remove the old key if it's not "operation"
                        if k != "operation":
                            op.pop(k, None)
                        break

            # Default based on presence of fields
            elif "id" in op and "content" in op:
                # Default to UPDATE if we have both id and content
                op["operation"] = "UPDATE"
            elif "content" in op:
                # Default to NEW if we only have content
                op["operation"] = "NEW"
            else:
                logger.warning(f"Cannot determine operation type for: {op}")
                return False

        # Normalize operation to uppercase
        if isinstance(op["operation"], str):
            op["operation"] = op["operation"].upper()

        if op["operation"] not in ["NEW", "UPDATE", "DELETE"]:
            logger.warning(f"Invalid operation type: {op['operation']}")
            return False

        if op["operation"] in ["UPDATE", "DELETE"] and "id" not in op:
            logger.warning(f"Missing ID for {op['operation']} operation: {op}")
            return False

        if op["operation"] in ["NEW", "UPDATE"] and "content" not in op:
            logger.warning(f"Missing content for {op['operation']} operation: {op}")
            return False

        # Tags are optional but should be a list if present
        if "tags" in op and not isinstance(op["tags"], list):
            # Try to fix if it's a string
            if isinstance(op["tags"], str):
                try:
                    # See if it's a JSON string
                    parsed_tags = json.loads(op["tags"])
                    if isinstance(parsed_tags, list):
                        op["tags"] = parsed_tags
                    else:
                        # If it parsed but isn't a list, handle that case
                        op["tags"] = [str(parsed_tags)]
                except json.JSONDecodeError:
                    # Split by comma if it looks like a comma-separated list
                    if "," in op["tags"]:
                        op["tags"] = [tag.strip() for tag in op["tags"].split(",")]
                    else:
                        # Just make it a single-item list
                        op["tags"] = [op["tags"]]
            else:
                logger.warning(
                    f"Invalid tags format, not a list or string: {op['tags']}"
                )
                op["tags"] = []  # Default to empty list

        # Validate memory_bank field
        provided_bank = None
        if "memory_bank" in op and isinstance(op["memory_bank"], str):
            raw_bank_value = op["memory_bank"]
            provided_bank = raw_bank_value.strip().capitalize() # Normalize

            # --- SIMPLIFIED VALIDATION LOGIC ---
            # Use the validated list directly from self.valves
            allowed_banks_list = self.valves.allowed_memory_banks

            if provided_bank in allowed_banks_list:
                 # Valid bank provided
                 op["memory_bank"] = provided_bank # Assign normalized valid bank
            else:
                 # Invalid bank provided
                logger.warning(
                    f"Invalid memory bank '{op['memory_bank']}' (normalized to '{provided_bank}'), not in allowed list {allowed_banks_list}. Using default '{self.valves.default_memory_bank}'"
                )
                op["memory_bank"] = self.valves.default_memory_bank
        else:
            # If memory_bank is missing or not a string, set default
            logger.debug(
                f"Memory bank missing or invalid type ({type(op.get('memory_bank'))}), using default '{self.valves.default_memory_bank}'"
            )
            op["memory_bank"] = self.valves.default_memory_bank

        # Validate confidence score
        if "confidence" in op:
            if isinstance(op["confidence"], (int, float)):
                if 0.0 <= op["confidence"] <= 1.0:
                    pass  # Valid confidence score
                else:
                    logger.warning(f"Invalid confidence score range: {op['confidence']}")
                    return False
            else:
                logger.warning(f"Invalid confidence score type: {type(op['confidence'])}")
                return False
        else:
            logger.warning("Missing confidence score")
            return False

        return True

    def _extract_and_parse_json(self, text: str) -> Union[List, Dict, None]:
        """Extracts JSON object or array from text, trying various methods."""
        if not text:
            return None

        # --- NEW: Log raw LLM response ---
        logger.debug(f"Raw LLM response content received:\\n>>>\\n{text}\\n<<<")
        # --- END NEW ---

        # 1. Attempt direct parsing (most common case)
        try:
            parsed = json.loads(text)
            logger.debug("Successfully parsed JSON directly after pre-processing.")
            # ---- NEW: unwrap single-key object -> list automatically ----
            if isinstance(parsed, dict) and len(parsed) == 1:
                sole_value = next(iter(parsed.values()))
                if isinstance(sole_value, list):
                    logger.debug("Unwrapped single-key object returned by LLM into list of operations.")
                    parsed = sole_value
            # ------------------------------------------------------------
            if parsed == {} or parsed == []:
                logger.info("LLM returned empty object/array, treating as empty memory list")
                return []
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parsing failed after pre-processing: {e}")
            # Continue to more specific extraction attempts if direct parsing fails


        # --- Stage 3: Specific Pattern Extraction (If direct parsing failed) ---

        # Try extracting from potential JSON code blocks (already handled by stripping, but as fallback)
        code_block_pattern = r"```(?:json)?\\s*(\\[[\\s\\S]*?\\]|\\{[\\s\\S]*?\\})\\s*```"
        matches = re.findall(code_block_pattern, text)
        if matches:
            logger.debug(f"Found {len(matches)} JSON code blocks (fallback check)")
            for i, match in enumerate(matches):
                try:
                    parsed = json.loads(match)
                    logger.debug(f"Successfully parsed JSON from code block {i+1} (fallback)")
                    if parsed == {} or parsed == []: continue
                    return parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from code block {i+1} (fallback): {e}")

        # Try finding JSON directly (more refined patterns)
        # Prioritize array of objects, then single object, then empty array
        direct_json_patterns = [
            r"(\\s*\\{\\s*\"operation\":.*?\\}\\s*,?)+", # Matches one or more operation objects
            r"\\[\\s*\\{\\s*\"operation\":.*?\\}\\s*\\]", # Full array of objects
            r"\\{\\s*\"operation\":.*?\\}", # Single operation object
            r"\\[\\s*\\]", # Empty array explicitly
        ]
        for pattern in direct_json_patterns:
             # Find the *first* potential match
             match = re.search(pattern, text)
             if match:
                 potential_json_str = match.group(0)
                 # If the pattern is for multiple objects, wrap in brackets if needed
                 if pattern == r"(\\s*\\{\\s*\"operation\":.*?\\}\\s*,?)+" and not potential_json_str.startswith('['):
                      # Remove trailing comma if present and wrap in brackets
                     potential_json_str = f"[{potential_json_str.strip().rstrip(',')}]"

                 logger.debug(f"Found potential direct JSON match with pattern: {pattern}")
                 try:
                     parsed = json.loads(potential_json_str)
                     logger.debug(f"Successfully parsed direct JSON match: {potential_json_str[:100]}...")
                     if parsed == {} or parsed == []:
                         logger.info("Parsed direct JSON match resulted in empty object/array.")
                         return [] # Explicit empty is valid
                     return parsed
                 except json.JSONDecodeError as e:
                     logger.warning(f"Failed to parse direct JSON match: {e}")
                     # Continue searching with other patterns


        # Handle Ollama's quoted JSON format
        if text.startswith('"') and text.endswith('"'):
            try:
                unescaped = json.loads(text) # Interpret as a JSON string
                if isinstance(unescaped, str):
                    try:
                        parsed = json.loads(unescaped) # Parse the content
                        logger.debug("Successfully parsed quoted JSON from Ollama")
                        if parsed == {} or parsed == []: return []
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse unescaped quoted JSON: {e}")
            except json.JSONDecodeError: pass # Not a valid JSON string

        # --- Stage 4: Final Checks and Failure ---

        # Check for explicit empty array token after all attempts
        if "[]" in text.replace(" ", ""):
            logger.info("Detected '[]' token in LLM response after exhaustive parsing. Treating as empty list.")
            return []

        # If all attempts failed
        self.error_counters["json_parse_errors"] += 1
        # Point 8: Record timestamp for guard mechanism
        self.error_timestamps["json_parse_errors"].append(time.time())

        self._error_message = "json_parse_error"
        logger.error("Failed to extract valid JSON from LLM response after all attempts.")
        logger.debug(f"Full text that failed JSON parsing: {text}") # Log full text on final failure
        return None

    def _calculate_memory_similarity(self, memory1: str, memory2: str) -> float:
        """
        Calculate similarity between two memory contents using a more robust method.
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        """
        if not memory1 or not memory2:
            return 0.0

        # Clean the memories - remove tags and normalize
        memory1_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory2).lower().strip()

        # Handle exact matches quickly
        if memory1_clean == memory2_clean:
            return 1.0

        # Handle near-duplicates with same meaning but minor differences
        # Split into words and compare overlap
        words1 = set(re.findall(r"\b\w+\b", memory1_clean))
        words2 = set(re.findall(r"\b\w+\b", memory2_clean))

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity for word overlap
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0

        # Use sequence matcher for more precise comparison
        seq_similarity = SequenceMatcher(None, memory1_clean, memory2_clean).ratio()

        # Combine both metrics, weighting sequence similarity higher
        combined_similarity = (0.4 * jaccard) + (0.6 * seq_similarity)

        return combined_similarity
        
    async def _calculate_embedding_similarity(self, memory1: str, memory2: str) -> float:
        """
        Calculate semantic similarity between two memory contents using embeddings.
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        
        This method uses the configured embedding provider (local or API)
        and calculates cosine similarity for more accurate semantic matching.
        """
        if not memory1 or not memory2:
            return 0.0
            
        # Clean the memories - remove tags and normalize
        memory1_clean = re.sub(r"\[Tags:.*?\]\\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\\s*", "", memory2).lower().strip()
        
        # Handle exact matches quickly
        if memory1_clean == memory2_clean:
            return 1.0
            
        try:
            # Get embeddings using the main dispatcher function
            mem1_embedding = await self._get_embedding(memory1_clean)
            mem2_embedding = await self._get_embedding(memory2_clean)
            
            # Check if embeddings were successfully generated
            if mem1_embedding is None or mem2_embedding is None:
                logger.warning("Could not generate embeddings for similarity calculation. Falling back to text-based similarity.")
                # Fallback to text-based on failure
                return self._calculate_memory_similarity(memory1, memory2)

            # Check for dimension mismatch before dot product
            if mem1_embedding.shape[0] != mem2_embedding.shape[0]:
                logger.warning(
                    f"Dimension mismatch in embedding similarity: mem1={mem1_embedding.shape[0]}, "
                    f"mem2={mem2_embedding.shape[0]}. Falling back to text similarity."
                )
                return self._calculate_memory_similarity(memory1, memory2)

            # Calculate cosine similarity (dot product of normalized vectors)
            # _get_embedding should return normalized vectors
            similarity = float(np.dot(mem1_embedding, mem2_embedding))
            
            # Clamp similarity to [0, 1] just in case of float precision issues
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}\n{traceback.format_exc()}")
            # Fall back to text-based similarity on unexpected error
            logger.info("Falling back to text-based similarity due to unexpected error.")
            return self._calculate_memory_similarity(memory1, memory2)

    async def get_relevant_memories(
        self, current_message: str, user_id: str, user_timezone: str = None
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to the current context"""
        # --- RELOAD VALVES --- REMOVED
        # Ensure we have the latest config potentially injected by OWUI
        # try:
        #     logger.debug("Reloading self.valves at start of get_relevant_memories")
        #     self.valves = self.Valves(**getattr(self, "config", {}).get("valves", {}))
        # except Exception as e:
        #      logger.error(f"Error reloading valves in get_relevant_memories: {e}")
        # --- END RELOAD --- REMOVED

        import time

        # Metrics instrumentation
        RETRIEVAL_REQUESTS.inc()
        _retrieval_start = time.perf_counter()
        start = _retrieval_start
        try:
            # Get all memories for the user
            existing_memories = await self._get_formatted_memories(user_id)

            if not existing_memories:
                logger.debug("No existing memories found for relevance assessment")
                return []

            # --- Local vector similarity filtering ---
            vector_similarities = []
            user_embedding = None # Initialize to handle potential errors
            user_embedding_dim = None # NEW: Store dimension
            try:
                # Obtain embedding using unified dispatcher (local or API)
                user_embedding = await self._get_embedding(current_message)

                # Bail out early if embedding failed and LLM relevance is disabled
                if user_embedding is None and not self.valves.use_llm_for_relevance:
                    logger.warning("Cannot calculate relevance — failed to generate embedding and LLM relevance is disabled.")
                    return []  # Cannot proceed without either method
                elif user_embedding is not None: # NEW: Get dimension if successful
                    user_embedding_dim = user_embedding.shape[0]
                    logger.debug(f"User message embedding dimension: {user_embedding_dim}")

            except Exception as e:
                self.error_counters["embedding_errors"] += 1
                logger.error(
                    f"Error computing embedding for user message: {e}\n{traceback.format_exc()}"
                )
                # Decide fallback based on config
                if not self.valves.use_llm_for_relevance:
                    logger.warning("Cannot calculate relevance due to embedding error and no LLM fallback.")
                    return [] # Cannot proceed

            if user_embedding is not None:
                # Calculate vector similarities only if user embedding was successful
                for mem in existing_memories:
                    mem_id = mem.get("id")
                    # Check RAM cache first
                    mem_emb = self.memory_embeddings.get(mem_id)
                    
                    # If not in RAM, try loading from database (NEW: eliminates re-embedding on restart)
                    if mem_emb is None:
                        mem_emb = await self._load_embedding_from_db(user_id, mem_id)
                        if mem_emb is not None:
                            # Cache in RAM for this session
                            self.memory_embeddings[mem_id] = mem_emb
                            logger.debug(f"Loaded embedding for memory {mem_id} from database")
                    
                    # Only re-embed if still not found (legacy memories without stored embeddings)
                    if mem_emb is None:
                        try:
                            mem_text = mem.get("memory") or ""
                            if mem_text:
                                mem_emb = await self._get_embedding(mem_text)
                                if mem_emb is not None:
                                    # Cache for future similarity checks
                                    self.memory_embeddings[mem_id] = mem_emb
                                    # Store for future persistence
                                    await self._store_embedding_in_db(user_id, mem_id, mem_text, mem_emb)
                                    logger.info(f"Generated and stored embedding for legacy memory {mem_id}")
                        except Exception as e:
                            logger.warning(
                                f"Error computing embedding for memory {mem_id}: {e}"
                            )

                    if mem_emb is not None:
                        try:
                            # --- NEW: Dimension Check with Re-embedding ---
                            if user_embedding_dim is not None and mem_emb.shape[0] != user_embedding_dim:
                                logger.warning(f"Dimension mismatch for memory {mem_id}: {mem_emb.shape[0]} vs user {user_embedding_dim}")

                                # Attempt to re-embed with current configured provider
                                mem_text = mem.get("memory", "")
                                if mem_text:
                                    new_emb = await self._re_embed_memory_if_needed(
                                        mem_id, mem_text, mem_emb.shape[0], user_embedding_dim
                                    )
                                    if new_emb is not None:
                                        mem_emb = new_emb  # Use re-embedded version
                                    else:
                                        logger.warning(f"Re-embedding failed for memory {mem_id}, skipping")
                                        continue # Skip this memory if re-embedding failed
                                else:
                                    logger.warning(f"Cannot re-embed memory {mem_id}: no memory text available")
                                    continue # Skip this memory

                            # Cosine similarity (embeddings are normalized in _get_embedding)
                            sim = float(np.dot(user_embedding, mem_emb))
                            vector_similarities.append((sim, mem))
                        except Exception as e:
                            logger.warning(
                                f"Error calculating similarity for memory {mem_id}: {e}"
                            )
                            continue  # Skip this memory if calculation fails
                        else:
                            logger.debug(
                                f"No embedding available for memory {mem_id} even after attempted computation."
                            )
                    else:
                        logger.debug(
                            f"No embedding available for memory {mem_id} even after attempted computation."
                        )

                # Sort by similarity descending
                vector_similarities.sort(reverse=True, key=lambda x: x[0])

                # Debug thresholds used
                logger.debug(f"DEBUG THRESHOLDS: vector_similarity_threshold={self.valves.vector_similarity_threshold}, relevance_threshold={self.valves.relevance_threshold}")

                # Filter by threshold
                sim_threshold = self.valves.vector_similarity_threshold
                top_n = self.valves.top_n_memories # Note: This top_n is applied BEFORE deciding on LLM/Vector scoring.
                filtered_by_vector = [mem for sim, mem in vector_similarities if sim >= sim_threshold][:top_n]
                logger.info(
                    f"Vector filter selected {len(filtered_by_vector)} of {len(existing_memories)} memories (Threshold: {sim_threshold}, Top N: {top_n})"
                )
            else:
                 # If user_embedding failed and LLM fallback is disabled, we already returned.
                 # If LLM fallback is enabled, proceed with all existing memories for LLM relevance check.
                 logger.warning("User embedding failed, proceeding with all memories for potential LLM check.")
                 filtered_by_vector = existing_memories # Pass all memories to LLM check if enabled


            # --- Decide Relevance Method ---
            if not self.valves.use_llm_for_relevance:
                # --- Use Vector Similarity Scores Directly ---
                logger.info("Using vector similarity directly for relevance scoring (LLM call skipped).")
                relevant_memories = []
                final_relevance_threshold = self.valves.relevance_threshold  # Use configured relevance threshold for vector-only filtering.

                # Use the already calculated and sorted vector similarities
                for sim_score, mem in vector_similarities: # Iterate through the originally sorted list
                    if sim_score >= final_relevance_threshold:
                         # Check if this memory was part of the top_n initially filtered by vector
                         # This ensures we respect the vector_similarity_threshold AND top_n_memories filter first
                         if any(filtered_mem['id'] == mem['id'] for filtered_mem in filtered_by_vector):
                            relevant_memories.append(
                                {"id": mem["id"], "memory": mem["memory"], "relevance": sim_score} # Use vector score as relevance
                            )

                # Sort again just to be sure (though vector_similarities was already sorted)
                relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

                # Limit to configured number
                final_top_n = self.valves.related_memories_n
                logger.info(
                    f"Found {len(relevant_memories)} relevant memories using vector similarity >= {final_relevance_threshold}"
                )
                duration = time.perf_counter() - _retrieval_start
                RETRIEVAL_LATENCY.observe(duration)
                logger.info(f"Memory retrieval (vector only) took {duration:.2f}s")
                return relevant_memories[:final_top_n]

            else:
                # --- Use LLM for Relevance Scoring (Optimised) ---
                logger.info("Proceeding with LLM call for relevance scoring.")

                # Optimisation: If the vector similarities for *all* candidate memories are above
                # `llm_skip_relevance_threshold`, we consider the vector score sufficiently
                # confident and *skip* the LLM call (Improvement #5).
                confident_threshold = self.valves.llm_skip_relevance_threshold

                # Build helper map id -> vector similarity for quick lookup
                id_to_vec_score = {mem['id']: sim for sim, mem in vector_similarities}

                if filtered_by_vector and all(
                    id_to_vec_score.get(mem['id'], 0.0) >= confident_threshold
                    for mem in filtered_by_vector
                ):
                    logger.info(
                        f"All {len(filtered_by_vector)} memories exceed confident vector threshold ({confident_threshold}). Skipping LLM relevance call."
                    )

                    relevant_memories = [
                        {
                            "id": mem["id"],
                            "memory": mem["memory"],
                            "relevance": id_to_vec_score.get(mem["id"], 0.0),
                        }
                        for mem in filtered_by_vector
                    ]
                    # Ensure sorted by relevance desc
                    relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
                    return relevant_memories[: self.valves.related_memories_n]

                # If not confident, fall back to existing LLM relevance path
                memories_for_llm = filtered_by_vector # Use the vector-filtered list

                if not memories_for_llm:
                     logger.debug("No memories passed vector filter for LLM relevance check.")
                     return []

                # Build the prompt for LLM
                memory_strings = []
                for mem in memories_for_llm:
                    memory_strings.append(f"ID: {mem['id']}, CONTENT: {mem['memory']}")

                system_prompt = self.valves.memory_relevance_prompt
                user_prompt = f"""Current user message: "{current_message}"

Available memories (pre-filtered by vector similarity):
{json.dumps(memory_strings)}

Rate the relevance of EACH memory to the current user message based *only* on the provided content and message context."""

                # Add current datetime for context
                current_datetime = self.get_formatted_datetime(user_timezone)
                user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""

                # Check cache or call LLM for relevance score
                import time as time_module

                now = time_module.time()
                ttl_seconds = self.valves.cache_ttl_seconds

                relevance_data = []
                uncached_memories = [] # Memories needing LLM call
                uncached_ids = set() # Track IDs needing LLM call

                # Check cache first
                if user_embedding is not None: # Can only use cache if we have user embedding
                    for mem in memories_for_llm:
                        mem_id = mem.get("id")
                        mem_emb = self.memory_embeddings.get(mem_id)
                        if mem_emb is None:
                             # If memory embedding is missing, cannot use cache, must call LLM
                             if mem_id not in uncached_ids:
                                 uncached_memories.append(mem)
                                 uncached_ids.add(mem_id)
                             continue

                        key = hash((user_embedding.tobytes(), mem_emb.tobytes()))
                        cached = self.relevance_cache.get(key)
                        if cached:
                            score, ts = cached
                            if now - ts < ttl_seconds:
                                logger.info(f"Cache hit for memory {mem_id} (LLM relevance)")
                                relevance_data.append(
                                    {"memory": mem["memory"], "id": mem_id, "relevance": score}
                                )
                                continue  # use cached score

                        # Cache miss or expired, add to uncached list if not already there
                        if mem_id not in uncached_ids:
                             uncached_memories.append(mem)
                             uncached_ids.add(mem_id)
                else:
                     # No user embedding, cannot use cache, all need LLM call
                     logger.warning("Cannot use relevance cache as user embedding failed.")
                     uncached_memories = memories_for_llm # Send all vector-filtered memories to LLM


                # If any uncached memories, call LLM
                if uncached_memories:
                    logger.info(f"Calling LLM for relevance on {len(uncached_memories)} uncached memories.")
                    # Build prompt with only uncached memories
                    uncached_memory_strings = [
                        f"ID: {mem['id']}, CONTENT: {mem['memory']}"
                        for mem in uncached_memories
                    ]
                    # Reuse system_prompt, construct user_prompt specifically for uncached items
                    uncached_user_prompt = f"""Current user message: "{current_message}"

Available memories (evaluate relevance for these specific IDs):
{json.dumps(uncached_memory_strings)}

Rate the relevance of EACH listed memory to the current user message based *only* on the provided content and message context."""
                    current_datetime = self.get_formatted_datetime(user_timezone)
                    uncached_user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""

                    llm_response_text = await self.query_llm_with_retry(
                        system_prompt, uncached_user_prompt # Use the specific uncached prompt
                    )

                    if not llm_response_text or llm_response_text.startswith("Error:"):
                        if llm_response_text:
                            logger.error(
                                f"Error from LLM during memory relevance: {llm_response_text}"
                            )
                        # If LLM fails, we might return empty or potentially fall back
                        # For now, return empty to indicate failure
                        return []

                    # Parse the LLM response for the uncached items
                    llm_relevance_results = self._extract_and_parse_json(
                        llm_response_text
                    )

                    if not llm_relevance_results or not isinstance(
                        llm_relevance_results, list
                    ):
                        logger.warning("Failed to parse relevance data from LLM response for uncached items.")
                        # Decide how to handle partial failure - return only cached? or empty?
                        # Returning only cached for now
                    else:
                         # Process successful LLM results
                         for item in llm_relevance_results:
                            mem_id = item.get("id")
                            score = item.get("relevance")
                            mem_text = item.get("memory") # Use memory text from LLM response if available
                            if mem_id and isinstance(score, (int, float)):
                                relevance_data.append(
                                    {"memory": mem_text or f"Content for {mem_id}", # Fallback if memory text missing
                                     "id": mem_id,
                                     "relevance": score}
                                )
                                # Save to cache if possible
                                if user_embedding is not None:
                                    mem_emb = self.memory_embeddings.get(mem_id)
                                    if mem_emb is not None:
                                        key = hash((user_embedding.tobytes(), mem_emb.tobytes()))
                                        self.relevance_cache[key] = (score, now)
                                    else:
                                         logger.debug(f"Cannot cache relevance for {mem_id}, embedding missing.")
                            else:
                                logger.warning(f"Invalid item format in LLM relevance response: {item}")


                # Combine cached and newly fetched results, filter by relevance threshold
                final_relevant_memories = []
                final_relevance_threshold = self.valves.relevance_threshold  # Use configured relevance threshold for LLM-score filtering.

                seen_ids = set() # Ensure unique IDs in final list
                for item in relevance_data:
                    if not isinstance(item, dict): continue # Skip invalid entries

                    memory_content = item.get("memory")
                    relevance_score = item.get("relevance")
                    mem_id = item.get("id")

                    if memory_content and isinstance(relevance_score, (int, float)) and mem_id:
                        # Use the final_relevance_threshold determined earlier (should be self.valves.relevance_threshold)
                        if relevance_score >= final_relevance_threshold and mem_id not in seen_ids:
                            final_relevant_memories.append(
                                {"id": mem_id, "memory": memory_content, "relevance": relevance_score}
                            )
                            seen_ids.add(mem_id)

                # Sort final list by relevance (descending)
                final_relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

                # Limit to configured number
                final_top_n = self.valves.related_memories_n
                logger.info(
                    f"Found {len(final_relevant_memories)} relevant memories using LLM score >= {final_relevance_threshold}"
                )
                logger.info(f"Memory retrieval (LLM scoring) took {time.perf_counter() - start:.2f}s")
                return final_relevant_memories[:final_top_n]

        except Exception as e:
            logger.error(
                f"Error getting relevant memories: {e}\n{traceback.format_exc()}"
            )
            return []

    async def process_memories(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> List[Dict[str, Any]]:  # Return list of successfully processed operations
        """Process memory operations"""
        successfully_saved_ops = []
        try:
            user = Users.get_user_by_id(user_id)
            if not user:
                logger.error(f"User not found: {user_id}")
                return []

            # Get existing memories for deduplication
            existing_memories = []
            if self.valves.deduplicate_memories:
                existing_memories = await self._get_formatted_memories(user_id)

            logger.debug(f"Processing {len(memories)} memory operations")

            # First filter for duplicates if enabled
            processed_memories = []
            if self.valves.deduplicate_memories and existing_memories:
                # Store all existing contents for quick lookup
                existing_contents = []
                for mem in existing_memories:
                    existing_contents.append(mem["memory"])

                logger.debug(f"[DEDUPE] Existing memories being checked against: {existing_contents}")

                # Decide similarity method and corresponding threshold
                use_embeddings = self.valves.use_embeddings_for_deduplication
                threshold_to_use = (
                    self.valves.embedding_similarity_threshold
                    if use_embeddings
                    else self.valves.similarity_threshold
                )
                logger.debug(
                    f"Using {'embedding-based' if use_embeddings else 'text-based'} similarity for deduplication. "
                    f"Threshold: {threshold_to_use}"
                )

                # Check each new memory against existing ones
                for new_memory_idx, memory_dict in enumerate(memories):
                    if memory_dict["operation"] == "NEW":
                        logger.debug(f"[DEDUPE CHECK {new_memory_idx+1}/{len(memories)}] Processing NEW memory: {memory_dict}") # LOG START
                        # Format the memory content
                        operation = MemoryOperation(**memory_dict)
                        formatted_content = self._format_memory_content(operation)

                        # --- BYPASS: Skip dedup for short preference statements ---
                        if (
                            self.valves.enable_short_preference_shortcut
                            and len(formatted_content) <= self.valves.short_preference_no_dedupe_length
                        ):
                            pref_kwds = [kw.strip() for kw in self.valves.preference_keywords_no_dedupe.split(',') if kw.strip()]
                            if any(kw in formatted_content.lower() for kw in pref_kwds):
                                logger.debug("Bypassing deduplication for short preference statement: '%s'", formatted_content)
                                processed_memories.append(memory_dict)
                                continue  # Skip duplicate checking entirely for this memory

                        is_duplicate = False
                        similarity_score = 0.0 # Track similarity score for logging
                        similarity_method = 'none' # Track method used

                        if use_embeddings:
                            # Precompute embedding for the new memory once
                            try:
                                new_embedding = await self._get_embedding(
                                    formatted_content.lower().strip()
                                )
                                if new_embedding is None:
                                    raise ValueError("Failed to generate embedding")
                            except Exception as e:
                                logger.warning(f"Failed to encode new memory for deduplication; falling back to text sim. Error: {e}")
                                use_embeddings = False  # fall back

                        for existing_idx, existing_content in enumerate(existing_contents):
                            if use_embeddings:
                                # Retrieve or compute embedding for the existing memory content
                                existing_mem_dict = existing_memories[existing_idx]
                                existing_id = existing_mem_dict.get("id")
                                existing_emb = self.memory_embeddings.get(existing_id)
                                if existing_emb is None:
                                    try:
                                        existing_emb = await self._get_embedding(
                                            existing_content.lower().strip()
                                        )
                                        if existing_emb is not None:
                                            self.memory_embeddings[existing_id] = existing_emb
                                    except Exception:
                                        # On failure, mark duplicate check using text sim for this item
                                        existing_emb = None
                                if existing_emb is not None:
                                    # Check for dimension mismatch before dot product
                                    if new_embedding.shape[0] != existing_emb.shape[0]:
                                        logger.warning(
                                            f"Dimension mismatch during deduplication: new={new_embedding.shape[0]}, "
                                            f"existing={existing_emb.shape[0]}. Attempting to re-embed existing memory."
                                        )

                                        # Try to re-embed the existing memory with current provider
                                        new_existing_emb = await self._re_embed_memory_if_needed(
                                            existing_id, existing_content, existing_emb.shape[0], new_embedding.shape[0]
                                        )

                                        if new_existing_emb is not None:
                                            # Successfully re-embedded, use embedding similarity
                                            existing_emb = new_existing_emb
                                            similarity = float(np.dot(new_embedding, existing_emb))
                                            similarity_score = similarity
                                            similarity_method = 'embedding'
                                        else:
                                            # Re-embedding failed, fall back to text similarity
                                            logger.warning(f"Re-embedding failed for existing memory {existing_id}, using text similarity")
                                            similarity = self._calculate_memory_similarity(
                                                formatted_content, existing_content
                                            )
                                            similarity_score = similarity
                                            similarity_method = 'text_fallback'
                                    else:
                                        similarity = float(np.dot(new_embedding, existing_emb))
                                        similarity_score = similarity # Store score
                                        similarity_method = 'embedding'
                                else:
                                    similarity = self._calculate_memory_similarity(
                                        formatted_content, existing_content
                                    )
                                    similarity_score = similarity # Store score
                                    similarity_method = 'text'
                            else:
                                # Choose the appropriate similarity calculation method
                                similarity = self._calculate_memory_similarity(
                                    formatted_content, existing_content
                                )
                                
                            if similarity >= threshold_to_use:
                                logger.debug(
                                    f"  -> Duplicate found vs existing mem {existing_idx} (Similarity: {similarity_score:.3f}, Method: {similarity_method}, Threshold: {threshold_to_use})"
                                )
                                logger.debug(
                                    f"Skipping duplicate NEW memory (similarity: {similarity_score:.2f}, method: {similarity_method}): {formatted_content[:50]}..."
                                )
                                is_duplicate = True
                                # Increment duplicate skipped counter for status reporting
                                self._duplicate_skipped += 1
                                break # Stop checking against other existing memories for this new one

                        if not is_duplicate:
                            logger.debug(f"  -> No duplicate found. Adding to processed list: {formatted_content[:50]}...")
                            processed_memories.append(memory_dict)
                        else:
                             logger.debug(f"NEW memory was identified as duplicate and skipped: {formatted_content[:50]}...")
                    else:
                        # Keep all UPDATE and DELETE operations
                        logger.debug(f"Keeping non-NEW operation: {memory_dict['operation']} ID: {memory_dict.get('id', 'N/A')}")
                        processed_memories.append(memory_dict)
            else:
                logger.debug("Deduplication skipped (valve disabled or no existing memories). Processing all operations.")
                processed_memories = memories

            # Process the filtered memories
            logger.debug(f"Executing {len(processed_memories)} filtered memory operations.")
            for idx, memory_dict in enumerate(processed_memories):
                logger.debug(f"Executing operation {idx + 1}/{len(processed_memories)}: {memory_dict}")
                try:
                    # Validate memory operation
                    operation = MemoryOperation(**memory_dict)
                    # Execute the memory operation
                    await self._execute_memory_operation(operation, user)
                    # If successful, add to our list
                    logger.debug(f"Successfully executed operation: {operation.operation} ID: {operation.id}")
                    successfully_saved_ops.append(memory_dict)
                except ValueError as e:
                    logger.error(f"Invalid memory operation during execution phase: {e} {memory_dict}")
                    self.error_counters["memory_crud_errors"] += 1 # Increment error counter
                    continue
                except Exception as e:
                    logger.error(f"Error executing memory operation in process_memories: {e} {memory_dict}")
                    self.error_counters["memory_crud_errors"] += 1 # Increment error counter
                    continue

            logger.debug(
                f"Successfully executed {len(successfully_saved_ops)} memory operations out of {len(processed_memories)} processed.")
            # Add confirmation message if any memory was added or updated
            if successfully_saved_ops:
                # Check if any operation was NEW or UPDATE
                if any(op.get("operation") in ["NEW", "UPDATE"] for op in successfully_saved_ops):
                    logger.debug("Attempting to add confirmation message.") # Log confirmation attempt
                    try:
                        from fastapi.requests import Request  # ensure import

                        # Find the last assistant message and append confirmation
                        # This is a safe operation, no error if no assistant message
                        for i in reversed(range(len(self._last_body.get("messages", [])))):
                            msg = self._last_body["messages"][i]
                            if msg.get("role") == "assistant":
                                # Do nothing here
                                break
                    except Exception:
                        pass
            return successfully_saved_ops
        except Exception as e:
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            return []  # Return empty list on major error

    async def _execute_memory_operation(
        self, operation: MemoryOperation, user: Any
    ) -> None:
        """Execute a memory operation (NEW, UPDATE, DELETE)"""
        formatted_content = self._format_memory_content(operation)

        if operation.operation == "NEW":
            try:
                result = await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}), # Add missing request object
                    user=user, # Pass the full user object
                    form_data=AddMemoryForm(
                        content=formatted_content,
                        metadata={
                            "tags": operation.tags,
                            "memory_bank": operation.memory_bank or self.valves.default_memory_bank,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source": "adaptive_memory_v3",
                        },
                    ),
                )
                logger.info(f"NEW memory created: {formatted_content[:50]}...")

                # Generate and cache embedding for new memory if embedding model is available
                # This helps with future deduplication checks when using embedding-based similarity
                # Handle both Pydantic model and dict response forms
                mem_id = getattr(result, "id", None)
                if mem_id is None and isinstance(result, dict):
                    mem_id = result.get("id")
                if mem_id is not None:
                    try:
                        memory_clean = re.sub(r"\[Tags:.*?\]\s*", "", formatted_content).lower().strip()
                        memory_embedding = await self._get_embedding(memory_clean)
                        if memory_embedding is not None:
                            # Cache in RAM for immediate use
                            self.memory_embeddings[mem_id] = memory_embedding
                            
                            # Store persistently in vector database
                            await self._store_embedding_in_db(user.id, mem_id, memory_clean, memory_embedding)
                            logger.debug(f"Generated, cached, and stored embedding for new memory ID: {mem_id}")
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for new memory: {e}")
                        # Non-critical error, don't raise

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error creating memory (operation=NEW, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

        elif operation.operation == "UPDATE" and operation.id:
            try:
                # Delete existing memory
                deleted = await delete_memory_by_id(operation.id, user=user)
                if deleted:
                    # Create new memory with updated content
                    result = await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=formatted_content),
                        user=user,
                    )
                    logger.info(
                        f"UPDATE memory {operation.id}: {formatted_content[:50]}..."
                    )

                    # Update embedding for modified memory
                    # Handle both Pydantic model and dict response forms
                    new_mem_id = getattr(result, "id", None)
                    if new_mem_id is None and isinstance(result, dict):
                        new_mem_id = result.get("id")

                    if new_mem_id is not None:
                        try:
                            memory_clean = re.sub(r"\[Tags:.*?\]\s*", "", formatted_content).lower().strip()
                            memory_embedding = await self._get_embedding(memory_clean)
                            if memory_embedding is not None:
                                # Cache in RAM
                                self.memory_embeddings[new_mem_id] = memory_embedding
                                
                                # Store persistently in vector database
                                await self._store_embedding_in_db(user.id, new_mem_id, memory_clean, memory_embedding)
                                logger.debug(
                                    f"Updated and stored embedding for memory ID: {new_mem_id} (was: {operation.id})"
                                )

                                # Remove old embedding if ID changed
                                if operation.id != new_mem_id and operation.id in self.memory_embeddings:
                                    del self.memory_embeddings[operation.id]
                        except Exception as e:
                            logger.warning(
                                f"Failed to update embedding for memory ID {new_mem_id}: {e}"
                            )
                            # Non-critical error, don't raise

                else:
                    logger.warning(f"Memory {operation.id} not found for UPDATE")
            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error updating memory (operation=UPDATE, memory_id={operation.id}, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

            # Invalidate cache entries involving this memory
            mem_emb = self.memory_embeddings.get(operation.id)
            if mem_emb is not None:
                keys_to_delete = []
                for key, (score, ts) in self.relevance_cache.items():
                    # key is hash of (user_emb, mem_emb)
                    # We can't extract mem_emb from key, so approximate by deleting all keys with this mem_emb
                    # Since we can't reverse hash, we skip this for now
                    # Future: store reverse index or use tuple keys
                    pass  # Placeholder for future precise invalidation

        elif operation.operation == "DELETE" and operation.id:
            try:
                deleted = await delete_memory_by_id(operation.id, user=user)
                logger.info(f"DELETE memory {operation.id}: {deleted}")

                # Invalidate cache entries involving this memory
                mem_emb = self.memory_embeddings.get(operation.id)
                if mem_emb is not None:
                    keys_to_delete = []
                    for key, (score, ts) in self.relevance_cache.items():
                        # Same as above, placeholder
                        pass

                # Remove embedding
                if operation.id in self.memory_embeddings:
                    del self.memory_embeddings[operation.id]
                    logger.debug(f"Removed embedding for deleted memory ID: {operation.id}")

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error deleting memory (operation=DELETE, memory_id={operation.id}, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

    def _format_memory_content(self, operation: MemoryOperation) -> str:
        """Format memory content with tags, memory bank, and confidence for saving / display"""
        content = operation.content or ""
        tag_part = f"[Tags: {', '.join(operation.tags)}] " if operation.tags else ""
        bank_part = f" [Memory Bank: {operation.memory_bank or self.valves.default_memory_bank}]"
        # Format confidence score, handling None case
        confidence_score = operation.confidence if operation.confidence is not None else 0.0 # Default to 0.0 if None for formatting
        confidence_part = f" [Confidence: {confidence_score:.2f}]" # Format to 2 decimal places
        return f"{tag_part}{content}{bank_part}{confidence_part}".strip()

    async def query_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Query LLM with retry logic, supporting multiple provider types.

        Args:
            system_prompt: System prompt for context/instructions
            user_prompt: User prompt/query

        Returns:
            String response from LLM or error message
        """
        # Get configuration from valves
        provider_type = self.valves.llm_provider_type 
        model = self.valves.llm_model_name
        api_url = self.valves.llm_api_endpoint_url
        api_key = self.valves.llm_api_key
        max_retries = self.valves.max_retries
        retry_delay = self.valves.retry_delay

        logger.info(
            f"LLM Query: Provider={provider_type}, Model={model}, URL={api_url}"
        )
        logger.debug(
            f"System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}"
        )

        # ---- Improvement #5: Track LLM call frequency ----
        try:
            # Use dict to avoid attribute errors if metrics removed/reset elsewhere
            self.metrics["llm_call_count"] = self.metrics.get("llm_call_count", 0) + 1
        except Exception as metric_err:
            # Non-critical; log at DEBUG level to avoid clutter
            logger.debug(f"Unable to increment llm_call_count metric: {metric_err}")

        # Ensure we have a valid aiohttp session
        session = await self._get_aiohttp_session()

        # Add the current datetime to system prompt for time awareness
        system_prompt_with_date = system_prompt
        try:
            now = self.get_formatted_datetime()
            tzname = now.tzname() or "UTC"
            system_prompt_with_date = f"{system_prompt}\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} {tzname}"
        except Exception as e:
            logger.warning(f"Could not add date to system prompt: {e}")

        headers = {"Content-Type": "application/json"}

        # Add API key if provided (required for OpenAI-compatible APIs)
        if provider_type == "openai_compatible" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(
            1, max_retries + 2
        ):  # +2 because we start at 1 and want max_retries+1 attempts
            logger.debug(f"LLM query attempt {attempt}/{max_retries+1}")
            try:
                if provider_type == "ollama":
                    # Prepare the request body for Ollama
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt_with_date},
                            {"role": "user", "content": user_prompt},
                        ],
                        # Set some parameters to encourage consistent outputs
                        "options": {
                            "temperature": 0.1,  # Lower temperature for more deterministic responses
                            "top_p": 0.95,  # Slightly constrain token selection
                            "top_k": 80,  # Reasonable top_k value
                            "num_predict": 2048,  # Reasonable length limit
                            "format": "json",  # Request JSON format
                        },
                        # Disable streaming so we get a single JSON response; newer Ollama respects this flag.
                        "stream": False,
                    }
                    logger.debug(f"Ollama request data: {json.dumps(data)[:500]}...")
                elif provider_type == "openai_compatible":
                    # Prepare the request body for OpenAI-compatible API
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt_with_date},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": 1024,
                        "response_format": {"type": "json_object"},  # Force JSON mode
                        "seed": 42,
                        "stream": False,
                    }
                    logger.debug(
                        f"OpenAI-compatible request data: {json.dumps(data)[:500]}..."
                    )
                else:
                    error_msg = f"Unsupported provider type: {provider_type}"
                    logger.error(error_msg)
                    return error_msg

                # Log the API call attempt
                logger.info(
                    f"Making API request to {api_url} (attempt {attempt}/{max_retries+1})"
                )

                # Make the API call with timeout
                async with session.post(
                    api_url, json=data, headers=headers, timeout=60
                ) as response:
                    # Log the response status
                    logger.info(f"API response status: {response.status}")

                    if response.status == 200:
                        # Success - parse the response, handling both JSON and NDJSON
                        content_type = response.headers.get("content-type", "")
                        if "application/x-ndjson" in content_type:
                            # Ollama may still return NDJSON even with stream=False; aggregate lines
                            raw_text = await response.text()
                            logger.debug(
                                f"Received NDJSON response length: {len(raw_text)}"
                            )
                            last_json = None
                            for line in raw_text.strip().splitlines():
                                try:
                                    last_json = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                            if last_json is None:
                                error_msg = "Could not decode NDJSON response from LLM"
                                logger.error(error_msg)
                                if attempt > max_retries:
                                    return error_msg
                                else:
                                    continue
                            data = last_json
                        else:
                            # Regular JSON
                            data = await response.json()

                        # Extract content based on provider type
                        content = None

                        # Log the raw response for debugging
                        logger.debug(f"Raw API response: {json.dumps(data)[:500]}...")

                        if provider_type == "openai_compatible":
                            if (
                                data.get("choices")
                                and data["choices"][0].get("message")
                                and data["choices"][0]["message"].get("content")
                            ):
                                content = data["choices"][0]["message"]["content"]
                                logger.info(
                                    f"Retrieved content from OpenAI-compatible response (length: {len(content)})"
                                )
                        elif provider_type == "ollama":
                            if data.get("message") and data["message"].get("content"):
                                content = data["message"]["content"]
                                logger.info(
                                    f"Retrieved content from Ollama response (length: {len(content)})"
                                )

                        if content:
                            return content
                        else:
                            error_msg = f"Could not extract content from {provider_type} response format"
                            logger.error(f"{error_msg}: {data}")

                            # If we're on the last attempt, return the error message
                            if attempt > max_retries:
                                return error_msg
                    else:
                        # Handle error response
                        error_text = await response.text()
                        error_msg = f"Error: LLM API ({provider_type}) returned {response.status}: {error_text}"
                        logger.warning(f"API error: {error_msg}")

                        # Determine if we should retry based on status code
                        is_retryable = response.status in [429, 500, 502, 503, 504]

                        if is_retryable and attempt <= max_retries:
                            sleep_time = retry_delay * (
                                2 ** (attempt - 1)
                            ) + random.uniform(
                                0, 1.0
                            )  # Longer backoff for rate limits/server errors
                            logger.warning(f"Retrying in {sleep_time:.2f} seconds...")
                            await asyncio.sleep(sleep_time)
                            continue  # Retry
                        else:
                            return error_msg  # Final failure

            except asyncio.TimeoutError:
                logger.warning(f"Attempt {attempt} failed: LLM API request timed out")
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue  # Retry on timeout
                else:
                    return "Error: LLM API request timed out after multiple retries."
            except ClientError as e:
                logger.warning(
                    f"Attempt {attempt} failed: LLM API connection error: {str(e)}"
                )
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue  # Retry on connection error
                else:
                    # Return specific error code for connection failure
                    return f"Error: LLM_CONNECTION_FAILED after multiple retries: {str(e)}"
            except Exception as e:
                logger.error(
                    f"Attempt {attempt} failed: Unexpected error during LLM query: {e}\n{traceback.format_exc()}"
                )
                if attempt <= max_retries:
                    # Generic retry for unexpected errors
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    return f"Error: UNEXPECTED_LLM_ERROR after {max_retries} attempts: {str(e)}"

        return f"Error: LLM query failed after {max_retries} attempts."

    async def _add_confirmation_message(self, body: Dict[str, Any]) -> None:
        """Add a confirmation message about memory operations"""
        if (
            not body
            or "messages" not in body
            or not body["messages"]
            or not self.valves.show_status
        ):
            return

        # Prepare the confirmation message
        confirmation = ""

        if self._error_message:
            confirmation = f"(Memory error: {self._error_message})"
        elif self.stored_memories:
            # Count operations by type
            new_count = 0
            update_count = 0
            delete_count = 0

            for memory in self.stored_memories:
                if memory["operation"] == "NEW":
                    new_count += 1
                elif memory["operation"] == "UPDATE":
                    update_count += 1
                elif memory["operation"] == "DELETE":
                    delete_count += 1

            # Build the confirmation message in new styled format
            total_saved = new_count + update_count + delete_count

            # Use bold italic styling with an emoji as requested
            confirmation = f"**_Memory: 🧠 Saved {total_saved} memories..._**"

        # If no confirmation necessary, exit early
        if not confirmation:
            logger.debug("No memory confirmation message needed")
            return

        # Critical fix: Make a complete deep copy of the messages array
        try:
            logger.debug("Making deep copy of messages array for safe modification")
            messages_copy = copy.deepcopy(body["messages"])

            # Find the last assistant message
            last_assistant_idx = -1
            for i in range(len(messages_copy) - 1, -1, -1):
                if messages_copy[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break

            # If found, modify the copy
            if last_assistant_idx != -1:
                # Get the original content
                original_content = messages_copy[last_assistant_idx].get("content", "")

                # Append the confirmation message
                messages_copy[last_assistant_idx]["content"] = (
                    original_content + f" {confirmation}"
                )

                # Replace the entire messages array in body
                logger.debug(
                    f"Replacing messages array with modified copy containing confirmation: {confirmation}"
                )
                body["messages"] = messages_copy
            else:
                logger.debug("No assistant message found to append confirmation")

        except Exception as e:
            logger.error(f"Error adding confirmation message: {e}")
            # Don't modify anything if there's an error

    # Cleanup method for aiohttp session and background tasks
    async def cleanup(self):
        """Clean up resources when filter is being shut down"""
        logger.info("Cleaning up Adaptive Memory Filter")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done() and not task.cancelled():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Expected when cancelling
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling task: {e}")
        
        # Clear task tracking set
        self._background_tasks.clear()
        
        # Close any open sessions
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()

        # Clear shared class-level memory caches to help with GC
        Filter._memory_embeddings.clear()
        Filter._relevance_cache.clear()

        logger.info("Adaptive Memory Filter cleanup complete")

    def _convert_dict_to_memory_operations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert a dictionary returned by the LLM into the expected list of memory operations.

        Handles cases where the LLM returns a dict containing a list (e.g., {"memories": [...]})
        or a flatter structure. Avoids adding unnecessary prefixes.
        """
        if not isinstance(data, dict) or not data:
            return []

        operations: List[Dict[str, Any]] = []
        seen_content = set()

        # --- Primary Handling: Check for a key containing a list of operations ---
        # Common keys LLMs might use: "memories", "memory_operations", "results", "operations"
        list_keys = ["memories", "memory_operations", "results", "operations"]
        processed_primary = False
        for key in list_keys:
            if key in data and isinstance(data[key], list):
                logger.info(
                    f"Found list of operations under key '{key}', processing directly."
                )
                for item in data[key]:
                    if isinstance(item, dict):
                        # Extract fields directly, provide defaults
                        op = item.get("operation", "NEW").upper()  # Default to NEW
                        content = item.get(
                            "content", item.get("memory", item.get("value"))
                        )  # Check common content keys
                        tags = item.get("tags", [])
                        memory_bank = item.get("memory_bank", self.valves.default_memory_bank)
                        
                        # Validate memory_bank
                        if memory_bank not in self.valves.allowed_memory_banks:
                            memory_bank = self.valves.default_memory_bank

                        # Basic validation
                        if op not in ["NEW", "UPDATE", "DELETE"]:
                            continue
                        if (
                            not content
                            or not isinstance(content, str)
                            or len(content) < 5
                        ):
                            continue  # Skip empty/short content
                        if not isinstance(tags, list):
                            tags = [str(tags)]  # Ensure tags is a list

                        # Add if content is unique
                        if content not in seen_content:
                            operations.append(
                                {
                                    "operation": op, 
                                    "content": content, 
                                    "tags": tags,
                                    "memory_bank": memory_bank
                                }
                            )
                            seen_content.add(content)
                processed_primary = True
                break  # Stop after processing the first found list

        # --- Fallback Handling: If no primary list found, try simple key-value flattening ---
        if not processed_primary:
            logger.info(
                "No primary operations list found, attempting fallback key-value flattening."
            )
            # Helper maps for simple tag inference (less critical now)
            identity_keys = {"name", "username", "location", "city", "country", "age"}
            goal_keys = {"goal", "objective", "plan"}
            preference_keys = {
                "likes",
                "dislikes",
                "interests",
                "hobbies",
                "favorite",
                "preference",
            }
            relationship_keys = {"family", "friend", "brother", "sister"}
            ignore_keys = {"notes", "meta", "trivia"}

            # Bank inference based on key name
            work_keys = {"job", "profession", "career", "work", "office", "business", "project"}
            personal_keys = {"home", "family", "hobby", "personal", "like", "enjoy", "love", "hate", "friend"}

            for key, value in data.items():
                lowered_key = key.lower()
                if (
                    lowered_key in ignore_keys
                    or not isinstance(value, (str, int, float, bool))
                    or not str(value).strip()
                ):
                    continue

                content = str(value).strip()
                if len(content) > 5 and content not in seen_content:
                    # Simple tag inference
                    tag = "preference"  # Default tag
                    if lowered_key in identity_keys:
                        tag = "identity"
                    elif lowered_key in goal_keys:
                        tag = "goal"
                    elif lowered_key in relationship_keys:
                        tag = "relationship"
                    
                    # Simple bank inference
                    memory_bank = self.valves.default_memory_bank
                    if lowered_key in work_keys:
                        memory_bank = "Work"
                    elif lowered_key in personal_keys:
                        memory_bank = "Personal"

                    # Format simply: "Key: Value" unless key is generic
                    generic_keys = {"content", "memory", "text", "value", "result", "data"}
                    if key.lower() in generic_keys:
                        content_to_save = content # Use content directly
                    else:
                        # Prepend the key for non-generic keys
                        content_to_save = f"{key.replace('_', ' ').capitalize()}: {content}"

                    operations.append(
                        {
                            "operation": "NEW",
                            "content": content_to_save,
                            "tags": [tag],
                            "memory_bank": memory_bank,
                            "confidence": 0.5 # --- Assign default confidence --- NEW
                        }
                    )
                    seen_content.add(content)

        logger.info(f"Converted dict response into {len(operations)} memory operations")
        return operations

    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Helper: Increment named error counter safely
    # ------------------------------------------------------------------
    def _increment_error_counter(self, counter_name: str) -> None:
        """Increment an error counter defined in `self.error_counters`.

        Args:
            counter_name: The key identifying the counter to increment.
        """
        try:
            if counter_name not in self.error_counters:
                # Lazily create unknown counters so callers don't crash
                self.error_counters[counter_name] = 0
            self.error_counters[counter_name] += 1
        except Exception as e:
            # Should never fail, but guard to avoid cascading errors
            logger.debug(f"_increment_error_counter failed for '{counter_name}': {e}")

    # --- NEW Embedding Functions --- ADDED
    async def _get_embedding_from_api(self, text: str) -> Optional[np.array]:
        """Helper to get embedding from a configured OpenAI-compatible API."""
        provider_type = self.valves.embedding_provider_type # Keep for check, but not used further
        api_url = self.valves.embedding_api_url
        api_key = self.valves.embedding_api_key
        model_name = self.valves.embedding_model_name
        max_retries = self.valves.max_retries # Reuse LLM retries for now
        retry_delay = self.valves.retry_delay

        if not api_url or not api_key:
            logger.error("Attempted to call embedding API without proper URL or Key configuration.")
            return None
        if not text:
            logger.debug("Skipping embedding API call for empty text.")
            return None

        logger.info(f"Getting embedding via API: URL={api_url}, Model={model_name}")
        session = await self._get_aiohttp_session()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # Common OpenAI embedding request format
        data = {
            "input": text,
            "model": model_name
            # Add other API specific params like 'encoding_format': 'float' if needed
        }

        for attempt in range(1, max_retries + 2):
            logger.debug(f"Embedding API request attempt {attempt}/{max_retries+1}")
            try:
                async with session.post(api_url, json=data, headers=headers, timeout=30) as response:
                    logger.debug(f"Embedding API response status: {response.status}")
                    if response.status == 200:
                        response_data = await response.json()
                        # Standard OpenAI format: {"object":"list","data":[{"object":"embedding","index":0,"embedding":[...]}],"model":"...","usage":{...}}
                        if response_data.get("data") and isinstance(response_data["data"], list) and len(response_data["data"]) > 0:
                            embedding_list = response_data["data"][0].get("embedding")
                            if embedding_list and isinstance(embedding_list, list):
                                logger.debug(f"Successfully received embedding vector of dimension {len(embedding_list)} from API.")
                                return np.array(embedding_list, dtype=np.float32)
                        logger.error(f"Could not extract embedding from API response format: {str(response_data)[:200]}...")
                        return None # Invalid format
                    else:
                        error_text = await response.text()
                        logger.warning(f"Embedding API returned error {response.status}: {error_text[:200]}... (Attempt {attempt})")
                        is_retryable = response.status in [429, 500, 502, 503, 504]
                        if is_retryable and attempt <= max_retries:
                             sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                             logger.warning(f"Retrying embedding API call in {sleep_time:.2f} seconds...")
                             await asyncio.sleep(sleep_time)
                             continue
                        else:
                            logger.error(f"Non-retryable embedding API error or max retries reached.")
                            return None # Non-retryable error or max retries exceeded

            except asyncio.TimeoutError:
                logger.warning(f"Embedding API request timed out (Attempt {attempt})")
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    logger.error("Embedding API request timed out after multiple retries.")
                    return None
            except ClientError as e:
                 logger.warning(f"Embedding API connection error: {e} (Attempt {attempt})")
                 if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(sleep_time)
                    continue
                 else:
                    logger.error("Embedding API connection error after multiple retries.")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error getting embedding from API: {e}\n{traceback.format_exc()}")
                # Don't retry on unexpected errors during parsing/processing
                return None # Return None on unexpected error
        
        logger.error(f"Embedding API query failed after {max_retries + 1} attempts.")
        return None

    async def _get_embedding(self, text: str) -> Optional[np.array]:
        """Get embedding using configured provider (local or API) with metrics and retries."""
        if not text:
            logger.debug("Skipping embedding for empty text.")
            return None

        # Metrics instrumentation
        provider_type = self.valves.embedding_provider_type
        EMBEDDING_REQUESTS.labels(provider_type).inc()
        start_time = time.perf_counter()

        embedding_vector = None
        try:
            if provider_type == "local":
                local_model = self._local_embedding_model # Access the renamed property
                if local_model:
                    # Ensure text is not excessively long for local model
                    # Simple truncation, might need smarter chunking for production
                    max_local_len = 512 # Adjust based on model limits
                    truncated_text = text[:max_local_len]
                    if len(text) > max_local_len:
                        logger.warning(f"Truncating text for local embedding model (>{max_local_len} chars): {text[:60]}...")
                    
                    embedding_vector = local_model.encode(truncated_text, normalize_embeddings=True)
                else:
                    logger.error("Local embedding provider configured, but model failed to load.")
                    self.error_counters["embedding_errors"] += 1 # Count as error
            
            elif provider_type == "openai_compatible":
                embedding_vector = await self._get_embedding_from_api(text)
                if embedding_vector is None: # API call failed
                     self.error_counters["embedding_errors"] += 1 # Count as error
            
            else:
                logger.error(f"Invalid embedding_provider_type configured: {provider_type}")
                self.error_counters["embedding_errors"] += 1 # Count as error

        except Exception as e:
            logger.error(f"Error during embedding generation ({provider_type}): {e}\n{traceback.format_exc()}")
            self.error_counters["embedding_errors"] += 1 # Count as error
            embedding_vector = None # Ensure None is returned on error

        end_time = time.time()
        if embedding_vector is not None:
            # Ensure it's a numpy array of float32 for consistency
            if not isinstance(embedding_vector, np.ndarray):
                 try:
                     embedding_vector = np.array(embedding_vector, dtype=np.float32)
                 except Exception as array_err:
                      logger.error(f"Failed to convert embedding to numpy array: {array_err}")
                      self.error_counters["embedding_errors"] += 1
                      return None
            # Normalize just in case encode/API didn't
            norm = np.linalg.norm(embedding_vector)
            if norm > 1e-6: # Avoid division by zero or near-zero
                embedding_vector = embedding_vector / norm
            else:
                logger.warning("Generated embedding vector has near-zero norm. Cannot normalize.")
            
            logger.debug(f"Generated embedding via {provider_type} in {end_time - start_time:.3f}s, dim: {embedding_vector.shape}")
            EMBEDDING_LATENCY.labels(provider_type).observe(time.perf_counter() - start_time)
            return embedding_vector
        else:
            logger.warning(f"Failed to generate embedding via {provider_type} in {end_time - start_time:.3f}s")
            EMBEDDING_ERRORS.labels(provider_type).inc()
            EMBEDDING_LATENCY.labels(provider_type).observe(time.perf_counter() - start_time)
            return None

    async def _re_embed_memory_if_needed(self, memory_id: str, memory_text: str, current_dim: int, expected_dim: int) -> Optional[np.array]:
        """
        Re-embed a memory with dimension mismatch using the current configured provider.
        Returns the new embedding or None if re-embedding fails.
        """
        try:
            logger.info(
                f"Re-embedding memory {memory_id} due to dimension mismatch "
                f"(old: {current_dim}, expected: {expected_dim})"
            )

            # Get new embedding using current configured provider
            new_embedding = await self._get_embedding(memory_text)

            if new_embedding is not None and new_embedding.shape[0] == expected_dim:
                # Update cache with new embedding
                self.memory_embeddings[memory_id] = new_embedding
                logger.info(
                    f"Successfully re-embedded memory {memory_id} with new dimension: {new_embedding.shape[0]}"
                )
                return new_embedding
            elif new_embedding is not None:
                logger.warning(
                    f"Re-embedding produced unexpected dimension {new_embedding.shape[0]}, expected {expected_dim}"
                )
                return None
            else:
                logger.warning(f"Failed to re-embed memory {memory_id}")
                return None

        except Exception as e:
            logger.error(f"Error re-embedding memory {memory_id}: {e}\n{traceback.format_exc()}")
            return None

    async def _store_embedding_in_db(self, user_id: str, memory_id: str, memory_text: str, embedding: np.array) -> None:
        """Store memory embedding in a persistent JSON file for reload across restarts.
        
        Args:
            user_id: User ID for file naming
            memory_id: Memory ID to use as the key
            memory_text: The memory text content (not stored, just for logging)
            embedding: The embedding vector to store
        """
        try:
            # Use data directory for persistence
            cache_dir = "/app/backend/data/cache/embeddings"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{user_id}_embeddings.json")
            
            # Load existing cache
            cache = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading embedding cache, starting fresh: {e}")
            
            # Convert numpy array to list for JSON storage
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Store embedding
            cache[memory_id] = {
                "embedding": embedding_list,
                "model": self.valves.embedding_model_name,
                "provider": self.valves.embedding_provider_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save cache
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            
            logger.debug(f"Stored embedding for memory {memory_id} in file cache")
        except Exception as e:
            logger.warning(f"Failed to store embedding in file cache for memory {memory_id}: {e}")
            # Non-critical - embedding will be regenerated if needed

    async def _load_embedding_from_db(self, user_id: str, memory_id: str) -> Optional[np.array]:
        """Load a stored embedding from the persistent JSON file.
        
        Args:
            user_id: User ID for file naming
            memory_id: Memory ID to retrieve
            
        Returns:
            The embedding vector if found, None otherwise
        """
        try:
            cache_dir = "/app/backend/data/cache/embeddings"
            cache_file = os.path.join(cache_dir, f"{user_id}_embeddings.json")
            
            if not os.path.exists(cache_file):
                return None
            
            # Load cache
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            
            if memory_id in cache:
                embedding_list = cache[memory_id]["embedding"]
                embedding = np.array(embedding_list, dtype=np.float32)
                return embedding
            
            return None
        except Exception as e:
            logger.warning(f"ERROR loading embedding from file cache for memory {memory_id}: {e}\n{traceback.format_exc()}")
            return None

    # --- END NEW Embedding Functions ---
