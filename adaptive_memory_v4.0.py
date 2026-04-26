import json
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    Set,
    Tuple,
)
import logging
import re
import asyncio
import inspect
import pytz
import difflib
import time
import os
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ----------------------------
# Metrics & Monitoring Imports
# ----------------------------
try:
    from prometheus_client import Counter, Histogram  # type: ignore[import-not-found]
except ImportError:
    # Fallback: define dummy Counter/Histogram if prometheus_client not installed
    class _NoOpMetric:
        def __init__(self, *_args, **_kwargs):
            pass

        def labels(self, *_args, **_kwargs):
            return self

        def inc(self, *_args, **_kwargs):
            pass

        def observe(self, *_args, **_kwargs):
            pass

    Counter = Histogram = _NoOpMetric

# Define Prometheus metrics
EMBEDDING_REQUESTS = Counter(
    "adaptive_memory_embedding_requests_total",
    "Total number of embedding requests",
    ["provider"],
)
EMBEDDING_ERRORS = Counter(
    "adaptive_memory_embedding_errors_total",
    "Total number of embedding errors",
    ["provider"],
)
EMBEDDING_LATENCY = Histogram(
    "adaptive_memory_embedding_latency_seconds",
    "Latency of embedding generation",
    ["provider"],
)

RETRIEVAL_REQUESTS = Counter(
    "adaptive_memory_retrieval_requests_total",
    "Total number of get_relevant_memories calls",
    [],
)
RETRIEVAL_ERRORS = Counter(
    "adaptive_memory_retrieval_errors_total", "Total number of retrieval errors", []
)
RETRIEVAL_LATENCY = Histogram(
    "adaptive_memory_retrieval_latency_seconds",
    "Latency of get_relevant_memories execution",
    [],
)

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
except ImportError:
    SentenceTransformer = None

import numpy as np
import aiohttp
from pydantic import BaseModel, Field, model_validator, field_validator

# OpenWebUI Imports
try:
    from open_webui.config import DATA_DIR  # type: ignore[import-not-found]
except ImportError:
    from pathlib import Path
    DATA_DIR = Path("/app/backend/data")

from open_webui.models.memories import Memories  # type: ignore[import-not-found]
from open_webui.models.users import Users  # type: ignore[import-not-found]
from open_webui.main import app as webui_app  # type: ignore[import-not-found]

# --- Router & Mock Imports for Vector Indexing ---
try:
    from open_webui.routers.memories import add_memory, AddMemoryForm  # type: ignore[import-not-found]
except ImportError:
    add_memory = None
    AddMemoryForm = None

# --- Vector Database Client for Synchronization ---
try:
    from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT  # type: ignore[import-not-found]
except ImportError:
    VECTOR_DB_CLIENT = None
    # Note: Custom logger not yet defined, will log via standard logging

# --- Advanced Mock Infrastructure for Router Compatibility ---
class MockConfig:
    def __init__(self):
        # Flags required by router.add_memory
        self.ENABLE_MEMORIES = True
        self.USER_PERMISSIONS = {"features": {"memories": True}}

class MockAppState:
    def __init__(self, embedding_function):
        self.config = MockConfig()
        self.EMBEDDING_FUNCTION = embedding_function

class MockApp:
    def __init__(self, embedding_function):
        self.state = MockAppState(embedding_function)

class MockState:
    def __init__(self, user):
        self.user = user

class MockRequest:
    def __init__(self, user, embedding_function):
        self.app = MockApp(embedding_function)
        self.state = MockState(user)
        self.user = user

# Set up logging with versioned adapter
_raw_logger = logging.getLogger("openwebui.plugins.adaptive_memory")
if not _raw_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    _raw_logger.addHandler(handler)
_raw_logger.setLevel(logging.INFO)
_raw_logger.propagate = False

class AMAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[AM v4.0.2] {msg}", kwargs

logger = AMAdapter(_raw_logger, {})

# ------------------------------------------------------------------------------
# Data Models and Helper Classes
# ------------------------------------------------------------------------------


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    memory_bank: Optional[str] = None
    confidence: Optional[float] = None


class LocalAddMemoryForm(BaseModel):
    content: str


class ErrorManager:
    """Centralized error tracking and reporting."""

    def __init__(self):
        self.counters: Dict[str, int] = {
            "embedding_errors": 0,
            "llm_call_errors": 0,
            "json_parse_errors": 0,
            "memory_crud_errors": 0,
        }

    def increment(self, counter_name: str):
        self.counters[counter_name] = self.counters.get(counter_name, 0) + 1

    def get_counters(self) -> Dict[str, int]:
        return self.counters


class JSONParser:
    """Robust JSON parsing utilities."""

    @staticmethod
    def extract_and_parse(text: str) -> Union[List, Dict, None]:
        if not text:
            return None

        # 1. Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Extract from code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Extract from raw brackets
        bracket_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(1))
            except json.JSONDecodeError:
                pass

        return None


SUPPORTED_MEMORY_TAGS = {
    "identity",
    "behavior",
    "preference",
    "goal",
    "relationship",
    "possession",
    "summary",
}
MEMORY_STORAGE_PATTERN = re.compile(
    r"^\[Tags:\s*(?P<tags>[^\]]*)\]\s*(?P<content>.*?)\s*\[Memory Bank:\s*(?P<memory_bank>[^\]]+)\]\s*\[Confidence:\s*(?P<confidence>[^\]]+)\]\s*$",
    re.DOTALL,
)


@dataclass
class StoredMemoryRecord:
    content: str
    tags: List[str] = field(default_factory=list)
    memory_bank: str = "General"
    confidence: Optional[float] = None


def normalize_memory_id(memory_id: Any) -> str:
    return str(memory_id)


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text") or item.get("content") or ""
            if text:
                text_parts.append(str(text).strip())
        return " ".join(part for part in text_parts if part).strip()
    if content is None:
        return ""
    return str(content).strip()


def parse_stored_memory(memory_text: Any) -> StoredMemoryRecord:
    if memory_text is None:
        return StoredMemoryRecord(content="")

    text = str(memory_text).strip()
    if not text:
        return StoredMemoryRecord(content="")

    match = MEMORY_STORAGE_PATTERN.match(text)
    if not match:
        return StoredMemoryRecord(content=text)

    tags_raw = match.group("tags").strip()
    tags = [
        tag.strip().lower()
        for tag in tags_raw.split(",")
        if tag.strip() and tag.strip().lower() != "none"
    ]

    confidence = None
    confidence_raw = match.group("confidence").strip()
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = None

    return StoredMemoryRecord(
        content=match.group("content").strip(),
        tags=tags,
        memory_bank=match.group("memory_bank").strip() or "General",
        confidence=confidence,
    )


def format_memory_content(
    content: str,
    tags: List[str],
    memory_bank: str,
    confidence: Optional[float],
) -> str:
    cleaned_content = re.sub(r"\s+", " ", str(content or "")).strip()
    cleaned_tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
    tags_str = ", ".join(cleaned_tags) if cleaned_tags else "none"
    safe_confidence = 1.0 if confidence is None else float(confidence)
    return (
        f"[Tags: {tags_str}] {cleaned_content} "
        f"[Memory Bank: {memory_bank}] [Confidence: {safe_confidence:.2f}]"
    )


def truncate_text(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    stripped = str(text or "").strip()
    if len(stripped) <= max_length:
        return stripped
    return stripped[: max_length - 3].rstrip() + "..."


async def _resolve_owui_result(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _call_owui_method(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await _resolve_owui_result(method(*args, **kwargs))


async def get_user_by_id_compat(user_id: str) -> Optional[Any]:
    return await _call_owui_method(Users.get_user_by_id, user_id)


def extract_memory_id(memory: Any) -> Optional[str]:
    if memory is None:
        return None

    if hasattr(memory, "id"):
        memory_id = getattr(memory, "id")
    elif hasattr(memory, "get"):
        try:
            memory_id = memory.get("id")
        except Exception:
            return None
    else:
        return None

    if memory_id is None:
        return None
    return normalize_memory_id(memory_id)


async def get_memories_by_user_id_compat(user_id: str) -> List[Any]:
    memories = await _call_owui_method(Memories.get_memories_by_user_id, user_id)
    if memories is None:
        return []
    return list(memories)


async def insert_new_memory_compat(user_id: str, content: str) -> Any:
    return await _call_owui_method(Memories.insert_new_memory, user_id, content)


async def update_memory_by_id_and_user_id_compat(
    memory_id: str, user_id: str, content: str
) -> Any:
    return await _call_owui_method(
        Memories.update_memory_by_id_and_user_id, memory_id, user_id, content
    )


async def delete_memory_by_id_compat(memory_id: str) -> Any:
    return await _call_owui_method(Memories.delete_memory_by_id, memory_id)


async def delete_memory_by_id_and_user_id_compat(memory_id: str, user_id: str) -> bool:
    memory_id = normalize_memory_id(memory_id)
    if not memory_id or not user_id:
        return False

    existing_memories = await get_memories_by_user_id_compat(user_id)
    existing_ids = {
        existing_id
        for existing_id in (extract_memory_id(memory) for memory in existing_memories)
        if existing_id is not None
    }
    if memory_id not in existing_ids:
        return False

    await delete_memory_by_id_compat(memory_id)

    refreshed_memories = await get_memories_by_user_id_compat(user_id)
    refreshed_ids = {
        refreshed_id
        for refreshed_id in (extract_memory_id(memory) for memory in refreshed_memories)
        if refreshed_id is not None
    }
    return memory_id not in refreshed_ids


class LRUCache:
    """A simple LRU (Least Recently Used) cache with bounded size.
    
    Entries are evicted when the cache reaches max_size. Most recently
    accessed items are kept, oldest items are removed first.
    """
    
    def __init__(self, max_size: int = 10000):
        """Initialize LRU cache with maximum size.
        
        Args:
            max_size: Maximum number of entries to keep in cache
        """
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()  # Thread-safe for concurrent async access
    
    async def get(self, key: str) -> Optional[np.ndarray]:
        """Get value from cache, moving it to end (most recently used).
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value if found, None otherwise
        """
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: np.ndarray) -> None:
        """Set value in cache, evicting oldest entry if at capacity.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[key] = value

    async def delete(self, key: str) -> None:
        """Remove a value from cache if it exists."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._lock:
            self._cache.clear()


# ------------------------------------------------------------------------------
# Embedding Management
# ------------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embedding(self, text: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[np.ndarray]:
        del text, session
        raise NotImplementedError

    @abstractmethod
    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        del texts, session
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer:
            try:
                logger.info(f"Loading local embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.exception(f"Failed to load local SentenceTransformer model: {e}")

    async def get_embedding(self, text: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[np.ndarray]:
        del session
        if not self.model:
            return None
        try:
            # Run blocking call in executor
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode(text, normalize_embeddings=True)
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.exception(f"Local embedding error: {e}")
            return None

    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        del session
        if not self.model or not texts:
            return [None] * len(texts)
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                ),
            )
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            logger.exception(f"Local batch embedding error: {e}")
            return [None] * len(texts)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    async def get_embedding(self, text: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[np.ndarray]:
        try:
            # Use provided session or create one (fallback)
            inner_session = session if session else aiohttp.ClientSession()
            should_close = session is None
            
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {"input": text, "model": self.model_name}
                async with inner_session.post(
                    self.api_url, json=data, headers=headers, timeout=30
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        if "data" in res_json and len(res_json["data"]) > 0:
                            emb = res_json["data"][0]["embedding"]
                            return np.array(emb, dtype=np.float32)
                    return None
            finally:
                if should_close:
                    await inner_session.close()
        except Exception as e:
            logger.exception(f"API embedding error: {e}")
            return None

    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        try:
            inner_session = session if session else aiohttp.ClientSession()
            should_close = session is None
            
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {"input": texts, "model": self.model_name}
                async with inner_session.post(
                    self.api_url, json=data, headers=headers, timeout=60
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        if "data" in res_json:
                            # Correct indexing: map by the 'index' field to ensure positional alignment
                            results = [None] * len(texts)
                            for item in res_json["data"]:
                                idx = item.get("index")
                                embedding_data = item.get("embedding")
                                if idx is not None and 0 <= idx < len(results) and embedding_data is not None:
                                    results[idx] = np.array(embedding_data, dtype=np.float32)
                                elif idx is not None and 0 <= idx < len(results):
                                    logger.warning(f"Missing embedding data for item at index {idx} in batch response")
                            return results
                    return [None] * len(texts)
            finally:
                if should_close:
                    await inner_session.close()
        except Exception as e:
            logger.exception(f"API batch embedding error: {e}")
            return [None] * len(texts)


class EmbeddingManager:
    """Manages embedding generation, caching, and persistence."""

    def __init__(self, get_valves: Callable[[], Any], error_manager: ErrorManager):
        self.get_valves = get_valves
        self.error_manager = error_manager
        self.cache = LRUCache()  # Bounded LRU cache (default max_size=10000)
        self.provider: Optional[EmbeddingProvider] = None
        self._provider_signature: Optional[Tuple[Any, ...]] = None
        self._session: Optional[aiohttp.ClientSession] = None
        # WeakValueDictionary allows garbage collection of locks when no longer referenced,
        # preventing unbounded growth of the locks dict
        # Use regular dict instead of WeakValueDictionary to avoid premature GC of Lock objects
        self._locks: Dict[str, asyncio.Lock] = {}
        self._cache_root = os.path.join(DATA_DIR, "cache")
        self._legacy_cache_dir = os.path.join(self._cache_root, "embeddings")
        self._sqlite_cache_file = os.path.join(self._cache_root, "embeddings.sqlite")

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a lock for the given user_id."""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]
    
    def _cleanup_lock(self, user_id: str) -> None:
        """Remove lock for user_id if it's not locked and has no waiters.
        
        This prevents unbounded growth of the locks dict while ensuring
        we don't delete locks that are still in use.
        """
        if user_id in self._locks:
            lock = self._locks[user_id]
            # Only delete if lock is not currently held and has no tasks waiting
            if not lock.locked() and (not hasattr(lock, '_waiters') or not lock._waiters):
                del self._locks[user_id]


    async def cleanup(self):
        """Clean up resources like the shared HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _ensure_session(self):
        """Ensure a shared aiohttp session exists."""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    def _ensure_provider(self):
        valves = self.get_valves()
        provider_signature = (
            getattr(valves, "embedding_source", "auto"),
            valves.embedding_provider_type,
            valves.embedding_model_name,
            valves.embedding_api_url,
            valves.embedding_api_key,
        )
        if not self.provider or self._provider_signature != provider_signature:
            if self._provider_signature and self._provider_signature != provider_signature:
                logger.info(
                    "Embedding provider configuration changed; recreating provider and clearing in-memory embedding cache"
                )
            self._provider_signature = provider_signature
            self.cache = LRUCache()
            if not self._should_use_plugin_embeddings():
                self.provider = None
                return
            if valves.embedding_provider_type == "local":
                self.provider = LocalEmbeddingProvider(valves.embedding_model_name)
            elif valves.embedding_provider_type == "openai_compatible":
                self.provider = OpenAICompatibleEmbeddingProvider(
                    valves.embedding_api_url,
                    valves.embedding_api_key,
                    valves.embedding_model_name,
                )

    def _embedding_source_mode(self) -> str:
        return getattr(self.get_valves(), "embedding_source", "auto")

    def _should_use_open_webui_embeddings(self) -> bool:
        return self._embedding_source_mode() in {"auto", "owui"}

    def _should_use_plugin_embeddings(self) -> bool:
        return self._embedding_source_mode() in {"auto", "plugin"}

    def _metrics_provider_label(self, source: str) -> str:
        if source == "open_webui":
            return "open_webui"
        return self.get_valves().embedding_provider_type

    def _default_record_identity(self) -> Tuple[str, str]:
        valves = self.get_valves()
        if self._embedding_source_mode() == "owui":
            return "open_webui", "open_webui"
        return valves.embedding_model_name, valves.embedding_provider_type

    def _get_open_webui_embedding_function(self) -> Optional[Callable[..., Any]]:
        app_state = getattr(webui_app, "state", None)
        return getattr(app_state, "EMBEDDING_FUNCTION", None)

    async def _get_open_webui_embedding(
        self, text: str, user: Any = None
    ) -> Optional[np.ndarray]:
        if not text:
            return None

        embedding_function = self._get_open_webui_embedding_function()
        if embedding_function is None:
            return None

        try:
            try:
                result = embedding_function(text, user=user)
            except TypeError:
                result = embedding_function(text)

            if inspect.isawaitable(result):
                result = await result

            if result is None:
                return None

            embedding = np.asarray(result, dtype=np.float32)
            if embedding.ndim != 1 or embedding.size == 0:
                logger.warning(
                    "Open WebUI embedding function returned an invalid embedding shape"
                )
                return None

            return embedding
        except Exception as e:
            logger.warning(f"Open WebUI embedding function failed: {e}")
            return None

    async def get_embedding(self, text: str, user: Any = None) -> Optional[np.ndarray]:
        if not text:
            return None

        metric_source = (
            "open_webui" if self._should_use_open_webui_embeddings() else "plugin"
        )
        EMBEDDING_REQUESTS.labels(self._metrics_provider_label(metric_source)).inc()
        start = time.perf_counter()

        emb = None
        if self._should_use_open_webui_embeddings():
            emb = await self._get_open_webui_embedding(text, user=user)
        if emb is not None:
            EMBEDDING_LATENCY.labels(self._metrics_provider_label("open_webui")).observe(
                time.perf_counter() - start
            )
            return emb

        if not self._should_use_plugin_embeddings():
            self.error_manager.increment("embedding_errors")
            EMBEDDING_ERRORS.labels(self._metrics_provider_label("open_webui")).inc()
            return None

        if not self.provider:
            self._ensure_provider()

        if not self.provider:
            self.error_manager.increment("embedding_errors")
            EMBEDDING_ERRORS.labels(self._metrics_provider_label("plugin")).inc()
            return None

        self._ensure_session()
        emb = await self.provider.get_embedding(text, session=self._session)

        if emb is not None:
            EMBEDDING_LATENCY.labels(self._metrics_provider_label("plugin")).observe(
                time.perf_counter() - start
            )
        else:
            self.error_manager.increment("embedding_errors")
            EMBEDDING_ERRORS.labels(self._metrics_provider_label("plugin")).inc()

        return emb

    async def get_embeddings_batch(
        self, texts: List[str], user: Any = None
    ) -> List[Optional[np.ndarray]]:
        if not texts:
            return []

        open_webui_results = [None] * len(texts)
        if self._should_use_open_webui_embeddings():
            open_webui_results = await asyncio.gather(
                *[self._get_open_webui_embedding(text, user=user) for text in texts]
            )
        if all(result is not None for result in open_webui_results):
            return list(open_webui_results)

        if not self._should_use_plugin_embeddings():
            return list(open_webui_results)

        if not self.provider:
            self._ensure_provider()

        if not self.provider:
            return list(open_webui_results)

        self._ensure_session()
        missing_indices = [
            idx for idx, result in enumerate(open_webui_results) if result is None
        ]
        if not missing_indices:
            return list(open_webui_results)

        provider_results = await self.provider.get_embeddings_batch(
            [texts[idx] for idx in missing_indices], session=self._session
        )

        combined_results = list(open_webui_results)
        for idx, provider_embedding in zip(
            missing_indices, provider_results, strict=True
        ):
            combined_results[idx] = provider_embedding
        return combined_results

    def _get_legacy_cache_file(self, user_id: str) -> str:
        return os.path.join(self._legacy_cache_dir, f"{user_id}_embeddings.json")

    def _connect_cache_db(self) -> sqlite3.Connection:
        os.makedirs(self._cache_root, exist_ok=True)
        conn = sqlite3.connect(self._sqlite_cache_file, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_cache_db_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                user_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                model TEXT,
                provider TEXT,
                timestamp TEXT,
                PRIMARY KEY (user_id, memory_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS legacy_cache_migrations (
                user_id TEXT PRIMARY KEY,
                legacy_cache_file TEXT NOT NULL,
                legacy_mtime REAL NOT NULL,
                migrated_at TEXT NOT NULL
            )
            """
        )

    def _build_embedding_record(
        self,
        embedding: np.ndarray,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        embedding_array = np.asarray(embedding, dtype=np.float32)
        default_model, default_provider = self._default_record_identity()
        return {
            "embedding_json": json.dumps(embedding_array.tolist()),
            "model": model or default_model,
            "provider": provider or default_provider,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }

    def _record_to_embedding(self, record: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            embedding_json = record.get("embedding_json")
            if embedding_json is None and "embedding" in record:
                embedding_json = json.dumps(record["embedding"])
            if embedding_json is None:
                return None
            return np.array(json.loads(embedding_json), dtype=np.float32)
        except Exception:
            return None

    def _load_legacy_cache_sync(self, user_id: str) -> Dict[str, Any]:
        cache_file = self._get_legacy_cache_file(user_id)
        if not os.path.exists(cache_file):
            return {}
        with open(cache_file, "r") as f:
            return json.load(f)

    def _save_legacy_cache_sync(self, user_id: str, cache: Dict[str, Any]) -> None:
        os.makedirs(self._legacy_cache_dir, exist_ok=True)
        cache_file = self._get_legacy_cache_file(user_id)
        tmp_file = cache_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(cache, f)
        os.replace(tmp_file, cache_file)

    def _store_embedding_legacy_json_sync(
        self, user_id: str, memory_id: str, embedding: np.ndarray
    ) -> None:
        memory_id_str = normalize_memory_id(memory_id)
        cache = self._load_legacy_cache_sync(user_id)
        model, provider = self._default_record_identity()
        cache[memory_id_str] = {
            "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
            "model": model,
            "provider": provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save_legacy_cache_sync(user_id, cache)

    def _store_embeddings_batch_legacy_json_sync(
        self, user_id: str, ids: List[str], embeddings: List[np.ndarray]
    ) -> None:
        if not ids:
            return
        cache = self._load_legacy_cache_sync(user_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        model, provider = self._default_record_identity()
        for memory_id, embedding in zip(ids, embeddings, strict=True):
            if embedding is None:
                continue
            cache[normalize_memory_id(memory_id)] = {
                "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                "model": model,
                "provider": provider,
                "timestamp": timestamp,
            }
        self._save_legacy_cache_sync(user_id, cache)

    def _load_embedding_legacy_json_sync(
        self, user_id: str, memory_id: str
    ) -> Optional[Dict[str, Any]]:
        memory_id_str = normalize_memory_id(memory_id)
        cache = self._load_legacy_cache_sync(user_id)
        embedding_data = cache.get(memory_id_str)
        if not embedding_data:
            return None
        return {
            "embedding_json": json.dumps(embedding_data.get("embedding")),
            "model": embedding_data.get("model"),
            "provider": embedding_data.get("provider"),
            "timestamp": embedding_data.get("timestamp"),
        }

    def _delete_embedding_legacy_json_sync(self, user_id: str, memory_id: str) -> None:
        cache_file = self._get_legacy_cache_file(user_id)
        if not os.path.exists(cache_file):
            return
        cache = self._load_legacy_cache_sync(user_id)
        memory_id_str = normalize_memory_id(memory_id)
        if memory_id_str not in cache:
            return
        cache.pop(memory_id_str, None)
        if cache:
            self._save_legacy_cache_sync(user_id, cache)
        else:
            try:
                os.remove(cache_file)
            except FileNotFoundError:
                pass

    def _store_embedding_sqlite_sync(
        self, user_id: str, memory_id: str, embedding: np.ndarray
    ) -> None:
        memory_id_str = normalize_memory_id(memory_id)
        record = self._build_embedding_record(embedding)
        with self._connect_cache_db() as conn:
            self._ensure_cache_db_schema(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                    (user_id, memory_id, embedding_json, model, provider, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    memory_id_str,
                    record["embedding_json"],
                    record["model"],
                    record["provider"],
                    record["timestamp"],
                ),
            )
            conn.commit()

    def _store_embeddings_batch_sqlite_sync(
        self, user_id: str, ids: List[str], embeddings: List[np.ndarray]
    ) -> int:
        rows = []
        timestamp = datetime.now(timezone.utc).isoformat()
        model = self.get_valves().embedding_model_name
        provider = self.get_valves().embedding_provider_type
        for memory_id, embedding in zip(ids, embeddings, strict=True):
            if embedding is None:
                continue
            record = self._build_embedding_record(
                embedding, model=model, provider=provider, timestamp=timestamp
            )
            rows.append(
                (
                    user_id,
                    normalize_memory_id(memory_id),
                    record["embedding_json"],
                    record["model"],
                    record["provider"],
                    record["timestamp"],
                )
            )

        if not rows:
            return 0

        with self._connect_cache_db() as conn:
            self._ensure_cache_db_schema(conn)
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                    (user_id, memory_id, embedding_json, model, provider, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def _load_embedding_sqlite_sync(
        self, user_id: str, memory_id: str
    ) -> Optional[Dict[str, Any]]:
        memory_id_str = normalize_memory_id(memory_id)
        with self._connect_cache_db() as conn:
            self._ensure_cache_db_schema(conn)
            row = conn.execute(
                """
                SELECT embedding_json, model, provider, timestamp
                FROM embeddings
                WHERE user_id = ? AND memory_id = ?
                """,
                (user_id, memory_id_str),
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def _delete_embedding_sqlite_sync(self, user_id: str, memory_id: str) -> None:
        memory_id_str = normalize_memory_id(memory_id)
        with self._connect_cache_db() as conn:
            self._ensure_cache_db_schema(conn)
            conn.execute(
                "DELETE FROM embeddings WHERE user_id = ? AND memory_id = ?",
                (user_id, memory_id_str),
            )
            conn.commit()

    def _migrate_legacy_cache_if_needed_sync(self, user_id: str) -> int:
        cache_file = self._get_legacy_cache_file(user_id)
        if not os.path.exists(cache_file):
            return 0

        legacy_mtime = os.path.getmtime(cache_file)
        with self._connect_cache_db() as conn:
            self._ensure_cache_db_schema(conn)
            existing = conn.execute(
                """
                SELECT legacy_mtime
                FROM legacy_cache_migrations
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            if existing and float(existing["legacy_mtime"]) == float(legacy_mtime):
                return 0

            cache = self._load_legacy_cache_sync(user_id)
            rows = []
            for memory_id, embedding_data in cache.items():
                embedding = self._record_to_embedding(embedding_data)
                if embedding is None:
                    continue
                record = self._build_embedding_record(
                    embedding,
                    model=embedding_data.get("model"),
                    provider=embedding_data.get("provider"),
                    timestamp=embedding_data.get("timestamp"),
                )
                rows.append(
                    (
                        user_id,
                        normalize_memory_id(memory_id),
                        record["embedding_json"],
                        record["model"],
                        record["provider"],
                        record["timestamp"],
                    )
                )

            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings
                        (user_id, memory_id, embedding_json, model, provider, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

            conn.execute(
                """
                INSERT OR REPLACE INTO legacy_cache_migrations
                    (user_id, legacy_cache_file, legacy_mtime, migrated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    cache_file,
                    legacy_mtime,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return len(rows)

    def _is_record_compatible(self, record: Dict[str, Any]) -> bool:
        expected_model, expected_provider = self._default_record_identity()
        return (
            record.get("model") == expected_model
            and record.get("provider") == expected_provider
        )

    async def store_embedding_persistent(self, user_id: str, memory_id: str, _memory_text: str, embedding: np.ndarray) -> None:
        """Store memory embedding in the plugin's persistent sidecar cache."""

        async with self._get_lock(user_id):
            try:
                migrated_count = await asyncio.to_thread(
                    self._migrate_legacy_cache_if_needed_sync, user_id
                )
                if migrated_count:
                    logger.info(
                        f"Migrated {migrated_count} legacy embeddings into SQLite cache for user {user_id}"
                    )

                await asyncio.to_thread(
                    self._store_embedding_sqlite_sync, user_id, memory_id, embedding
                )
                logger.debug(f"Stored embedding for memory {memory_id} in SQLite cache")
            except Exception as e:
                logger.warning(
                    f"SQLite cache store failed for memory {memory_id}; falling back to legacy JSON cache: {e}"
                )
                try:
                    await asyncio.to_thread(
                        self._store_embedding_legacy_json_sync, user_id, memory_id, embedding
                    )
                except Exception as legacy_err:
                    logger.warning(
                        f"Failed to store embedding in persistent cache for memory {memory_id}: {legacy_err}"
                    )
            finally:
                self._cleanup_lock(user_id)

    async def store_embeddings_batch_persistent(self, user_id: str, ids: List[str], _texts: List[str], embeddings: List[np.ndarray]) -> None:
        """Store multiple embeddings in the plugin's persistent sidecar cache."""
        if not ids:
            return

        async with self._get_lock(user_id):
            try:
                migrated_count = await asyncio.to_thread(
                    self._migrate_legacy_cache_if_needed_sync, user_id
                )
                if migrated_count:
                    logger.info(
                        f"Migrated {migrated_count} legacy embeddings into SQLite cache for user {user_id}"
                    )

                stored_count = await asyncio.to_thread(
                    self._store_embeddings_batch_sqlite_sync, user_id, ids, embeddings
                )
                logger.info(
                    f"Batched stored {stored_count} embeddings in SQLite cache for user {user_id}"
                )
            except Exception as e:
                logger.warning(
                    f"SQLite batch cache store failed; falling back to legacy JSON cache: {e}"
                )
                try:
                    await asyncio.to_thread(
                        self._store_embeddings_batch_legacy_json_sync,
                        user_id,
                        ids,
                        embeddings,
                    )
                except Exception as legacy_err:
                    logger.warning(
                        f"Failed to store batch embeddings in persistent cache: {legacy_err}"
                    )
            finally:
                self._cleanup_lock(user_id)

    async def load_embedding_persistent(self, user_id: str, memory_id: str) -> Optional[np.ndarray]:
        """Load a stored embedding from the plugin's persistent sidecar cache."""
        result = None
        memory_id_str = normalize_memory_id(memory_id)

        async with self._get_lock(user_id):
            try:
                migrated_count = await asyncio.to_thread(
                    self._migrate_legacy_cache_if_needed_sync, user_id
                )
                if migrated_count:
                    logger.info(
                        f"Migrated {migrated_count} legacy embeddings into SQLite cache for user {user_id}"
                    )

                sqlite_record = await asyncio.to_thread(
                    self._load_embedding_sqlite_sync, user_id, memory_id_str
                )
                if sqlite_record and self._is_record_compatible(sqlite_record):
                    result = self._record_to_embedding(sqlite_record)
                elif sqlite_record:
                    logger.debug(
                        f"Cache miss for {memory_id_str}: Model/provider changed"
                    )

                if result is None:
                    legacy_record = await asyncio.to_thread(
                        self._load_embedding_legacy_json_sync, user_id, memory_id_str
                    )
                    if legacy_record and self._is_record_compatible(legacy_record):
                        result = self._record_to_embedding(legacy_record)
                        if result is not None:
                            try:
                                await asyncio.to_thread(
                                    self._store_embedding_sqlite_sync,
                                    user_id,
                                    memory_id_str,
                                    result,
                                )
                            except Exception as sqlite_err:
                                logger.debug(
                                    f"Failed to hydrate SQLite cache from legacy JSON for {memory_id_str}: {sqlite_err}"
                                )
            except Exception as e:
                logger.warning(
                    f"Error loading embedding from persistent cache for memory {memory_id}: {e}"
                )
                result = None
            finally:
                self._cleanup_lock(user_id)
        return result

    async def delete_embedding_persistent(self, user_id: str, memory_id: str) -> None:
        """Delete a stored embedding from memory and all persistent cache backends."""
        memory_id_str = normalize_memory_id(memory_id)
        await self.cache.delete(memory_id_str)

        async with self._get_lock(user_id):
            try:
                await asyncio.to_thread(
                    self._delete_embedding_sqlite_sync, user_id, memory_id_str
                )
            except Exception as e:
                logger.warning(
                    f"Failed to delete embedding from SQLite cache for memory {memory_id_str}: {e}"
                )

            try:
                await asyncio.to_thread(
                    self._delete_embedding_legacy_json_sync, user_id, memory_id_str
                )
            except Exception as e:
                logger.warning(
                    f"Failed to delete embedding from legacy JSON cache for memory {memory_id_str}: {e}"
                )
            finally:
                self._cleanup_lock(user_id)

    async def get_embedding_with_persistence(
        self, text: str, user_id: str, memory_id: str, user: Any = None
    ) -> Optional[np.ndarray]:
        """Get embedding with full caching hierarchy: memory -> persistent -> generate."""
        if not text:
            return None

        # Ensure memory_id is a string for consistent caching
        memory_id_str = str(memory_id)

        # 1. Check in-memory cache first
        cached_emb = await self.cache.get(memory_id_str)
        if cached_emb is not None:
            return cached_emb

        # 2. Check persistent cache
        persistent_emb = await self.load_embedding_persistent(user_id, memory_id_str)
        if persistent_emb is not None:
            # Cache in memory for this session
            await self.cache.set(memory_id_str, persistent_emb)
            return persistent_emb

        # 3. Generate new embedding
        new_emb = await self.get_embedding(text, user=user)
        if new_emb is not None:
            # Cache in memory
            await self.cache.set(memory_id_str, new_emb)
            # Store persistently
            await self.store_embedding_persistent(user_id, memory_id_str, text, new_emb)
        
        return new_emb


class Mem0SyncManager:
    """Best-effort mirror of local memories into a Mem0 instance."""

    def __init__(self, get_valves: Callable[[], Any]):
        self.get_valves = get_valves
        self._session: Optional[aiohttp.ClientSession] = None
        self._db_lock = asyncio.Lock()
        self._cache_root = os.path.join(DATA_DIR, "cache")
        self._sqlite_file = os.path.join(self._cache_root, "mem0_sync.sqlite")
        self._reconcile_locks: Dict[str, asyncio.Lock] = {}
        self._last_reconcile_check: Dict[str, float] = {}
        self._reconcile_tasks: Dict[str, asyncio.Task] = {}

    async def cleanup(self):
        """Clean up resources like the shared HTTP session."""
        reconcile_tasks = [
            task for task in self._reconcile_tasks.values() if not task.done()
        ]
        for task in reconcile_tasks:
            task.cancel()
        if reconcile_tasks:
            await asyncio.gather(*reconcile_tasks, return_exceptions=True)
        self._reconcile_tasks.clear()

        if self._session:
            await self._session.close()
            self._session = None
        self._reconcile_locks.clear()
        self._last_reconcile_check.clear()

    def _is_enabled(self) -> bool:
        valves = self.get_valves()
        return bool(
            getattr(valves, "enable_mem0_sync", False)
            and str(getattr(valves, "mem0_api_base_url", "") or "").strip()
            and str(getattr(valves, "mem0_api_key", "") or "").strip()
        )

    def should_use_background_sync(self) -> bool:
        strategy = str(
            getattr(self.get_valves(), "mem0_sync_strategy", "background")
            or "background"
        ).strip()
        return self._is_enabled() and strategy == "background"

    def should_reconcile_in_request_path(self) -> bool:
        return self._is_enabled() and not self.should_use_background_sync()

    def _get_sync_batch_size(self) -> int:
        try:
            batch_size = int(getattr(self.get_valves(), "mem0_sync_batch_size", 10))
        except (TypeError, ValueError):
            batch_size = 10
        return max(1, batch_size)

    def _get_sync_batch_interval_seconds(self) -> float:
        try:
            interval = float(
                getattr(self.get_valves(), "mem0_sync_batch_interval_seconds", 7200.0)
            )
        except (TypeError, ValueError):
            interval = 7200.0
        return max(0.1, interval)

    def _get_sync_retry_delay_seconds(self) -> float:
        try:
            retry_delay = float(
                getattr(self.get_valves(), "mem0_sync_retry_delay_seconds", 15.0)
            )
        except (TypeError, ValueError):
            retry_delay = 15.0
        return max(1.0, retry_delay)

    def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    def _base_url(self) -> str:
        return str(self.get_valves().mem0_api_base_url).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.get_valves().mem0_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=float(self.get_valves().mem0_timeout_seconds))

    def _connect_db(self) -> sqlite3.Connection:
        os.makedirs(self._cache_root, exist_ok=True)
        conn = sqlite3.connect(self._sqlite_file, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_db_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mem0_memory_mappings (
                user_id TEXT NOT NULL,
                owui_memory_id TEXT NOT NULL,
                mem0_memory_id TEXT NOT NULL,
                synced_at TEXT NOT NULL,
                PRIMARY KEY (user_id, owui_memory_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mem0_user_mappings (
                owui_user_id TEXT PRIMARY KEY,
                mem0_user_id TEXT NOT NULL,
                synced_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mem0_sync_jobs (
                user_id TEXT NOT NULL,
                owui_memory_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                payload TEXT,
                queued_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                available_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                PRIMARY KEY (user_id, owui_memory_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mem0_sync_jobs_available_at
            ON mem0_sync_jobs (available_at, queued_at)
            """
        )

    def _upsert_mapping_sync(
        self, user_id: str, owui_memory_id: str, mem0_memory_id: str
    ) -> None:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO mem0_memory_mappings
                    (user_id, owui_memory_id, mem0_memory_id, synced_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    normalize_memory_id(owui_memory_id),
                    str(mem0_memory_id),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def _get_mapping_sync(self, user_id: str, owui_memory_id: str) -> Optional[str]:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            row = conn.execute(
                """
                SELECT mem0_memory_id
                FROM mem0_memory_mappings
                WHERE user_id = ? AND owui_memory_id = ?
                """,
                (user_id, normalize_memory_id(owui_memory_id)),
            ).fetchone()
            return str(row["mem0_memory_id"]) if row is not None else None

    def _delete_mapping_sync(self, user_id: str, owui_memory_id: str) -> None:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            conn.execute(
                """
                DELETE FROM mem0_memory_mappings
                WHERE user_id = ? AND owui_memory_id = ?
                """,
                (user_id, normalize_memory_id(owui_memory_id)),
            )
            conn.commit()

    def _list_mappings_sync(self, user_id: str) -> List[Tuple[str, str]]:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            rows = conn.execute(
                """
                SELECT owui_memory_id, mem0_memory_id
                FROM mem0_memory_mappings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()
            return [
                (str(row["owui_memory_id"]), str(row["mem0_memory_id"]))
                for row in rows
            ]

    def _upsert_user_mapping_sync(self, user_id: str, mem0_user_id: str) -> None:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO mem0_user_mappings
                    (owui_user_id, mem0_user_id, synced_at)
                VALUES (?, ?, ?)
                """,
                (
                    user_id,
                    str(mem0_user_id),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def _get_user_mapping_sync(self, user_id: str) -> Optional[str]:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            row = conn.execute(
                """
                SELECT mem0_user_id
                FROM mem0_user_mappings
                WHERE owui_user_id = ?
                """,
                (user_id,),
            ).fetchone()
            return str(row["mem0_user_id"]) if row is not None else None

    def _enqueue_job_sync(
        self,
        user_id: str,
        owui_memory_id: str,
        operation: str,
        payload: Optional[Dict[str, Any]],
        available_at: datetime,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        available_at_iso = available_at.astimezone(timezone.utc).isoformat()
        payload_json = json.dumps(payload) if payload is not None else None
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            conn.execute(
                """
                INSERT INTO mem0_sync_jobs (
                    user_id,
                    owui_memory_id,
                    operation,
                    payload,
                    queued_at,
                    updated_at,
                    available_at,
                    attempt_count,
                    last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)
                ON CONFLICT(user_id, owui_memory_id) DO UPDATE SET
                    operation = excluded.operation,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at,
                    available_at = excluded.available_at,
                    attempt_count = 0,
                    last_error = NULL
                """,
                (
                    user_id,
                    normalize_memory_id(owui_memory_id),
                    str(operation).upper(),
                    payload_json,
                    now_iso,
                    now_iso,
                    available_at_iso,
                ),
            )
            conn.commit()

    def _fetch_ready_jobs_sync(self, limit: int) -> List[Dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    user_id,
                    owui_memory_id,
                    operation,
                    payload,
                    queued_at,
                    updated_at,
                    available_at,
                    attempt_count,
                    last_error
                FROM mem0_sync_jobs
                WHERE available_at <= ?
                ORDER BY queued_at ASC
                LIMIT ?
                """,
                (now_iso, int(limit)),
            ).fetchall()

        jobs: List[Dict[str, Any]] = []
        for row in rows:
            payload_value = row["payload"]
            parsed_payload: Optional[Dict[str, Any]] = None
            if payload_value:
                try:
                    loaded_payload = json.loads(str(payload_value))
                    if isinstance(loaded_payload, dict):
                        parsed_payload = loaded_payload
                except json.JSONDecodeError:
                    parsed_payload = None

            jobs.append(
                {
                    "user_id": str(row["user_id"]),
                    "owui_memory_id": str(row["owui_memory_id"]),
                    "operation": str(row["operation"]),
                    "payload": parsed_payload,
                    "queued_at": str(row["queued_at"]),
                    "updated_at": str(row["updated_at"]),
                    "available_at": str(row["available_at"]),
                    "attempt_count": int(row["attempt_count"] or 0),
                    "last_error": row["last_error"],
                }
            )
        return jobs

    def _delete_job_sync(self, user_id: str, owui_memory_id: str) -> None:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            conn.execute(
                """
                DELETE FROM mem0_sync_jobs
                WHERE user_id = ? AND owui_memory_id = ?
                """,
                (user_id, normalize_memory_id(owui_memory_id)),
            )
            conn.commit()

    def _reschedule_job_sync(
        self,
        user_id: str,
        owui_memory_id: str,
        attempt_count: int,
        available_at: datetime,
        last_error: str,
    ) -> None:
        with self._connect_db() as conn:
            self._ensure_db_schema(conn)
            conn.execute(
                """
                UPDATE mem0_sync_jobs
                SET attempt_count = ?,
                    available_at = ?,
                    updated_at = ?,
                    last_error = ?
                WHERE user_id = ? AND owui_memory_id = ?
                """,
                (
                    int(attempt_count),
                    available_at.astimezone(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    truncate_text(last_error, 500),
                    user_id,
                    normalize_memory_id(owui_memory_id),
                ),
            )
            conn.commit()

    async def _store_mapping(
        self, user_id: str, owui_memory_id: str, mem0_memory_id: str
    ) -> None:
        async with self._db_lock:
            await asyncio.to_thread(
                self._upsert_mapping_sync, user_id, owui_memory_id, mem0_memory_id
            )

    async def _get_mapping(self, user_id: str, owui_memory_id: str) -> Optional[str]:
        async with self._db_lock:
            return await asyncio.to_thread(
                self._get_mapping_sync, user_id, owui_memory_id
            )

    async def _delete_mapping(self, user_id: str, owui_memory_id: str) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._delete_mapping_sync, user_id, owui_memory_id)

    async def _list_mappings(self, user_id: str) -> List[Tuple[str, str]]:
        async with self._db_lock:
            return await asyncio.to_thread(self._list_mappings_sync, user_id)

    async def _store_user_mapping(self, user_id: str, mem0_user_id: str) -> None:
        async with self._db_lock:
            await asyncio.to_thread(
                self._upsert_user_mapping_sync, user_id, mem0_user_id
            )

    async def _get_user_mapping(self, user_id: str) -> Optional[str]:
        async with self._db_lock:
            return await asyncio.to_thread(self._get_user_mapping_sync, user_id)

    async def _enqueue_job(
        self,
        user_id: str,
        owui_memory_id: str,
        operation: str,
        payload: Optional[Dict[str, Any]],
        *,
        available_at: Optional[datetime] = None,
    ) -> None:
        available_at = available_at or datetime.now(timezone.utc)
        async with self._db_lock:
            await asyncio.to_thread(
                self._enqueue_job_sync,
                user_id,
                owui_memory_id,
                operation,
                payload,
                available_at,
            )

    async def _fetch_ready_jobs(self, limit: int) -> List[Dict[str, Any]]:
        async with self._db_lock:
            return await asyncio.to_thread(self._fetch_ready_jobs_sync, limit)

    async def _delete_job(self, user_id: str, owui_memory_id: str) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._delete_job_sync, user_id, owui_memory_id)

    async def _reschedule_job(
        self,
        user_id: str,
        owui_memory_id: str,
        attempt_count: int,
        available_at: datetime,
        last_error: str,
    ) -> None:
        async with self._db_lock:
            await asyncio.to_thread(
                self._reschedule_job_sync,
                user_id,
                owui_memory_id,
                attempt_count,
                available_at,
                last_error,
            )

    def _get_reconcile_lock(self, user_id: str) -> asyncio.Lock:
        lock = self._reconcile_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._reconcile_locks[user_id] = lock
        return lock

    def _cleanup_reconcile_state(self, user_id: str) -> None:
        """Remove reconciliation state for user_id if it is no longer in use."""
        lock = self._reconcile_locks.get(user_id)
        if lock is None:
            self._last_reconcile_check.pop(user_id, None)
            return

        has_waiters = bool(getattr(lock, "_waiters", None))
        if not lock.locked() and not has_waiters:
            del self._reconcile_locks[user_id]
            self._last_reconcile_check.pop(user_id, None)

    def _cleanup_finished_reconcile_task(
        self, user_id: str, task: Optional[asyncio.Task] = None
    ) -> None:
        current_task = self._reconcile_tasks.get(user_id)
        if current_task is None:
            return
        if task is not None and current_task is not task:
            return
        if current_task.done() and not current_task.cancelled():
            try:
                exc = current_task.exception()
            except (asyncio.InvalidStateError, asyncio.CancelledError):
                exc = None
            if exc is not None:
                logger.warning(
                    f"Background Mem0 reconcile failed for user {user_id}: {exc}"
                )
        if current_task.done():
            self._reconcile_tasks.pop(user_id, None)

    def _get_reconcile_cooldown_seconds(self) -> float:
        try:
            cooldown = float(
                getattr(self.get_valves(), "mem0_reconcile_cooldown_seconds", 30.0)
            )
        except (TypeError, ValueError):
            cooldown = 30.0
        return max(0.0, cooldown)

    def _render_mem0_user_id(self, user_id: str) -> str:
        template = str(
            getattr(self.get_valves(), "mem0_user_id_template", "owui:{user_id}")
            or "owui:{user_id}"
        )
        try:
            rendered = template.format(user_id=user_id)
        except Exception as e:
            logger.warning(
                f"Invalid mem0_user_id_template '{template}', falling back to raw user_id: {e}"
            )
            rendered = str(user_id)

        rendered = str(rendered).strip()
        return rendered or str(user_id)

    async def _resolve_global_mem0_override(
        self, user_id: str
    ) -> Tuple[Optional[str], Set[str]]:
        raw_override = str(
            getattr(self.get_valves(), "mem0_user_id_override", "") or ""
        ).strip()
        if not raw_override:
            return None, set()

        entries = [
            entry.strip()
            for entry in re.split(r"[\r\n,;]+", raw_override)
            if entry and entry.strip()
        ]
        if not entries:
            return None, set()

        ignored_literal_overrides: Set[str] = set()
        for entry in entries:
            if ":" not in entry:
                ignored_literal_overrides.add(entry)
                logger.warning(
                    f"Ignoring legacy mem0_user_id_override value '{entry}'. This valve now only accepts per-user mappings in the form 'owui_user_id:mem0_user_id'."
                )
                continue

            source_user_id, mapped_mem0_user_id = entry.split(":", 1)
            source_user_id = str(source_user_id).strip()
            mapped_mem0_user_id = str(mapped_mem0_user_id).strip()
            if not source_user_id or not mapped_mem0_user_id:
                logger.warning(
                    f"Ignoring invalid mem0_user_id_override entry '{entry}'; expected 'owui_user_id:mem0_user_id'"
                )
                continue

            try:
                is_known_user = bool(await get_user_by_id_compat(source_user_id))
            except Exception as e:
                logger.warning(
                    f"Failed to validate mem0_user_id_override entry '{entry}' against Open WebUI users: {e}"
                )
                is_known_user = False

            if not is_known_user:
                continue

            if source_user_id == str(user_id):
                return mapped_mem0_user_id, ignored_literal_overrides

        return None, ignored_literal_overrides

    async def _resolve_mem0_user_id(
        self, user_id: str, override: Optional[str] = None
    ) -> str:
        override_value = str(override or "").strip()
        global_override, ignored_literal_overrides = await self._resolve_global_mem0_override(
            user_id
        )
        existing_mapping = await self._get_user_mapping(user_id)
        if override_value:
            mem0_user_id = override_value
        elif global_override:
            mem0_user_id = global_override
        elif existing_mapping and existing_mapping not in ignored_literal_overrides:
            mem0_user_id = existing_mapping
        else:
            mem0_user_id = self._render_mem0_user_id(user_id)

        if existing_mapping != mem0_user_id:
            await self._store_user_mapping(user_id, mem0_user_id)
        return mem0_user_id

    def _build_metadata(
        self,
        user_id: str,
        mem0_user_id: str,
        owui_memory_id: str,
        tags: List[str],
        memory_bank: str,
        confidence: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "source": "adaptive_memory",
            "owui_user_id": str(user_id),
            "mem0_user_id": str(mem0_user_id),
            "owui_memory_id": normalize_memory_id(owui_memory_id),
            "tags": list(tags or []),
            "memory_bank": str(memory_bank or "General"),
            "confidence": self._normalize_confidence(confidence),
        }

    def _normalize_confidence(self, value: Any) -> float:
        try:
            confidence = float(1.0 if value is None else value)
        except (TypeError, ValueError):
            confidence = 1.0
        return max(0.0, min(1.0, confidence))

    def _summarize_payload_for_logs(
        self, payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return a safe, compact summary of a Mem0 payload for debugging."""
        if not isinstance(payload, dict):
            return {"has_payload": False}

        messages = payload.get("messages")
        return {
            "keys": sorted(payload.keys()),
            "has_user_id": bool(str(payload.get("user_id") or "").strip()),
            "has_app_id": bool(str(payload.get("app_id") or "").strip()),
            "has_agent_id": bool(str(payload.get("agent_id") or "").strip()),
            "has_run_id": bool(str(payload.get("run_id") or "").strip()),
            "messages_count": len(messages) if isinstance(messages, list) else 0,
            "infer": payload.get("infer"),
            "async_mode": payload.get("async_mode"),
        }

    async def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        expected_statuses: Optional[Set[int]] = None,
    ) -> Tuple[int, Any]:
        if not self._is_enabled():
            return 0, None

        self._ensure_session()
        expected_statuses = expected_statuses or set()
        request_kwargs: Dict[str, Any] = {
            "headers": self._headers(),
            "timeout": self._timeout(),
        }
        if payload is not None:
            request_kwargs["json"] = payload

        url = f"{self._base_url()}{path}"
        try:
            async with self._session.request(method, url, **request_kwargs) as response:
                text = await response.text()
                data: Any = None
                if text.strip():
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        data = text

                if not (200 <= response.status < 300):
                    payload_summary = self._summarize_payload_for_logs(payload)
                    if response.status in expected_statuses:
                        logger.debug(
                            f"Mem0 mirror {method} {path} returned expected status {response.status}: "
                            f"{truncate_text(text, 300)}"
                        )
                    else:
                        logger.warning(
                            f"Mem0 mirror {method} {path} failed with status {response.status}: "
                            f"{truncate_text(text, 300)}"
                        )
                        logger.warning(
                            f"Mem0 request summary for {method} {path}: {payload_summary}"
                        )
                    if (
                        method == "POST"
                        and path.rstrip("/") == "/v1/memories"
                        and isinstance(data, list)
                        and any(
                            "One of the filters" in str(item)
                            for item in data
                        )
                    ):
                        logger.warning(
                            "Mem0 returned a filter-validation error for a create request. "
                            "This often means the API/proxy treated '/v1/memories' like the retrieval route. "
                            "Retrying or configuring the documented trailing-slash endpoint '/v1/memories/' is recommended."
                        )
                return response.status, data
        except Exception as e:
            logger.warning(f"Mem0 mirror {method} {path} failed: {e}")
            return 0, None

    def _extract_mem0_memory_id(self, response_data: Any) -> Optional[str]:
        if isinstance(response_data, dict):
            if response_data.get("id"):
                return str(response_data["id"])
            results = response_data.get("results")
            if isinstance(results, list):
                return self._extract_mem0_memory_id(results)
        elif isinstance(response_data, list):
            for item in response_data:
                if isinstance(item, dict) and item.get("id"):
                    return str(item["id"])
        return None

    async def _mem0_memory_exists(self, mem0_memory_id: str) -> Optional[bool]:
        if not mem0_memory_id:
            return None

        paths_to_try = [
            f"/v1/memories/{mem0_memory_id}",
            f"/v1/memories/{mem0_memory_id}/",
        ]
        saw_not_found = False

        for path in paths_to_try:
            status, _ = await self._request(
                "GET", path, expected_statuses={404}
            )
            if 200 <= status < 300:
                return True
            if status == 404:
                saw_not_found = True
                continue
            if status in {0, 405}:
                continue
            logger.warning(
                f"Mem0 existence check for memory {mem0_memory_id} returned status {status}; skipping delete reconciliation for now"
            )
            return None

        if saw_not_found:
            return False
        return None

    async def reconcile_deleted_memories(
        self,
        user_id: str,
        local_memory_ids: Set[str],
        delete_local_memory: Callable[[str], Awaitable[bool]],
        force: bool = False,
    ) -> Dict[str, int]:
        result = {
            "checked": 0,
            "deleted": 0,
            "stale_mappings": 0,
            "skipped": 0,
        }
        if not self._is_enabled():
            return result

        lock = self._get_reconcile_lock(user_id)
        try:
            async with lock:
                now = time.monotonic()
                last_check = self._last_reconcile_check.get(user_id, 0.0)
                cooldown_seconds = self._get_reconcile_cooldown_seconds()
                if not force and (now - last_check) < cooldown_seconds:
                    result["skipped"] = 1
                    return result
                self._last_reconcile_check[user_id] = now

                mappings = await self._list_mappings(user_id)
                if not mappings:
                    return result

                for owui_memory_id, mem0_memory_id in mappings:
                    if owui_memory_id not in local_memory_ids:
                        await self._delete_mapping(user_id, owui_memory_id)
                        result["stale_mappings"] += 1
                        continue

                    exists_in_mem0 = await self._mem0_memory_exists(mem0_memory_id)
                    if exists_in_mem0 is None:
                        continue

                    result["checked"] += 1
                    if exists_in_mem0:
                        continue

                    deleted = await delete_local_memory(owui_memory_id)
                    if deleted:
                        local_memory_ids.discard(owui_memory_id)
                        await self._delete_mapping(user_id, owui_memory_id)
                        result["deleted"] += 1
                        logger.info(
                            f"Reconciled deleted Mem0 memory back into Open WebUI (Open WebUI ID: {owui_memory_id}, Mem0 ID: {mem0_memory_id})"
                        )
                    else:
                        logger.warning(
                            f"Mem0 memory {mem0_memory_id} is gone but local cleanup failed for Open WebUI memory {owui_memory_id}"
                        )

                return result
        finally:
            self._cleanup_reconcile_state(user_id)

    def schedule_reconcile_deleted_memories(
        self,
        user_id: str,
        local_memory_ids: Set[str],
        delete_local_memory: Callable[[str], Awaitable[bool]],
        force: bool = False,
    ) -> bool:
        if not self._is_enabled():
            return False

        existing_task = self._reconcile_tasks.get(user_id)
        if existing_task and not existing_task.done():
            return False

        task = asyncio.create_task(
            self.reconcile_deleted_memories(
                user_id=user_id,
                local_memory_ids=set(local_memory_ids),
                delete_local_memory=delete_local_memory,
                force=force,
            )
        )
        self._reconcile_tasks[user_id] = task
        task.add_done_callback(
            lambda finished_task, uid=user_id: self._cleanup_finished_reconcile_task(
                uid, finished_task
            )
        )
        return True

    async def sync_memory_create(
        self,
        user_id: str,
        owui_memory_id: str,
        content: str,
        tags: List[str],
        memory_bank: str,
        confidence: Optional[float],
        mem0_user_id_override: Optional[str] = None,
    ) -> Optional[str]:
        if not self._is_enabled() or not content or not owui_memory_id:
            return None

        existing_mapping = await self._get_mapping(user_id, owui_memory_id)
        if existing_mapping:
            return existing_mapping

        mem0_user_id = await self._resolve_mem0_user_id(
            user_id, override=mem0_user_id_override
        )

        payload = {
            "user_id": mem0_user_id,
            "app_id": str(self.get_valves().mem0_app_id),
            "messages": [{"role": "user", "content": content}],
            "infer": bool(getattr(self.get_valves(), "mem0_infer_on_create", True)),
            "async_mode": False,
            "metadata": self._build_metadata(
                user_id, mem0_user_id, owui_memory_id, tags, memory_bank, confidence
            ),
        }
        status, response_data = await self._request("POST", "/v1/memories/", payload)
        if status in {404, 405}:
            logger.warning(
                "Mem0 create request to '/v1/memories/' was not accepted; retrying legacy path '/v1/memories'"
            )
            status, response_data = await self._request(
                "POST", "/v1/memories", payload
            )
        if not (200 <= status < 300):
            return None

        mem0_memory_id = self._extract_mem0_memory_id(response_data)
        if mem0_memory_id:
            await self._store_mapping(user_id, owui_memory_id, mem0_memory_id)
            logger.info(
                f"Mirrored memory to Mem0 (Open WebUI ID: {owui_memory_id}, Mem0 ID: {mem0_memory_id})"
            )
        else:
            logger.warning(
                f"Mem0 mirror add succeeded but no memory ID was returned for Open WebUI memory {owui_memory_id}"
            )
        return mem0_memory_id

    async def sync_memory_update(
        self,
        user_id: str,
        owui_memory_id: str,
        content: str,
        tags: List[str],
        memory_bank: str,
        confidence: Optional[float],
        mem0_user_id_override: Optional[str] = None,
    ) -> bool:
        if not self._is_enabled() or not content or not owui_memory_id:
            return False

        mem0_memory_id = await self._get_mapping(user_id, owui_memory_id)
        if not mem0_memory_id:
            logger.info(
                f"Mem0 mapping missing for memory {owui_memory_id}; creating a fresh mirrored record"
            )
            created_id = await self.sync_memory_create(
                user_id,
                owui_memory_id,
                content,
                tags,
                memory_bank,
                confidence,
                mem0_user_id_override=mem0_user_id_override,
            )
            return created_id is not None

        mem0_user_id = await self._resolve_mem0_user_id(
            user_id, override=mem0_user_id_override
        )

        payload = {
            "text": content,
            "metadata": self._build_metadata(
                user_id, mem0_user_id, owui_memory_id, tags, memory_bank, confidence
            ),
        }
        status, _ = await self._request(
            "PUT", f"/v1/memories/{mem0_memory_id}", payload
        )
        if 200 <= status < 300:
            logger.info(
                f"Updated mirrored Mem0 memory (Open WebUI ID: {owui_memory_id}, Mem0 ID: {mem0_memory_id})"
            )
            return True

        if status == 404:
            logger.warning(
                f"Mem0 mirror mapping was stale for Open WebUI memory {owui_memory_id}; recreating mirror"
            )
            await self._delete_mapping(user_id, owui_memory_id)
            created_id = await self.sync_memory_create(
                user_id,
                owui_memory_id,
                content,
                tags,
                memory_bank,
                confidence,
                mem0_user_id_override=mem0_user_id_override,
            )
            return created_id is not None

        return False

    async def sync_memory_delete(self, user_id: str, owui_memory_id: str) -> bool:
        if not self._is_enabled() or not owui_memory_id:
            return False

        mem0_memory_id = await self._get_mapping(user_id, owui_memory_id)
        if not mem0_memory_id:
            return False

        status, _ = await self._request(
            "DELETE",
            f"/v1/memories/{mem0_memory_id}",
            expected_statuses={404},
        )
        if 200 <= status < 300 or status == 404:
            await self._delete_mapping(user_id, owui_memory_id)
            logger.info(
                f"Deleted mirrored Mem0 memory (Open WebUI ID: {owui_memory_id}, Mem0 ID: {mem0_memory_id})"
            )
            return True

        return False

    async def enqueue_memory_upsert(
        self,
        user_id: str,
        owui_memory_id: str,
        content: str,
        tags: List[str],
        memory_bank: str,
        confidence: Optional[float],
        mem0_user_id_override: Optional[str] = None,
    ) -> bool:
        if not self._is_enabled() or not content or not owui_memory_id:
            return False

        payload = {
            "content": str(content),
            "tags": list(tags or []),
            "memory_bank": str(memory_bank or "General"),
            "confidence": self._normalize_confidence(confidence),
            "mem0_user_id_override": str(mem0_user_id_override or "").strip(),
        }
        await self._enqueue_job(
            user_id=user_id,
            owui_memory_id=owui_memory_id,
            operation="UPSERT",
            payload=payload,
        )
        logger.debug(
            f"Queued Mem0 UPSERT for Open WebUI memory {normalize_memory_id(owui_memory_id)}"
        )
        return True

    async def enqueue_memory_delete(self, user_id: str, owui_memory_id: str) -> bool:
        if not self._is_enabled() or not owui_memory_id:
            return False

        await self._enqueue_job(
            user_id=user_id,
            owui_memory_id=owui_memory_id,
            operation="DELETE",
            payload=None,
        )
        logger.debug(
            f"Queued Mem0 DELETE for Open WebUI memory {normalize_memory_id(owui_memory_id)}"
        )
        return True

    async def _process_upsert_job(self, job: Dict[str, Any]) -> bool:
        payload = job.get("payload")
        if not isinstance(payload, dict):
            logger.warning(
                f"Mem0 queued UPSERT for memory {job.get('owui_memory_id')} has no usable payload; dropping job"
            )
            return True

        content = str(payload.get("content") or "").strip()
        if not content:
            logger.warning(
                f"Mem0 queued UPSERT for memory {job.get('owui_memory_id')} has empty content; dropping job"
            )
            return True

        return await self.sync_memory_update(
            user_id=str(job["user_id"]),
            owui_memory_id=str(job["owui_memory_id"]),
            content=content,
            tags=list(payload.get("tags") or []),
            memory_bank=str(payload.get("memory_bank") or "General"),
            confidence=payload.get("confidence"),
            mem0_user_id_override=str(
                payload.get("mem0_user_id_override") or ""
            ).strip()
            or None,
        )

    async def _process_delete_job(self, job: Dict[str, Any]) -> bool:
        user_id = str(job["user_id"])
        owui_memory_id = str(job["owui_memory_id"])
        existing_mapping = await self._get_mapping(user_id, owui_memory_id)
        if not existing_mapping:
            logger.debug(
                f"Skipping queued Mem0 DELETE for {owui_memory_id}; no mirror mapping exists"
            )
            return True
        return await self.sync_memory_delete(user_id, owui_memory_id)

    async def process_sync_queue_batch(self) -> Dict[str, int]:
        result = {
            "fetched": 0,
            "processed": 0,
            "succeeded": 0,
            "retried": 0,
        }
        if not self._is_enabled():
            return result

        jobs = await self._fetch_ready_jobs(self._get_sync_batch_size())
        result["fetched"] = len(jobs)
        if not jobs:
            return result

        for job in jobs:
            operation = str(job.get("operation") or "").upper()
            job_error = "Unknown error"
            success = False
            result["processed"] += 1
            try:
                if operation == "UPSERT":
                    success = await self._process_upsert_job(job)
                elif operation == "DELETE":
                    success = await self._process_delete_job(job)
                else:
                    logger.warning(
                        f"Mem0 queued job for memory {job.get('owui_memory_id')} has unsupported operation '{operation}'; dropping job"
                    )
                    success = True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                job_error = str(e)
                logger.warning(
                    f"Mem0 queued job {operation} failed for memory {job.get('owui_memory_id')}: {e}"
                )

            if success:
                await self._delete_job(
                    str(job["user_id"]), str(job["owui_memory_id"])
                )
                result["succeeded"] += 1
                continue

            retry_at = datetime.now(timezone.utc) + timedelta(
                seconds=self._get_sync_retry_delay_seconds()
            )
            await self._reschedule_job(
                str(job["user_id"]),
                str(job["owui_memory_id"]),
                int(job.get("attempt_count", 0)) + 1,
                retry_at,
                job_error,
            )
            result["retried"] += 1

        return result

    async def run_sync_loop(self) -> None:
        logger.info(
            f"Mem0 background sync loop started with interval: {self._get_sync_batch_interval_seconds()} seconds"
        )
        try:
            while True:
                await asyncio.sleep(self._get_sync_batch_interval_seconds())
                processed = 0
                succeeded = 0
                retried = 0
                batches = 0

                while True:
                    result = await self.process_sync_queue_batch()
                    if result["fetched"] == 0:
                        break

                    batches += 1
                    processed += result["processed"]
                    succeeded += result["succeeded"]
                    retried += result["retried"]

                    if result["fetched"] < self._get_sync_batch_size():
                        break

                if processed > 0 or retried > 0:
                    logger.info(
                        "Mem0 background sync cycle complete: "
                        f"processed={processed}, "
                        f"succeeded={succeeded}, "
                        f"retried={retried}, "
                        f"batches={batches}"
                    )
        except asyncio.CancelledError:
            logger.info("Mem0 background sync loop stopped")
            raise


# ------------------------------------------------------------------------------
# Memory Pipeline
# ------------------------------------------------------------------------------


class MemoryPipeline:
    """Core logic for extracting, retrieving, and processing memories."""

    def __init__(
        self,
        valves: Any,
        embedding_manager: EmbeddingManager,
        error_manager: ErrorManager,
        mem0_sync_manager: Optional[Mem0SyncManager] = None,
    ):
        self.valves = valves
        self.embedding_manager = embedding_manager
        self.error_manager = error_manager
        self.mem0_sync_manager = mem0_sync_manager

    def _get_mem0_user_id_override(self, user_valves: Any = None) -> Optional[str]:
        if user_valves is None:
            return None
        override = getattr(user_valves, "mem0_user_id_override", "")
        override = str(override or "").strip()
        return override or None

    def _get_memory_id(self, memory: Any) -> Optional[str]:
        return extract_memory_id(memory)

    def _get_memory_record(self, memory: Any) -> StoredMemoryRecord:
        memory_content = memory.content if hasattr(memory, "content") else memory.get("content")
        return parse_stored_memory(memory_content)

    async def _get_user_object(self, user_id: str) -> Optional[Any]:
        try:
            return await get_user_by_id_compat(user_id)
        except Exception as e:
            logger.warning(f"Failed to fetch user object for {user_id}: {e}")
            return None

    def _log_memory_save_user_id(self, user_id: str, memory_id: Optional[str]) -> None:
        if not getattr(self.valves, "log_user_id_on_memory_save", False):
            return
        logger.info(
            f"Memory save user_id={user_id} memory_id={normalize_memory_id(memory_id) if memory_id is not None else 'unknown'}"
        )

    async def _mirror_memory_upsert(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        tags: List[str],
        memory_bank: str,
        confidence: Optional[float],
        mem0_user_id_override: Optional[str] = None,
        log_context: str = "Memory",
    ) -> None:
        if not self.mem0_sync_manager or not memory_id or not content:
            return

        if self.mem0_sync_manager.should_use_background_sync():
            queued = await self.mem0_sync_manager.enqueue_memory_upsert(
                user_id=user_id,
                owui_memory_id=memory_id,
                content=content,
                tags=tags,
                memory_bank=memory_bank,
                confidence=confidence,
                mem0_user_id_override=mem0_user_id_override,
            )
            if queued:
                return

        await self.mem0_sync_manager.sync_memory_update(
            user_id=user_id,
            owui_memory_id=memory_id,
            content=content,
            tags=tags,
            memory_bank=memory_bank,
            confidence=confidence,
            mem0_user_id_override=mem0_user_id_override,
        )

    async def _mirror_memory_delete(
        self,
        user_id: str,
        memory_id: str,
        log_context: str = "Memory",
    ) -> None:
        if not self.mem0_sync_manager or not memory_id:
            return

        if self.mem0_sync_manager.should_use_background_sync():
            queued = await self.mem0_sync_manager.enqueue_memory_delete(
                user_id, memory_id
            )
            if queued:
                return

        deleted = await self.mem0_sync_manager.sync_memory_delete(user_id, memory_id)
        if not deleted:
            logger.debug(
                f"{log_context}: No Mem0 mapping found while deleting memory {memory_id}"
            )

    async def _delete_local_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        mirror_to_mem0: bool = True,
        log_context: str = "Memory",
    ) -> bool:
        memory_id = normalize_memory_id(memory_id)
        try:
            deleted = await delete_memory_by_id_and_user_id_compat(memory_id, user_id)
        except Exception as e:
            logger.exception(f"{log_context}: Failed to delete memory {memory_id}: {e}")
            return False
        if not deleted:
            logger.warning(
                f"{log_context}: Memory {memory_id} was not deleted because it was missing, belonged to another user, or the delete did not complete"
            )
            return False

        if VECTOR_DB_CLIENT:
            try:
                VECTOR_DB_CLIENT.delete(
                    collection_name=f"user-memory-{user_id}",
                    ids=[memory_id],
                )
                logger.debug(f"{log_context}: Deleted memory {memory_id} from vector DB")
            except Exception as vec_err:
                logger.warning(
                    f"{log_context}: Failed to delete memory {memory_id} from vector DB: {vec_err}"
                )

        await self.embedding_manager.delete_embedding_persistent(user_id, memory_id)

        if mirror_to_mem0 and self.mem0_sync_manager:
            try:
                await self._mirror_memory_delete(
                    user_id,
                    memory_id,
                    log_context=log_context,
                )
            except Exception as mem0_err:
                logger.warning(
                    f"{log_context}: Mem0 mirror delete failed for memory {memory_id}: {mem0_err}"
                )

        logger.info(f"{log_context}: Deleted memory {memory_id}")
        return True

    async def reconcile_mem0_deleted_memories(
        self, user_id: str, all_memories: List[Any]
    ) -> List[Any]:
        if not self.mem0_sync_manager:
            return all_memories

        local_memory_ids = {
            memory_id
            for memory_id in (self._get_memory_id(memory) for memory in all_memories)
            if memory_id is not None
        }
        if not self.mem0_sync_manager.should_reconcile_in_request_path():
            scheduled = self.mem0_sync_manager.schedule_reconcile_deleted_memories(
                user_id=user_id,
                local_memory_ids=local_memory_ids,
                delete_local_memory=lambda memory_id: self._delete_local_memory(
                    user_id,
                    memory_id,
                    mirror_to_mem0=False,
                    log_context="Mem0 reconcile",
                ),
            )
            if scheduled:
                logger.debug(
                    f"Mem0 reconcile scheduled in background for user {user_id}"
                )
            return all_memories

        reconciliation = await self.mem0_sync_manager.reconcile_deleted_memories(
            user_id=user_id,
            local_memory_ids=local_memory_ids,
            delete_local_memory=lambda memory_id: self._delete_local_memory(
                user_id,
                memory_id,
                mirror_to_mem0=False,
                log_context="Mem0 reconcile",
            ),
        )
        if reconciliation["deleted"] > 0:
            try:
                refreshed_memories = await get_memories_by_user_id_compat(user_id)
                logger.info(
                    f"Mem0 reconcile: removed {reconciliation['deleted']} stale local memories for user {user_id}"
                )
                return refreshed_memories
            except Exception as e:
                logger.warning(
                    f"Mem0 reconcile: deleted local memories but failed to refresh memory list for user {user_id}: {e}"
                )
                remaining_ids = local_memory_ids
                return [
                    memory
                    for memory in all_memories
                    if self._get_memory_id(memory) in remaining_ids
                ]

        if reconciliation["stale_mappings"] > 0:
            logger.debug(
                f"Mem0 reconcile: pruned {reconciliation['stale_mappings']} stale mirror mappings for user {user_id}"
            )
        return all_memories

    def _find_memory_by_exact_content(
        self, memories: List[Any], content: str, excluded_ids: Optional[Set[str]] = None
    ) -> Optional[Any]:
        excluded_ids = excluded_ids or set()
        for memory in memories:
            memory_id = self._get_memory_id(memory)
            if memory_id and memory_id in excluded_ids:
                continue
            stored_content = (
                memory.content if hasattr(memory, "content") else memory.get("content")
            )
            if str(stored_content or "") == content:
                return memory
        return None

    async def _sync_memory_vector(
        self, user_id: str, memory: Any, text_for_vector: str, user_obj: Any = None
    ) -> bool:
        if not VECTOR_DB_CLIENT:
            return False

        memory_id = self._get_memory_id(memory)
        if not memory_id:
            return False

        vector_embedding = await self.embedding_manager.get_embedding(
            text_for_vector, user=user_obj
        )
        if vector_embedding is None:
            logger.warning(
                f"Could not generate embedding for vector sync (memory ID: {memory_id})"
            )
            return False

        metadata = {}
        created_at = getattr(memory, "created_at", None)
        updated_at = getattr(memory, "updated_at", None)
        if created_at is not None:
            metadata["created_at"] = created_at
        if updated_at is not None:
            metadata["updated_at"] = updated_at

        try:
            VECTOR_DB_CLIENT.upsert(
                collection_name=f"user-memory-{user_id}",
                items=[
                    {
                        "id": memory_id,
                        "text": text_for_vector,
                        "vector": (
                            vector_embedding.tolist()
                            if hasattr(vector_embedding, "tolist")
                            else vector_embedding
                        ),
                        "metadata": metadata,
                    }
                ],
            )
            logger.info(f"Memory synced to vector DB (ID: {memory_id})")
            return True
        except Exception as e:
            logger.warning(f"Failed to sync memory to vector DB (ID: {memory_id}): {e}")
            return False

    def _coerce_created_at(self, created_at: Any) -> Optional[datetime]:
        if created_at is None:
            return None

        if isinstance(created_at, datetime):
            return (
                created_at
                if created_at.tzinfo is not None
                else created_at.replace(tzinfo=timezone.utc)
            )

        if isinstance(created_at, (int, float)):
            timestamp = float(created_at)
            candidate_timestamps = [timestamp]
            if abs(timestamp) > 1e11:
                candidate_timestamps.insert(0, timestamp / 1000.0)

            for candidate in candidate_timestamps:
                try:
                    return datetime.fromtimestamp(candidate, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    continue

            return None

        if isinstance(created_at, str):
            normalized = created_at.strip()
            if not normalized:
                return None

            try:
                parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
                return (
                    parsed
                    if parsed.tzinfo is not None
                    else parsed.replace(tzinfo=timezone.utc)
                )
            except ValueError:
                try:
                    return self._coerce_created_at(float(normalized))
                except ValueError:
                    return None

        return None

    def _memory_created_at_sort_key(self, memory: Any) -> datetime:
        created_at = (
            getattr(memory, "created_at", None)
            if hasattr(memory, "created_at")
            else memory.get("created_at")
        )

        normalized_created_at = self._coerce_created_at(created_at)
        if normalized_created_at is not None:
            return normalized_created_at

        return datetime.min.replace(tzinfo=timezone.utc)

    def _enabled_memory_tags(self) -> Set[str]:
        enabled_tags = {"summary"}
        tag_flag_map = {
            "identity": self.valves.enable_identity_memories,
            "behavior": self.valves.enable_behavior_memories,
            "preference": self.valves.enable_preference_memories,
            "goal": self.valves.enable_goal_memories,
            "relationship": self.valves.enable_relationship_memories,
            "possession": self.valves.enable_possession_memories,
        }
        for tag, enabled in tag_flag_map.items():
            if enabled:
                enabled_tags.add(tag)
        return enabled_tags

    def _normalize_tags(self, tags: Any) -> List[str]:
        if tags is None:
            return []
        if not isinstance(tags, list):
            tags = [tags]

        enabled_tags = self._enabled_memory_tags()
        cleaned_tags = []
        seen = set()
        for tag in tags:
            normalized_tag = str(tag).strip().lower()
            if (
                normalized_tag
                and normalized_tag in SUPPORTED_MEMORY_TAGS
                and normalized_tag in enabled_tags
                and normalized_tag not in seen
            ):
                seen.add(normalized_tag)
                cleaned_tags.append(normalized_tag)
        return cleaned_tags

    def _normalize_memory_bank(self, memory_bank: Any) -> str:
        allowed_banks = [
            str(bank).strip() for bank in self.valves.allowed_memory_banks if str(bank).strip()
        ]
        default_bank = str(self.valves.default_memory_bank or "General").strip() or "General"

        if not allowed_banks:
            return default_bank

        requested_bank = str(memory_bank or "").strip()
        for allowed_bank in allowed_banks:
            if requested_bank.lower() == allowed_bank.lower():
                return allowed_bank

        for allowed_bank in allowed_banks:
            if default_bank.lower() == allowed_bank.lower():
                return allowed_bank

        return allowed_banks[0]

    def _normalize_confidence(self, value: Any, default: float = 0.0) -> float:
        try:
            confidence = float(default if value is None else value)
        except (TypeError, ValueError):
            confidence = float(default)
        return max(0.0, min(1.0, confidence))

    def _parse_csv_keywords(self, value: Optional[str]) -> Set[str]:
        if not value:
            return set()
        return {item.strip().lower() for item in str(value).split(",") if item.strip()}

    def _has_whitelist_keyword(self, content: str) -> bool:
        lowered = content.lower()
        return any(
            keyword in lowered for keyword in self._parse_csv_keywords(self.valves.whitelist_keywords)
        )

    def _looks_like_trivia(self, content: str) -> bool:
        lowered = content.strip().lower()
        if not lowered:
            return False
        trivia_patterns = [
            r"^(what|who|where|when|why|how)\b",
            r"\b(the capital of|world war|boiling point|photosynthesis|periodic table)\b",
        ]
        return lowered.endswith("?") or any(
            re.search(pattern, lowered) for pattern in trivia_patterns
        )

    def _passes_memory_filters(self, content: str) -> bool:
        if not content:
            return False

        if self._has_whitelist_keyword(content):
            return True

        lowered = content.lower()
        blacklist_topics = self._parse_csv_keywords(self.valves.blacklist_topics)
        if blacklist_topics and any(topic in lowered for topic in blacklist_topics):
            return False

        if self.valves.filter_trivia and self._looks_like_trivia(content):
            return False

        return True

    def _should_skip_dedupe_for_short_preference(self, content: str) -> bool:
        if not content or len(content) > self.valves.short_preference_no_dedupe_length:
            return False

        lowered = content.lower()
        preference_keywords = self._parse_csv_keywords(
            self.valves.preference_keywords_no_dedupe
        )
        if not preference_keywords:
            return False

        has_first_person_marker = bool(re.search(r"\b(i|i'm|im|my)\b", lowered))
        return has_first_person_marker and any(
            keyword in lowered for keyword in preference_keywords
        )

    def _extract_fallback_operations(self, response: str) -> List[Dict[str, Any]]:
        operations = []
        for match in re.finditer(
            r'"operation"\s*:\s*"(?P<operation>NEW|UPDATE|DELETE)"(?P<body>.*?)(?=\{?"operation"\s*:|$)',
            response,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            body = match.group("body")
            op = {"operation": match.group("operation").upper()}

            id_match = re.search(r'"id"\s*:\s*"(?P<id>(?:\\.|[^"\\])*)"', body)
            if id_match:
                op["id"] = bytes(id_match.group("id"), "utf-8").decode("unicode_escape")

            content_match = re.search(
                r'"content"\s*:\s*"(?P<content>(?:\\.|[^"\\])*)"', body
            )
            if content_match:
                op["content"] = bytes(
                    content_match.group("content"), "utf-8"
                ).decode("unicode_escape")

            tags_match = re.search(r'"tags"\s*:\s*\[(?P<tags>[^\]]*)\]', body)
            if tags_match:
                op["tags"] = re.findall(r'"([^"]+)"', tags_match.group("tags"))

            bank_match = re.search(
                r'"memory_bank"\s*:\s*"(?P<memory_bank>(?:\\.|[^"\\])*)"', body
            )
            if bank_match:
                op["memory_bank"] = bytes(
                    bank_match.group("memory_bank"), "utf-8"
                ).decode("unicode_escape")

            confidence_match = re.search(
                r'"confidence"\s*:\s*(?P<confidence>-?\d+(?:\.\d+)?)', body
            )
            if confidence_match:
                try:
                    op["confidence"] = float(confidence_match.group("confidence"))
                except ValueError:
                    pass

            operations.append(op)
        return operations

    def _normalize_operation(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        operation = str(item.get("operation", "")).upper().strip()
        if operation not in {"NEW", "UPDATE", "DELETE"}:
            return None

        if operation == "DELETE":
            memory_id = item.get("id")
            if not memory_id:
                return None
            return {"operation": "DELETE", "id": normalize_memory_id(memory_id)}

        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not content or len(content) < self.valves.min_memory_length:
            return None
        if not self._passes_memory_filters(content):
            return None

        confidence = self._normalize_confidence(item.get("confidence"), default=0.0)
        if confidence < self.valves.min_confidence_threshold:
            return None

        tags = self._normalize_tags(item.get("tags", []))
        if item.get("tags") and not tags:
            return None

        normalized_op = {
            "operation": operation,
            "content": content,
            "tags": tags,
            "memory_bank": self._normalize_memory_bank(item.get("memory_bank")),
            "confidence": confidence,
        }

        if operation == "UPDATE" and item.get("id"):
            normalized_op["id"] = normalize_memory_id(item["id"])

        return normalized_op

    def _build_short_preference_operation(self, user_message: str) -> Optional[Dict[str, Any]]:
        if not self.valves.enable_short_preference_shortcut:
            return None

        content = re.sub(r"\s+", " ", user_message).strip()
        if not self._should_skip_dedupe_for_short_preference(content):
            return None

        return self._normalize_operation(
            {
                "operation": "NEW",
                "content": content,
                "tags": ["preference"],
                "memory_bank": self._normalize_memory_bank(self.valves.default_memory_bank),
                "confidence": 0.95,
            }
        )

    # --- Memory Identification ---
    async def identify_memories(
        self,
        user_message: str,
        context_memories: Optional[List[Dict[str, Any]]] = None,
        query_llm_func: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Identify potential memories from user message using LLM."""
        if not user_message:
            return []

        # Construct prompt
        system_prompt = self.valves.memory_identification_prompt
        now = datetime.now(timezone.utc)
        system_prompt += f"\n\nCurrent Date: {now.strftime('%Y-%m-%d %H:%M:%S')}"

        user_prompt = f"User Message: {user_message}"
        if context_memories:
            context_lines = []
            for memory in context_memories:
                memory_id = self._get_memory_id(memory)
                memory_record = self._get_memory_record(memory)
                if not memory_id or not memory_record.content:
                    continue

                metadata = []
                if memory_record.memory_bank:
                    metadata.append(f"bank={memory_record.memory_bank}")
                if memory_record.tags:
                    metadata.append(f"tags={', '.join(memory_record.tags)}")
                metadata_suffix = f" ({'; '.join(metadata)})" if metadata else ""
                context_lines.append(
                    f"- ID: {memory_id}{metadata_suffix} | Content: {memory_record.content}"
                )

            if context_lines:
                user_prompt += (
                    "\n\nRelevant Existing Memories (use these IDs for UPDATE/DELETE when appropriate):\n"
                    + "\n".join(context_lines)
                )

        fallback_operation = self._build_short_preference_operation(user_message)

        # Call LLM
        if not query_llm_func:
            return [fallback_operation] if fallback_operation else []

        try:
            response = await query_llm_func(system_prompt, user_prompt)
            parsed_operations = []

            if response:
                data = (
                    JSONParser.extract_and_parse(response)
                    if self.valves.enable_json_stripping
                    else json.loads(response)
                )
                if isinstance(data, list):
                    parsed_operations = data
                elif isinstance(data, dict):
                    parsed_operations = [data]
                elif self.valves.enable_fallback_regex:
                    parsed_operations = self._extract_fallback_operations(response)
            elif fallback_operation:
                return [fallback_operation]
            else:
                return []

            valid_ops = []
            for item in parsed_operations:
                if not isinstance(item, dict):
                    continue
                normalized_op = self._normalize_operation(item)
                if normalized_op:
                    valid_ops.append(normalized_op)

            if valid_ops:
                return valid_ops

            return [fallback_operation] if fallback_operation else []

        except Exception as e:
            self.error_manager.increment("json_parse_errors")
            logger.warning(f"Identify memories parsing failed, attempting fallback: {e}")
            if fallback_operation:
                return [fallback_operation]
            self.error_manager.increment("llm_call_errors")
            logger.exception(f"Identify memories failed: {e}")
            return []

    # --- Relevance Retrieval ---
    async def get_relevant_memories(
        self, query: str, user_id: str, all_memories: List[Any]
    ) -> List[Any]:
        """Retrieve relevant memories using vector similarity + optional LLM ranking."""
        if not query or not all_memories:
            return []

        RETRIEVAL_REQUESTS.inc()
        start_time = time.perf_counter()
        user_obj = await self._get_user_object(user_id)

        # 1. Vector Search
        try:
            query_embedding = await self.embedding_manager.get_embedding(
                query, user=user_obj
            )
            if query_embedding is None:
                return []
        except Exception as e:
            RETRIEVAL_ERRORS.inc()
            logger.exception(f"Error generating query embedding: {e}")
            return []

        scored_memories = []

        # Batch embedding for memories without cached embeddings
        # This assumes all_memories are custom objects or dicts.
        # OpenWebUI Memories are SQLModel objects usually.
        mem_objects = []
        texts_to_embed = []
        ids_to_embed = []

        for mem in all_memories:
            memory_record = self._get_memory_record(mem)
            mem_content = memory_record.content
            mem_id = self._get_memory_id(mem)

            if not mem_id or not mem_content:
                continue

            # Check in-memory cache first
            cached_emb = await self.embedding_manager.cache.get(mem_id)
            if cached_emb is not None:
                sim = self._cosine_similarity(query_embedding, cached_emb)
                if sim >= self.valves.vector_similarity_threshold:
                    scored_memories.append((sim, mem))
            else:
                # Check persistent cache
                persistent_emb = await self.embedding_manager.load_embedding_persistent(user_id, mem_id)
                if persistent_emb is not None:
                    # Cache in memory for this session
                    await self.embedding_manager.cache.set(mem_id, persistent_emb)
                    sim = self._cosine_similarity(query_embedding, persistent_emb)
                    if sim >= self.valves.vector_similarity_threshold:
                        scored_memories.append((sim, mem))
                else:
                    # Need to generate embedding
                    mem_objects.append(mem)
                    texts_to_embed.append(mem_content)
                    ids_to_embed.append(mem_id)

        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} memories (using cache for {len(all_memories) - len(texts_to_embed)})")
            # Batch generate
            new_embeddings = await self.embedding_manager.get_embeddings_batch(
                texts_to_embed, user=user_obj
            )
            for i, emb in enumerate(new_embeddings):
                if emb is not None:
                    # Update in-memory cache
                    await self.embedding_manager.cache.set(ids_to_embed[i], emb)
                    # Score
                    sim = self._cosine_similarity(query_embedding, emb)
                    if sim >= self.valves.vector_similarity_threshold:
                        scored_memories.append((sim, mem_objects[i]))
            
            # Store all newly generated embeddings persistently in one go
            if any(e is not None for e in new_embeddings):
                await self.embedding_manager.store_embeddings_batch_persistent(
                    user_id, ids_to_embed, texts_to_embed, new_embeddings
                )
        else:
            logger.info(f"Using cached embeddings for all {len(all_memories)} memories")

        # Sort by similarity
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [
            mem
            for sim, mem in scored_memories
            if sim >= self.valves.relevance_threshold
        ][: self.valves.related_memories_n]
        
        RETRIEVAL_LATENCY.observe(time.perf_counter() - start_time)
        return top_memories

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.shape != v2.shape:
            logger.debug(f"Cosine similarity dimension mismatch: {v1.shape} vs {v2.shape}")
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing punctuation, articles, intensifiers, and extra spaces."""
        # Remove punctuation, extra spaces, convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', text.strip().lower())
        
        # Handle common plural variations
        normalized = re.sub(r'\bs\b', '', normalized)  # Remove standalone 's'
        
        # Remove articles (a, an, the)
        normalized = re.sub(r'\b(a|an|the)\b', '', normalized)
        
        # Remove intensifiers (but keep adjectives like 'cold', 'hot')
        normalized = re.sub(r'\b(really|very|quite|pretty|so|totally|absolutely)\b', '', normalized)
        
        # Clean up extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    # --- Memory Operations ---
    async def process_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        skip_deduplication: bool = False,
        user_valves: Any = None,
    ) -> List[Dict[str, Any]]:
        """Execute valid memory operations (NEW, UPDATE, DELETE)."""
        # Fetch full user object for Router DI (MockRequest)
        user_obj = await get_user_by_id_compat(user_id)
        mem0_user_id_override = self._get_mem0_user_id_override(user_valves)
        success_ops = []
        for op in operations:
            try:
                normalized_op = self._normalize_operation(op)
                if normalized_op is None:
                    logger.debug(f"Skipping invalid memory operation payload: {op}")
                    continue
                kind = normalized_op.get("operation")
                content = normalized_op.get("content")

                if kind == "NEW" and content:
                    tags = self._normalize_tags(normalized_op.get("tags", []))
                    bank = self._normalize_memory_bank(normalized_op.get("memory_bank"))
                    confidence = self._normalize_confidence(
                        normalized_op.get("confidence"), default=1.0
                    )

                    dedup_embedding = None
                    skip_preference_dedupe = self._should_skip_dedupe_for_short_preference(content)
                    if (
                        self.valves.deduplicate_memories
                        and not skip_deduplication
                        and not skip_preference_dedupe
                    ):
                        is_dupe, dedup_embedding = await self._is_duplicate(content, user_id)
                        if is_dupe:
                            logger.info(f"Skipping duplicate memory (length: {len(content)})")
                            continue

                    final_content = format_memory_content(content, tags, bank, confidence)

                    try:
                        mem_obj = None
                        vector_sync_needed = False
                        existing_memory_ids = {
                            memory_id
                            for memory_id in (
                                self._get_memory_id(memory)
                                for memory in await get_memories_by_user_id_compat(user_id)
                            )
                            if memory_id is not None
                        }
                        try:
                            logger.info("Attempting to add memory via router add_memory (Vector-Aware)...")

                            if add_memory and (AddMemoryForm or LocalAddMemoryForm):
                                FormClass = AddMemoryForm if AddMemoryForm else LocalAddMemoryForm
                                form = FormClass(content=final_content)

                                async def mock_embedding_function(content: str, user=None):
                                    return await self.embedding_manager.get_embedding(
                                        content, user=user
                                    )

                                req = (
                                    MockRequest(user_obj, mock_embedding_function)
                                    if "MockRequest" in globals()
                                    else None
                                )
                                if req:
                                    mem_obj = await add_memory(
                                        request=req, form_data=form, user=user_obj
                                    )
                                else:
                                    raise ImportError("MockRequest not available")
                            else:
                                raise ImportError("Router add_memory not successfully imported")

                        except Exception as add_err:
                            logger.warning(
                                f"Router add_memory failed ({add_err}), falling back to insert_new_memory"
                            )
                            current_memories = await get_memories_by_user_id_compat(user_id)
                            mem_obj = self._find_memory_by_exact_content(
                                current_memories,
                                final_content,
                                excluded_ids=existing_memory_ids,
                            )
                            if mem_obj is not None:
                                vector_sync_needed = True
                                logger.warning(
                                    "Router add_memory inserted the memory before vector sync failed; reusing that row to avoid creating a duplicate"
                                )
                            else:
                                mem_obj = await insert_new_memory_compat(
                                    user_id, final_content
                                )
                                vector_sync_needed = True

                        memory_id = self._get_memory_id(mem_obj)
                        if mem_obj is None:
                            raise ValueError("Memory insert returned no record")
                        if memory_id is None:
                            raise ValueError("Memory insert returned a record without an ID")
                        if vector_sync_needed:
                            await self._sync_memory_vector(
                                user_id, mem_obj, final_content, user_obj=user_obj
                            )
                        success_ops.append(normalized_op)
                        logger.info(
                            f"Memory saved (ID: {memory_id}) [Bank: {bank}] [Confidence: {confidence:.2f}]"
                        )
                        self._log_memory_save_user_id(user_id, memory_id)

                        if dedup_embedding is not None and memory_id:
                            memory_key = normalize_memory_id(memory_id)
                            logger.debug(
                                f"Caching embedding from deduplication check for memory {memory_key}"
                            )
                            await self.embedding_manager.cache.set(memory_key, dedup_embedding)
                            await self.embedding_manager.store_embedding_persistent(
                                user_id, memory_key, content, dedup_embedding
                            )

                        if self.mem0_sync_manager and memory_id:
                            try:
                                await self._mirror_memory_upsert(
                                    user_id=user_id,
                                    memory_id=memory_id,
                                    content=content,
                                    tags=tags,
                                    memory_bank=bank,
                                    confidence=confidence,
                                    mem0_user_id_override=mem0_user_id_override,
                                )
                            except Exception as mem0_err:
                                logger.warning(
                                    f"Mem0 mirror create failed for memory {memory_id}: {mem0_err}"
                                )
                    except Exception as ins_err:
                        logger.error(f"Failed to insert memory: {ins_err}")

                elif kind == "UPDATE" and normalized_op.get("id") and content:
                    try:
                        memory_id = normalize_memory_id(normalized_op["id"])
                        new_content = content
                        existing_memories = await get_memories_by_user_id_compat(user_id)
                        existing_memory = next(
                            (
                                memory
                                for memory in existing_memories
                                if self._get_memory_id(memory) == memory_id
                            ),
                            None,
                        )

                        if not existing_memory:
                            logger.warning(f"Memory not found for update (ID: {memory_id})")
                            continue

                        existing_record = self._get_memory_record(existing_memory)
                        tags = self._normalize_tags(
                            normalized_op.get("tags", existing_record.tags)
                        ) or existing_record.tags
                        bank = self._normalize_memory_bank(
                            normalized_op.get("memory_bank", existing_record.memory_bank)
                        )
                        confidence = self._normalize_confidence(
                            normalized_op.get("confidence", existing_record.confidence),
                            default=existing_record.confidence or 1.0,
                        )

                        new_embedding = None
                        skip_preference_dedupe = self._should_skip_dedupe_for_short_preference(
                            new_content
                        )
                        if (
                            self.valves.deduplicate_memories
                            and not skip_preference_dedupe
                        ):
                            is_dupe, new_embedding = await self._is_duplicate(
                                new_content, user_id, exclude_id=memory_id
                            )
                            if is_dupe:
                                logger.info(
                                    f"Skipping update because content would duplicate another memory (ID: {memory_id})"
                                )
                                continue

                        final_content = format_memory_content(
                            new_content, tags, bank, confidence
                        )
                        updated_memory = await update_memory_by_id_and_user_id_compat(
                            memory_id, user_id, final_content
                        )

                        if updated_memory:
                            if new_embedding is None:
                                new_embedding = await self.embedding_manager.get_embedding(
                                    new_content, user=user_obj
                                )

                            if new_embedding is not None:
                                await self.embedding_manager.cache.set(memory_id, new_embedding)
                                await self.embedding_manager.store_embedding_persistent(
                                    user_id, memory_id, new_content, new_embedding
                                )

                            await self._sync_memory_vector(
                                user_id,
                                updated_memory,
                                final_content,
                                user_obj=user_obj,
                            )

                            if self.mem0_sync_manager:
                                try:
                                    await self._mirror_memory_upsert(
                                        user_id=user_id,
                                        memory_id=memory_id,
                                        content=new_content,
                                        tags=tags,
                                        memory_bank=bank,
                                        confidence=confidence,
                                        mem0_user_id_override=mem0_user_id_override,
                                    )
                                except Exception as mem0_err:
                                    logger.warning(
                                        f"Mem0 mirror update failed for memory {memory_id}: {mem0_err}"
                                    )

                            success_ops.append(normalized_op)
                            logger.info(f"Memory updated (ID: {memory_id})")
                            self._log_memory_save_user_id(user_id, memory_id)
                        else:
                            logger.warning(f"Memory not found for update (ID: {memory_id})")

                    except Exception as upd_err:
                        logger.exception(f"Failed to update memory: {upd_err}")

                elif kind == "DELETE" and normalized_op.get("id"):
                    try:
                        memory_id = normalize_memory_id(normalized_op["id"])
                        deleted = await self._delete_local_memory(
                            user_id,
                            memory_id,
                            mirror_to_mem0=True,
                            log_context="Memory operation",
                        )
                        if deleted:
                            success_ops.append(normalized_op)
                    except Exception as del_err:
                        logger.exception(f"Failed to delete memory: {del_err}")

            except Exception as e:
                self.error_manager.increment("memory_crud_errors")
                logger.exception(f"Memory operation failed: {e}")

        # Prune old memories if we exceeded the limit
        if success_ops and user_id:
            try:
                deleted_count = await self._prune_old_memories(user_id)
                if deleted_count > 0:
                    logger.info(f"Pruned {deleted_count} old memories to maintain limit of {self.valves.max_total_memories}")
            except Exception as prune_err:
                logger.exception(f"Memory pruning failed: {prune_err}")

        return success_ops

    async def _is_duplicate(self, text: str, user_id: str, exclude_id: Optional[str] = None, all_memories_override: Optional[List[Any]] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if the given text is a duplicate of existing memories.
        
        Returns:
            Tuple of (is_duplicate: bool, embedding: Optional[np.ndarray])
            The embedding is returned so it can be cached after successful save.
        """
        if not text or not self.valves.deduplicate_memories:
            return False, None
            
        try:
            # Get all existing memories for the user (or use override for optimization)
            if all_memories_override is not None:
                all_memories = all_memories_override
            else:
                all_memories = await get_memories_by_user_id_compat(user_id)
                
            if not all_memories:
                return False, None

            user_obj = await self._get_user_object(user_id)

            if self.valves.use_embeddings_for_deduplication:
                new_embedding = await self.embedding_manager.get_embedding(
                    text, user=user_obj
                )
                if new_embedding is None:
                    logger.warning("Could not generate embedding for duplicate check, falling back to text similarity")
                    is_dup = await self._check_text_similarity(text, all_memories, exclude_id=exclude_id)
                    return is_dup, None
                    
                for memory in all_memories:
                    memory_id = self._get_memory_id(memory)
                    if not memory_id:
                        continue
                    if exclude_id and memory_id == normalize_memory_id(exclude_id):
                        continue

                    raw_memory_content = self._get_memory_record(memory).content
                    if not raw_memory_content:
                        continue

                    if self._normalize_text(text) == self._normalize_text(raw_memory_content):
                        logger.info(f"Exact match found for memory {memory_id}")
                        return True, new_embedding

                    existing_embedding = await self.embedding_manager.cache.get(memory_id)
                    if existing_embedding is None:
                        existing_embedding = await self.embedding_manager.load_embedding_persistent(user_id, memory_id)
                        if existing_embedding is not None:
                            await self.embedding_manager.cache.set(memory_id, existing_embedding)
                        else:
                            existing_embedding = await self.embedding_manager.get_embedding(
                                raw_memory_content, user=user_obj
                            )
                            if existing_embedding is not None:
                                await self.embedding_manager.cache.set(memory_id, existing_embedding)
                                await self.embedding_manager.store_embedding_persistent(
                                    user_id, memory_id, raw_memory_content, existing_embedding
                                )
                    
                    if existing_embedding is not None:
                        similarity = self._cosine_similarity(new_embedding, existing_embedding)
                        if similarity >= self.valves.embedding_similarity_threshold:
                            logger.info(f"Duplicate detected via embeddings (similarity: {similarity:.3f}) for memory {memory_id}")
                            return True, new_embedding
                    else:
                        logger.warning(f"Could not generate embedding for existing memory {memory_id}")
            else:
                is_dup = await self._check_text_similarity(text, all_memories, exclude_id=exclude_id)
                return is_dup, None
                        
            return False, new_embedding if self.valves.use_embeddings_for_deduplication else None
            
        except Exception as e:
            logger.error(f"Error during duplicate check: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, None

    async def _check_text_similarity(self, text: str, all_memories: List[Any], exclude_id: str = None) -> bool:
        """Check for text-based similarity using difflib."""

        normalized_text = self._normalize_text(text)
        
        for memory in all_memories:
            memory_id = self._get_memory_id(memory)
            if not memory_id:
                continue
            if exclude_id and memory_id == normalize_memory_id(exclude_id):
                continue

            raw_memory_content = self._get_memory_record(memory).content
            if not raw_memory_content:
                continue

            normalized_raw = self._normalize_text(raw_memory_content)
            similarity = difflib.SequenceMatcher(None, normalized_text, normalized_raw).ratio()
            
            if similarity >= self.valves.similarity_threshold:
                logger.info(f"Duplicate detected via text similarity (similarity: {similarity:.3f}) for memory {memory_id}")
                return True
                
        return False

    # --- Memory Pruning ---
    async def _prune_old_memories(self, user_id: str) -> int:
        """Prune memories when total count exceeds max_total_memories.
        
        Returns:
            Number of memories deleted
        """
        try:
            # Get all memories for the user
            all_memories = await get_memories_by_user_id_compat(user_id)
            
            if not all_memories:
                return 0
            
            total_count = len(all_memories)
            max_allowed = self.valves.max_total_memories
            
            if total_count <= max_allowed:
                logger.debug(f"Memory count ({total_count}) within limit ({max_allowed}), no pruning needed")
                return 0
            
            # Calculate how many to delete
            num_to_delete = total_count - max_allowed
            logger.info(f"Pruning {num_to_delete} memories (total: {total_count}, limit: {max_allowed})")
            
            # Select memories to delete based on strategy
            if self.valves.pruning_strategy == "fifo":
                # Sort by created_at (oldest first)
                sorted_memories = sorted(
                    all_memories, 
                    key=self._memory_created_at_sort_key
                )
                memories_to_delete = sorted_memories[:num_to_delete]
                logger.info(f"Using FIFO strategy: deleting {num_to_delete} oldest memories")
                
            elif self.valves.pruning_strategy == "least_relevant":
                scored_memories = []
                for m in all_memories:
                    confidence = self._get_memory_record(m).confidence or 1.0
                    normalized_created_at = self._coerce_created_at(
                        getattr(m, "created_at", None) if hasattr(m, "created_at") else None
                    )
                    if normalized_created_at is not None:
                        age_days = (datetime.now(timezone.utc) - normalized_created_at).days
                    else:
                        age_days = 9999

                    relevance_score = confidence - (age_days * 0.01)
                    scored_memories.append((relevance_score, m))
                
                # Sort by relevance score (lowest first)
                sorted_memories = sorted(scored_memories, key=lambda x: x[0])
                memories_to_delete = [m for _, m in sorted_memories[:num_to_delete]]
                logger.info(f"Using least_relevant strategy: deleting {num_to_delete} least relevant memories")
            else:
                logger.warning(f"Unknown pruning strategy: {self.valves.pruning_strategy}, defaulting to FIFO")
                sorted_memories = sorted(
                    all_memories,
                    key=self._memory_created_at_sort_key
                )
                memories_to_delete = sorted_memories[:num_to_delete]
            
            # Delete the selected memories
            deleted_count = 0
            for memory in memories_to_delete:
                try:
                    memory_id = self._get_memory_id(memory)
                    if not memory_id:
                        continue
                    deleted = await self._delete_local_memory(
                        user_id,
                        memory_id,
                        mirror_to_mem0=True,
                        log_context="Pruning",
                    )
                    if deleted:
                        deleted_count += 1
                    
                except Exception as del_err:
                    logger.error(
                        f"Pruning: Failed to delete memory {getattr(memory, 'id', 'unknown')}: {del_err}"
                    )
            
            logger.info(f"Pruning complete: deleted {deleted_count}/{num_to_delete} memories")
            return deleted_count
            
        except Exception as e:
            logger.exception(f"Error during memory pruning: {e}")
            return 0

    # --- Summarization ---
    async def cluster_and_summarize(
        self, user_id: str, query_llm_func: Callable
    ) -> Optional[str]:
        """Find clusters of memories and summarize them."""
        logger.info(f"Starting summarization for user {user_id}")
        user_obj = await self._get_user_object(user_id)
        
        # 1. Fetch memories
        try:
            all_memories = await get_memories_by_user_id_compat(user_id)
            logger.info(f"Found {len(all_memories) if all_memories else 0} memories for user {user_id}")

            if not all_memories:
                logger.info(f"No memories found for user {user_id}, skipping summarization")
                return

            memories = []
            now = datetime.now(timezone.utc)
            for memory in all_memories:
                memory_record = self._get_memory_record(memory)
                if not memory_record.content or "summary" in memory_record.tags:
                    continue

                created_at = getattr(memory, "created_at", None)
                if self.valves.summarization_min_memory_age_days > 0:
                    normalized_created_at = self._coerce_created_at(created_at)
                    if normalized_created_at is None:
                        continue
                    age_days = (now - normalized_created_at).days
                    if age_days < self.valves.summarization_min_memory_age_days:
                        continue

                memories.append(memory)

            if len(memories) < self.valves.summarization_min_cluster_size:
                logger.info(
                    f"Only {len(memories)} eligible memories found for user {user_id}, need at least {self.valves.summarization_min_cluster_size} for clustering"
                )
                return
        except Exception as e:
            logger.exception(f"Summarization fetch failed: {e}")
            return

        logger.info(f"Processing embeddings for {len(memories)} memories")
        contents = [self._get_memory_record(m).content for m in memories]
        ids = [self._get_memory_id(m) for m in memories]

        embeddings = []
        uncached_indices = []
        uncached_contents = []

        for i, (memory_id, content) in enumerate(zip(ids, contents, strict=True)):
            if not memory_id or not content:
                embeddings.append(None)
                continue
            cached_embedding = await self.embedding_manager.cache.get(memory_id)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                persistent_embedding = await self.embedding_manager.load_embedding_persistent(user_id, memory_id)
                if persistent_embedding is not None:
                    await self.embedding_manager.cache.set(memory_id, persistent_embedding)
                    embeddings.append(persistent_embedding)
                else:
                    embeddings.append(None)
                    uncached_indices.append(i)
                    uncached_contents.append(content)

        if uncached_contents:
            logger.info(f"Generating embeddings for {len(uncached_contents)} uncached memories (using cache for {len(memories) - len(uncached_contents)})")
            new_embeddings = await self.embedding_manager.get_embeddings_batch(
                uncached_contents, user=user_obj
            )
            
            for idx, new_emb in zip(uncached_indices, new_embeddings, strict=True):
                if new_emb is not None:
                    embeddings[idx] = new_emb
                    await self.embedding_manager.cache.set(ids[idx], new_emb)
            
            await self.embedding_manager.store_embeddings_batch_persistent(
                user_id, 
                [str(ids[idx]) for idx in uncached_indices],
                uncached_contents,
                new_embeddings
            )
        else:
            logger.info(f"Using cached embeddings for all {len(memories)} memories")
            new_embeddings = []

        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        newly_generated_count = len([e for e in new_embeddings if e is not None]) if new_embeddings else 0
        logger.info(f"Ready for clustering: {len(valid_indices)} valid embeddings ({len(memories) - len(uncached_contents)} from cache, {newly_generated_count} newly generated)")
        
        if len(valid_indices) < self.valves.summarization_min_cluster_size:
            logger.info(f"Only {len(valid_indices)} valid embeddings, need at least {self.valves.summarization_min_cluster_size} for clustering")
            return

        logger.info(f"Starting complete linkage clustering with similarity threshold {self.valves.summarization_similarity_threshold}")
        clusters = []
        visited = set()

        for i in valid_indices:
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in valid_indices:
                if j in visited:
                    continue

                vec_j = embeddings[j]
                similar_to_all = True
                for cluster_idx in cluster:
                    vec_cluster = embeddings[cluster_idx]
                    sim = self._cosine_similarity(vec_j, vec_cluster)
                    
                    if sim < self.valves.summarization_similarity_threshold:
                        similar_to_all = False
                        break
                
                if similar_to_all:
                    cluster.append(j)
                    visited.add(j)

                    if len(cluster) >= self.valves.summarization_max_cluster_size:
                        break

            if len(cluster) >= self.valves.summarization_min_cluster_size:
                clusters.append(cluster)
                logger.info(f"Found cluster with {len(cluster)} memories (all members mutually similar >= {self.valves.summarization_similarity_threshold})")

        logger.info(f"Found {len(clusters)} clusters ready for summarization")

        summaries_created = 0
        source_memories_deleted = 0
        for cluster_indices in clusters:
            try:
                cluster_memories = [memories[i] for i in cluster_indices]
                cluster_text = "\n".join(
                    [f"- {self._get_memory_record(m).content}" for m in cluster_memories]
                )

                summary = await query_llm_func(
                    self.valves.summarization_memory_prompt,
                    f"Memories to summarize:\n{cluster_text}",
                )

                if summary:
                    confidence_scores = []
                    for m in cluster_memories:
                        conf = self._get_memory_record(m).confidence
                        if conf is not None:
                            confidence_scores.append(conf)

                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.85

                    op = {
                        "operation": "NEW",
                        "content": re.sub(r"\s+", " ", summary).strip(),
                        "tags": ["summary"],
                        "memory_bank": "General",
                        "confidence": avg_confidence,
                    }

                    success_ops = await self.process_memory_operations([op], user_id, skip_deduplication=True)

                    if success_ops:
                        logger.info(
                            f"Summarization: New consolidated summary saved successfully. Now removing {len(cluster_memories)} source memories."
                        )
                        for m in cluster_memories:
                            try:
                                memory_id = self._get_memory_id(m)
                                if not memory_id:
                                    continue
                                deleted = await self._delete_local_memory(
                                    user_id,
                                    memory_id,
                                    mirror_to_mem0=True,
                                    log_context="Summarization",
                                )
                                if deleted:
                                    source_memories_deleted += 1
                            except Exception as del_err:
                                logger.error(
                                    f"Summarization: Failed to delete source memory {getattr(m, 'id', 'unknown')}: {del_err}"
                                )

                        summaries_created += 1
                        logger.info(
                            f"Summarized {len(cluster_memories)} memories into new summary (Confidence {avg_confidence:.2f})"
                        )
                    else:
                        logger.error(
                            "Summarization: Failed to save new summary. Aborting source memory deletion to prevent data loss."
                        )

            except Exception as e:
                self.error_manager.increment("memory_crud_errors")
                logger.exception(f"Memory operation failed: {e}")

        if summaries_created:
            return (
                f"Consolidated {source_memories_deleted} memories into "
                f"{summaries_created} summary {'memory' if summaries_created == 1 else 'memories'}."
            )

        return None


class TaskManager:
    """Manages background tasks."""

    def __init__(self, filter_instance: Any):
        self.filter = filter_instance
        self.tasks: Set[asyncio.Task] = set()

    def start_tasks(self) -> bool:
        """Attempt to start background tasks. Returns True if successful."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "TaskManager: No running event loop found. Tasks will be retried on next request."
            )
            return False

        # Kill rogue ghost tasks from previous versions before starting new ones
        scavenger_task = asyncio.create_task(self._scavenge_rogue_tasks())
        self.tasks.add(scavenger_task)
        scavenger_task.add_done_callback(self.tasks.discard)

        valves = self.filter.valves
        logger.info(f"Starting background tasks: summarization={valves.enable_summarization_task}")

        if valves.enable_summarization_task:
            task = asyncio.create_task(self.filter._summarize_old_memories_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_mem0_sync:
            task = asyncio.create_task(self.filter.mem0_sync_manager.run_sync_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_error_logging_task:
            task = asyncio.create_task(self.filter._log_error_counters_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_vector_cleanup_task:
            task = asyncio.create_task(self.filter._cleanup_vectors_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        logger.info(f"Background tasks started: {len(self.tasks)} active tasks")
        return True

    async def stop_tasks(self):
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def _scavenge_rogue_tasks(self):
        """Find and terminate any orphaned background tasks from previous versions."""
        logger.info("TaskManager: Starting rogue task scavenger...")
        current_task = asyncio.current_task()
        all_tasks = asyncio.all_tasks()

        scavenged_count = 0
        for task in all_tasks:
            if task == current_task:
                continue

            # Look for tasks running functions related to adaptive memory loops
            task_repr = repr(task)
            # We target the specific function names used in v3.1 and v4.0
            ghost_indicators = [
                "_summarize_old_memories_loop",
                "_deduplicate_memories_loop",
                "_remove_duplicate_memories",
                "function_adaptive_memory_v31",
            ]

            if any(indicator in task_repr for indicator in ghost_indicators):
                # If it's not one of OUR currently tracked tasks, it's a ghost
                if task not in self.tasks:
                    logger.warning(
                        f"TaskManager: Found potentially rogue ghost task: {task_repr}. Requesting cancellation."
                    )
                    task.cancel()
                    scavenged_count += 1

        if scavenged_count > 0:
            logger.info(
                f"TaskManager: Scavenger requested cancellation of {scavenged_count} rogue tasks."
            )
        else:
            logger.info("TaskManager: No rogue tasks detected.")


# ------------------------------------------------------------------------------
# Main Filter Class
# ------------------------------------------------------------------------------


class Filter:
    # --------------------------------------------------------------------------
    # Configuration / Valves (PRESERVED EXACTLY)
    # --------------------------------------------------------------------------
    class Valves(BaseModel):
        """Configuration valves for the filter"""

        # Embedding Model Configuration
        embedding_source: Literal["auto", "owui", "plugin"] = Field(
            default="auto",
            description="Embedding source mode: 'auto' prefers Open WebUI's configured embedding function and falls back to the plugin provider, 'owui' uses only Open WebUI embeddings, and 'plugin' uses only the plugin-configured provider.",
        )
        embedding_provider_type: Literal["local", "openai_compatible"] = Field(
            default="local",
            description="Plugin-side embedding provider type used when embedding_source is 'auto' and Open WebUI embeddings are unavailable, or when embedding_source is 'plugin'.",
        )
        embedding_model_name: str = Field(
            default="all-MiniLM-L6-v2",
            description="Plugin-side embedding model name used for the fallback/internal provider when embedding_source allows plugin embeddings.",
        )
        embedding_api_url: Optional[str] = Field(
            default=None,
            description="Plugin-side embedding API endpoint used when embedding_source allows plugin embeddings and embedding_provider_type is 'openai_compatible'.",
        )
        embedding_api_key: Optional[str] = Field(
            default=None,
            description="Plugin-side embedding API key used when embedding_source allows plugin embeddings and embedding_provider_type is 'openai_compatible'.",
        )

        # Optional Mem0 Mirror Configuration
        enable_mem0_sync: bool = Field(
            default=False,
            description="Optionally mirror local memory CRUD operations to Mem0. Disabled by default so memories stay local unless explicitly enabled.",
        )
        mem0_api_base_url: str = Field(
            default="https://api.mem0.ai",
            description="Base URL for the Mem0 API when Mem0 mirroring is enabled.",
        )
        mem0_api_key: Optional[str] = Field(
            default=None,
            description="API key for Mem0. Required only when enable_mem0_sync is enabled.",
        )
        mem0_app_id: str = Field(
            default="openwebui-adaptive-memory",
            description="Mem0 app_id used to namespace memories mirrored from this plugin.",
        )
        mem0_timeout_seconds: int = Field(
            default=30,
            description="Timeout in seconds for Mem0 API requests.",
        )
        mem0_reconcile_cooldown_seconds: float = Field(
            default=30.0,
            description="Minimum seconds between Mem0 delete-reconciliation checks for the same user during inbound requests.",
        )
        mem0_sync_strategy: Literal["background", "inline"] = Field(
            default="background",
            description="How Mem0 mirroring runs: 'background' queues CRUD changes for batched async syncing, while 'inline' performs Mem0 requests during the request path.",
        )
        mem0_sync_batch_size: int = Field(
            default=10,
            description="Maximum number of queued Mem0 jobs to process in one background batch.",
        )
        mem0_sync_batch_interval_seconds: float = Field(
            default=7200.0,
            description="Interval in seconds between scheduled Mem0 background sync runs.",
        )
        mem0_sync_retry_delay_seconds: float = Field(
            default=15.0,
            description="Delay before retrying a failed queued Mem0 sync job.",
        )
        mem0_user_id_template: str = Field(
            default="owui:{user_id}",
            description="Template used to map an Open WebUI user id into a Mem0 user id. May include {user_id} for per-user mapping, or be a fixed string such as 'jefe' to force all mirrored memories into one Mem0 user/entity.",
        )
        mem0_user_id_override: str = Field(
            default="",
            description="Optional per-user Mem0 mapping table shown in the main valve UI. Use targeted mappings like 'owui_user_id:jefe' (comma, semicolon, or newline separated) to route specific Open WebUI users to specific Mem0 users. Plain values like 'jefe' are ignored.",
        )
        mem0_infer_on_create: bool = Field(
            default=False,
            description="When true, mirrored Mem0 create requests use infer=true so Mem0 can extract, deduplicate, and resolve conflicts from the provided message. Disabled by default; when false, mirrored text is stored more literally with infer=false.",
        )

        # Background Task Management Configuration
        enable_summarization_task: bool = Field(
            default=True,
            description="Enable or disable the background memory summarization task",
        )
        summarization_interval: int = Field(
            default=7200,
            description="Interval in seconds between memory summarization runs",
        )
        enable_error_logging_task: bool = Field(
            default=True,
            description="Enable or disable the background error counter logging task",
        )
        error_logging_interval: int = Field(
            default=1800,
            description="Interval in seconds between error counter log entries",
        )

        enable_vector_cleanup_task: bool = Field(
            default=True,
            description="Enable or disable the background vector cleanup task",
        )
        vector_cleanup_interval: int = Field(
            default=7200,
            description="Interval in seconds between vector cleanup runs (removes orphaned embeddings)",
        )

        enable_date_update_task: bool = Field(
            default=True,
            description="Enable or disable the background date update task",
        )
        date_update_interval: int = Field(
            default=3600,
            description="Interval in seconds between date information updates",
        )
        enable_model_discovery_task: bool = Field(
            default=True,
            description="Enable or disable the background model discovery task",
        )
        model_discovery_interval: int = Field(
            default=7200, description="Interval in seconds between model discovery runs"
        )

        # Summarization Configuration
        summarization_min_cluster_size: int = Field(
            default=3,
            description="Minimum number of memories in a cluster for summarization",
        )
        summarization_similarity_threshold: float = Field(
            default=0.7,
            description="Threshold for considering memories related when using embedding similarity",
        )
        summarization_max_cluster_size: int = Field(
            default=8,
            description="Maximum memories to include in one summarization batch",
        )
        summarization_min_memory_age_days: int = Field(
            default=7,
            description="Minimum age in days for memories to be considered for summarization",
        )
        summarization_strategy: Literal["embeddings", "tags", "hybrid"] = Field(
            default="hybrid",
            description="Strategy for clustering memories: 'embeddings' (semantic similarity), 'tags' (shared tags), or 'hybrid' (combination)",
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
            description="System prompt for summarizing clusters of related memories",
        )

        # Filtering & Saving Configuration
        enable_json_stripping: bool = Field(
            default=True,
            description="Attempt to strip non-JSON text before/after the main JSON object/array from LLM responses.",
        )
        enable_fallback_regex: bool = Field(
            default=True,
            description="If primary JSON parsing fails, attempt a simple regex fallback to extract at least one memory.",
        )
        enable_short_preference_shortcut: bool = Field(
            default=True,
            description="If JSON parsing fails for a short message containing preference keywords, directly save the message content.",
        )
        short_preference_no_dedupe_length: int = Field(
            default=100,
            description="If a NEW memory's content length is below this threshold and contains preference keywords, skip deduplication checks to avoid false positives.",
        )
        preference_keywords_no_dedupe: str = Field(
            default="favorite,love,like,prefer,enjoy",
            description="Comma-separated keywords indicating user preferences that, when present in a short statement, trigger deduplication bypass.",
        )
        blacklist_topics: Optional[str] = Field(
            default=None,
            description="Optional: Comma-separated list of topics to ignore during memory extraction",
        )
        filter_trivia: bool = Field(
            default=True,
            description="Enable filtering of trivia/general knowledge memories after extraction",
        )
        whitelist_keywords: Optional[str] = Field(
            default=None,
            description="Optional: Comma-separated keywords that force-save a memory even if blacklisted",
        )
        max_total_memories: int = Field(
            default=200,
            description="Maximum number of memories per user; prune oldest beyond this",
        )
        pruning_strategy: Literal["fifo", "least_relevant"] = Field(
            default="fifo",
            description="Strategy for pruning memories when max_total_memories is exceeded: 'fifo' (oldest first) or 'least_relevant' (lowest relevance to current message first).",
        )
        min_memory_length: int = Field(
            default=8,
            description="Minimum length of memory content to be saved",
        )
        min_confidence_threshold: float = Field(
            default=0.5,
            description="Minimum confidence score (0-1) required for an extracted memory to be saved. Scores below this are discarded.",
        )
        recent_messages_n: int = Field(
            default=5,
            description="Number of recent user messages to include in extraction prompt context",
        )
        save_relevance_threshold: float = Field(
            default=0.8,
            description="Minimum relevance score (based on relevance calculation method) to save a memory",
        )
        max_injected_memory_length: int = Field(
            default=300,
            description="Maximum length of each injected memory snippet",
        )

        # Generic LLM Provider Configuration
        llm_provider_type: Literal["ollama", "openai_compatible"] = Field(
            default="ollama",
            description="Type of LLM provider ('ollama' or 'openai_compatible')",
        )
        llm_model_name: str = Field(
            default="llama3:latest",
            description="Name of the LLM model to use (e.g., 'llama3:latest', 'gpt-4o')",
        )
        llm_api_endpoint_url: str = Field(
            default="http://host.docker.internal:11434/api/chat",
            description="API endpoint URL for the LLM provider (e.g., 'http://host.docker.internal:11434/api/chat', 'https://api.openai.com/v1/chat/completions')",
        )
        llm_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the LLM provider (required if type is 'openai_compatible')",
        )

        # Memory processing settings
        related_memories_n: int = Field(
            default=5,
            description="Number of related memories to consider",
        )
        relevance_threshold: float = Field(
            default=0.60,
            description="Minimum relevance score (0-1) for memories to be considered relevant for injection after scoring",
        )
        memory_threshold: float = Field(
            default=0.6,
            description="Threshold for similarity when comparing memories (0-1)",
        )
        vector_similarity_threshold: float = Field(
            default=0.60,
            description="Minimum cosine similarity for initial vector filtering (0-1)",
        )
        llm_skip_relevance_threshold: float = Field(
            default=0.93,
            description="If *all* vector-filtered memories have similarity >= this threshold, treat the vector score as final relevance and skip the additional LLM call.",
        )
        top_n_memories: int = Field(
            default=3,
            description="Number of top similar memories to pass to LLM",
        )
        cache_ttl_seconds: int = Field(
            default=86400,
            description="Cache time-to-live in seconds (default 24 hours)",
        )
        use_llm_for_relevance: bool = Field(
            default=False,
            description="Use LLM call for final relevance scoring (if False, relies solely on vector similarity + relevance_threshold)",
        )
        deduplicate_memories: bool = Field(
            default=True,
            description="Prevent storing duplicate or very similar memories",
        )
        use_embeddings_for_deduplication: bool = Field(
            default=True,
            description="Use embedding-based similarity for more accurate semantic duplicate detection (if False, uses text-based similarity)",
        )
        embedding_similarity_threshold: float = Field(
            default=0.75,
            description="Threshold (0-1) for considering two memories duplicates when using embedding similarity.",
        )
        similarity_threshold: float = Field(
            default=0.95,
            description="Threshold for detecting similar memories (0-1) using text or embeddings",
        )
        timezone: str = Field(
            default="Asia/Dubai",
            description="Timezone for date/time processing (e.g., 'America/New_York', 'Europe/London')",
        )
        show_status: bool = Field(
            default=True, description="Show memory operations status in chat"
        )
        show_memories: bool = Field(
            default=True, description="Show relevant memories in context"
        )
        log_user_id_on_memory_save: bool = Field(
            default=False,
            description="Log only the Open WebUI user_id and memory_id whenever a memory save or update succeeds. Useful for admin debugging.",
        )
        memory_format: Literal["bullet", "paragraph", "numbered"] = Field(
            default="bullet", description="Format for displaying memories in context"
        )
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
        max_retries: int = Field(
            default=2, description="Maximum number of retries for API calls"
        )
        retry_delay: float = Field(
            default=1.0, description="Delay between retries (seconds)"
        )

        # Prompts
        memory_identification_prompt: str = Field(
            default="""You are an automated JSON data extraction system. Your ONLY function is to identify user-specific, persistent facts, preferences, goals, relationships, or interests from the user's messages and output them STRICTLY as a JSON array of operations.

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

Analyze the following user message(s) and provide **ONLY** the JSON array output. Double-check your response starts with `[` and ends with `]` and contains **NO** other text whatsoever.""",
            description="System prompt for memory identification",
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

        # Memory Bank Config
        allowed_memory_banks: List[str] = Field(
            default=["General", "Personal", "Work"],
            description="List of allowed memory bank names for categorization.",
        )
        default_memory_bank: str = Field(
            default="General",
            description="Default memory bank assigned when LLM omits or supplies an invalid bank.",
        )

        # Error Guard Config
        enable_error_counter_guard: bool = Field(
            default=True,
            description="Enable guard to temporarily disable LLM/embedding features if specific error rates spike.",
        )
        error_guard_threshold: int = Field(
            default=5,
            description="Number of errors within the window required to activate the guard.",
        )
        error_guard_window_seconds: int = Field(
            default=600,
            description="Rolling time-window (in seconds) over which errors are counted for guarding logic.",
        )
        debug_error_counter_logs: bool = Field(
            default=False,
            description="Emit detailed error counter logs at DEBUG level (set to True for troubleshooting).",
        )

        # Validators
        @field_validator(
            "summarization_interval",
            "error_logging_interval",
            "date_update_interval",
            "model_discovery_interval",
            "max_total_memories",
            "min_memory_length",
            "recent_messages_n",
            "related_memories_n",
            "top_n_memories",
            "cache_ttl_seconds",
            "max_retries",
            "max_injected_memory_length",
            "summarization_min_cluster_size",
            "summarization_max_cluster_size",
            "summarization_min_memory_age_days",
            "mem0_timeout_seconds",
            "mem0_sync_batch_size",
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{info.field_name} must be a non-negative integer")
            return v

        @field_validator(
            "save_relevance_threshold",
            "relevance_threshold",
            "memory_threshold",
            "vector_similarity_threshold",
            "similarity_threshold",
            "summarization_similarity_threshold",
            "llm_skip_relevance_threshold",
            "embedding_similarity_threshold",
            "min_confidence_threshold",
            check_fields=False,
        )
        def check_threshold_float(cls, v, info):
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
                )
            return v

        @field_validator("retry_delay")
        def check_non_negative_float(cls, v, info):
            if not isinstance(v, (int, float)) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return float(v)

        @field_validator("mem0_reconcile_cooldown_seconds")
        def check_non_negative_mem0_cooldown(cls, v, info):
            if not isinstance(v, (int, float)) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return float(v)

        @field_validator(
            "mem0_sync_batch_interval_seconds", "mem0_sync_retry_delay_seconds"
        )
        def check_non_negative_mem0_sync_float(cls, v, info):
            if not isinstance(v, (int, float)) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return float(v)

        @field_validator("timezone")
        def check_valid_timezone(cls, v):
            try:
                pytz.timezone(v)
            except Exception as e:
                raise ValueError(f"Invalid timezone string in config: {v}") from e
            return v

        @model_validator(mode="after")
        def check_llm_config(self):
            if self.llm_provider_type == "openai_compatible" and not self.llm_api_key:
                raise ValueError(
                    "API Key is required when llm_provider_type is 'openai_compatible'"
                )
            return self

        @model_validator(mode="after")
        def check_mem0_config(self):
            if self.enable_mem0_sync:
                if not self.mem0_api_key:
                    raise ValueError(
                        "Mem0 API key is required when enable_mem0_sync is enabled"
                    )
                if not str(self.mem0_user_id_template or "").strip():
                    raise ValueError(
                        "mem0_user_id_template must not be empty when Mem0 mirroring is enabled"
                    )
            return self

        @field_validator("allowed_memory_banks", check_fields=False)
        def check_allowed_memory_banks(cls, v):
            if not isinstance(v, list) or not v or v == [""]:
                return cls.model_fields["allowed_memory_banks"].default
            cleaned_list = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned_list:
                return cls.model_fields["allowed_memory_banks"].default
            return cleaned_list

        @model_validator(mode="after")
        def check_embedding_config(self):
            if (
                self.embedding_source in {"auto", "plugin"}
                and self.embedding_provider_type == "openai_compatible"
            ):
                if not self.embedding_api_key:
                    raise ValueError(
                        "API Key required for openai_compatible embedding provider"
                    )
            return self

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable or disable the memory function"
        )
        show_status: bool = Field(
            default=True, description="Show memory processing status updates"
        )
        mem0_user_id_override: str = Field(
            default="",
            description="Optional per-user Mem0 user/entity id override. Leave empty to use the global mem0_user_id_template; set a value like 'jefe' to route only this user's mirrored memories to that exact Mem0 entity.",
        )
        timezone: str = Field(
            default="",
            description="User's timezone (overrides global setting if provided)",
        )

    # --------------------------------------------------------------------------
    # Main Filter Initialization
    # --------------------------------------------------------------------------

    def __init__(self):
        logger.info("Initializing Adaptive Memory Filter v4.0.2")
        self.valves = self.Valves()
        self.error_manager = ErrorManager()
        # Pass a lambda to always get the current valves state
        self.embedding_manager = EmbeddingManager(
            lambda: self.valves, self.error_manager
        )
        self.mem0_sync_manager = Mem0SyncManager(lambda: self.valves)
        self.task_manager = TaskManager(self)

        # Initialize internal state
        self._processed_messages = set()
        self._last_body = {}
        self.memory_embeddings = {}  # Local in-memory cache
        self.seen_users = set()  # Track active users for background tasks
        self.notification_queue = []  # Queue for background task notifications
        self._tasks_started = False
        self._valve_hash = None  # Track valve changes

        logger.info("Adaptive Memory Filter v4.0.2 initialized")
    def _check_and_handle_valve_changes(self):
        """Detect if valves have changed and restart tasks if needed."""
        # Hash important valve settings that affect background tasks
        valve_str = (
            f"{self.valves.enable_summarization_task}_{self.valves.summarization_interval}_"
            f"{self.valves.enable_error_logging_task}_{self.valves.error_logging_interval}_"
            f"{self.valves.enable_vector_cleanup_task}_{self.valves.vector_cleanup_interval}_"
            f"{self.valves.enable_mem0_sync}_{self.valves.mem0_sync_strategy}_"
            f"{self.valves.mem0_sync_batch_size}_{self.valves.mem0_sync_batch_interval_seconds}_"
            f"{self.valves.mem0_sync_retry_delay_seconds}"
        )
        new_hash = hashlib.sha256(valve_str.encode()).hexdigest()
        
        if self._valve_hash is None:
            self._valve_hash = new_hash
            return False
        
        if new_hash != self._valve_hash:
            logger.info("Valve changes detected! Restarting background tasks...")
            self._valve_hash = new_hash
            # Restart tasks with new valve values
            if self._tasks_started:
                # Cancel existing restart task if running
                if hasattr(self, '_restart_task') and self._restart_task and not self._restart_task.done():
                    self._restart_task.cancel()
                
                # Create managed task with error callback
                self._restart_task = asyncio.create_task(self._restart_tasks())
                
                def _log_restart_exception(task):
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        pass  # Expected when task is cancelled
                    except Exception as e:
                        logger.exception(f"Background restart task failed: {e}")
                
                self._restart_task.add_done_callback(_log_restart_exception)
            return True
        return False
    
    async def _restart_tasks(self):
        """Restart background tasks with new valve settings."""
        await self.task_manager.stop_tasks()
        self._tasks_started = False
        self._tasks_started = self.task_manager.start_tasks()
        logger.info("Background tasks restarted with new valve values")

    async def cleanup(self):
        await self.task_manager.stop_tasks()
        await self.embedding_manager.cleanup()
        await self.mem0_sync_manager.cleanup()

    def _load_user_valves(self, __user__: Optional[Dict[str, Any]]) -> "Filter.UserValves":
        raw_valves = (__user__ or {}).get("valves", {})
        if hasattr(raw_valves, "model_dump"):
            return self.UserValves(**raw_valves.model_dump())
        if hasattr(raw_valves, "dict"):
            return self.UserValves(**raw_valves.dict())
        if isinstance(raw_valves, dict):
            return self.UserValves(**raw_valves)
        if isinstance(raw_valves, self.UserValves):
            return raw_valves
        return self.UserValves()

    def _get_recent_user_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        user_messages = []
        for message in messages:
            if message.get("role") != "user":
                continue
            text = extract_message_text(message.get("content"))
            if text:
                user_messages.append(text)
        return user_messages[-self.valves.recent_messages_n :]

    def _format_relevant_memories(self, relevant_memories: List[Any]) -> str:
        formatted_memories = []
        for index, memory in enumerate(relevant_memories, start=1):
            memory_record = parse_stored_memory(
                memory.content if hasattr(memory, "content") else memory.get("content")
            )
            if not memory_record.content:
                continue
            content = truncate_text(
                memory_record.content, self.valves.max_injected_memory_length
            )
            if self.valves.memory_format == "numbered":
                formatted_memories.append(f"{index}. {content}")
            elif self.valves.memory_format == "paragraph":
                formatted_memories.append(content)
            else:
                formatted_memories.append(f"- {content}")

        if not formatted_memories:
            return ""

        if self.valves.memory_format == "paragraph":
            return "User Memories:\n" + " ".join(formatted_memories)
        return "User Memories:\n" + "\n".join(formatted_memories)

    async def _emit_queued_notifications(self, __event_emitter__) -> None:
        while self.notification_queue:
            msg = self.notification_queue.pop(0)
            bg_status_dict = {
                "type": "status",
                "data": {"description": f"🧹 {msg}", "done": True},
            }
            if __event_emitter__:
                await __event_emitter__(bg_status_dict)

    # --------------------------------------------------------------------------
    # Helper: LLM Query Wrapper
    # --------------------------------------------------------------------------
    async def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Unified LLM query method with retries and metrics."""
        valves = self.valves
        
        for attempt in range(valves.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    url = valves.llm_api_endpoint_url
                    headers = {"Content-Type": "application/json"}
                    if valves.llm_api_key:
                        headers["Authorization"] = f"Bearer {valves.llm_api_key}"

                    payload = {
                        "model": valves.llm_model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                    }

                    if valves.llm_provider_type == "openai_compatible":
                        # For identify_memories, we expect an array. 
                        # openai_compatible's 'json_object' format requires a root object, not array.
                        # Only use it if not identification or if the prompt can be adjusted.
                        # For now, following CodeRabbit: remove to avoid array errors.
                        pass

                    async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Extract content logic...
                            if "choices" in data:
                                return data["choices"][0]["message"]["content"]
                            elif "message" in data:
                                return data["message"]["content"]
                        elif resp.status >= 500:
                            # Server error - retry
                            raise aiohttp.ClientError(f"Server error: {resp.status}")
                        else:
                            # Client error (4xx) - don't retry
                            logger.warning(f"LLM API client error {resp.status}, not retrying")
                            return None
                            
            except Exception as e:
                if attempt < valves.max_retries:
                    logger.warning(f"LLM query attempt {attempt + 1}/{valves.max_retries + 1} failed, retrying in {valves.retry_delay}s: {e}")
                    await asyncio.sleep(valves.retry_delay)
                else:
                    logger.exception(f"LLM Query failed after {valves.max_retries + 1} attempts: {e}")
                    self.error_manager.increment("llm_call_errors")
        
        return None

    # --------------------------------------------------------------------------
    # Core Pipeline: Inlet (Incoming Message)
    # --------------------------------------------------------------------------
    async def inlet(
        self, body: Dict[str, Any], __event_emitter__=None, __user__=None
    ) -> Dict[str, Any]:
        """Process incoming message: Identify user, inject context memories."""
        if not __user__ or not body.get("messages"):
            return body

        user_valves = self._load_user_valves(__user__)
        if not user_valves.enabled:
            return body

        if not self._tasks_started:
            self._tasks_started = self.task_manager.start_tasks()
        
        # Check if valves have changed and restart tasks if needed
        self._check_and_handle_valve_changes()

        user_id = __user__["id"]
        self.seen_users.add(user_id)  # Track active user
        messages = body["messages"]
        last_message = extract_message_text(messages[-1].get("content"))

        # Skip command processing
        if last_message.startswith("/"):
            logger.info("Skipping memory processing for command")
            if user_valves.show_status:
                await self._emit_queued_notifications(__event_emitter__)
            return body

        if not last_message:
            logger.debug("Skipping memory processing for empty message.")
            if user_valves.show_status:
                await self._emit_queued_notifications(__event_emitter__)
            return body

        # Pipeline
        pipeline = MemoryPipeline(
            self.valves,
            self.embedding_manager,
            self.error_manager,
            self.mem0_sync_manager,
        )

        # 1. Retrieve all memories
        try:
            all_memories = await get_memories_by_user_id_compat(user_id)
        except Exception as e:
            logger.error(f"Failed to fetch memories: {e}")
            all_memories = []

        if all_memories and self.mem0_sync_manager:
            try:
                all_memories = await pipeline.reconcile_mem0_deleted_memories(
                    user_id, all_memories
                )
            except Exception as e:
                logger.warning(
                    f"Mem0 reconcile failed for user {user_id}; continuing with local memories: {e}"
                )

        # 2. Filter relevant memories
        relevant_memories = []
        if all_memories:
            relevant_memories = await pipeline.get_relevant_memories(
                last_message, user_id, all_memories
            )
            logger.info(f"Memory retrieval: found {len(relevant_memories)} relevant memories from {len(all_memories)} total memories")
        else:
            logger.debug(f"Memory retrieval: no existing memories found for user {user_id}")

        # 3. Inject into system prompt
        if relevant_memories and self.valves.show_memories:
            context_text = self._format_relevant_memories(relevant_memories)

            if context_text:
                if messages[0]["role"] == "system":
                    messages[0]["content"] += f"\n\n{context_text}"
                else:
                    messages.insert(0, {"role": "system", "content": context_text})

        if user_valves.show_status:
            count = len(relevant_memories)
            if count > 0:
                suffix = "memory" if count == 1 else "memories"
                status_dict = {
                    "type": "status",
                    "data": {
                        "description": f"🧠 Recalled {count} {suffix}.",
                        "done": True,
                    },
                }
                if __event_emitter__:
                    await __event_emitter__(status_dict)

            await self._emit_queued_notifications(__event_emitter__)

        return body

    # --------------------------------------------------------------------------
    # Core Pipeline: Outlet (Response Processing)
    # --------------------------------------------------------------------------
    async def outlet(
        self, body: Dict[str, Any], __event_emitter__=None, __user__=None
    ) -> Dict[str, Any]:
        """Process outgoing response: Extract memories, update status."""

        if not __user__ or not body.get("messages"):
            return body

        user_valves = self._load_user_valves(__user__)
        if not user_valves.enabled:
            return body

        user_id = __user__["id"]
        messages = body["messages"]

        recent_user_messages = self._get_recent_user_messages(messages)
        user_message = "\n".join(recent_user_messages).strip()
        retrieval_query = recent_user_messages[-1] if recent_user_messages else ""

        if retrieval_query.startswith("/"):
            logger.info("Skipping memory extraction for command")
            return body

        # Pipeline
        pipeline = MemoryPipeline(
            self.valves,
            self.embedding_manager,
            self.error_manager,
            self.mem0_sync_manager,
        )

        # Identify Memories
        if user_message:
            try:
                all_memories = await get_memories_by_user_id_compat(user_id)
            except Exception as e:
                logger.error(f"Failed to fetch memories for extraction context: {e}")
                all_memories = []

            context_memories = []
            if all_memories and retrieval_query:
                context_memories = await pipeline.get_relevant_memories(
                    retrieval_query, user_id, all_memories
                )

            # Pass our _query_llm as callback
            ops = await pipeline.identify_memories(
                user_message,
                context_memories=context_memories,
                query_llm_func=self._query_llm,
            )
            logger.info(f"Memory extraction: identified {len(ops)} potential memories from user message")

            success_ops = []
            if ops:
                # Process Operations (Save/Delete)
                success_ops = await pipeline.process_memory_operations(
                    ops, user_id, user_valves=user_valves
                )
                
            if len(success_ops) > 0:
                logger.info(f"Memory operations: saved {len(success_ops)} new memories (skipped {len(ops) - len(success_ops)} duplicates)")
            elif len(ops) > 0:
                logger.info(f"Memory operations: all {len(ops)} identified memories were duplicates, none saved")
            else:
                logger.debug("Memory operations: no memories identified from user message")

            # Show status if enabled
            if user_valves.show_status:
                count = len(success_ops)
                if count > 0:
                    suffix = "memory" if count == 1 else "memories"
                    description = f"🧠 Saved {count} {suffix}."
                else:
                    description = "No memories saved."

                status_dict = {
                    "type": "status",
                    "data": {"description": description, "done": True},
                }
                if __event_emitter__:
                    await __event_emitter__(status_dict)
                else:
                    logger.warning("Outlet: No event emitter available for status.")

        return body

    # ... Placeholder for other required methods (referenced by TaskManager) ...
    async def _summarize_old_memories_loop(self):
        """Background task for summarization."""
        logger.info(f"Summarization background task launched with interval: {self.valves.summarization_interval} seconds")
        while True:
            try:
                # Always get the current valve value in case it changed
                interval = self.valves.summarization_interval
                await asyncio.sleep(interval)
                logger.info(
                    f"Summarization task running. Active users: {len(self.seen_users)}, enabled: {self.valves.enable_summarization_task}"
                )

                if self.valves.enable_summarization_task and self.seen_users:
                    logger.info("Background summarization: starting scan...")
                    pipeline = MemoryPipeline(
                        self.valves,
                        self.embedding_manager,
                        self.error_manager,
                        self.mem0_sync_manager,
                    )

                    # Copy set to avoid size change during iteration
                    active_users = list(self.seen_users)
                    for user_id in active_users:
                        try:
                            logger.info(f"Background summarization: processing user {user_id}")
                            # Use _query_llm as callback
                            result_msg = await pipeline.cluster_and_summarize(
                                user_id, self._query_llm
                            )
                            if result_msg and isinstance(result_msg, str):
                                self.notification_queue.append(result_msg)
                                logger.info(f"Background summarization: {result_msg}")
                            else:
                                logger.debug(f"Background summarization: no clusters found for user {user_id}")
                        except Exception as u_err:
                            logger.exception(
                                f"Background summarization error for user {user_id}: {u_err}"
                            )

                    logger.info("Background summarization: cycle complete")
                else:
                    logger.debug(f"Background summarization: skipped (enabled: {self.valves.enable_summarization_task}, users: {len(self.seen_users)})")

            except asyncio.CancelledError:
                logger.info("Summarization task cancelled")
                break
            except Exception as e:
                logger.exception(f"Summarization task error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(60)

    async def _log_error_counters_loop(self):
        """Periodically log error counters."""
        try:
            while True:
                await asyncio.sleep(self.valves.error_logging_interval)
                logger.debug(f"Error Counters: {self.error_manager.get_counters()}")
        except asyncio.CancelledError:
            logger.info("Error logging task cancelled")
        except Exception as e:
            logger.exception(f"Error in error logging loop: {e}")

    async def cleanup_orphaned_vectors(self, user_id: str) -> Dict[str, Union[int, str]]:
        """
        Audit and clean up orphaned vector embeddings.
        
        Returns dict with:
        - db_memories: count of memories in database
        - orphans_deleted: count of orphaned vectors removed
        - error: (optional) error message if cleanup failed
        """
        if not VECTOR_DB_CLIENT:
            logger.warning("Vector DB not available - cannot cleanup orphaned vectors")
            return {"error": "Vector DB not available", "orphans_deleted": 0}
        
        try:
            # Get all memory IDs from database
            db_memories = await get_memories_by_user_id_compat(user_id)
            valid_ids = {
                memory_id
                for memory_id in (extract_memory_id(memory) for memory in db_memories)
                if memory_id is not None
            }
            
            collection_name = f"user-memory-{user_id}"
            
            # Get all vector IDs - method depends on vector DB implementation
            try:
                # Try to get all items from collection
                result = VECTOR_DB_CLIENT.get(collection_name=collection_name)
                vector_ids = []
                if isinstance(result, dict):
                    vector_ids = result.get("ids") or []
                elif hasattr(result, "ids"):
                    vector_ids = getattr(result, "ids") or []
                elif isinstance(result, list):
                    vector_ids = [
                        item.get("id", item) if isinstance(item, dict) else item
                        for item in result
                    ]

                if vector_ids and isinstance(vector_ids[0], list):
                    vector_ids = vector_ids[0]

                vector_ids = [normalize_memory_id(vector_id) for vector_id in vector_ids]

                if not vector_ids:
                    logger.warning(f"Unable to retrieve vector IDs for cleanup - collection may not exist")
                    return {"db_memories": len(valid_ids), "orphans_deleted": 0}
            except Exception as e:
                logger.error(f"Failed to retrieve vector IDs: {e}")
                return {"error": str(e), "orphans_deleted": 0}
            
            # Find orphans (vectors without corresponding database entry)
            orphaned_ids = [vid for vid in vector_ids if vid not in valid_ids]
            
            # Delete orphans
            if orphaned_ids:
                try:
                    VECTOR_DB_CLIENT.delete(
                        collection_name=collection_name,
                        ids=orphaned_ids
                    )
                    logger.info(f"Deleted {len(orphaned_ids)} orphaned vectors for user {user_id}")
                except Exception as del_err:
                    logger.error(f"Failed to delete orphaned vectors: {del_err}")
                    return {
                        "db_memories": len(valid_ids),
                        "orphans_found": len(orphaned_ids),
                        "orphans_deleted": 0,
                        "error": str(del_err)
                    }
            else:
                logger.info(f"No orphaned vectors found for user {user_id}")
            
            return {
                "db_memories": len(valid_ids),
                "vector_count": len(vector_ids),
                "orphans_deleted": len(orphaned_ids)
            }
            
        except Exception as e:
            logger.exception(f"Error during vector cleanup: {e}")
            return {"error": str(e), "orphans_deleted": 0}

    async def _cleanup_vectors_loop(self):
        """Background task for cleaning up orphaned vectors."""
        logger.info(f"Vector cleanup background task launched with interval: {self.valves.vector_cleanup_interval} seconds")
        while True:
            try:
                interval = self.valves.vector_cleanup_interval
                await asyncio.sleep(interval)
                logger.info(
                    f"Vector cleanup task running. Active users: {len(self.seen_users)}, enabled: {self.valves.enable_vector_cleanup_task}"
                )

                if self.valves.enable_vector_cleanup_task and self.seen_users:
                    logger.info("Background vector cleanup: starting scan...")
                    
                    # Copy set to avoid size change during iteration
                    active_users = list(self.seen_users)
                    for user_id in active_users:
                        try:
                            logger.info(f"Background vector cleanup: processing user {user_id}")
                            result = await self.cleanup_orphaned_vectors(user_id)
                            
                            if "orphans_deleted" in result and result["orphans_deleted"] > 0:
                                msg = f"Cleaned up {result['orphans_deleted']} orphaned vectors for user {user_id}"
                                self.notification_queue.append(msg)
                                logger.info(f"Background vector cleanup: {msg}")
                            else:
                                logger.debug(f"Background vector cleanup: no orphans found for user {user_id}")
                        except Exception as u_err:
                            logger.exception(
                                f"Background vector cleanup error for user {user_id}: {u_err}"
                            )

                    logger.info("Background vector cleanup: cycle complete")
                else:
                    logger.debug(f"Background vector cleanup: skipped (enabled: {self.valves.enable_vector_cleanup_task}, users: {len(self.seen_users)})")

            except asyncio.CancelledError:
                logger.info("Vector cleanup task cancelled")
                break
            except Exception as e:
                logger.exception(f"Vector cleanup task error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(60)

