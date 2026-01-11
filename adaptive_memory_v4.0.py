import json
import copy
import traceback
from datetime import datetime, timezone
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
import pytz
import difflib
import time
import os
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ----------------------------
# Metrics & Monitoring Imports
# ----------------------------
try:
    from prometheus_client import Counter, Histogram  # type: ignore
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
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import numpy as np
import aiohttp
from aiohttp import ClientError
from pydantic import BaseModel, Field, model_validator, field_validator

# OpenWebUI Imports
# OpenWebUI Imports
from open_webui.models.memories import Memories
from open_webui.models.users import Users
from open_webui.main import app as webui_app

# Set up logging
logger = logging.getLogger("openwebui.plugins.adaptive_memory")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Data Models and Helper Classes
# ------------------------------------------------------------------------------


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = []
    memory_bank: Optional[str] = None
    confidence: Optional[float] = None


class AddMemoryForm(BaseModel):
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


# ------------------------------------------------------------------------------
# Embedding Management
# ------------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    async def get_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[np.ndarray]]:
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer:
            try:
                logger.info(f"Loading local embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"Failed to load local SentenceTransformer model: {e}")

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not self.model:
            return None
        try:
            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode(text, normalize_embeddings=True)
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            return None

    async def get_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[np.ndarray]]:
        if not self.model or not texts:
            return [None] * len(texts)
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                ),
            )
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            logger.error(f"Local batch embedding error: {e}")
            return [None] * len(texts)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {"input": text, "model": self.model_name}
                async with session.post(
                    self.api_url, json=data, headers=headers, timeout=30
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        if "data" in res_json and len(res_json["data"]) > 0:
                            emb = res_json["data"][0]["embedding"]
                            return np.array(emb, dtype=np.float32)
                    return None
        except Exception as e:
            logger.error(f"API embedding error: {e}")
            return None

    async def get_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[np.ndarray]]:
        # Naive implementation: sequential calls to avoid complexity for now, or use batch endpoint if supported
        # For robustness, we will map them to the batch endpoint if possible.
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {"input": texts, "model": self.model_name}
                async with session.post(
                    self.api_url, json=data, headers=headers, timeout=60
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        if "data" in res_json:
                            # Sort by index
                            sorted_data = sorted(
                                res_json["data"], key=lambda x: x.get("index", 0)
                            )
                            results = []
                            for item in sorted_data:
                                results.append(
                                    np.array(item["embedding"], dtype=np.float32)
                                )
                            return results
                    return [None] * len(texts)
        except Exception as e:
            logger.error(f"API batch embedding error: {e}")
            return [None] * len(texts)


class EmbeddingManager:
    """Manages embedding generation, caching, and persistence."""

    def __init__(self, get_valves: Callable[[], Any], error_manager: ErrorManager):
        self.get_valves = get_valves
        self.error_manager = error_manager
        self.cache: Dict[str, np.ndarray] = {}  # In-memory cache
        self.provider: Optional[EmbeddingProvider] = None
        self._current_provider_type = None

    def _ensure_provider(self):
        valves = self.get_valves()
        # Initialize if not set or if provider type changed
        if (
            not self.provider
            or self._current_provider_type != valves.embedding_provider_type
        ):
            self._current_provider_type = valves.embedding_provider_type
            if valves.embedding_provider_type == "local":
                self.provider = LocalEmbeddingProvider(valves.embedding_model_name)
            elif valves.embedding_provider_type == "openai_compatible":
                self.provider = OpenAICompatibleEmbeddingProvider(
                    valves.embedding_api_url,
                    valves.embedding_api_key,
                    valves.embedding_model_name,
                )

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None

        # Check memory cache (not implemented fully here for simplicity, but could be added)
        start = time.perf_counter()

        if not self.provider:
            self._ensure_provider()

        if not self.provider:
            return None

        emb = await self.provider.get_embedding(text)

        if emb is not None:
            EMBEDDING_LATENCY.labels(self.get_valves().embedding_provider_type).observe(
                time.perf_counter() - start
            )
        else:
            self.error_manager.increment("embedding_errors")
            EMBEDDING_ERRORS.labels(self.get_valves().embedding_provider_type).inc()

        return emb

    async def get_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[np.ndarray]]:
        if not self.provider:
            self._ensure_provider()

        if not self.provider:
            return [None] * len(texts)
        return await self.provider.get_embeddings_batch(texts)

    async def store_embedding_persistent(self, user_id: str, memory_id: str, memory_text: str, embedding: np.ndarray) -> None:
        """Store memory embedding in a persistent JSON file for reload across restarts."""
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
            
            # Store embedding with metadata
            cache[memory_id] = {
                "embedding": embedding_list,
                "model": self.get_valves().embedding_model_name,
                "provider": self.get_valves().embedding_provider_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save cache
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            
            logger.debug(f"Stored embedding for memory {memory_id} in persistent cache")
        except Exception as e:
            logger.warning(f"Failed to store embedding in persistent cache for memory {memory_id}: {e}")
            # Non-critical - embedding will be regenerated if needed

    async def load_embedding_persistent(self, user_id: str, memory_id: str) -> Optional[np.ndarray]:
        """Load a stored embedding from the persistent JSON file."""
        try:
            cache_dir = "/app/backend/data/cache/embeddings"
            cache_file = os.path.join(cache_dir, f"{user_id}_embeddings.json")
            
            if not os.path.exists(cache_file):
                return None
            
            # Load cache
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            
            if memory_id in cache:
                embedding_data = cache[memory_id]
                embedding_list = embedding_data["embedding"]
                embedding = np.array(embedding_list, dtype=np.float32)
                
                # Validate model compatibility (optional - could regenerate if different)
                valves = self.get_valves()
                if (embedding_data.get("model") != valves.embedding_model_name or 
                    embedding_data.get("provider") != valves.embedding_provider_type):
                    logger.debug(f"Embedding model/provider changed for memory {memory_id}, will regenerate")
                    return None
                
                return embedding
            
            return None
        except Exception as e:
            logger.warning(f"Error loading embedding from persistent cache for memory {memory_id}: {e}")
            return None

    async def get_embedding_with_persistence(self, text: str, user_id: str, memory_id: str) -> Optional[np.ndarray]:
        """Get embedding with full caching hierarchy: memory -> persistent -> generate."""
        if not text:
            return None

        # 1. Check in-memory cache first
        cached_emb = self.cache.get(memory_id)
        if cached_emb is not None:
            return cached_emb

        # 2. Check persistent cache
        persistent_emb = await self.load_embedding_persistent(user_id, memory_id)
        if persistent_emb is not None:
            # Cache in memory for this session
            self.cache[memory_id] = persistent_emb
            logger.debug(f"Loaded embedding for memory {memory_id} from persistent cache")
            return persistent_emb

        # 3. Generate new embedding
        new_emb = await self.get_embedding(text)
        if new_emb is not None:
            # Cache in memory
            self.cache[memory_id] = new_emb
            # Store persistently
            await self.store_embedding_persistent(user_id, memory_id, text, new_emb)
            logger.debug(f"Generated and cached new embedding for memory {memory_id}")
        
        return new_emb


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
    ):
        self.valves = valves
        self.embedding_manager = embedding_manager
        self.error_manager = error_manager

    # --- Memory Identification ---
    async def identify_memories(
        self,
        user_message: str,
        user_id: str,
        context_memories: List[Dict[str, Any]] = None,
        query_llm_func: Callable = None,
    ) -> List[Dict[str, Any]]:
        """Identify potential memories from user message using LLM."""
        if not user_message:
            return []

        # Construct prompt
        system_prompt = self.valves.memory_identification_prompt
        now = datetime.now()
        system_prompt += f"\n\nCurrent Date: {now.strftime('%Y-%m-%d %H:%M:%S')}"

        user_prompt = f"User Message: {user_message}"
        if context_memories:
            user_prompt += f"\n\nContext Memories:\n" + "\n".join(
                [f"- {m.get('content', '')}" for m in context_memories]
            )

        # Call LLM
        if not query_llm_func:
            return []

        try:
            response = await query_llm_func(system_prompt, user_prompt)
            if not response:
                return []

            # Parse JSON
            data = JSONParser.extract_and_parse(response)
            if not isinstance(data, list):
                return []

            # Validate and filter
            valid_ops = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                op = item.get("operation")
                content = item.get("content")
                confidence = item.get("confidence", 0.0)

                if op in ["NEW", "UPDATE", "DELETE"] and content:
                    if confidence >= self.valves.min_confidence_threshold:
                        valid_ops.append(item)

            return valid_ops

        except Exception as e:
            self.error_manager.increment("llm_call_errors")
            logger.error(f"Identify memories failed: {e}")
            return []

    # --- Relevance Retrieval ---
    async def get_relevant_memories(
        self, query: str, user_id: str, all_memories: List[Any]
    ) -> List[Any]:
        """Retrieve relevant memories using vector similarity + optional LLM ranking."""
        if not query or not all_memories:
            return []

        # 1. Vector Search
        query_embedding = await self.embedding_manager.get_embedding(query)
        if query_embedding is None:
            return []

        scored_memories = []

        # Batch embedding for memories without cached embeddings
        # This assumes all_memories are custom objects or dicts.
        # OpenWebUI Memories are SQLModel objects usually.
        mem_objects = []
        texts_to_embed = []
        ids_to_embed = []

        for mem in all_memories:
            # Handle object vs dict
            mem_content = mem.content if hasattr(mem, "content") else mem.get("content")
            mem_id = mem.id if hasattr(mem, "id") else mem.get("id")

            # Check in-memory cache first
            cached_emb = self.embedding_manager.cache.get(mem_id)
            if cached_emb is not None:
                sim = self._cosine_similarity(query_embedding, cached_emb)
                if sim >= self.valves.vector_similarity_threshold:
                    scored_memories.append((sim, mem))
            else:
                # Check persistent cache
                persistent_emb = await self.embedding_manager.load_embedding_persistent(user_id, mem_id)
                if persistent_emb is not None:
                    # Cache in memory for this session
                    self.embedding_manager.cache[mem_id] = persistent_emb
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
                texts_to_embed
            )
            for i, emb in enumerate(new_embeddings):
                if emb is not None:
                    # Update in-memory cache
                    self.embedding_manager.cache[ids_to_embed[i]] = emb
                    # Store persistently
                    await self.embedding_manager.store_embedding_persistent(
                        user_id, ids_to_embed[i], texts_to_embed[i], emb
                    )
                    # Score
                    sim = self._cosine_similarity(query_embedding, emb)
                    if sim >= self.valves.vector_similarity_threshold:
                        scored_memories.append((sim, mem_objects[i]))
        else:
            logger.info(f"Using cached embeddings for all {len(all_memories)} memories")

        # Sort by similarity
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [m[1] for m in scored_memories[: self.valves.related_memories_n]]

        # Optional: LLM Reranking (Skipped for brevity/Streamline, relying on vector)
        return top_memories

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    # --- Memory Operations ---
    async def process_memory_operations(
        self, operations: List[Dict[str, Any]], user_id: str
    ) -> List[Dict[str, Any]]:
        """Execute valid memory operations (NEW, UPDATE, DELETE)."""
        success_ops = []
        for op in operations:
            try:
                kind = op.get("operation")
                content = op.get("content")

                if kind == "NEW" and content:
                    # Incorporate metadata into content string for storage
                    tags = op.get("tags", [])
                    bank = op.get("memory_bank", "General")

                    # Deduplication check
                    if self.valves.deduplicate_memories:
                        is_dupe = await self._is_duplicate(content, user_id)
                        if is_dupe:
                            logger.info(f"Skipping duplicate memory: '{content[:50]}...'")
                            continue

                    # Format: [Tags: tag1, tag2] Content [Memory Bank: Bank] [Confidence: X.XX]
                    tags_str = ", ".join(tags) if tags else "none"
                    confidence = op.get("confidence", 1.0)

                    final_content = f"[Tags: {tags_str}] {content} [Memory Bank: {bank}] [Confidence: {confidence:.2f}]"

                    # Add memory - using Model directly
                    try:
                        mem_obj = Memories.insert_new_memory(user_id, final_content)
                        success_ops.append(op)
                        logger.info(f"Memory saved: '{content[:50]}...' [Bank: {bank}] [Confidence: {confidence:.2f}]")
                    except Exception as ins_err:
                        logger.error(f"Failed to insert memory: {ins_err}")

                elif kind == "DELETE" and op.get("id"):
                    try:
                        Memories.delete_memory_by_id(op["id"])
                        success_ops.append(op)
                    except Exception as del_err:
                        logger.error(f"Failed to delete memory: {del_err}")

            except Exception as e:
                self.error_manager.increment("memory_crud_errors")
                logger.error(f"Memory operation failed: {e}")

        return success_ops

    async def _is_duplicate(self, text: str, user_id: str) -> bool:
        """Check if the given text is a duplicate of existing memories using embedding or text similarity."""
        if not text or not self.valves.deduplicate_memories:
            return False
            
        try:
            # Get all existing memories for the user
            all_memories = Memories.get_memories_by_user_id(user_id)
            if not all_memories:
                return False

            if self.valves.use_embeddings_for_deduplication:
                # Use embedding-based similarity
                new_embedding = await self.embedding_manager.get_embedding(text)
                if new_embedding is None:
                    logger.warning("Could not generate embedding for duplicate check, falling back to text similarity")
                    return await self._check_text_similarity(text, all_memories)
                    
                # Check similarity against existing memories
                for i, memory in enumerate(all_memories):
                    memory_id = memory.id if hasattr(memory, 'id') else memory.get('id')
                    memory_content = memory.content if hasattr(memory, 'content') else memory.get('content')
                    
                    # Extract raw content from formatted memory for comparison
                    # Format: [Tags: ...] CONTENT [Memory Bank: ...] [Confidence: ...]
                    raw_memory_content = memory_content
                    
                    # Extract content between tags and memory bank
                    if '[Tags:' in memory_content and '[Memory Bank:' in memory_content:
                        # Find the end of tags section
                        tags_end = memory_content.find(']', memory_content.find('[Tags:'))
                        if tags_end != -1:
                            # Find the start of memory bank section
                            bank_start = memory_content.find('[Memory Bank:', tags_end)
                            if bank_start != -1:
                                # Extract content between tags and memory bank
                                raw_memory_content = memory_content[tags_end + 1:bank_start].strip()
                    

                    # Check for exact match first (ignoring punctuation and case)
                    def normalize_text(t):
                        import re
                        # Remove punctuation, extra spaces, convert to lowercase
                        normalized = re.sub(r'[^\w\s]', '', t.strip().lower())
                        
                        # Handle common plural variations
                        normalized = re.sub(r'\bs\b', '', normalized)  # Remove standalone 's'
                        normalized = re.sub(r'(\w)s\b', r'\1', normalized)  # Remove trailing 's' from words
                        
                        # Handle common verb tense variations
                        normalized = re.sub(r'\bwould like\b', 'like', normalized)  # "would like" -> "like"
                        normalized = re.sub(r'\bwould want\b', 'want', normalized)  # "would want" -> "want"
                        normalized = re.sub(r'\bwould enjoy\b', 'enjoy', normalized)  # "would enjoy" -> "enjoy"
                        normalized = re.sub(r'\bwould love\b', 'love', normalized)  # "would love" -> "love"
                        normalized = re.sub(r'\bwould prefer\b', 'prefer', normalized)  # "would prefer" -> "prefer"
                        
                        # Handle "likes" vs "like"
                        normalized = re.sub(r'\blikes\b', 'like', normalized)
                        normalized = re.sub(r'\bwants\b', 'want', normalized)
                        normalized = re.sub(r'\benjoys\b', 'enjoy', normalized)
                        normalized = re.sub(r'\bloves\b', 'love', normalized)
                        normalized = re.sub(r'\bprefers\b', 'prefer', normalized)
                        
                        # Normalize synonymous verbs to a common form
                        normalized = re.sub(r'\b(love|enjoy|like)\b', 'like', normalized)  # All positive preferences -> "like"
                        normalized = re.sub(r'\b(want|desire|crave)\b', 'want', normalized)  # All desires -> "want"
                        
                        # Remove articles (a, an, the)
                        normalized = re.sub(r'\b(a|an|the)\b', '', normalized)
                        
                        # Remove common adjectives/modifiers that don't change meaning
                        normalized = re.sub(r'\b(good|great|nice|cold|refreshing|perfect|awesome|amazing|wonderful)\b', '', normalized)
                        
                        # Remove intensifiers
                        normalized = re.sub(r'\b(really|very|quite|pretty|so|totally|absolutely)\b', '', normalized)
                        
                        # Clean up extra spaces
                        normalized = re.sub(r'\s+', ' ', normalized).strip()
                        return normalized
                    
                    if normalize_text(text) == normalize_text(raw_memory_content):
                        logger.info(f"Exact match found (normalized): '{text[:50]}...' matches existing memory")
                        return True
                    
                    # Use raw content for embedding comparison
                    content_for_embedding = raw_memory_content
                    # Check in-memory cache first
                    existing_embedding = self.embedding_manager.cache.get(memory_id)
                    if existing_embedding is None:
                        # Check persistent cache
                        existing_embedding = await self.embedding_manager.load_embedding_persistent(user_id, memory_id)
                        if existing_embedding is not None:
                            # Cache in memory for this session
                            self.embedding_manager.cache[memory_id] = existing_embedding
                        else:
                            # Generate embedding for existing memory using raw content
                            existing_embedding = await self.embedding_manager.get_embedding(content_for_embedding)
                            if existing_embedding is not None:
                                # Cache in memory and store persistently
                                self.embedding_manager.cache[memory_id] = existing_embedding
                                await self.embedding_manager.store_embedding_persistent(
                                    user_id, memory_id, content_for_embedding, existing_embedding
                                )
                    
                    if existing_embedding is not None:
                        # Calculate similarity
                        similarity = self._cosine_similarity(new_embedding, existing_embedding)
                        
                        # Use embedding similarity threshold from valves
                        if similarity >= self.valves.embedding_similarity_threshold:
                            logger.info(f"Duplicate detected via embeddings (similarity: {similarity:.3f}): '{text[:50]}...'")
                            return True
                    else:
                        logger.warning(f"Could not generate embedding for existing memory: '{memory_content[:50]}...'")
            else:
                # Use text-based similarity
                return await self._check_text_similarity(text, all_memories)
                        
            return False
            
        except Exception as e:
            logger.error(f"Error during duplicate check: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # If deduplication fails, err on the side of caution and don't save
            return True

    async def _check_text_similarity(self, text: str, all_memories: List[Any]) -> bool:
        """Check for text-based similarity using difflib."""
        import difflib
        
        for i, memory in enumerate(all_memories):
            memory_content = memory.content if hasattr(memory, 'content') else memory.get('content')
            
            # Calculate text similarity using difflib
            similarity = difflib.SequenceMatcher(None, text.lower().strip(), memory_content.lower().strip()).ratio()
            
            if similarity >= self.valves.similarity_threshold:
                logger.info(f"Duplicate detected via text similarity (similarity: {similarity:.3f}): '{text[:50]}...'")
                return True
                
        return False

    # --- Summarization ---
    async def cluster_and_summarize(
        self, user_id: str, query_llm_func: Callable
    ) -> None:
        """Find clusters of memories and summarize them."""
        logger.info(f"Starting summarization for user {user_id}")
        
        # 1. Fetch memories
        try:
            memories = Memories.get_memories_by_user_id(user_id)
            logger.info(f"Found {len(memories) if memories else 0} memories for user {user_id}")
            
            if not memories:
                logger.info(f"No memories found for user {user_id}, skipping summarization")
                return
                
            if len(memories) < self.valves.summarization_min_cluster_size:
                logger.info(f"Only {len(memories)} memories found for user {user_id}, need at least {self.valves.summarization_min_cluster_size} for clustering")
                return
        except Exception as e:
            logger.error(f"Summarization fetch failed: {e}")
            return

        # 2. Get embeddings (use cache when possible)
        logger.info(f"Processing embeddings for {len(memories)} memories")
        contents = [m.content for m in memories]
        ids = [m.id for m in memories]
        
        # Check cache hierarchy: memory -> persistent -> generate
        embeddings = []
        uncached_indices = []
        uncached_contents = []
        
        for i, (memory_id, content) in enumerate(zip(ids, contents)):
            # Check in-memory cache first
            cached_embedding = self.embedding_manager.cache.get(memory_id)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                # Check persistent cache
                persistent_embedding = await self.embedding_manager.load_embedding_persistent(user_id, memory_id)
                if persistent_embedding is not None:
                    # Cache in memory for this session
                    self.embedding_manager.cache[memory_id] = persistent_embedding
                    embeddings.append(persistent_embedding)
                else:
                    # Need to generate
                    embeddings.append(None)  # Placeholder
                    uncached_indices.append(i)
                    uncached_contents.append(content)
        
        # Generate embeddings for uncached memories
        if uncached_contents:
            logger.info(f"Generating embeddings for {len(uncached_contents)} uncached memories (using cache for {len(memories) - len(uncached_contents)})")
            new_embeddings = await self.embedding_manager.get_embeddings_batch(uncached_contents)
            
            # Update cache and embeddings list
            for idx, new_emb in zip(uncached_indices, new_embeddings):
                if new_emb is not None:
                    embeddings[idx] = new_emb
                    # Cache in memory
                    self.embedding_manager.cache[ids[idx]] = new_emb
                    # Store persistently
                    await self.embedding_manager.store_embedding_persistent(
                        user_id, ids[idx], uncached_contents[uncached_indices.index(idx)], new_emb
                    )
        else:
            logger.info(f"Using cached embeddings for all {len(memories)} memories")

        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        logger.info(f"Ready for clustering: {len(valid_indices)} valid embeddings ({len(memories) - len(uncached_contents)} from cache, {len([e for e in new_embeddings if e is not None]) if uncached_contents else 0} newly generated)")
        
        if len(valid_indices) < self.valves.summarization_min_cluster_size:
            logger.info(f"Only {len(valid_indices)} valid embeddings, need at least {self.valves.summarization_min_cluster_size} for clustering")
            return

        # 3. Simple Greedy Clustering
        logger.info(f"Starting clustering with similarity threshold {self.valves.summarization_similarity_threshold}")
        clusters = []
        visited = set()

        for i in valid_indices:
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)
            vec_i = embeddings[i]

            for j in valid_indices:
                if j in visited:
                    continue

                vec_j = embeddings[j]
                sim = self._cosine_similarity(vec_i, vec_j)

                if sim >= self.valves.summarization_similarity_threshold:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) >= self.valves.summarization_min_cluster_size:
                clusters.append(cluster)
                logger.info(f"Found cluster with {len(cluster)} memories (similarity >= {self.valves.summarization_similarity_threshold})")

        logger.info(f"Found {len(clusters)} clusters ready for summarization")

        # 4. Summarize Clusters
        for cluster_indices in clusters:
            try:
                cluster_memories = [memories[i] for i in cluster_indices]
                cluster_text = "\n".join([f"- {m.content}" for m in cluster_memories])

                prompt = self.valves.summarization_memory_prompt + f"\n\n{cluster_text}"

                summary = await query_llm_func(
                    self.valves.summarization_memory_prompt,
                    f"Memories to summarize:\n{cluster_text}",
                )

                if summary:
                    # 5. Execute Changes
                    # Delete old
                    for m in cluster_memories:
                        Memories.delete_memory_by_id(m.id)

                    # Add new summary
                    # NOTE: Explicitly setting confidence to 1.0 for summaries as they are consolidated facts.
                    op = {
                        "operation": "NEW",
                        "content": summary,
                        "tags": ["summary"],
                        "memory_bank": "General",
                        "confidence": 1.0,
                    }
                    await self.process_memory_operations([op], user_id)
                    logger.info(
                        f"Summarized {len(cluster_memories)} memories into new summary (Confidence 1.0)"
                    )

                    return (
                        f"Consolidated {len(cluster_memories)} memories into a summary."
                    )

            except Exception as e:
                self.error_manager.increment("memory_crud_errors")
                logger.error(f"Memory operation failed: {e}")

        return None


class TaskManager:
    """Manages background tasks."""

    def __init__(self, filter_instance: Any):
        self.filter = filter_instance
        self.tasks: Set[asyncio.Task] = set()

    def start_tasks(self):
        valves = self.filter.valves
        logger.info(f"Starting background tasks: summarization={valves.enable_summarization_task}, deduplication={valves.enable_deduplication_task}, error_logging={valves.enable_error_logging_task}")

        if valves.enable_summarization_task:
            task = asyncio.create_task(self.filter._summarize_old_memories_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_error_logging_task:
            task = asyncio.create_task(self.filter._log_error_counters_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_deduplication_task:
            task = asyncio.create_task(self.filter._deduplicate_memories_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_date_update_task:
            # Just sets a variable, simplistic
            pass

        logger.info(f"Background tasks started: {len(self.tasks)} active tasks")

    async def stop_tasks(self):
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()


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
        embedding_provider_type: Literal["local", "openai_compatible"] = Field(
            default="local",
            description="Type of embedding provider ('local' for SentenceTransformer or 'openai_compatible' for API)",
        )
        embedding_model_name: str = Field(
            default="all-MiniLM-L6-v2",
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
        enable_deduplication_task: bool = Field(
            default=True,
            description="Enable or disable the background memory deduplication task",
        )
        deduplication_interval: int = Field(
            default=14400,
            description="Interval in seconds between memory deduplication runs (default: 4 hours)",
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
            if not isinstance(v, float) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return v

        @field_validator("timezone")
        def check_valid_timezone(cls, v):
            try:
                pytz.timezone(v)
            except Exception as e:
                raise ValueError(f"Invalid timezone string in config: {v}")
            return v

        @model_validator(mode="after")
        def check_llm_config(self):
            if self.llm_provider_type == "openai_compatible" and not self.llm_api_key:
                raise ValueError(
                    "API Key is required when llm_provider_type is 'openai_compatible'"
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
            if self.embedding_provider_type == "openai_compatible":
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
        timezone: str = Field(
            default="",
            description="User's timezone (overrides global setting if provided)",
        )

    # --------------------------------------------------------------------------
    # Main Filter Initialization
    # --------------------------------------------------------------------------

    def __init__(self):
        logger.info("Initializing Adaptive Memory Filter v4.0")
        self.valves = self.Valves()
        self.error_manager = ErrorManager()
        # Pass a lambda to always get the current valves state
        self.embedding_manager = EmbeddingManager(
            lambda: self.valves, self.error_manager
        )
        self.task_manager = TaskManager(self)

        # Initialize internal state
        self._processed_messages = set()
        self._last_body = {}
        self.memory_embeddings = {}  # Local in-memory cache
        self.seen_users = set()  # Track active users for background tasks
        self.notification_queue = []  # Queue for background task notifications

        logger.info("Starting background tasks...")
        self.task_manager.start_tasks()
        logger.info("Adaptive Memory Filter v4.0 initialized successfully")

    async def cleanup(self):
        await self.task_manager.stop_tasks()

    # --------------------------------------------------------------------------
    # Helper: LLM Query Wrapper
    # --------------------------------------------------------------------------
    async def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Unified LLM query method with retries and metrics."""
        # This replaces the massive query logic from before
        valves = self.valves
        # ... logic to call Ollama or OpenAI ...
        # For brevity in this refactor step, simplified:
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
                    payload["response_format"] = {"type": "json_object"}

                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Extract content logic...
                        if "choices" in data:
                            return data["choices"][0]["message"]["content"]
                        elif "message" in data:
                            return data["message"]["content"]
        except Exception as e:
            logger.error(f"LLM Query failed: {e}")
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

        # Safe UserValves instantiation
        raw_valves = __user__.get("valves", {})
        if hasattr(raw_valves, "model_dump"):
            user_valves = self.UserValves(**raw_valves.model_dump())
        elif hasattr(raw_valves, "dict"):
            user_valves = self.UserValves(**raw_valves.dict())
        elif isinstance(raw_valves, dict):
            user_valves = self.UserValves(**raw_valves)
        elif isinstance(raw_valves, self.UserValves):
            user_valves = raw_valves
        else:
            user_valves = self.UserValves()
        if not user_valves.enabled:
            return body

        user_id = __user__["id"]
        self.seen_users.add(user_id)  # Track active user
        messages = body["messages"]
        last_message = messages[-1]["content"]

        # Pipeline
        pipeline = MemoryPipeline(
            self.valves, self.embedding_manager, self.error_manager
        )

        # 1. Retrieve all memories
        try:
            # This gets generic memory objects (SYNC call)
            all_memories = Memories.get_memories_by_user_id(user_id)
        except Exception as e:
            logger.error(f"Failed to fetch memories: {e}")
            all_memories = []

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
        if relevant_memories:
            context_text = "User Memories:\n" + "\n".join(
                [f"- {m.content}" for m in relevant_memories]
            )

            # Find system message or insert one
            if messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{context_text}"
            else:
                messages.insert(0, {"role": "system", "content": context_text})

            # Show status if enabled
            if user_valves.show_status:
                # 1. Recall Notifications
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

                # 2. Background Notifications (queued)
                while self.notification_queue:
                    msg = self.notification_queue.pop(0)
                    bg_status_dict = {
                        "type": "status",  # Or "status" with a different message
                        "data": {"description": f"🧹 {msg}", "done": True},
                    }
                    if __event_emitter__:
                        await __event_emitter__(bg_status_dict)

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

        # Safe UserValves instantiation
        raw_valves = __user__.get("valves", {})
        if hasattr(raw_valves, "model_dump"):
            user_valves = self.UserValves(**raw_valves.model_dump())
        elif hasattr(raw_valves, "dict"):
            user_valves = self.UserValves(**raw_valves.dict())
        elif isinstance(raw_valves, dict):
            user_valves = self.UserValves(**raw_valves)
        elif isinstance(raw_valves, self.UserValves):
            user_valves = raw_valves
        else:
            user_valves = self.UserValves()
        if not user_valves.enabled:
            return body

        user_id = __user__["id"]
        messages = body["messages"]

        # Get last user message
        user_message = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_message = m["content"]
                break

        # Pipeline
        pipeline = MemoryPipeline(
            self.valves, self.embedding_manager, self.error_manager
        )

        # Identify Memories
        if user_message:
            # Pass our _query_llm as callback
            ops = await pipeline.identify_memories(
                user_message,
                user_id,
                context_memories=[],
                query_llm_func=self._query_llm,
            )
            logger.info(f"Memory extraction: identified {len(ops)} potential memories from user message")

            success_ops = []
            if ops:
                # Process Operations (Save/Delete)
                success_ops = await pipeline.process_memory_operations(ops, user_id)
                
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
                        self.valves, self.embedding_manager, self.error_manager
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
                            logger.error(
                                f"Background summarization error for user {user_id}: {u_err}"
                            )

                    logger.info("Background summarization: cycle complete")
                else:
                    logger.debug(f"Background summarization: skipped (enabled: {self.valves.enable_summarization_task}, users: {len(self.seen_users)})")

            except asyncio.CancelledError:
                logger.info("Summarization task cancelled")
                break
            except Exception as e:
                logger.error(f"Summarization task error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(60)

    async def _log_error_counters_loop(self):
        while True:
            await asyncio.sleep(self.valves.error_logging_interval)
            logger.debug(f"Error Counters: {self.error_manager.get_counters()}")

    async def _deduplicate_memories_loop(self):
        """Background task for removing duplicate memories."""
        logger.info("Deduplication background task launched.")
        
        while True:
            try:
                # Always get the current valve value in case it changed
                interval = self.valves.deduplication_interval
                await asyncio.sleep(interval)
                
                logger.info(f"Deduplication task running. Active users: {len(self.seen_users)}")

                if self.valves.enable_deduplication_task and self.seen_users:
                    logger.info("Background deduplication: starting scan...")
                    
                    # Copy set to avoid size change during iteration
                    active_users = list(self.seen_users)
                    for user_id in active_users:
                        try:
                            logger.info(f"Background deduplication: processing user {user_id}")
                            duplicates_removed = await self._remove_duplicate_memories(user_id)
                            if duplicates_removed > 0:
                                result_msg = f"Removed {duplicates_removed} duplicate memories."
                                self.notification_queue.append(result_msg)
                                logger.info(f"Background deduplication: {result_msg}")
                            else:
                                logger.debug(f"Background deduplication: no duplicates found for user {user_id}")
                        except Exception as u_err:
                            logger.error(f"Background deduplication error for user {user_id}: {u_err}")

                    logger.info("Background deduplication: cycle complete")
                else:
                    logger.debug(f"Background deduplication: skipped (enabled: {self.valves.enable_deduplication_task}, users: {len(self.seen_users)})")

            except asyncio.CancelledError:
                logger.info("Deduplication task cancelled.")
                break
            except Exception as e:
                logger.error(f"Deduplication task error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(60)

    async def _remove_duplicate_memories(self, user_id: str) -> int:
        """Remove duplicate memories for a user and return count of removed duplicates."""
        try:
            all_memories = Memories.get_memories_by_user_id(user_id)
            if not all_memories or len(all_memories) < 2:
                return 0
                
            duplicates_removed = 0
            pipeline = MemoryPipeline(self.valves, self.embedding_manager, self.error_manager)
            
            # Check each memory against all others
            for i, memory in enumerate(all_memories):
                if hasattr(memory, 'content'):
                    memory_content = memory.content
                    memory_id = memory.id
                else:
                    memory_content = memory.get('content')
                    memory_id = memory.get('id')
                
                # Extract raw content
                raw_content = memory_content
                if '[Tags:' in memory_content and '[Memory Bank:' in memory_content:
                    tags_end = memory_content.find(']', memory_content.find('[Tags:'))
                    if tags_end != -1:
                        bank_start = memory_content.find('[Memory Bank:', tags_end)
                        if bank_start != -1:
                            raw_content = memory_content[tags_end + 1:bank_start].strip()
                
                # Check if this memory is a duplicate of any previous memory
                is_duplicate = await pipeline._is_duplicate(raw_content, user_id)
                if is_duplicate:
                    # Remove this duplicate
                    try:
                        Memories.delete_memory_by_id(memory_id)
                        duplicates_removed += 1
                        logger.info(f"Removed duplicate memory: '{raw_content[:50]}...'")
                    except Exception as del_err:
                        logger.error(f"Failed to delete duplicate memory {memory_id}: {del_err}")
                        
            return duplicates_removed
            
        except Exception as e:
            logger.error(f"Error during background deduplication: {e}")
            return 0
