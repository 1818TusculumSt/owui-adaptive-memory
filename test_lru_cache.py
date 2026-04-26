import sys
import unittest
import asyncio
from unittest.mock import MagicMock

# 1. Mock all dependencies before importing the module
mock_modules = [
    'pytz',
    'numpy',
    'aiohttp',
    'pydantic',
    'prometheus_client',
    'sentence_transformers',
    'open_webui',
    'open_webui.config',
    'open_webui.models.memories',
    'open_webui.models.users',
    'open_webui.main',
    'open_webui.routers.memories',
    'open_webui.retrieval.vector.factory',
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# Specifically mock np.ndarray for type hinting and isinstance checks if any
import numpy as np
np.ndarray = MagicMock

# Mock pydantic.BaseModel and related
import pydantic
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @classmethod
    def model_validate(cls, obj):
        return obj

pydantic.BaseModel = MockBaseModel
pydantic.Field = MagicMock
pydantic.model_validator = lambda **kwargs: lambda f: f
pydantic.field_validator = lambda *args, **kwargs: lambda f: f

# 2. Import LRUCache from the module
import importlib.util
spec = importlib.util.spec_from_file_location("adaptive_memory", "adaptive_memory_v4.0.py")
adaptive_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive_memory)
LRUCache = adaptive_memory.LRUCache

class TestLRUCache(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_get_empty_cache(self):
        cache = LRUCache(max_size=2)
        result = self.loop.run_until_complete(cache.get("key1"))
        self.assertIsNone(result)

    def test_set_and_get(self):
        cache = LRUCache(max_size=2)
        val1 = "value1"
        self.loop.run_until_complete(cache.set("key1", val1))

        result = self.loop.run_until_complete(cache.get("key1"))
        self.assertEqual(result, val1)

    def test_get_nonexistent_key(self):
        cache = LRUCache(max_size=2)
        self.loop.run_until_complete(cache.set("key1", "value1"))
        result = self.loop.run_until_complete(cache.get("key2"))
        self.assertIsNone(result)

    def test_lru_eviction(self):
        # Max size 2
        cache = LRUCache(max_size=2)
        self.loop.run_until_complete(cache.set("key1", "val1"))
        self.loop.run_until_complete(cache.set("key2", "val2"))

        # Access key1 to make it MRU
        self.loop.run_until_complete(cache.get("key1"))

        # Set key3, should evict key2 (since key1 was moved to end by get)
        self.loop.run_until_complete(cache.set("key3", "val3"))

        self.assertIsNone(self.loop.run_until_complete(cache.get("key2")))
        self.assertEqual(self.loop.run_until_complete(cache.get("key1")), "val1")
        self.assertEqual(self.loop.run_until_complete(cache.get("key3")), "val3")

    def test_get_updates_mru(self):
        cache = LRUCache(max_size=2)
        self.loop.run_until_complete(cache.set("key1", "val1"))
        self.loop.run_until_complete(cache.set("key2", "val2"))

        # Order should be key1, key2 (key2 is MRU)
        # Verify order internally
        keys = list(cache._cache.keys())
        self.assertEqual(keys, ["key1", "key2"])

        # Get key1, should move it to end
        self.loop.run_until_complete(cache.get("key1"))
        keys = list(cache._cache.keys())
        self.assertEqual(keys, ["key2", "key1"])

if __name__ == "__main__":
    unittest.main()
