import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib.util
import asyncio
import os

# Mock dependencies before importing the module
mock_modules = [
    'numpy',
    'aiohttp',
    'pydantic',
    'open_webui',
    'open_webui.config',
    'open_webui.models',
    'open_webui.models.memories',
    'open_webui.models.users',
    'open_webui.main',
    'open_webui.routers',
    'open_webui.routers.memories',
    'open_webui.retrieval',
    'open_webui.retrieval.vector',
    'open_webui.retrieval.vector.factory',
    'pytz',
]

for module_name in mock_modules:
    sys.modules[module_name] = MagicMock()

# Pydantic specifically needs some classes
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @classmethod
    def model_validate(cls, obj):
        return obj

sys.modules['pydantic'].BaseModel = MockBaseModel
sys.modules['pydantic'].Field = MagicMock()
sys.modules['pydantic'].model_validator = lambda **kwargs: lambda x: x
sys.modules['pydantic'].field_validator = lambda *args, **kwargs: lambda x: x

# Mock numpy.ndarray for type hinting if needed, though mostly it's used as a type
class MockNdArray:
    pass
sys.modules['numpy'].ndarray = MockNdArray

# Now import the module
module_name = "adaptive_memory"
file_path = "adaptive_memory_v4.0.py"
spec = importlib.util.spec_from_file_location(module_name, file_path)
adaptive_memory = importlib.util.module_from_spec(spec)
sys.modules[module_name] = adaptive_memory
spec.loader.exec_module(adaptive_memory)

LRUCache = adaptive_memory.LRUCache

class TestLRUCache(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_set_and_get_basic(self):
        async def run_test():
            cache = LRUCache(max_size=2)
            await cache.set("key1", "value1")
            val = await cache.get("key1")
            self.assertEqual(val, "value1")

        self.loop.run_until_complete(run_test())

    def test_set_updates_existing_key(self):
        async def run_test():
            cache = LRUCache(max_size=2)
            await cache.set("key1", "value1")
            await cache.set("key1", "value1_updated")
            val = await cache.get("key1")
            self.assertEqual(val, "value1_updated")

        self.loop.run_until_complete(run_test())

    def test_set_evicts_lru(self):
        async def run_test():
            # max_size=2
            cache = LRUCache(max_size=2)
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")

            # This should evict key1 if we add key3
            await cache.set("key3", "value3")

            self.assertIsNone(await cache.get("key1"))
            self.assertEqual(await cache.get("key2"), "value2")
            self.assertEqual(await cache.get("key3"), "value3")

        self.loop.run_until_complete(run_test())

    def test_get_updates_mru(self):
        async def run_test():
            cache = LRUCache(max_size=2)
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")

            # Access key1, making it MRU
            await cache.get("key1")

            # Now add key3, it should evict key2 (the new LRU)
            await cache.set("key3", "value3")

            self.assertEqual(await cache.get("key1"), "value1")
            self.assertIsNone(await cache.get("key2"))
            self.assertEqual(await cache.get("key3"), "value3")

        self.loop.run_until_complete(run_test())

    def test_set_existing_updates_mru(self):
        async def run_test():
            cache = LRUCache(max_size=2)
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")

            # Update key1, making it MRU
            await cache.set("key1", "value1_new")

            # Now add key3, it should evict key2
            await cache.set("key3", "value3")

            self.assertEqual(await cache.get("key1"), "value1_new")
            self.assertIsNone(await cache.get("key2"))
            self.assertEqual(await cache.get("key3"), "value3")

        self.loop.run_until_complete(run_test())

if __name__ == "__main__":
    unittest.main()
