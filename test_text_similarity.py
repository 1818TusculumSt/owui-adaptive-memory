import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import importlib.util
import os
import asyncio

# --- 1. Mock External Dependencies BEFORE importing the main module ---

# Mock numpy
mock_np = MagicMock()
mock_np.ndarray = type('ndarray', (), {})
mock_np.array.return_value = MagicMock()
mock_np.float32 = 'float32'
mock_np.linalg.norm.return_value = 1.0
mock_np.dot.return_value = 1.0
sys.modules['numpy'] = mock_np

# Mock pytz
mock_pytz = MagicMock()
sys.modules['pytz'] = mock_pytz

# Mock pydantic
mock_pydantic = MagicMock()
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @classmethod
    def model_validate(cls, data):
        return cls(**data)
mock_pydantic.BaseModel = MockBaseModel
mock_pydantic.Field = MagicMock()
mock_pydantic.model_validator = lambda **kwargs: lambda f: f
mock_pydantic.field_validator = lambda *args, **kwargs: lambda f: f
sys.modules['pydantic'] = mock_pydantic

# Mock aiohttp
mock_aiohttp = MagicMock()
mock_aiohttp.ClientSession = MagicMock()
mock_aiohttp.ClientError = Exception
sys.modules['aiohttp'] = mock_aiohttp

# Mock prometheus_client
mock_prometheus = MagicMock()
sys.modules['prometheus_client'] = mock_prometheus

# Mock sentence_transformers
sys.modules['sentence_transformers'] = MagicMock()

# Mock open_webui modules
mock_ow_config = MagicMock()
mock_ow_config.DATA_DIR = "/tmp/data"
sys.modules['open_webui.config'] = mock_ow_config

mock_ow_models_memories = MagicMock()
sys.modules['open_webui.models.memories'] = mock_ow_models_memories

mock_ow_models_users = MagicMock()
sys.modules['open_webui.models.users'] = mock_ow_models_users

mock_ow_main = MagicMock()
sys.modules['open_webui.main'] = mock_ow_main

# --- 2. Dynamic Import of the main module ---
spec = importlib.util.spec_from_file_location("adaptive_memory", "adaptive_memory_v4.0.py")
am_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am_module)

# --- 3. Test Cases ---

class TestTextSimilarity(unittest.TestCase):
    def setUp(self):
        # Setup mocks for MemoryPipeline
        self.mock_valves = MagicMock()
        self.mock_valves.similarity_threshold = 0.95
        self.mock_valves.deduplicate_memories = True

        self.mock_embedding_manager = MagicMock()
        self.mock_error_manager = MagicMock()

        # Instantiate MemoryPipeline
        self.pipeline = am_module.MemoryPipeline(
            self.mock_valves,
            self.mock_embedding_manager,
            self.mock_error_manager
        )

    def run_async(self, coro):
        return asyncio.run(coro)

    def test_exact_match(self):
        """Test that two identical strings return True."""
        text = "I love programming in Python"
        all_memories = [
            {'id': '1', 'content': "I love programming in Python"}
        ]

        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories))
        self.assertTrue(result)

    def test_normalization_match(self):
        """Test that normalization handles minor differences like articles."""
        text = "coffee is great"
        all_memories = [
            {'id': '1', 'content': "The coffee is really great"}
        ]
        # _normalize_text should remove 'the' and 'really'
        # "coffee is great" vs "coffee is great"

        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories))
        self.assertTrue(result)

    def test_formatted_content_match(self):
        """Test extraction of raw content from formatted memory string."""
        text = "User enjoys hiking"
        # Format: [Tags: tag1] User enjoys hiking [Memory Bank: General]
        formatted_content = "[Tags: behavior] User enjoys hiking [Memory Bank: General] [Confidence: 0.90]"
        all_memories = [
            {'id': '1', 'content': formatted_content}
        ]

        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories))
        self.assertTrue(result)

    def test_similarity_above_threshold(self):
        """Test strings with high similarity above the default 0.95 threshold."""
        text = "I have a cat named Whiskers"
        all_memories = [
            {'id': '1', 'content': "I have cat named Whiskers"} # Missing 'a'
        ]
        # Normalization removes 'a' anyway

        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories))
        self.assertTrue(result)

    def test_similarity_below_threshold(self):
        """Test strings with low similarity below the threshold."""
        text = "I love apples"
        all_memories = [
            {'id': '1', 'content': "I love oranges"}
        ]

        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories))
        self.assertFalse(result)

    def test_exclude_id(self):
        """Test that memory with exclude_id is ignored."""
        text = "Exact same text"
        all_memories = [
            {'id': 'exclude_me', 'content': "Exact same text"}
        ]

        # Should be False because the only matching memory is excluded
        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories, exclude_id='exclude_me'))
        self.assertFalse(result)

    def test_object_based_memory(self):
        """Test that object-based memories (not just dicts) are handled."""
        class MockMemory:
            def __init__(self, id, content):
                self.id = id
                self.content = content

        text = "Hello world"
        all_memories = [
            MockMemory('1', "Hello world")
        ]

        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories))
        self.assertTrue(result)

    def test_multiple_memories(self):
        """Test with multiple memories where only one matches."""
        text = "Match me"
        all_memories = [
            {'id': '1', 'content': "First memory"},
            {'id': '2', 'content': "Second memory"},
            {'id': '3', 'content': "Match me"},
            {'id': '4', 'content': "Fourth memory"}
        ]

        result = self.run_async(self.pipeline._check_text_similarity(text, all_memories))
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
