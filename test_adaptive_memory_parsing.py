import unittest
import sys
from unittest.mock import MagicMock
import importlib.util
import os

# Mock external dependencies
mock_modules = [
    "numpy",
    "aiohttp",
    "pydantic",
    "pytz",
    "open_webui",
    "open_webui.config",
    "open_webui.models.memories",
    "open_webui.models.users",
    "open_webui.main",
    "open_webui.retrieval.vector.factory",
    "open_webui.routers.memories",
    "prometheus_client",
    "sentence_transformers"
]

for mod_name in mock_modules:
    sys.modules[mod_name] = MagicMock()

# Pydantic specific mocks needed for the module to load
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @classmethod
    def model_validate(cls, obj):
        return obj

sys.modules["pydantic"].BaseModel = MockBaseModel
sys.modules["pydantic"].Field = MagicMock()
sys.modules["pydantic"].model_validator = lambda *args, **kwargs: lambda x: x
sys.modules["pydantic"].field_validator = lambda *args, **kwargs: lambda x: x

# Mock numpy.ndarray for isinstance checks
class MockNdArray:
    pass
sys.modules["numpy"].ndarray = MockNdArray

# Import the module with dot in name
module_name = "adaptive_memory"
file_path = "adaptive_memory_v4.0.py"
spec = importlib.util.spec_from_file_location(module_name, file_path)
am = importlib.util.module_from_spec(spec)
sys.modules[module_name] = am
spec.loader.exec_module(am)

class TestMemoryParsing(unittest.TestCase):
    def test_parse_stored_memory_none(self):
        result = am.parse_stored_memory(None)
        self.assertEqual(result.content, "")
        self.assertEqual(result.tags, [])
        self.assertEqual(result.memory_bank, "General")
        self.assertIsNone(result.confidence)

    def test_parse_stored_memory_empty(self):
        result = am.parse_stored_memory("")
        self.assertEqual(result.content, "")
        self.assertEqual(result.tags, [])

    def test_parse_stored_memory_plain_text(self):
        text = "Just some plain text without formatting"
        result = am.parse_stored_memory(text)
        self.assertEqual(result.content, text)
        self.assertEqual(result.tags, [])
        self.assertEqual(result.memory_bank, "General")

    def test_parse_stored_memory_happy_path(self):
        text = "[Tags: coding, python] I love programming [Memory Bank: Personal] [Confidence: 0.95]"
        result = am.parse_stored_memory(text)
        self.assertEqual(result.content, "I love programming")
        self.assertEqual(result.tags, ["coding", "python"])
        self.assertEqual(result.memory_bank, "Personal")
        self.assertEqual(result.confidence, 0.95)

    def test_parse_stored_memory_with_none_tag(self):
        text = "[Tags: none, Python] Content [Memory Bank: General] [Confidence: 0.8]"
        result = am.parse_stored_memory(text)
        self.assertEqual(result.tags, ["python"])

    def test_parse_stored_memory_case_and_whitespace(self):
        text = "[Tags:  Python , AI  ]  The content  [Memory Bank: Work] [Confidence: 1.0]"
        result = am.parse_stored_memory(text)
        self.assertEqual(result.content, "The content")
        self.assertEqual(result.tags, ["python", "ai"])
        self.assertEqual(result.memory_bank, "Work")
        self.assertEqual(result.confidence, 1.0)

    def test_parse_stored_memory_invalid_confidence(self):
        text = "[Tags: test] Content [Memory Bank: General] [Confidence: high]"
        result = am.parse_stored_memory(text)
        self.assertEqual(result.content, "Content")
        self.assertIsNone(result.confidence)

    def test_parse_stored_memory_multiline_content(self):
        text = "[Tags: work]\nLine 1\nLine 2\n[Memory Bank: Project X] [Confidence: 0.7]"
        result = am.parse_stored_memory(text)
        self.assertEqual(result.content, "Line 1\nLine 2")
        self.assertEqual(result.tags, ["work"])
        self.assertEqual(result.memory_bank, "Project X")
        self.assertEqual(result.confidence, 0.7)

    def test_parse_stored_memory_missing_bank_or_confidence(self):
        # The regex requires all components to be present
        text = "[Tags: work] Content [Memory Bank: General]"
        result = am.parse_stored_memory(text)
        # Should fall back to plain text parsing if pattern doesn't match
        self.assertEqual(result.content, text)
        self.assertEqual(result.tags, [])

if __name__ == "__main__":
    unittest.main()
