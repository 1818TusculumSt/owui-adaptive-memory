import sys
from unittest.mock import MagicMock
import importlib.util
import os

# Create mock objects for the nested structure
mock_open_webui = MagicMock()
mock_models = MagicMock()
mock_open_webui.models = mock_models

# Mocking modules that are not available or would cause issues during import
mock_modules = {
    "numpy": MagicMock(),
    "aiohttp": MagicMock(),
    "pydantic": MagicMock(),
    "open_webui": mock_open_webui,
    "open_webui.config": MagicMock(),
    "open_webui.models": mock_models,
    "open_webui.models.memories": MagicMock(),
    "open_webui.models.users": MagicMock(),
    "open_webui.main": MagicMock(),
    "open_webui.routers": MagicMock(),
    "open_webui.routers.memories": MagicMock(),
    "open_webui.retrieval": MagicMock(),
    "open_webui.retrieval.vector": MagicMock(),
    "open_webui.retrieval.vector.factory": MagicMock(),
    "pytz": MagicMock(),
    "prometheus_client": MagicMock(),
    "sentence_transformers": MagicMock()
}

for module_name, mock_obj in mock_modules.items():
    sys.modules[module_name] = mock_obj

# Specifically mock Pydantic and other used features if needed
import pydantic
pydantic.BaseModel = MagicMock
pydantic.Field = MagicMock
pydantic.model_validator = lambda **kwargs: lambda x: x
pydantic.field_validator = lambda *args, **kwargs: lambda x: x
pydantic.SecretStr = MagicMock

# Mock numpy.ndarray for isinstance checks
import numpy as np
class MockNdarray: pass
np.ndarray = MockNdarray

# Load the module
module_path = os.path.join(os.getcwd(), "adaptive_memory_v4.0.py")
spec = importlib.util.spec_from_file_location("adaptive_memory", module_path)
adaptive_memory = importlib.util.module_from_spec(spec)
sys.modules["adaptive_memory"] = adaptive_memory
spec.loader.exec_module(adaptive_memory)
truncate_text = adaptive_memory.truncate_text

def test_truncate_text_basic():
    assert truncate_text("Hello World", 11) == "Hello World"
    assert truncate_text("Hello World", 12) == "Hello World"
    assert truncate_text("Hello World", 10) == "Hello W..."
    assert truncate_text("  Hello World  ", 11) == "Hello World"

def test_truncate_text_edge_cases():
    assert truncate_text("", 10) == ""
    assert truncate_text(None, 10) == ""
    assert truncate_text("Hello", 0) == ""
    assert truncate_text("Hello", -1) == ""

def test_truncate_text_boundary():
    # New behavior for max_length=3 and length > 3:
    # stripped[:3] -> "Hel"
    assert truncate_text("Hello", 3) == "Hel"

    # Corrected behavior for max_length < 3:
    assert truncate_text("Hello", 2) == "He"
    assert truncate_text("Hello", 1) == "H"

def test_truncate_text_exact_length():
    assert truncate_text("12345", 5) == "12345"
    assert truncate_text("12345", 4) == "1..."
