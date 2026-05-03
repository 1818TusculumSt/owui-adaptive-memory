import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class MockSecretStr:
    def __init__(self, value):
        self._value = value

    def get_secret_value(self):
        return self._value

    def __str__(self):
        return "**********"


def _install_module(name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


def load_adaptive_memory():
    for name in list(sys.modules):
        if name == "adaptive_memory_test" or name.startswith("open_webui"):
            sys.modules.pop(name, None)

    np_module = _install_module(
        "numpy",
        ndarray=object,
        float32=float,
        array=lambda value, dtype=None: value,
        asarray=lambda value, dtype=None: value,
    )
    np_module.linalg = types.SimpleNamespace(norm=lambda value: 1.0)

    _install_module("aiohttp", ClientSession=MagicMock, ClientError=Exception, ClientTimeout=MagicMock)
    _install_module("pytz", timezone=lambda value: value)
    _install_module("sentence_transformers", SentenceTransformer=None)

    class NoOpMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    _install_module("prometheus_client", Counter=NoOpMetric, Histogram=NoOpMetric)

    def field(default=None, **kwargs):
        return default

    def decorator_factory(*args, **kwargs):
        return lambda func: func

    class MockBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    _install_module(
        "pydantic",
        BaseModel=MockBaseModel,
        Field=field,
        SecretStr=MockSecretStr,
        model_validator=decorator_factory,
        field_validator=decorator_factory,
    )

    _install_module("open_webui")
    _install_module("open_webui.config", DATA_DIR=Path("/tmp/owui-test"))
    _install_module("open_webui.models")
    _install_module(
        "open_webui.models.memories",
        Memories=types.SimpleNamespace(),
    )
    _install_module("open_webui.models.users", Users=types.SimpleNamespace())
    _install_module("open_webui.main", app=types.SimpleNamespace(state=types.SimpleNamespace()))
    _install_module("open_webui.routers")
    _install_module("open_webui.routers.memories", add_memory=None, AddMemoryForm=None)
    _install_module("open_webui.retrieval")
    _install_module("open_webui.retrieval.vector")
    _install_module("open_webui.retrieval.vector.factory", VECTOR_DB_CLIENT=None)

    module_path = Path(__file__).resolve().parents[1] / "adaptive_memory_v4.0.py"
    spec = importlib.util.spec_from_file_location("adaptive_memory_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["adaptive_memory_test"] = module
    spec.loader.exec_module(module)
    return module


am = load_adaptive_memory()


class TestExtractMemoryId(unittest.TestCase):
    def test_extract_memory_id_none(self):
        self.assertIsNone(am.extract_memory_id(None))

    def test_extract_memory_id_has_id_attr(self):
        class Memory:
            def __init__(self, mem_id):
                self.id = mem_id

        mem = Memory("test-id")
        self.assertEqual(am.extract_memory_id(mem), "test-id")

    def test_extract_memory_id_has_id_attr_none(self):
        class Memory:
            def __init__(self, mem_id):
                self.id = mem_id

        mem = Memory(None)
        self.assertIsNone(am.extract_memory_id(mem))

    def test_extract_memory_id_has_get_method(self):
        mem = {"id": "test-id-dict"}
        self.assertEqual(am.extract_memory_id(mem), "test-id-dict")

    def test_extract_memory_id_has_get_method_none(self):
        mem = {"id": None}
        self.assertIsNone(am.extract_memory_id(mem))

    def test_extract_memory_id_has_get_raises_exception(self):
        class BadMemory:
            def get(self, key):
                raise Exception("Something went wrong")

        mem = BadMemory()
        self.assertIsNone(am.extract_memory_id(mem))

    def test_extract_memory_id_no_id_no_get(self):
        class NoIdNoGet:
            pass

        mem = NoIdNoGet()
        self.assertIsNone(am.extract_memory_id(mem))

    def test_extract_memory_id_normalizes_id(self):
        mem = {"id": 12345}
        # normalize_memory_id converts to str
        self.assertEqual(am.extract_memory_id(mem), "12345")


if __name__ == "__main__":
    unittest.main()
