import asyncio
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

    _install_module("aiohttp", ClientSession=MagicMock, ClientError=Exception)
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

    open_webui = _install_module("open_webui")
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


class TestHelpers(unittest.TestCase):
    def test_truncate_text_respects_short_limits(self):
        self.assertEqual(am.truncate_text("Hello", 3), "Hel")
        self.assertEqual(am.truncate_text("Hello", 2), "He")
        self.assertEqual(am.truncate_text("Hello", 1), "H")
        self.assertEqual(am.truncate_text("Hello", 0), "")

    def test_truncate_text_uses_ellipsis_when_room_exists(self):
        self.assertEqual(am.truncate_text("Hello World", 10), "Hello W...")
        self.assertEqual(am.truncate_text("Hello", 5), "Hello")

    def test_parse_stored_memory_formatted_text(self):
        record = am.parse_stored_memory(
            "[Tags: coding, none, Python] I like Python [Memory Bank: Work] [Confidence: 0.95]"
        )
        self.assertEqual(record.content, "I like Python")
        self.assertEqual(record.tags, ["coding", "python"])
        self.assertEqual(record.memory_bank, "Work")
        self.assertEqual(record.confidence, 0.95)

    def test_secret_value_unwraps_secretstr(self):
        self.assertEqual(am.secret_value(MockSecretStr("abc123")), "abc123")
        self.assertEqual(am.secret_value("plain"), "plain")
        self.assertIsNone(am.secret_value(None))


class TestLRUCache(unittest.TestCase):
    def test_get_updates_recently_used_entry(self):
        async def run_test():
            cache = am.LRUCache(max_size=2)
            await cache.set("a", "first")
            await cache.set("b", "second")
            self.assertEqual(await cache.get("a"), "first")
            await cache.set("c", "third")
            self.assertIsNone(await cache.get("b"))
            self.assertEqual(await cache.get("a"), "first")
            self.assertEqual(await cache.get("c"), "third")

        asyncio.run(run_test())

    def test_set_existing_key_updates_value_and_recentness(self):
        async def run_test():
            cache = am.LRUCache(max_size=2)
            await cache.set("a", "first")
            await cache.set("b", "second")
            await cache.set("a", "updated")
            await cache.set("c", "third")
            self.assertEqual(await cache.get("a"), "updated")
            self.assertIsNone(await cache.get("b"))

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
