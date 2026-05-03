import asyncio
import json
import types
import unittest
from unittest.mock import mock_open, patch

from adaptive_memory_loader import MockSecretStr, load_adaptive_memory


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

    def test_api_key_valves_remain_plain_strings_for_persistence(self):
        annotations = am.Filter.Valves.__annotations__
        self.assertNotIn("SecretStr", str(annotations["embedding_api_key"]))
        self.assertNotIn("SecretStr", str(annotations["mem0_api_key"]))
        self.assertNotIn("SecretStr", str(annotations["llm_api_key"]))

    def test_tag_summarization_clusters_include_prior_summaries(self):
        valves = types.SimpleNamespace(
            summarization_strategy="tags",
            summarization_similarity_threshold=0.7,
            summarization_min_cluster_size=2,
            summarization_max_cluster_size=8,
            allowed_memory_banks=["General", "Personal", "Work"],
            default_memory_bank="General",
            enable_identity_memories=True,
            enable_behavior_memories=True,
            enable_preference_memories=True,
            enable_goal_memories=True,
            enable_relationship_memories=True,
            enable_possession_memories=True,
        )
        pipeline = am.MemoryPipeline(valves, None, am.ErrorManager())
        records = [
            am.StoredMemoryRecord(
                content="User prefers Open WebUI memory tooling.",
                tags=["summary", "preference"],
                memory_bank="Work",
            ),
            am.StoredMemoryRecord(
                content="User likes concise memory debug logs.",
                tags=["preference"],
                memory_bank="Work",
            ),
            am.StoredMemoryRecord(
                content="User wants memory retrieval quality tuned.",
                tags=["preference"],
                memory_bank="Work",
            ),
        ]

        clusters = pipeline._build_summarization_clusters(
            records, [None, None, None], [0, 1, 2]
        )

        self.assertEqual(clusters, [[0, 1, 2]])

    def test_tag_summarization_does_not_merge_summary_tag_alone(self):
        valves = types.SimpleNamespace(
            summarization_strategy="tags",
            summarization_similarity_threshold=0.7,
            summarization_min_cluster_size=2,
            summarization_max_cluster_size=8,
            allowed_memory_banks=["General", "Personal", "Work"],
            default_memory_bank="General",
            enable_identity_memories=True,
            enable_behavior_memories=True,
            enable_preference_memories=True,
            enable_goal_memories=True,
            enable_relationship_memories=True,
            enable_possession_memories=True,
        )
        pipeline = am.MemoryPipeline(valves, None, am.ErrorManager())
        records = [
            am.StoredMemoryRecord(
                content="User has a prior summary.",
                tags=["summary"],
                memory_bank="Work",
            ),
            am.StoredMemoryRecord(
                content="User likes concise memory debug logs.",
                tags=["preference"],
                memory_bank="Work",
            ),
        ]

        clusters = pipeline._build_summarization_clusters(
            records, [None, None], [0, 1]
        )

        self.assertEqual(clusters, [])


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


class TestEmbeddingManager(unittest.TestCase):
    def test_get_embedding_rebuilds_provider_when_valves_change(self):
        class FakeProvider:
            def __init__(self, api_url, api_key, model_name):
                self.api_url = api_url
                self.api_key = api_key
                self.model_name = model_name

            async def get_embedding(self, text, session=None):
                return self.model_name

            async def get_embeddings_batch(self, texts, session=None):
                return [self.model_name for _ in texts]

        original_provider = am.OpenAICompatibleEmbeddingProvider
        am.OpenAICompatibleEmbeddingProvider = FakeProvider
        try:
            valves = types.SimpleNamespace(
                embedding_source="plugin",
                embedding_provider_type="openai_compatible",
                embedding_model_name="model-a",
                embedding_api_url="https://example.invalid/embeddings",
                embedding_api_key="key-a",
            )
            manager = am.EmbeddingManager(lambda: valves, am.ErrorManager())

            async def run_test():
                self.assertEqual(await manager.get_embedding("hello"), "model-a")
                valves.embedding_model_name = "model-b"
                self.assertEqual(await manager.get_embedding("hello"), "model-b")

            asyncio.run(run_test())
        finally:
            am.OpenAICompatibleEmbeddingProvider = original_provider

    def test_legacy_cache_loader_reads_unhashed_filename(self):
        manager = am.EmbeddingManager(
            lambda: types.SimpleNamespace(
                embedding_source="plugin",
                embedding_model_name="model",
                embedding_provider_type="local",
            ),
            am.ErrorManager(),
        )
        manager._legacy_cache_dir = "cache"
        unhashed_file = manager._get_unhashed_legacy_cache_file("legacy-user")
        file_data = json.dumps({"mem1": {"embedding": [1.0]}})
        with patch.object(
            am.os.path,
            "exists",
            side_effect=lambda path: path == unhashed_file,
        ), patch("builtins.open", mock_open(read_data=file_data)):
            cache = manager._load_legacy_cache_sync("legacy-user")
        self.assertEqual(cache["mem1"]["embedding"], [1.0])

    def test_legacy_cache_loader_rejects_unsafe_unhashed_filename(self):
        manager = am.EmbeddingManager(
            lambda: types.SimpleNamespace(
                embedding_source="plugin",
                embedding_model_name="model",
                embedding_provider_type="local",
            ),
            am.ErrorManager(),
        )
        self.assertIsNone(manager._get_unhashed_legacy_cache_file("../unsafe"))


if __name__ == "__main__":
    unittest.main()
