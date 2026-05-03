import unittest

from adaptive_memory_loader import load_adaptive_memory


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
