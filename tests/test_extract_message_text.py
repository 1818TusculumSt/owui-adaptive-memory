import unittest
from adaptive_memory_loader import load_adaptive_memory

am = load_adaptive_memory()

class TestExtractMessageText(unittest.TestCase):
    def test_extract_message_text_string(self):
        self.assertEqual(am.extract_message_text("  hello world  "), "hello world")

    def test_extract_message_text_list_of_text_parts(self):
        content = [
            {"type": "text", "text": " hello "},
            {"type": "text", "content": " world "}
        ]
        self.assertEqual(am.extract_message_text(content), "hello world")

    def test_extract_message_text_list_with_non_text_parts(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image", "url": "http://example.com/img.png"}
        ]
        self.assertEqual(am.extract_message_text(content), "hello")

    def test_extract_message_text_list_with_non_dict_items(self):
        content = [
            "unexpected string",
            {"type": "text", "text": "valid text"}
        ]
        self.assertEqual(am.extract_message_text(content), "valid text")

    def test_extract_message_text_none(self):
        self.assertEqual(am.extract_message_text(None), "")

    def test_extract_message_text_other_types(self):
        self.assertEqual(am.extract_message_text(123), "123")
        self.assertEqual(am.extract_message_text(True), "True")

    def test_extract_message_text_empty_list(self):
        self.assertEqual(am.extract_message_text([]), "")

    def test_extract_message_text_list_with_missing_keys(self):
        content = [
            {"type": "text"},
            {"type": "text", "text": None},
            {"type": "text", "content": ""}
        ]
        self.assertEqual(am.extract_message_text(content), "")

if __name__ == "__main__":
    unittest.main()
