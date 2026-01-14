## 2025-02-18 - Path Traversal in File-Based Cache
**Vulnerability:** The `store_embedding_persistent` method used an unsanitized `user_id` to construct file paths for caching embeddings. This allowed potential path traversal attacks where a malicious `user_id` could cause files to be written outside the intended cache directory.
**Learning:** Assumptions about the safety of inputs (like `user_id` coming from a trusted source) can be dangerous in plugin architectures where context might change. File system operations are high-risk sinks.
**Prevention:** Always apply strict sanitization (e.g., `os.path.basename`, regex allowlists) to any dynamic input used in file paths, regardless of its origin.
