import re
import difflib
from typing import List, Any, Optional

# Mocking the necessary parts of the MemoryPipeline and Valves
class MockValves:
    def __init__(self):
        self.deduplicate_memories = True
        self.use_embeddings_for_deduplication = False # Use text for this test
        self.similarity_threshold = 0.95
        self.embedding_similarity_threshold = 0.75

class MockMemory:
    def __init__(self, id, content):
        self.id = id
        self.content = content

class MockPipeline:
    def __init__(self, valves):
        self.valves = valves

    def normalize_text(self, t):
        # Implementation from the modified adaptive_memory_v4.0.py
        normalized = re.sub(r'[^\w\s]', '', t.strip().lower())
        normalized = re.sub(r'\bs\b', '', normalized)
        normalized = re.sub(r'\b(a|an|the)\b', '', normalized)
        normalized = re.sub(r'\b(really|very|quite|pretty|so|totally|absolutely)\b', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    async def _is_duplicate(self, text: str, memories: List[Any], exclude_id: str = None) -> bool:
        if not text or not self.valves.deduplicate_memories:
            return False

        # Simplified logic for testing normalization and exclude_id
        normalized_new = self.normalize_text(text)
        
        for memory in memories:
            if exclude_id and str(memory.id) == str(exclude_id):
                continue
            
            # Extract raw content logic (simplified for test)
            raw_content = memory.content
            if '[Tags:' in memory.content and '[Memory Bank:' in memory.content:
                tags_end = memory.content.find(']', memory.content.find('[Tags:'))
                if tags_end != -1:
                    bank_start = memory.content.find('[Memory Bank:', tags_end)
                    if bank_start != -1:
                        raw_content = memory.content[tags_end + 1:bank_start].strip()

            if normalized_new == self.normalize_text(raw_content):
                return True
            
            # difflib check
            sim = difflib.SequenceMatcher(None, normalized_new, self.normalize_text(raw_content)).ratio()
            if sim >= self.valves.similarity_threshold:
                return True
                
        return False

async def run_tests():
    valves = MockValves()
    pipeline = MockPipeline(valves)
    
    # Test 1: Self-deletion prevention
    mem1 = MockMemory("1", "[Tags: none] User likes cold coffee [Memory Bank: General] [Confidence: 1.00]")
    memories = [mem1]
    
    # This should be True if exclude_id is NOT passed (the bug)
    is_dupe_with_bug = await pipeline._is_duplicate("User likes cold coffee", memories)
    print(f"Test 1a (Buggy behavior): {is_dupe_with_bug} (Expected: True)")
    
    # This should be False because we exclude the memory itself
    is_dupe_fixed = await pipeline._is_duplicate("User likes cold coffee", memories, exclude_id="1")
    print(f"Test 1b (Fixed behavior): {is_dupe_fixed} (Expected: False)")
    
    # Test 2: Aggressive normalization softening
    # Old logic would strip 'cold' and match 'User likes cold coffee' with 'User likes hot coffee' if both were normalized to 'user like coffee'
    # New logic keeps adjectives.
    
    t1 = "User likes cold coffee"
    t2 = "User likes hot coffee"
    
    print(f"Test 2: Normalization comparison")
    print(f"  '{t1}' -> '{pipeline.normalize_text(t1)}'")
    print(f"  '{t2}' -> '{pipeline.normalize_text(t2)}'")
    
    if pipeline.normalize_text(t1) != pipeline.normalize_text(t2):
        print("Success: Adjectives preserved, distinct meanings maintained.")
    else:
        print("Failure: Adjectives stripped, distinct meanings collapsed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_tests())
