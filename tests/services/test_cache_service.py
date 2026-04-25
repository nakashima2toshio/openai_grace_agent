import time
import pytest
from services.cache_service import MemoryCache, cache_result

class TestMemoryCache:
    
    def test_basic_ops(self):
        cache = MemoryCache()
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"
        assert cache.has("k1") is True
        
        assert cache.get("k2") is None
        assert cache.has("k2") is False
        
        cache.delete("k1")
        assert cache.get("k1") is None

    def test_ttl(self):
        # Short TTL
        cache = MemoryCache(ttl=0.1)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"
        
        time.sleep(0.2)
        assert cache.get("k1") is None

    def test_max_size(self):
        cache = MemoryCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        
        # Access k1 to update timestamp? 
        # The implementation uses set timestamp. accessing doesn't update timestamp in _storage for eviction in this simple impl?
        # Let's check impl: `_evict_oldest` uses `timestamp`. `set` sets `timestamp`. `get` does not update it.
        
        # Add k3, should evict oldest (k1)
        time.sleep(0.01)
        cache.set("k3", "v3")
        
        assert cache.size() == 2
        assert cache.get("k1") is None # Evicted
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    def test_decorator(self):
        
        cache = MemoryCache()
        call_count = 0
        
        @cache_result(cache=cache)
        def expensive_func(arg):
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"
        
        # First call
        res1 = expensive_func("a")
        assert res1 == "result-a"
        assert call_count == 1
        
        # Second call (should be cached)
        res2 = expensive_func("a")
        assert res2 == "result-a"
        assert call_count == 1
        
        # Different arg
        res3 = expensive_func("b")
        assert res3 == "result-b"
        assert call_count == 2
