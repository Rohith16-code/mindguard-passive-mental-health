"""LRU cache implementation for recent features."""
from typing import Any, Dict, List, Optional
from collections import OrderedDict


class LRUCache:
    """Thread-safe LRU cache for storing recent feature data."""

    def __init__(self, max_size: int = 100):
        """Initialize LRU cache with maximum size."""
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = None

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache and move to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Add or update item in cache, evicting least recently used if necessary."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache

    def keys(self) -> List[str]:
        """Return list of keys in order from oldest to newest."""
        return list(self.cache.keys())

    def values(self) -> List[Any]:
        """Return list of values in order from oldest to newest."""
        return list(self.cache.values())


class CacheClient:
    """Redis-like cache client wrapping LRUCache for local use."""

    def __init__(self, max_size: int = 1000):
        self._lru = LRUCache(max_size=max_size)

    def get(self, key: str) -> Optional[Any]:
        return self._lru.get(key)

    def set(self, key: str, value: Any) -> None:
        self._lru.put(key, value)

    def delete(self, key: str) -> bool:
        if key in self._lru:
            self._lru.cache.pop(key, None)
            return True
        return False

    def exists(self, key: str) -> bool:
        return key in self._lru

    def keys(self, pattern: str = "*") -> List[str]:
        return self._lru.keys()

    def flushall(self) -> None:
        self._lru.clear()