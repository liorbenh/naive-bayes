"""LRU Cache implementation for model caching."""
from collections import OrderedDict
from typing import Any, List, Optional


class LRUCache:
    """Simple LRU Cache implementation."""
    
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache and move to end."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def keys(self) -> List[str]:
        """Get cache keys."""
        return list(self.cache.keys())
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def remove(self, key: str) -> bool:
        """Remove specific key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False