"""Database package — re-exports key components."""
from .cache import CacheClient, LRUCache

# Aliases for backward compat
db = CacheClient
redis_client = CacheClient

def init_db():
    """Initialize database connection."""
    pass

def close_db():
    """Close database connection."""
    pass

__all__ = ["CacheClient", "LRUCache", "db", "redis_client", "init_db", "close_db"]
