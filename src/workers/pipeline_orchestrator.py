"""Pipeline orchestrator module for coordinating async data flow in mental health crisis detection."""
from typing import List, Dict, Any, Callable, Optional
from src.db.async_db import AsyncDatabase
from src.redis.client import RedisClient


class PipelineOrchestrator:
    """Coordinates data flow through the mental health crisis detection pipeline."""

    def __init__(self, db: AsyncDatabase, redis: RedisClient):
        """Initialize orchestrator with database and Redis clients."""
        self.db = db
        self.redis = redis

    async def fetch_data(self, table: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch raw data from database."""
        try:
            return await self.db.fetch(table, limit=limit, offset=offset)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from {table}: {e}")

    async def transform_data(
        self,
        data: List[Dict[str, Any]],
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply transformation function to each data record."""
        if not data:
            return []
        try:
            return [transform_fn(record) for record in data]
        except Exception as e:
            raise RuntimeError(f"Failed to transform data: {e}")

    async def store_data(self, table: str, data: List[Dict[str, Any]]) -> bool:
        """Store processed data to database."""
        if not data:
            return True
        try:
            await self.db.batch_insert(table, data)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to store data to {table}: {e}")

    async def cache_results(self, key: str, results: List[Dict[str, Any]], ttl: int = 3600) -> bool:
        """Cache processed results in Redis."""
        try:
            await self.redis.set(key, results, ex=ttl)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to cache results for key {key}: {e}")

    async def get_cached_results(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached results from Redis."""
        try:
            return await self.redis.get(key)
        except Exception as e:
            raise RuntimeError(f"Failed to get cached results for key {key}: {e}")

    async def run_pipeline(
        self,
        fetch_fn: Callable[[], Any],
        transform_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
        store_fn: Callable[[List[Dict[str, Any]]], Any],
        cache_key: Optional[str] = None,
        ttl: int = 3600
    ) -> bool:
        """Run a complete pipeline: fetch → transform → store → cache."""
        try:
            data = await fetch_fn()
            if not data:
                return True
            transformed = await self.transform_data(data, transform_fn)
            await store_fn(transformed)
            if cache_key:
                await self.cache_results(cache_key, transformed, ttl)
            return True
        except Exception as e:
            raise RuntimeError(f"Pipeline execution failed: {e}")


async def run_pipeline(
    orchestrator: PipelineOrchestrator,
    fetch_fn: Callable[[], Any],
    transform_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    store_fn: Callable[[List[Dict[str, Any]]], Any],
    cache_key: Optional[str] = None,
    ttl: int = 3600
) -> bool:
    """Run a complete pipeline using the orchestrator."""
    return await orchestrator.run_pipeline(fetch_fn, transform_fn, store_fn, cache_key, ttl)


async def process_batch(
    orchestrator: PipelineOrchestrator,
    table: str,
    transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    store_table: str,
    batch_size: int = 100,
    offset: int = 0
) -> int:
    """Process data in batches and return total records processed."""
    total_processed = 0
    while True:
        data = await orchestrator.fetch_data(table, limit=batch_size, offset=offset)
        if not data:
            break
        transformed = await orchestrator.transform_data(data, transform_fn)
        await orchestrator.store_data(store_table, transformed)
        total_processed += len(transformed)
        offset += batch_size
    return total_processed