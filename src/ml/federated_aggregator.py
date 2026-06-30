"""Federated learning aggregator for synthesizing model updates from anonymized cohorts."""
import json
import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.db.cache import CacheClient
from src.ml.model_arch import build_lstm_attention_model


class FederatedAggregator:
    """Aggregates federated learning updates from client cohorts."""

    def __init__(self, aggregation_method: str = "weighted_average"):
        self.aggregation_method = aggregation_method

    async def aggregate_updates(
        self, updates: List[Dict[str, Any]], model: Optional[BaseModel] = None
    ) -> Optional[Dict[str, Any]]:
        """Aggregate model updates from multiple clients.

        Args:
            updates: List of client updates, each containing 'weights' and 'count'
            model: Optional model instance to apply aggregated weights to

        Returns:
            Aggregated update dict or None if no updates provided
        """
        if not updates:
            return None

        aggregated = aggregate_updates(updates)

        if model and aggregated:
            model.update_weights(aggregated[0]["weights"])

        return aggregated[0] if aggregated else None


def aggregate_updates(updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate client updates using weighted averaging.

    Args:
        updates: List of client updates, each containing 'weights' and 'count'

    Returns:
        List with single aggregated update containing combined weights and total count
    """
    if not updates:
        return []

    if len(updates) == 1:
        return updates

    total_count = sum(update.get("count", 0) for update in updates)
    if total_count == 0:
        return []

    num_weights = len(updates[0].get("weights", []))
    aggregated_weights = [0.0] * num_weights

    for update in updates:
        weights = update.get("weights", [])
        count = update.get("count", 0)

        if len(weights) != num_weights:
            raise ValueError("Inconsistent weight dimensions across updates")

        for i in range(num_weights):
            aggregated_weights[i] += weights[i] * count

    aggregated_weights = [w / total_count for w in aggregated_weights]

    return [{"weights": aggregated_weights, "count": total_count}]


async def get_cohort_updates(
    cohort_id: str, timestamp: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Retrieve and aggregate updates for a specific cohort.

    Args:
        cohort_id: Identifier for the cohort
        timestamp: Optional cutoff timestamp for updates

    Returns:
        List of aggregated updates for the cohort
    """
    updates = []

    try:
        redis_key = f"cohort:{cohort_id}:updates"
        raw_updates = await redis_client.lrange(redis_key, 0, -1)

        for raw_update in raw_updates:
            update = json.loads(raw_update.decode("utf-8"))
            if timestamp is None or update.get("timestamp", 0) <= timestamp:
                updates.append(update)

    except Exception as e:
        raise RuntimeError(f"Failed to retrieve cohort updates: {e}")

    return aggregate_updates(updates) if updates else []