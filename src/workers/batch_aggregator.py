"""Sliding-window feature aggregation for mental health crisis detection."""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import asyncio
import logging

import sqlite3
from tortoise.models import Model
from tortoise.fields import (
    IntField,
    DatetimeField,
    JSONField,
    CharField,
    FloatField,
)


logger = logging.getLogger(__name__)


@dataclass
class AggregatedFeatures:
    """Aggregated feature window."""
    user_id: str
    window_start: datetime
    window_end: datetime
    features: Dict[str, float]
    sample_count: int
    aggregation_method: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class FeatureWindow(Model):
    """Database model for storing aggregated feature windows."""
    id = IntField(pk=True)
    user_id = CharField(max_length=255, index=True)
    window_start = DatetimeField()
    window_end = DatetimeField()
    features = JSONField()
    sample_count = IntField()
    aggregation_method = CharField(max_length=50)
    created_at = DatetimeField(auto_now_add=True)

    class Meta:
        table = "feature_windows"
        indexes = [("user_id", "window_start", "window_end")]


class BatchAggregator:
    """Sliding-window feature aggregator for sensor data."""

    def __init__(
        self,
        window_size_minutes: int = 60,
        step_size_minutes: int = 15,
        max_samples: int = 1000,
        aggregation_methods: Optional[Dict[str, str]] = None,
    ):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.step_size = timedelta(minutes=step_size_minutes)
        self.max_samples = max_samples
        self.aggregation_methods = aggregation_methods or {
            "mean": "mean",
            "std": "std",
            "min": "min",
            "max": "max",
            "sum": "sum",
            "count": "count",
        }
        self._buffers: Dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def add_samples(self, user_id: str, samples: List[Dict[str, Any]]) -> None:
        """Add sensor samples to aggregation buffer."""
        async with self._lock:
            if user_id not in self._buffers:
                self._buffers[user_id] = deque(maxlen=self.max_samples)

            for sample in samples:
                sample_time = sample.get("timestamp")
                if sample_time is None:
                    continue
                if isinstance(sample_time, str):
                    sample["timestamp"] = datetime.fromisoformat(
                        sample_time.replace("Z", "+00:00")
                    )
                self._buffers[user_id].append(sample)

    async def aggregate_window(
        self, user_id: str, window_start: datetime, window_end: datetime
    ) -> Optional[AggregatedFeatures]:
        """Aggregate features for a specific time window."""
        async with self._lock:
            if user_id not in self._buffers:
                return None

            buffer = self._buffers[user_id]
            filtered = [
                s
                for s in buffer
                if window_start <= s.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)) < window_end
            ]

            if not filtered:
                return None

            features = self._compute_features(filtered)
            return AggregatedFeatures(
                user_id=user_id,
                window_start=window_start,
                window_end=window_end,
                features=features,
                sample_count=len(filtered),
                aggregation_method="sliding_window",
            )

    def _compute_features(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute statistical features from samples."""
        features: Dict[str, float] = {}

        if not samples:
            return features

        # Extract numeric values from samples
        numeric_values: Dict[str, List[float]] = {}
        for sample in samples:
            for key, value in sample.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in numeric_values:
                        numeric_values[key] = []
                    numeric_values[key].append(value)

        # Compute aggregations per feature
        for feature_name, values in numeric_values.items():
            if not values:
                continue

            # Mean
            features[f"{feature_name}_mean"] = sum(values) / len(values)

            # Std (sample std)
            if len(values) > 1:
                mean = features[f"{feature_name}_mean"]
                variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                features[f"{feature_name}_std"] = variance ** 0.5

            # Min/Max
            features[f"{feature_name}_min"] = min(values)
            features[f"{feature_name}_max"] = max(values)

            # Sum
            features[f"{feature_name}_sum"] = sum(values)

        # Count features
        features["total_samples"] = len(samples)
        features["unique_features"] = len(numeric_values)

        return features

    async def process_all_windows(
        self, user_id: str, now: Optional[datetime] = None
    ) -> List[AggregatedFeatures]:
        """Process all sliding windows for a user up to now."""
        if now is None:
            now = datetime.now(timezone.utc)

        windows = []
        async with self._lock:
            if user_id not in self._buffers or not self._buffers[user_id]:
                return windows

            buffer = self._buffers[user_id]
            buffer_times = [s.get("timestamp") for s in buffer if s.get("timestamp")]
            if not buffer_times:
                return windows

            min_time = min(buffer_times)
            max_time = max(buffer_times)

            # Generate windows
            current_start = min_time.replace(
                minute=(min_time.minute // self.step_size.total_seconds() // 60) * self.step_size.total_seconds() // 60,
                second=0,
                microsecond=0
            )
            current_start = current_start - (current_start - min_time) % self.step_size
            if current_start > min_time:
                current_start -= self.step_size

            while current_start + self.window_size <= now:
                window_end = current_start + self.window_size
                if window_end > max_time:
                    break

                window = await self.aggregate_window(user_id, current_start, window_end)
                if window:
                    windows.append(window)

                current_start += self.step_size

        return windows

    async def persist_windows(
        self, windows: List[AggregatedFeatures]
    ) -> List[FeatureWindow]:
        """Persist aggregated windows to database."""
        persisted = []
        for window in windows:
            try:
                db_window = await FeatureWindow.create(
                    user_id=window.user_id,
                    window_start=window.window_start,
                    window_end=window.window_end,
                    features=window.features,
                    sample_count=window.sample_count,
                    aggregation_method=window.aggregation_method,
                )
                persisted.append(db_window)
            except Exception as e:
                logger.error(f"Failed to persist window: {e}")
        return persisted

    async def get_latest_window(
        self, user_id: str
    ) -> Optional[AggregatedFeatures]:
        """Get the most recent aggregated window for a user."""
        async with self._lock:
            if user_id not in self._buffers:
                return None

            buffer = self._buffers[user_id]
            if not buffer:
                return None

            now = datetime.now(timezone.utc)
            window_end = now
            window_start = now - self.window_size

            return await self.aggregate_window(user_id, window_start, window_end)

    async def clear_buffer(self, user_id: str) -> None:
        """Clear the buffer for a specific user."""
        async with self._lock:
            if user_id in self._buffers:
                self._buffers[user_id].clear()

    async def clear_all_buffers(self) -> None:
        """Clear all buffers."""
        async with self._lock:
            self._buffers.clear()

    def get_buffer_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        return {user_id: len(buffer) for user_id, buffer in self._buffers.items()}