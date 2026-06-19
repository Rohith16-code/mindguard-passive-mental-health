"""Anomaly detection worker for real-time signal deviation monitoring."""
import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config import settings
from src.db.models import SignalType, WellnessIndex


@dataclass
class RollingStats:
    """Rolling statistics for anomaly detection."""
    window_size: int
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 1.0
    count: int = 0

    def update(self, value: float) -> None:
        """Update rolling statistics with new value."""
        self.values.append(value)
        self.count += 1

        if len(self.values) > self.window_size:
            self.values.pop(0)

        if len(self.values) >= 2:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values, ddof=1) if len(self.values) > 1 else 0.0
            if self.std == 0.0:
                self.std = 1e-6

    def get_zscore(self, value: float) -> float:
        """Calculate z-score for a value."""
        return (value - self.mean) / self.std if self.std > 0 else 0.0


class AnomalyDetector:
    """Real-time anomaly detection for behavioral signals."""

    def __init__(self, signal_type: SignalType, window_size: int = 100):
        self.signal_type = signal_type
        self.window_size = window_size
        self.stats = RollingStats(window_size=window_size)
        self.anomaly_threshold = settings.ANOMALY_ZSCORE_THRESHOLD
        self.min_samples_for_detection = settings.MIN_SAMPLES_FOR_DETECTION
        self.anomaly_history: List[Tuple[float, float]] = []  # (timestamp, zscore)

    def process(self, value: float, timestamp: Optional[float] = None) -> Tuple[bool, float]:
        """
        Process a new signal value and detect anomalies.

        Returns:
            Tuple of (is_anomaly, zscore)
        """
        if timestamp is None:
            timestamp = time.time()

        if self.stats.count < self.min_samples_for_detection:
            self.stats.update(value)
            return False, 0.0

        zscore = self.stats.get_zscore(value)
        is_anomaly = abs(zscore) > self.anomaly_threshold

        self.stats.update(value)
        self.anomaly_history.append((timestamp, zscore))

        if len(self.anomaly_history) > self.window_size:
            self.anomaly_history.pop(0)

        return is_anomaly, zscore

    def get_current_zscore(self, value: float) -> float:
        """Get z-score without updating statistics."""
        return self.stats.get_zscore(value)


class MultiSignalAnomalyDetector:
    """Multi-signal anomaly detection with weighted aggregation."""

    def __init__(self):
        self.detectors: Dict[SignalType, AnomalyDetector] = {}
        self.signal_weights = {
            SignalType.TYPING_RHYTHM: settings.WEIGHT_TYPING_RHYTHM,
            SignalType.APP_USAGE: settings.WEIGHT_APP_USAGE,
            SignalType.SCREEN_EVENTS: settings.WEIGHT_SCREEN_EVENTS,
            SignalType.RESPONSE_LATENCY: settings.WEIGHT_RESPONSE_LATENCY,
        }
        self.last_detection_time: Dict[SignalType, float] = {}
        self.anomaly_cooldown = settings.ANOMALY_COOLDOWN_SECONDS

    def get_detector(self, signal_type: SignalType) -> AnomalyDetector:
        """Get or create detector for signal type."""
        if signal_type not in self.detectors:
            self.detectors[signal_type] = AnomalyDetector(
                signal_type=signal_type,
                window_size=settings.DETECTION_WINDOW_SIZE
            )
        return self.detectors[signal_type]

    def process_signal(self, signal_type: SignalType, value: float, timestamp: Optional[float] = None) -> Tuple[bool, float]:
        """
        Process a single signal and return anomaly status.

        Returns:
            Tuple of (is_anomaly, weighted_zscore)
        """
        detector = self.get_detector(signal_type)
        is_anomaly, zscore = detector.process(value, timestamp)

        if is_anomaly:
            current_time = timestamp or time.time()
            last_time = self.last_detection_time.get(signal_type, 0.0)
            if current_time - last_time < self.anomaly_cooldown:
                is_anomaly = False
            else:
                self.last_detection_time[signal_type] = current_time

        weighted_zscore = zscore * self.signal_weights.get(signal_type, 1.0)
        return is_anomaly, weighted_zscore

    def aggregate_anomalies(self, timestamp: Optional[float] = None) -> Tuple[float, List[Dict]]:
        """
        Aggregate anomalies across all signals to compute wellness index.

        Returns:
            Tuple of (anomaly_score, detailed_anomalies)
        """
        if not self.detectors:
            return 0.0, []

        current_time = timestamp or time.time()
        anomaly_details = []
        total_weighted_zscore = 0.0
        active_signals = 0

        for signal_type, detector in self.detectors.items():
            if detector.stats.count < settings.MIN_SAMPLES_FOR_DETECTION:
                continue

            recent_anomalies = [
                (t, z) for t, z in detector.anomaly_history
                if current_time - t < settings.AGGREGATION_WINDOW_SECONDS
            ]

            if recent_anomalies:
                max_zscore = max(abs(z) for _, z in recent_anomalies)
                weighted_zscore = max_zscore * self.signal_weights.get(signal_type, 1.0)
                total_weighted_zscore += weighted_zscore
                active_signals += 1

                anomaly_details.append({
                    "signal_type": signal_type.value,
                    "max_zscore": max_zscore,
                    "weighted_zscore": weighted_zscore,
                    "anomaly_count": len(recent_anomalies),
                })

        if active_signals == 0:
            return 0.0, []

        normalized_score = min(1.0, total_weighted_zscore / (settings.ANOMALY_ZSCORE_THRESHOLD * active_signals))
        return normalized_score, anomaly_details


async def run_anomaly_detection_worker(
    signal_queue: asyncio.Queue,
    wellness_index_callback,
    stop_event: asyncio.Event,
) -> None:
    """
    Worker loop for processing signals and detecting anomalies.

    Args:
        signal_queue: Queue of (signal_type, value, timestamp) tuples
        wellness_index_callback: Async function to update wellness index
        stop_event: Event to signal worker shutdown
    """
    detector = MultiSignalAnomalyDetector()

    while not stop_event.is_set():
        try:
            signal_data = await asyncio.wait_for(
                signal_queue.get(),
                timeout=settings.WORKER_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            continue

        signal_type, value, timestamp = signal_data
        is_anomaly, weighted_zscore = detector.process_signal(
            signal_type, value, timestamp
        )

        if is_anomaly:
            anomaly_score, details = detector.aggregate_anomalies(timestamp)
            await wellness_index_callback(
                WellnessIndex(
                    timestamp=timestamp,
                    score=1.0 - anomaly_score,
                    anomaly_details=details,
                    is_crisis=anomaly_score >= settings.CRISIS_THRESHOLD,
                )
            )

        signal_queue.task_done()