"""On-device feature preprocessing for mental health crisis detection."""
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, stdev
import numpy as np


@dataclass
class FeatureStats:
    """Statistics for feature normalization."""
    mean: float
    std: float
    min_val: float
    max_val: float
    count: int = 0


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    window_size: int = 300
    window_step: int = 60
    normalize: bool = True
    fill_missing: bool = True
    missing_value: float = 0.0
    max_std_multiplier: float = 3.0
    feature_names: List[str] = field(default_factory=lambda: [
        'typing_speed', 'typing_variability', 'app_usage_duration',
        'screen_on_duration', 'response_latency', 'interaction_frequency',
        'swipe_velocity', 'tap_pressure', 'app_entropy'
    ])


class FeatureExtractor:
    """Extracts behavioral features from raw sensor data."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._feature_stats: Dict[str, FeatureStats] = {}

    def extract_typing_features(self, keystrokes: List[Dict]) -> Dict[str, float]:
        """Extract typing rhythm features from keystroke events."""
        if not keystrokes:
            return {f'typing_{k}': 0.0 for k in ['speed', 'variability']}

        intervals = []
        for i in range(1, len(keystrokes)):
            try:
                t1 = datetime.fromisoformat(keystrokes[i-1]['timestamp'])
                t2 = datetime.fromisoformat(keystrokes[i]['timestamp'])
                intervals.append((t2 - t1).total_seconds())
            except (ValueError, KeyError):
                continue

        if not intervals:
            return {f'typing_{k}': 0.0 for k in ['speed', 'variability']}

        avg_interval = mean(intervals)
        std_interval = stdev(intervals) if len(intervals) > 1 else 0.0

        return {
            'typing_speed': 1.0 / max(avg_interval, 0.01),
            'typing_variability': std_interval / max(avg_interval, 0.01)
        }

    def extract_app_usage_features(self, app_events: List[Dict]) -> Dict[str, float]:
        """Extract app usage pattern features."""
        if not app_events:
            return {f'app_{k}': 0.0 for k in ['usage_duration', 'entropy']}

        durations = []
        app_counts: Dict[str, int] = {}
        for event in app_events:
            try:
                duration = float(event.get('duration', 0))
                app_name = event.get('app', 'unknown')
                durations.append(duration)
                app_counts[app_name] = app_counts.get(app_name, 0) + 1
            except (ValueError, TypeError):
                continue

        if not durations:
            return {f'app_{k}': 0.0 for k in ['usage_duration', 'entropy']}

        total_duration = sum(durations)
        total_events = len(app_events)
        entropy = 0.0
        for count in app_counts.values():
            p = count / total_events
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            'app_usage_duration': total_duration,
            'app_entropy': entropy
        }

    def extract_screen_features(self, screen_events: List[Dict]) -> Dict[str, float]:
        """Extract screen interaction features."""
        if not screen_events:
            return {f'screen_{k}': 0.0 for k in ['on_duration', 'frequency']}

        on_durations = []
        on_times = []
        for event in screen_events:
            try:
                if event.get('type') == 'screen_on':
                    on_times.append(datetime.fromisoformat(event['timestamp']))
                elif event.get('type') == 'screen_off':
                    off_time = datetime.fromisoformat(event['timestamp'])
                    if on_times:
                        on_durations.append((off_time - on_times.pop()).total_seconds())
            except (ValueError, KeyError):
                continue

        if not on_durations:
            return {f'screen_{k}': 0.0 for k in ['on_duration', 'frequency']}

        total_on_duration = sum(on_durations)
        on_events = len(on_times) + len(on_durations)

        return {
            'screen_on_duration': total_on_duration,
            'screen_frequency': on_events / max(total_on_duration / 3600, 0.01)
        }

    def extract_response_features(self, response_events: List[Dict]) -> Dict[str, float]:
        """Extract response latency features."""
        if not response_events:
            return {'response_latency': 0.0}

        latencies = []
        for event in response_events:
            try:
                latency = float(event.get('latency_ms', 0)) / 1000.0
                latencies.append(latency)
            except (ValueError, TypeError):
                continue

        if not latencies:
            return {'response_latency': 0.0}

        return {'response_latency': mean(latencies)}

    def extract_interaction_features(self, events: List[Dict]) -> Dict[str, float]:
        """Extract general interaction frequency features."""
        if not events:
            return {'interaction_frequency': 0.0}

        timestamps = []
        for event in events:
            try:
                timestamps.append(datetime.fromisoformat(event['timestamp']))
            except (ValueError, KeyError):
                continue

        if len(timestamps) < 2:
            return {'interaction_frequency': 0.0}

        timestamps.sort()
        total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
        frequency = len(timestamps) / max(total_duration / 3600, 0.01)

        return {'interaction_frequency': frequency}

    def extract_all_features(self, data: Dict[str, List]) -> Dict[str, float]:
        """Extract all features from raw data."""
        features = {}

        features.update(self.extract_typing_features(data.get('keystrokes', [])))
        features.update(self.extract_app_usage_features(data.get('app_events', [])))
        features.update(self.extract_screen_features(data.get('screen_events', [])))
        features.update(self.extract_response_features(data.get('response_events', [])))
        features.update(self.extract_interaction_features(data.get('all_events', [])))

        return features


class SlidingWindowProcessor:
    """Processes data using sliding windows for temporal feature extraction."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)

    def process(self, data: List[Dict]) -> List[Dict[str, Union[float, int]]]:
        """Process data using sliding windows."""
        if not data:
            return []

        windows = []
        for i in range(0, len(data) - self.config.window_size + 1, self.config.window_step):
            window_data = data[i:i + self.config.window_size]
            features = self.feature_extractor.extract_all_features(window_data)
            features['timestamp'] = i + self.config.window_size // 2
            windows.append(features)

        return windows


class FeatureNormalizer:
    """Normalizes features using z-score or min-max scaling."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._stats: Dict[str, FeatureStats] = {}

    def fit(self, features: List[Dict[str, float]]) -> None:
        """Fit normalization statistics to data."""
        if not features:
            return

        for feature_name in self.config.feature_names:
            values = [f.get(feature_name, self.config.missing_value) for f in features]
            if not values:
                continue

            values = [v for v in values if not math.isnan(v) and not math.isinf(v)]
            if not values:
                continue

            self._stats[feature_name] = FeatureStats(
                mean=mean(values),
                std=stdev(values) if len(values) > 1 else 1.0,
                min_val=min(values),
                max_val=max(values),
                count=len(values)
            )

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """Transform features using fitted statistics."""
        result = {}
        for feature_name, value in features.items():
            if feature_name not in self._stats:
                result[feature_name] = value
                continue

            stats = self._stats[feature_name]
            if not self.config.normalize:
                result[feature_name] = value
                continue

            if stats.std == 0:
                result[feature_name] = 0.0
                continue

            z_score = (value - stats.mean) / stats.std
            if abs(z_score) > self.config.max_std_multiplier:
                z_score = math.copysign(self.config.max_std_multiplier, z_score)

            result[feature_name] = z_score

        return result

    def fit_transform(self, features: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Fit and transform features."""
        self.fit(features)
        return [self.transform(f) for f in features]

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics."""
        return {
            name: {
                'mean': s.mean,
                'std': s.std,
                'min': s.min_val,
                'max': s.max_val,
                'count': s.count
            }
            for name, s in self._stats.items()
        }


class DataPreprocessor:
    """Main data preprocessor for mental health crisis detection."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.feature_extractor = FeatureExtractor(self.config)
        self.sliding_window = SlidingWindowProcessor(self.config)
        self.normalizer = FeatureNormalizer(self.config)
        self._fitted = False

    def fit(self, data: List[Dict[str, List]]) -> None:
        """Fit preprocessing pipeline to data."""
        extracted_features = [self.feature_extractor.extract_all_features(d) for d in data]
        self.normalizer.fit(extracted_features)
        self._fitted = True

    def preprocess(self, data: Dict[str, List]) -> Dict[str, float]:
        """Preprocess raw data into normalized features."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing data")

        features = self.feature_extractor.extract_all_features(data)
        return self.normalizer.transform(features)

    def preprocess_batch(self, data: List[Dict[str, List]]) -> List[Dict[str, float]]:
        """Preprocess multiple data samples."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing data")

        extracted_features = [self.feature_extractor.extract_all_features(d) for d in data]
        return [self.normalizer.transform(f) for f in extracted_features]

    def preprocess_sliding_window(self, data: List[Dict[str, List]]) -> List[Dict[str, float]]:
        """Preprocess data using sliding windows."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before preprocessing data")

        windows = self.sliding_window.process(data)
        return [self.normalizer.transform(w) for w in windows]

    def save_stats(self, path: str) -> None:
        """Save normalization statistics to file."""
        import json
        stats = self.normalizer.get_stats()
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

    def load_stats(self, path: str) -> None:
        """Load normalization statistics from file."""
        import json
        with open(path, 'r') as f:
            stats = json.load(f)

        for name, s in stats.items():
            self.normalizer._stats[name] = FeatureStats(
                mean=s['mean'],
                std=s['std'],
                min_val=s['min'],
                max_val=s['max'],
                count=s.get('count', 0)
            )
        self._fitted = True