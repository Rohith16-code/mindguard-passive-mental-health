"""Risk engine module for threshold evaluation and soft-check trigger logic."""
from typing import Literal, Optional
from dataclasses import dataclass
from src.db.cache import CacheClient as db
from src.db.cache import CacheClient as redis
import time


@dataclass
class ThresholdConfig:
    """Configuration for threshold-based evaluation."""
    metric: str
    threshold: float
    op: Literal["gt", "lt", "eq"]
    window_seconds: int = 60


@dataclass
class SoftCheckConfig:
    """Configuration for soft-check trigger logic."""
    metric: str
    soft_threshold: float
    hard_threshold: float
    window_seconds: int = 300
    cooldown_seconds: int = 600


def evaluate_threshold(config: ThresholdConfig) -> bool:
    """Evaluate if a metric exceeds its configured threshold.

    Args:
        config: ThresholdConfig with metric, threshold, operation, and window.

    Returns:
        True if threshold is exceeded, False otherwise.
    """
    try:
        avg_value = db.get_metric_avg(config.metric, config.window_seconds)
        if avg_value is None:
            return False

        if config.op == "gt":
            return avg_value > config.threshold
        elif config.op == "lt":
            return avg_value < config.threshold
        elif config.op == "eq":
            return abs(avg_value - config.threshold) < 1e-9
        else:
            raise ValueError(f"Unsupported operation: {config.op}")
    except Exception:
        return False


def trigger_soft_check(config: SoftCheckConfig) -> bool:
    """Trigger soft-check if metric exceeds soft threshold but not hard threshold.

    Args:
        config: SoftCheckConfig with metric, thresholds, window, and cooldown.

    Returns:
        True if soft-check should be triggered, False otherwise.
    """
    try:
        avg_value = db.get_metric_avg(config.metric, config.window_seconds)
        if avg_value is None:
            return False

        cache_key = f"soft_check_cooldown:{config.metric}"
        cooldown_until = redis.get(cache_key)
        if cooldown_until is not None and float(cooldown_until) > time.time():
            return False

        if avg_value >= config.hard_threshold:
            return False
        elif avg_value >= config.soft_threshold:
            redis.setex(cache_key, config.cooldown_seconds, time.time() + config.cooldown_seconds)
            return True
        else:
            return False
    except Exception:
        return False


class RiskEngine:
    """Engine for evaluating risk based on configurable thresholds and soft-checks."""

    def __init__(
        self,
        threshold_configs: Optional[list[ThresholdConfig]] = None,
        soft_check_configs: Optional[list[SoftCheckConfig]] = None,
    ):
        """Initialize the risk engine with configurations.

        Args:
            threshold_configs: List of threshold evaluation configs.
            soft_check_configs: List of soft-check trigger configs.
        """
        self.threshold_configs = threshold_configs or []
        self.soft_check_configs = soft_check_configs or []

    def evaluate_all_thresholds(self) -> dict[str, bool]:
        """Evaluate all configured thresholds.

        Returns:
            Dict mapping metric names to evaluation results.
        """
        results = {}
        for config in self.threshold_configs:
            results[config.metric] = evaluate_threshold(config)
        return results

    def trigger_all_soft_checks(self) -> dict[str, bool]:
        """Trigger soft-checks for all configured metrics.

        Returns:
            Dict mapping metric names to soft-check trigger decisions.
        """
        results = {}
        for config in self.soft_check_configs:
            results[config.metric] = trigger_soft_check(config)
        return results

    def get_risk_score(self) -> float:
        """Compute a normalized risk score based on threshold and soft-check results.

        Returns:
            Risk score between 0.0 and 1.0.
        """
        threshold_results = self.evaluate_all_thresholds()
        soft_check_results = self.trigger_all_soft_checks()

        score = 0.0
        if threshold_results:
            score += 0.5 * sum(threshold_results.values()) / len(threshold_results)
        if soft_check_results:
            score += 0.3 * sum(soft_check_results.values()) / len(soft_check_results)
        return min(score, 1.0)