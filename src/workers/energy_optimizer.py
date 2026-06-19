"""Energy optimization module for CPU/GPU throttling to improve battery efficiency."""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnergyOptimizerConfig:
    """Configuration for energy optimizer."""
    min_cpu_freq: int = 800
    max_cpu_freq: int = 2400
    min_gpu_freq: int = 200
    max_gpu_freq: int = 1200
    low_battery_threshold: int = 20
    high_battery_threshold: int = 80
    high_load_threshold: int = 70
    low_load_threshold: int = 30


class EnergyOptimizer:
    """Energy optimizer for managing CPU/GPU throttling based on battery and system load."""

    def __init__(
        self,
        db: Optional[Any] = None,
        redis: Optional[Any] = None,
        config: Optional[EnergyOptimizerConfig] = None
    ):
        """Initialize the energy optimizer."""
        self.db = db
        self.redis = redis
        self.config = config or EnergyOptimizerConfig()

    def get_battery_level(self) -> int:
        """Get current battery level from database."""
        if self.db is None:
            raise RuntimeError("Database client not initialized")
        try:
            return self.db.get_battery_level()
        except Exception as e:
            logger.error(f"Failed to get battery level: {e}")
            return 50

    def get_system_load(self) -> Dict[str, int]:
        """Get current system load from database."""
        if self.db is None:
            raise RuntimeError("Database client not initialized")
        try:
            return self.db.get_system_load()
        except Exception as e:
            logger.error(f"Failed to get system load: {e}")
            return {"cpu": 50, "gpu": 50}


def optimize_cpu_throttle(optimizer: EnergyOptimizer) -> Dict[str, Any]:
    """Optimize CPU throttling based on battery level and system load."""
    battery_level = optimizer.get_battery_level()
    system_load = optimizer.get_system_load()
    cpu_load = system_load.get("cpu", 50)

    if battery_level <= optimizer.config.low_battery_threshold and cpu_load > optimizer.config.high_load_threshold:
        target_freq = optimizer.config.min_cpu_freq
        action = "throttle"
        reason = "low_battery_high_load"
    elif battery_level <= optimizer.config.low_battery_threshold:
        target_freq = max(
            optimizer.config.min_cpu_freq,
            int(optimizer.config.min_cpu_freq + (optimizer.config.max_cpu_freq - optimizer.config.min_cpu_freq) * 0.3)
        )
        action = "throttle"
        reason = "low_battery_moderate_load"
    elif battery_level >= optimizer.config.high_battery_threshold and cpu_load < optimizer.config.low_load_threshold:
        target_freq = optimizer.config.max_cpu_freq
        action = "boost"
        reason = "high_battery_low_load"
    elif battery_level >= optimizer.config.high_battery_threshold:
        target_freq = max(
            int(optimizer.config.min_cpu_freq + (optimizer.config.max_cpu_freq - optimizer.config.min_cpu_freq) * 0.7),
            optimizer.config.max_cpu_freq - 200
        )
        action = "boost"
        reason = "high_battery_moderate_load"
    else:
        target_freq = int(optimizer.config.min_cpu_freq + (optimizer.config.max_cpu_freq - optimizer.config.min_cpu_freq) * (cpu_load / 100))
        action = "adjust"
        reason = "normal_operation"

    return {
        "action": action,
        "target_freq": target_freq,
        "reason": reason,
        "current_battery": battery_level,
        "cpu_load": cpu_load
    }


def optimize_gpu_throttle(optimizer: EnergyOptimizer) -> Dict[str, Any]:
    """Optimize GPU throttling based on battery level and system load."""
    battery_level = optimizer.get_battery_level()
    system_load = optimizer.get_system_load()
    gpu_load = system_load.get("gpu", 50)

    if battery_level <= optimizer.config.low_battery_threshold and gpu_load > optimizer.config.high_load_threshold:
        target_freq = optimizer.config.min_gpu_freq
        action = "throttle"
        reason = "low_battery_high_load"
    elif battery_level <= optimizer.config.low_battery_threshold:
        target_freq = max(
            optimizer.config.min_gpu_freq,
            int(optimizer.config.min_gpu_freq + (optimizer.config.max_gpu_freq - optimizer.config.min_gpu_freq) * 0.3)
        )
        action = "throttle"
        reason = "low_battery_moderate_load"
    elif battery_level >= optimizer.config.high_battery_threshold and gpu_load < optimizer.config.low_load_threshold:
        target_freq = optimizer.config.max_gpu_freq
        action = "boost"
        reason = "high_battery_low_load"
    elif battery_level >= optimizer.config.high_battery_threshold:
        target_freq = max(
            int(optimizer.config.min_gpu_freq + (optimizer.config.max_gpu_freq - optimizer.config.min_gpu_freq) * 0.7),
            optimizer.config.max_gpu_freq - 100
        )
        action = "boost"
        reason = "high_battery_moderate_load"
    else:
        target_freq = int(optimizer.config.min_gpu_freq + (optimizer.config.max_gpu_freq - optimizer.config.min_gpu_freq) * (gpu_load / 100))
        action = "adjust"
        reason = "normal_operation"

    return {
        "action": action,
        "target_freq": target_freq,
        "reason": reason,
        "current_battery": battery_level,
        "gpu_load": gpu_load
    }


def get_battery_level(optimizer: EnergyOptimizer) -> int:
    """Get current battery level."""
    return optimizer.get_battery_level()


def apply_throttling(optimizer: EnergyOptimizer, throttle_cmd: Dict[str, Any]) -> bool:
    """Apply throttling command to system."""
    try:
        if optimizer.redis is None:
            logger.warning("Redis client not initialized, throttling command not applied")
            return False

        key = f"throttle:{throttle_cmd['action']}"
        optimizer.redis.set(key, throttle_cmd["target_freq"], ex=300)
        logger.info(f"Applied throttling: {throttle_cmd['action']} at {throttle_cmd['target_freq']}MHz")
        return True
    except Exception as e:
        logger.error(f"Failed to apply throttling: {e}")
        return False


def restore_normal_performance(optimizer: EnergyOptimizer) -> bool:
    """Restore normal CPU/GPU performance settings."""
    try:
        if optimizer.redis is None:
            logger.warning("Redis client not initialized, cannot restore performance")
            return False

        optimizer.redis.delete("throttle:throttle", "throttle:boost", "throttle:adjust")
        logger.info("Restored normal performance settings")
        return True
    except Exception as e:
        logger.error(f"Failed to restore performance: {e}")
        return False