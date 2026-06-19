"""Configuration module for passive mental health crisis detection system."""
from typing import Dict, Any
import os


REQUIRED_THRESHOLD_KEYS = {
    "cpu_warning",
    "cpu_critical",
    "memory_warning",
    "memory_critical",
    "latency_warning_ms",
    "latency_critical_ms"
}


def get_config() -> Dict[str, Any]:
    """Return runtime configuration dictionary."""
    return {
        "db_url": os.getenv("DB_URL", "sqlite:///./app.db"),
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        "thresholds": get_thresholds(),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "model_path": os.getenv("MODEL_PATH", "./models/crisis_detector.tflite"),
        "inference_threshold": float(os.getenv("INFERENCE_THRESHOLD", "0.75")),
        "monitoring_interval_seconds": int(os.getenv("MONITORING_INTERVAL", "60")),
        "max_history_days": int(os.getenv("MAX_HISTORY_DAYS", "30")),
        "anonymize_data": os.getenv("ANONYMIZE_DATA", "true").lower() == "true"
    }


def get_thresholds() -> Dict[str, float]:
    """Return system monitoring thresholds."""
    return {
        "cpu_warning": float(os.getenv("CPU_WARNING", "70")),
        "cpu_critical": float(os.getenv("CPU_CRITICAL", "90")),
        "memory_warning": float(os.getenv("MEMORY_WARNING", "75")),
        "memory_critical": float(os.getenv("MEMORY_CRITICAL", "95")),
        "latency_warning_ms": float(os.getenv("LATENCY_WARNING_MS", "200")),
        "latency_critical_ms": float(os.getenv("LATENCY_CRITICAL_MS", "500"))
    }


def validate_thresholds(thresholds: Dict[str, Any]) -> bool:
    """Validate threshold configuration dictionary.
    
    Args:
        thresholds: Dictionary containing threshold values
        
    Returns:
        True if thresholds are valid
        
    Raises:
        ValueError: If thresholds are invalid
    """
    missing_keys = REQUIRED_THRESHOLD_KEYS - set(thresholds.keys())
    if missing_keys:
        raise ValueError(f"Missing required threshold keys: {missing_keys}")
    
    for key, value in thresholds.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Threshold '{key}' must be numeric, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"Threshold '{key}' must be non-negative, got {value}")
    
    if thresholds["cpu_warning"] >= thresholds["cpu_critical"]:
        raise ValueError("CPU warning threshold must be less than CPU critical threshold")
    if thresholds["memory_warning"] >= thresholds["memory_critical"]:
        raise ValueError("Memory warning threshold must be less than memory critical threshold")
    if thresholds["latency_warning_ms"] >= thresholds["latency_critical_ms"]:
        raise ValueError("Latency warning threshold must be less than latency critical threshold")
    
    return True


validate_thresholds(get_thresholds())