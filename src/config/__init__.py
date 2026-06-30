"""Central configuration package — exports settings and helpers."""
import os
from pathlib import Path
from typing import Dict, Any

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent.resolve()

# ── Environment ───────────────────────────────────────────────
ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Database ──────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "app.db"))

# ── Redis ─────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── ML Model ──────────────────────────────────────────────────
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", str(BASE_DIR / "models" / "model.pt"))
MODEL_PATH = ML_MODEL_PATH  # alias used by some modules
ML_VERSION = os.getenv("ML_VERSION", "1.0.0")
ENCRYPT_MODELS = os.getenv("ENCRYPT_MODELS", "false").lower() == "true"

# ── API ───────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Rate Limiting ─────────────────────────────────────────────
RATE_LIMIT_CAPACITY = int(os.getenv("RATE_LIMIT_CAPACITY", "100"))
RATE_LIMIT_REFILL_RATE = float(os.getenv("RATE_LIMIT_REFILL_RATE", "10.0"))

# ── Anomaly Detection ─────────────────────────────────────────
ANOMALY_WINDOW_SIZE = int(os.getenv("ANOMALY_WINDOW_SIZE", "300"))
ANOMALY_Z_THRESHOLD = float(os.getenv("ANOMALY_Z_THRESHOLD", "2.5"))

# ── Ingestion ─────────────────────────────────────────────────
INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "100"))
INGESTION_MAX_BUFFER_SIZE = int(os.getenv("INGESTION_MAX_BUFFER_SIZE", "10000"))
INGESTION_FLUSH_INTERVAL_SECONDS = int(os.getenv("INGESTION_FLUSH_INTERVAL_SECONDS", "30"))

# ── Training ──────────────────────────────────────────────────
TRAINING_SEED = int(os.getenv("TRAINING_SEED", "42"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "1000"))

# ── Workers ────────────────────────────────────────────────────
ANALYSIS_INTERVAL_SECONDS = int(os.getenv("ANALYSIS_INTERVAL_SECONDS", "60"))
MODEL_REFRESH_INTERVAL_SECONDS = int(os.getenv("MODEL_REFRESH_INTERVAL_SECONDS", "3600"))

# ── Feedback ──────────────────────────────────────────────────
FEEDBACK_TTL = int(os.getenv("FEEDBACK_TTL", "86400"))

# ── Inference ─────────────────────────────────────────────────
INFERENCE_THRESHOLD = float(os.getenv("INFERENCE_THRESHOLD", "0.75"))


class _Settings:
    """Lazy settings object that reads from os.environ."""
    def __getattr__(self, name: str) -> Any:
        return globals().get(name)


settings = _Settings()


# ── Threshold helpers (from old config.py) ────────────────────
REQUIRED_THRESHOLD_KEYS = {
    "cpu_warning", "cpu_critical",
    "memory_warning", "memory_critical",
    "latency_warning_ms", "latency_critical_ms"
}


def get_config() -> Dict[str, Any]:
    """Return runtime configuration dictionary."""
    return {
        "db_url": DATABASE_URL,
        "redis_url": REDIS_URL,
        "thresholds": get_thresholds(),
        "log_level": LOG_LEVEL,
        "model_path": ML_MODEL_PATH,
        "inference_threshold": INFERENCE_THRESHOLD,
        "monitoring_interval_seconds": ANALYSIS_INTERVAL_SECONDS,
        "max_history_days": 30,
        "anonymize_data": True,
    }


def get_thresholds() -> Dict[str, float]:
    """Return system monitoring thresholds."""
    return {
        "cpu_warning": float(os.getenv("CPU_WARNING", "70")),
        "cpu_critical": float(os.getenv("CPU_CRITICAL", "90")),
        "memory_warning": float(os.getenv("MEMORY_WARNING", "75")),
        "memory_critical": float(os.getenv("MEMORY_CRITICAL", "95")),
        "latency_warning_ms": float(os.getenv("LATENCY_WARNING_MS", "200")),
        "latency_critical_ms": float(os.getenv("LATENCY_CRITICAL_MS", "500")),
    }


def validate_thresholds(thresholds: Dict[str, Any]) -> bool:
    """Validate threshold configuration dictionary."""
    missing = REQUIRED_THRESHOLD_KEYS - set(thresholds.keys())
    if missing:
        raise ValueError(f"Missing required threshold keys: {missing}")
    for key, value in thresholds.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Threshold '{key}' must be numeric")
        if value < 0:
            raise ValueError(f"Threshold '{key}' must be non-negative")
    if thresholds["cpu_warning"] >= thresholds["cpu_critical"]:
        raise ValueError("CPU warning must be less than CPU critical")
    if thresholds["memory_warning"] >= thresholds["memory_critical"]:
        raise ValueError("Memory warning must be less than memory critical")
    if thresholds["latency_warning_ms"] >= thresholds["latency_critical_ms"]:
        raise ValueError("Latency warning must be less than latency critical")
    return True


__all__ = [
    "settings", "BASE_DIR", "ENV", "DEBUG", "LOG_LEVEL",
    "DATABASE_URL", "DB_PATH", "REDIS_URL",
    "ML_MODEL_PATH", "MODEL_PATH", "ML_VERSION", "ENCRYPT_MODELS",
    "API_HOST", "API_PORT",
    "RATE_LIMIT_CAPACITY", "RATE_LIMIT_REFILL_RATE",
    "ANOMALY_WINDOW_SIZE", "ANOMALY_Z_THRESHOLD",
    "INGESTION_BATCH_SIZE", "INGESTION_MAX_BUFFER_SIZE", "INGESTION_FLUSH_INTERVAL_SECONDS",
    "TRAINING_SEED", "LEARNING_RATE", "BUFFER_SIZE",
    "ANALYSIS_INTERVAL_SECONDS", "MODEL_REFRESH_INTERVAL_SECONDS",
    "FEEDBACK_TTL", "INFERENCE_THRESHOLD",
    "get_config", "get_thresholds", "validate_thresholds",
]
