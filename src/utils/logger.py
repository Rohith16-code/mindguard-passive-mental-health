"""Privacy-safe logging with rotation and structured output."""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


LOG_DIR = Path(os.getenv("LOG_DIR", str(Path(__file__).parent.parent.parent / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# Backward-compat names used by tests
Logger = logging.Logger


def log_event(message: str, level: str = "info"):
    getattr(logger, level.lower(), logger.info)(message)


def log_error(error: Exception, context: str = ""):
    logger.error(f"{context}: {error}")


def get_logger(name: str = "app") -> logging.Logger:
    """Get a configured logger with file rotation and console output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)
    
    # Rotating file handler
    log_file = LOG_DIR / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger


# Singleton logger for backward compat
logger = get_logger("mindguard")


def log_crisis_signal(user_id: str, signal_type: str, severity: float, metadata: dict = None):
    """Log a crisis signal event (privacy-safe — no PII)."""
    logger.warning(
        f"CRISIS_SIGNAL user={_hash_id(user_id)} type={signal_type} "
        f"severity={severity:.2f} meta={metadata or {}}"
    )


def log_model_inference(model_version: str, input_shape: tuple, output: float, latency_ms: float):
    """Log model inference metrics."""
    logger.info(
        f"INFERENCE model={model_version} shape={input_shape} "
        f"output={output:.4f} latency={latency_ms:.1f}ms"
    )


def log_data_collection(source: str, records: int, bytes_transferred: int):
    """Log data collection statistics."""
    logger.info(
        f"DATA_COLLECTION source={source} records={records} bytes={bytes_transferred}"
    )


def _hash_id(user_id: str) -> str:
    """Hash a user ID for privacy-safe logging."""
    import hashlib
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]
