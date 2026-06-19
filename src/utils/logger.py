"""Structured on-device logging utilities."""
import logging
import os
import sys
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    context: Dict[str, Any]
    thread_id: int
    file_path: str
    line_number: int
    function_name: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class StructuredLogger:
    """Thread-safe structured logger for on-device use."""
    
    _instance: Optional['StructuredLogger'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        level: Union[LogLevel, int] = LogLevel.INFO,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 3,
    ):
        if self._initialized:
            return
        
        self._initialized = True
        self._log_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_file_size = max_file_size
        self._backup_count = backup_count
        
        self._logger = logging.getLogger("mental_health_logger")
        self._logger.setLevel(self._get_logging_level(level))
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s | %(extra_context)s"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        log_file = self._log_dir / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self._max_file_size,
            backupCount=self._backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        
        self._thread_local = threading.local()
    
    def _get_logging_level(self, level: Union[LogLevel, int]) -> int:
        if isinstance(level, LogLevel):
            return level.value
        return level
    
    def _build_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thread_id": threading.current_thread().ident,
        }
        if extra:
            context.update(extra)
        return context
    
    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        context = self._build_context(extra)
        extra_context = json.dumps(context, default=str)
        
        frame = sys._getframe(2)
        record = logging.LogRecord(
            name=self._logger.name,
            level=level,
            pathname=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            msg=message,
            args=(),
            exc_info=None,
            func=frame.f_code.co_name,
            sinfo=None,
        )
        record.extra_context = extra_context
        
        self._logger.handle(record)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._log(logging.CRITICAL, message, extra)
    
    def log_entry(self, entry: LogEntry) -> None:
        self._logger.info(entry.message, extra={"extra_context": entry.to_json()})
    
    def log_exception(
        self,
        exception: Exception,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._log(logging.ERROR, str(exception), extra, exc_info=True)
    
    def get_log_dir(self) -> Path:
        return self._log_dir


logger = StructuredLogger()


def log_crisis_signal(
    signal_type: str,
    confidence: float,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log potential crisis detection signals."""
    logger.warning(
        f"CRISIS_SIGNAL: {signal_type}",
        extra={
            "confidence": confidence,
            "signal_type": signal_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(context or {}),
        },
    )


def log_model_inference(
    model_name: str,
    input_features: Dict[str, Any],
    output_probs: Dict[str, float],
    latency_ms: float,
) -> None:
    """Log model inference details."""
    logger.info(
        "Model inference completed",
        extra={
            "model_name": model_name,
            "input_features": input_features,
            "output_probs": output_probs,
            "latency_ms": latency_ms,
        },
    )


def log_data_collection(
    event_type: str,
    data_points: int,
    device_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Log data collection events."""
    logger.info(
        f"Data collection: {event_type}",
        extra={
            "event_type": event_type,
            "data_points": data_points,
            "device_state": device_state or {},
        },
    )