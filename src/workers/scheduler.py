"""Scheduler module for periodic mental health analysis and model refresh."""
import threading
import time
from datetime import datetime
from typing import Optional

from src.core.config import settings
from src.core.db import db
from src.core.redis import redis_client
from src.services.model_manager import model_manager
from src.services.analyzer import analyzer


scheduler_instance: Optional[threading.Thread] = None
_analyze_thread: Optional[threading.Thread] = None
_refresh_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _analyze_task() -> None:
    """Execute analysis task and update timestamps."""
    try:
        analyzer.analyze()
        redis_client.set("last_analysis_timestamp", datetime.utcnow().isoformat())
    except Exception as e:
        redis_client.set("last_analysis_error", str(e))
        redis_client.set("last_analysis_timestamp", datetime.utcnow().isoformat())


def _refresh_model_task() -> None:
    """Execute model refresh task and update timestamps."""
    try:
        model_manager.refresh()
        redis_client.set("last_model_refresh_timestamp", datetime.utcnow().isoformat())
    except Exception as e:
        redis_client.set("last_model_refresh_error", str(e))
        redis_client.set("last_model_refresh_timestamp", datetime.utcnow().isoformat())


def analyze_periodically(interval: int = settings.ANALYSIS_INTERVAL_SECONDS) -> None:
    """Run analysis task periodically."""
    while not _stop_event.is_set():
        _analyze_task()
        _stop_event.wait(timeout=interval)


def refresh_model_periodically(interval: int = settings.MODEL_REFRESH_INTERVAL_SECONDS) -> None:
    """Run model refresh task periodically."""
    while not _stop_event.is_set():
        _refresh_model_task()
        _stop_event.wait(timeout=interval)


def start_scheduler() -> None:
    """Start background scheduler threads."""
    global scheduler_instance, _analyze_thread, _refresh_thread, _stop_event
    _stop_event.clear()

    _analyze_thread = threading.Thread(target=analyze_periodically, daemon=True)
    _refresh_thread = threading.Thread(target=refresh_model_periodically, daemon=True)

    _analyze_thread.start()
    _refresh_thread.start()

    scheduler_instance = threading.Thread(
        target=lambda: None, daemon=True
    )


def stop_scheduler() -> None:
    """Stop background scheduler threads."""
    global scheduler_instance
    _stop_event.set()
    if _analyze_thread:
        _analyze_thread.join(timeout=5.0)
    if _refresh_thread:
        _refresh_thread.join(timeout=5.0)
    scheduler_instance = None