"""Local notification dispatch module for mental health crisis alerts."""
import asyncio
import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Severity levels for mental health alerts."""
    INFO = "info"
    MONITORING = "monitoring"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a mental health alert."""
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    id: Optional[str] = None


class Notifier:
    """Handles local notification dispatch for mental health alerts."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the notifier with optional storage for alert history."""
        self._storage_path = storage_path or Path.home() / ".mental_health" / "alerts"
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._alert_history: List[Alert] = []
        self._pending_alerts: List[Alert] = []
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        self._listener_callbacks: List[Any] = []
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the notifier subsystem."""
        if self._is_initialized:
            return
        try:
            self._is_initialized = True
            logger.info("Notifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize notifier: {e}")
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the notifier."""
        self._is_initialized = False
        logger.info("Notifier shutdown complete")

    async def create_alert(
        self,
        level: AlertLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and queue a new alert."""
        alert = Alert(
            level=level,
            message=message,
            context=context or {}
        )
        self._pending_alerts.append(alert)
        await self._notification_queue.put(alert)
        logger.info(f"Alert queued: {level.value.upper()} - {message}")
        return alert

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert to suppress repeated notifications."""
        for alert in self._pending_alerts + self._alert_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    async def get_pending_alerts(self) -> List[Alert]:
        """Get all unacknowledged pending alerts."""
        return [a for a in self._pending_alerts if not a.acknowledged]

    async def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return self._alert_history[-limit:]

    async def dispatch_notification(self, alert: Alert) -> None:
        """Dispatch a notification based on alert level and platform."""
        try:
            if not self._is_initialized:
                return

            # Platform-specific notification handling
            system = platform.system().lower()
            if system == "windows":
                await self._notify_windows(alert)
            elif system == "darwin":
                await self._notify_macos(alert)
            elif system == "linux":
                await self._notify_linux(alert)
            else:
                await self._notify_generic(alert)

            # Store in history
            self._alert_history.append(alert)
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-1000:]

            # Remove from pending
            if alert in self._pending_alerts:
                self._pending_alerts.remove(alert)

            # Notify listeners
            for callback in self._listener_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Error in notification listener: {e}")

        except Exception as e:
            logger.error(f"Failed to dispatch notification: {e}")

    async def _notify_generic(self, alert: Alert) -> None:
        """Generic notification fallback."""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        level = alert.level.value.upper()
        message = alert.message
        logger.info(f"[{level}] {timestamp}: {message}")

    async def _notify_windows(self, alert: Alert) -> None:
        """Windows notification implementation."""
        try:
            import win10toast
            toast = win10toast.ToastNotifier()
            level = alert.level.value.upper()
            toast.show_toast(
                f"Mental Health Alert: {level}",
                alert.message,
                duration=5,
                threaded=True
            )
        except ImportError:
            await self._notify_generic(alert)
        except Exception as e:
            logger.error(f"Windows notification failed: {e}")
            await self._notify_generic(alert)

    async def _notify_macos(self, alert: Alert) -> None:
        """macOS notification implementation."""
        try:
            import Foundation
            import AppKit
            ns_alert = NSUserNotification.alloc().init()
            ns_alert.setTitle_(f"Mental Health Alert: {alert.level.value.upper()}")
            ns_alert.setInformativeText_(alert.message)
            ns_alert.setDeliveryDate_(Foundation.NSDate.dateWithTimeInterval_sinceDate_(
                0, Foundation.NSDate.date()))
            ns_alert.setSoundName_("NSUserNotificationDefaultSoundName")
            ns_alert.setHasActionButton_(True)
            ns_alert.setActionButtonTitle_("Acknowledge")
            AppKit.NSUserNotificationCenter.defaultUserNotificationCenter().scheduleNotification_(ns_alert)
        except Exception as e:
            logger.error(f"macOS notification failed: {e}")
            await self._notify_generic(alert)

    async def _notify_linux(self, alert: Alert) -> None:
        """Linux notification implementation."""
        try:
            import subprocess
            level = alert.level.value.upper()
            subprocess.run([
                "notify-send",
                f"[{level}] Mental Health Alert",
                alert.message
            ], check=False, capture_output=True)
        except Exception as e:
            logger.error(f"Linux notification failed: {e}")
            await self._notify_generic(alert)

    def register_listener(self, callback) -> None:
        """Register a callback to be notified of new alerts."""
        self._listener_callbacks.append(callback)

    def unregister_listener(self, callback) -> None:
        """Remove a registered listener."""
        if callback in self._listener_callbacks:
            self._listener_callbacks.remove(callback)

    async def process_queue(self) -> None:
        """Process the notification queue."""
        while self._is_initialized:
            try:
                alert = await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=1.0
                )
                await self.dispatch_notification(alert)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing notification queue: {e}")
                await asyncio.sleep(0.1)