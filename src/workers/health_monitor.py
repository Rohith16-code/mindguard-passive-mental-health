"""Health monitoring module for on-device resource and battery-aware scheduling."""
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.config import settings


@dataclass
class SystemHealthStatus:
    """Represents the current system health status."""
    cpu_ok: bool
    memory_ok: bool
    battery_ok: bool
    overall_ok: bool
    cpu_usage: float
    memory_usage: float
    battery_percent: Optional[float]
    battery_charging: bool
    timestamp: datetime


def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    return psutil.cpu_percent(interval=0.5)


def get_memory_usage() -> float:
    """Get current memory usage percentage."""
    return psutil.virtual_memory().percent


def get_battery_status() -> Dict[str, Any]:
    """Get battery status information."""
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return {
                "percent": None,
                "secsleft": None,
                "power_plugged": None
            }
        return {
            "percent": battery.percent,
            "secsleft": battery.secsleft,
            "power_plugged": battery.power_plugged
        }
    except (AttributeError, NotImplementedError):
        return {
            "percent": None,
            "secsleft": None,
            "power_plugged": None
        }


def check_system_health() -> SystemHealthStatus:
    """Check system health against configured thresholds."""
    cpu_usage = get_cpu_usage()
    memory_usage = get_memory_usage()
    battery_info = get_battery_status()

    cpu_ok = cpu_usage < settings.CPU_THRESHOLD
    memory_ok = memory_usage < settings.MEMORY_THRESHOLD
    battery_ok = True

    if battery_info["percent"] is not None:
        battery_ok = (
            battery_info["percent"] >= settings.MIN_BATTERY_THRESHOLD or
            battery_info["power_plugged"] is True
        )

    overall_ok = cpu_ok and memory_ok and battery_ok

    return SystemHealthStatus(
        cpu_ok=cpu_ok,
        memory_ok=memory_ok,
        battery_ok=battery_ok,
        overall_ok=overall_ok,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        battery_percent=battery_info["percent"],
        battery_charging=battery_info["power_plugged"] if battery_info["power_plugged"] is not None else False,
        timestamp=datetime.utcnow()
    )


def schedule_task_if_healthy(task_name: str, health_status: SystemHealthStatus) -> bool:
    """Determine if a task should be scheduled based on system health."""
    if not health_status.overall_ok:
        return False

    if health_status.battery_percent is not None:
        if not health_status.battery_charging and health_status.battery_percent < settings.MIN_BATTERY_FOR_TASK:
            return False

    return True


class HealthMonitor:
    """Monitors system health and manages task scheduling based on resource availability."""

    def __init__(self):
        self.last_health_check: Optional[SystemHealthStatus] = None

    def check_health(self) -> SystemHealthStatus:
        """Perform a health check and cache the result."""
        self.last_health_check = check_system_health()
        return self.last_health_check

    def is_healthy(self) -> bool:
        """Check if the system is currently healthy."""
        if self.last_health_check is None:
            self.check_health()
        return self.last_health_check.overall_ok

    def should_schedule_task(self, task_name: str) -> bool:
        """Determine if a task should be scheduled based on current health."""
        if self.last_health_check is None:
            self.check_health()
        return schedule_task_if_healthy(task_name, self.last_health_check)

    def get_health_report(self) -> Dict[str, Any]:
        """Generate a health report dictionary."""
        if self.last_health_check is None:
            self.check_health()
        status = self.last_health_check
        return {
            "cpu_ok": status.cpu_ok,
            "memory_ok": status.memory_ok,
            "battery_ok": status.battery_ok,
            "overall_ok": status.overall_ok,
            "cpu_usage": status.cpu_usage,
            "memory_usage": status.memory_usage,
            "battery_percent": status.battery_percent,
            "battery_charging": status.battery_charging,
            "timestamp": status.timestamp.isoformat()
        }