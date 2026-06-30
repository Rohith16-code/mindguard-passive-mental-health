"""Health monitoring worker — checks system health metrics periodically."""
import asyncio
import psutil
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("mindguard.health_monitor")


class HealthMonitor:
    """Monitors system health metrics."""
    
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start monitoring."""
        self._running = True
        logger.info("Health monitor started")
        self._task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Monitor system health."""
        from src.config import ANALYSIS_INTERVAL_SECONDS
        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory().percent
                disk = psutil.disk_usage('/').percent
                logger.debug(f"Health: CPU={cpu}% MEM={mem}% DISK={disk}%")
                if cpu > 90 or mem > 95:
                    logger.warning(f"High resource usage: CPU={cpu}% MEM={mem}%")
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(ANALYSIS_INTERVAL_SECONDS)
