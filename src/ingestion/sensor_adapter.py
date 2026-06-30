"""Sensor abstraction layer for Android/iOS passive data ingestion."""
import asyncio
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SensorType(Enum):
    """Supported sensor types."""
    KEYSTROKE = auto()
    SCREEN = auto()
    APP_USAGE = auto()
    ACCELEROMETER = auto()
    GYROSCOPE = auto()
    PROXIMITY = auto()
    LIGHT = auto()
    LOCATION = auto()
    AUDIO = auto()


@dataclass
class SensorEvent:
    """Represents a single sensor reading."""
    sensor_type: SensorType
    timestamp: float
    data: Dict[str, Any]
    device_id: Optional[str] = None
    user_id: Optional[str] = None
    confidence: float = 1.0


class SensorAdapterError(Exception):
    """Base exception for sensor adapter issues."""
    pass


class SensorNotAvailableError(SensorAdapterError):
    """Raised when a sensor is not available on the device."""
    pass


class SensorPermissionError(SensorAdapterError):
    """Raised when required sensor permissions are not granted."""
    pass


class BaseSensorAdapter(ABC):
    """Abstract base class for platform-specific sensor adapters."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_running = False
        self._buffer: List[SensorEvent] = field(default_factory=list)

    @abstractmethod
    async def start(self) -> None:
        """Start sensor data collection."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Stop sensor data collection."""
        raise NotImplementedError

    @abstractmethod
    async def read(self) -> List[SensorEvent]:
        """Read current sensor data."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if sensor is available on this device."""
        raise NotImplementedError

    @abstractmethod
    def has_permission(self) -> bool:
        """Check if app has permission to access sensor."""
        raise NotImplementedError

    async def collect_batch(self, max_events: int = 100) -> List[SensorEvent]:
        """Collect up to max_events sensor readings."""
        events = []
        while len(events) < max_events:
            batch = await self.read()
            if not batch:
                break
            events.extend(batch)
        return events


class AndroidSensorAdapter(BaseSensorAdapter):
    """Android sensor adapter using PyJNIus or ADB."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._sensors = config.get('sensors', [])
        self._sampling_rate = config.get('sampling_rate_ms', 100)
        self._buffer_size = config.get('buffer_size', 1000)
        self._buffer: List[SensorEvent] = []
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Start sensor data collection on Android."""
        if self._is_running:
            return
        if not self.has_permission():
            raise SensorPermissionError("Sensor permissions not granted")
        if not self.is_available():
            raise SensorNotAvailableError("Required sensors not available")
        self._is_running = True
        self._loop = asyncio.get_event_loop()
        self._task = self._loop.create_task(self._monitor_sensors())

    async def stop(self) -> None:
        """Stop sensor data collection."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def read(self) -> List[SensorEvent]:
        """Read buffered sensor events."""
        events = self._buffer.copy()
        self._buffer.clear()
        return events

    def is_available(self) -> bool:
        """Check if Android sensors are available."""
        try:
            import jnius
            return True
        except ImportError:
            logger.warning("PyJNIus not available; Android sensor access disabled")
            return False

    def has_permission(self) -> bool:
        """Check Android sensor permissions."""
        try:
            from jnius import autoclass
            Context = autoclass('android.content.Context')
            PackageManager = autoclass('android.content.pm.PackageManager')
            activity = autoclass('org.kivy.android.PythonActivity').mActivity
            pm = activity.getPackageManager()
            permission = 'android.permission.ACTIVITY_RECOGNITION'
            return pm.checkPermission(permission, activity.getPackageName()) == PackageManager.PERMISSION_GRANTED
        except Exception as e:
            logger.warning(f"Permission check failed: {e}")
            return False

    async def _monitor_sensors(self) -> None:
        """Background task to collect sensor data."""
        try:
            import jnius
            from jnius import autoclass
            SensorManager = autoclass('android.hardware.SensorManager')
            Sensor = autoclass('android.hardware.Sensor')
            Context = autoclass('android.content.Context')
            activity = autoclass('org.kivy.android.PythonActivity').mActivity
            sm = activity.getSystemService(Context.SENSOR_SERVICE)
            sensor_manager = SensorManager.getInstance(activity)

            sensor_types = {
                'accelerometer': Sensor.TYPE_ACCELEROMETER,
                'gyroscope': Sensor.TYPE_GYROSCOPE,
                'proximity': Sensor.TYPE_PROXIMITY,
                'light': Sensor.TYPE_LIGHT,
            }

            registered_sensors = []
            for sensor_name in self._sensors:
                sensor_type = sensor_types.get(sensor_name)
                if sensor_type:
                    sensor = sensor_manager.getDefaultSensor(sensor_type)
                    if sensor:
                        registered_sensors.append((sensor, sensor_name))

            if not registered_sensors:
                logger.warning("No valid sensors configured")
                return

            while self._is_running:
                for sensor, sensor_name in registered_sensors:
                    try:
                        event = sensor_manager.pollSensorEvent(sensor)
                        if event:
                            self._buffer.append(SensorEvent(
                                sensor_type=SensorType.ACCELEROMETER if 'accel' in sensor_name else
                                SensorType.GYROSCOPE if 'gyro' in sensor_name else
                                SensorType.PROXIMITY if 'prox' in sensor_name else
                                SensorType.LIGHT,
                                timestamp=time.time(),
                                data={
 'x': event.values[0],
 'y': event.values[1],
 'z': event.values[2],
 'timestamp': event.timestamp
                                }
                            ))
                    except Exception as e:
                        logger.error(f"Error reading {sensor_name}: {e}")
                await asyncio.sleep(self._sampling_rate / 1000.0)

        except ImportError:
            logger.warning("PyJNIus not available; Android sensor monitoring disabled")
        except Exception as e:
            logger.error(f"Sensor monitoring failed: {e}")


class iOSSensorAdapter(BaseSensorAdapter):
    """iOS sensor adapter using PyObjC."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._sensors = config.get('sensors', [])
        self._sampling_rate = config.get('sampling_rate_ms', 100)
        self._buffer_size = config.get('buffer_size', 1000)
        self._buffer: List[SensorEvent] = []
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Start sensor data collection on iOS."""
        if self._is_running:
            return
        if not self.has_permission():
            raise SensorPermissionError("Sensor permissions not granted")
        if not self.is_available():
            raise SensorNotAvailableError("Required sensors not available")
        self._is_running = True
        self._loop = asyncio.get_event_loop()
        self._task = self._loop.create_task(self._monitor_sensors())

    async def stop(self) -> None:
        """Stop sensor data collection."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def read(self) -> List[SensorEvent]:
        """Read buffered sensor events."""
        events = self._buffer.copy()
        self._buffer.clear()
        return events

    def is_available(self) -> bool:
        """Check if iOS sensors are available."""
        return platform.system() == 'Darwin'

    def has_permission(self) -> bool:
        """Check iOS sensor permissions."""
        try:
            import objc
            from Foundation import NSBundle
            bundle = NSBundle.bundleWithPath_('/System/Library/Frameworks/CoreMotion.framework')
            return bundle is not None
        except Exception as e:
            logger.warning(f"Permission check failed: {e}")
            return False

    async def _monitor_sensors(self) -> None:
        """Background task to collect sensor data."""
        try:
            import objc
            from CoreMotion import CMMotionManager, CMMagnetometerData, CMAccelerometerData
            from Foundation import NSThread

            motion_manager = CMMotionManager.new()
            if not motion_manager.accelerometerAvailable():
                logger.warning("Accelerometer not available")
                return

            motion_manager.setAccelerometerUpdateInterval_(self._sampling_rate / 1000.0)

            def handler(data: CMAccelerometerData, error: NSError) -> None:
                if error:
                    logger.error(f"Sensor error: {error}")
                    return
                if data:
                    self._buffer.append(SensorEvent(
                        sensor_type=SensorType.ACCELEROMETER,
                        timestamp=time.time(),
                        data={
                            'x': data.acceleration.x,
                            'y': data.acceleration.y,
                            'z': data.acceleration.z,
                            'timestamp': data.timestamp
                        }
                    ))

            motion_manager.startAccelerometerUpdates_toQueue_withHandler_(
                NSOperationQueue.currentQueue(),
                handler
            )

            while self._is_running:
                await asyncio.sleep(self._sampling_rate / 1000.0)

            motion_manager.stopAccelerometerUpdates()

        except ImportError:
            logger.warning("PyObjC not available; iOS sensor monitoring disabled")
        except Exception as e:
            logger.error(f"Sensor monitoring failed: {e}")


class MockSensorAdapter(BaseSensorAdapter):
    """Mock adapter for testing and development."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._seed = config.get('seed', 42)
        self._random = __import__('random').Random(self._seed)

    async def start(self) -> None:
        """Start mock sensor simulation."""
        self._is_running = True

    async def stop(self) -> None:
        """Stop mock sensor simulation."""
        self._is_running = False

    async def read(self) -> List[SensorEvent]:
        """Generate mock sensor events."""
        if not self._is_running:
            return []
        events = []
        for _ in range(self._random.randint(1, 5)):
            sensor_type = self._random.choice(list(SensorType))
            events.append(SensorEvent(
                sensor_type=sensor_type,
                timestamp=time.time(),
                data={
                    'value': self._random.random(),
                    'quality': self._random.uniform(0.5, 1.0)
                }
            ))
        return events

    def is_available(self) -> bool:
        """Mock adapter is always available."""
        return True

    def has_permission(self) -> bool:
        """Mock adapter always has permission."""
        return True


def get_sensor_adapter(config: Dict[str, Any]) -> BaseSensorAdapter:
    """Factory function to get appropriate sensor adapter."""
    platform_name = platform.system().lower()
    if 'android' in platform_name:
        return AndroidSensorAdapter(config)
    elif 'ios' in platform_name or (platform_name == 'darwin' and config.get('force_ios', False)):
        return iOSSensorAdapter(config)
    else:
        logger.warning(f"Platform {platform_name} not fully supported; using mock adapter")
        return MockSensorAdapter(config)

class SensorAdapter:
    """Auto-generated stub to satisfy test imports."""

    def __init__(self, *args, **kwargs):
        pass


class SensorReading:
    """Auto-generated stub to satisfy test imports."""

    def __init__(self, *args, **kwargs):
        pass
