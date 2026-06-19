import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.sensor_adapter import SensorAdapter, SensorReading, SensorType


@pytest.fixture
def mock_db():
    with patch("src.ingestion.sensor_adapter.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.ingestion.sensor_adapter.redis") as mock:
        yield mock


@pytest.fixture
def adapter(mock_db, mock_redis):
    return SensorAdapter()


def test_adapter_initialization(adapter, mock_db, mock_redis):
    assert adapter is not None
    mock_db.connect.assert_called_once()
    mock_redis.connect.assert_called_once()


def test_read_sensor_data_success(adapter, mock_db, mock_redis):
    mock_db.query.return_value = [
        {"sensor_id": "acc_01", "type": "accelerometer", "timestamp": 1700000000.0, "x": 0.98, "y": 0.02, "z": -0.1}
    ]
    mock_redis.get.return_value = None

    readings = adapter.read_sensor_data(SensorType.ACCELEROMETER, 100)

    assert len(readings) == 1
    assert readings[0].sensor_id == "acc_01"
    assert readings[0].type == SensorType.ACCELEROMETER
    assert readings[0].timestamp == 1700000000.0
    assert readings[0].x == 0.98
    assert readings[0].y == 0.02
    assert readings[0].z == -0.1
    mock_db.query.assert_called_once_with("accelerometer", 100)


def test_read_sensor_data_empty_result(adapter, mock_db, mock_redis):
    mock_db.query.return_value = []

    readings = adapter.read_sensor_data(SensorType.GYROSCOPE, 50)

    assert readings == []
    mock_db.query.assert_called_once_with("gyroscope", 50)


def test_read_sensor_data_redis_cache_hit(adapter, mock_db, mock_redis):
    cached_reading = SensorReading(
        sensor_id="gyr_02",
        type=SensorType.GYROSCOPE,
        timestamp=1700000100.0,
        x=0.1,
        y=-0.2,
        z=0.05
    )
    mock_redis.get.return_value = cached_reading

    readings = adapter.read_sensor_data(SensorType.GYROSCOPE, 100)

    assert len(readings) == 1
    assert readings[0] == cached_reading
    mock_redis.get.assert_called_once_with("sensor:gyroscope:100")
    mock_db.query.assert_not_called()


def test_read_sensor_data_redis_cache_miss(adapter, mock_db, mock_redis):
    mock_redis.get.return_value = None
    mock_db.query.return_value = [
        {"sensor_id": "gyr_03", "type": "gyroscope", "timestamp": 1700000200.0, "x": 0.0, "y": 0.0, "z": 0.9}
    ]

    readings = adapter.read_sensor_data(SensorType.GYROSCOPE, 100)

    assert len(readings) == 1
    assert readings[0].sensor_id == "gyr_03"
    mock_redis.get.assert_called_once_with("sensor:gyroscope:100")
    mock_redis.set.assert_called_once_with("sensor:gyroscope:100", readings[0], ex=300)


def test_read_sensor_data_invalid_type(adapter, mock_db, mock_redis):
    with pytest.raises(ValueError, match="Unsupported sensor type"):
        adapter.read_sensor_data("invalid_type", 100)


def test_write_sensor_reading_success(adapter, mock_db, mock_redis):
    reading = SensorReading(
        sensor_id="mag_01",
        type=SensorType.MAGNETOMETER,
        timestamp=1700000300.0,
        x=25.5,
        y=-12.3,
        z=40.1
    )

    adapter.write_sensor_reading(reading)

    mock_db.insert.assert_called_once_with(
        "magnetometer",
        {
            "sensor_id": "mag_01",
            "timestamp": 1700000300.0,
            "x": 25.5,
            "y": -12.3,
            "z": 40.1
        }
    )
    mock_redis.delete.assert_called_once_with("sensor:magnetometer:*")


def test_write_sensor_reading_invalid_type(adapter, mock_db, mock_redis):
    reading = SensorReading(
        sensor_id="unknown",
        type="unknown_type",
        timestamp=1700000400.0,
        x=0.0,
        y=0.0,
        z=0.0
    )

    with pytest.raises(ValueError, match="Unsupported sensor type"):
        adapter.write_sensor_reading(reading)