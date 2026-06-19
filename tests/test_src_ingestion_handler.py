import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.handler import (
    IngestionHandler,
    IngestionError,
    validate_sensor_data,
    process_sensor_batch,
    store_sensor_data,
    notify_ingestion_complete
)


@pytest.fixture
def mock_db():
    with patch('src.ingestion.handler.DatabaseClient') as mock:
        yield mock.return_value


@pytest.fixture
def mock_redis():
    with patch('src.ingestion.handler.RedisClient') as mock:
        yield mock.return_value


@pytest.fixture
def handler(mock_db, mock_redis):
    return IngestionHandler(db_client=mock_db, redis_client=mock_redis)


class TestValidateSensorData:
    def test_valid_data(self):
        data = {
            "sensor_id": "sensor_001",
            "timestamp": "2024-01-01T12:00:00Z",
            "value": 42.5,
            "unit": "celsius"
        }
        assert validate_sensor_data(data) is True

    def test_missing_required_field(self):
        data = {
            "sensor_id": "sensor_001",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        with pytest.raises(IngestionError, match="Missing required field"):
            validate_sensor_data(data)

    def test_invalid_timestamp(self):
        data = {
            "sensor_id": "sensor_001",
            "timestamp": "not-a-timestamp",
            "value": 42.5,
            "unit": "celsius"
        }
        with pytest.raises(IngestionError, match="Invalid timestamp"):
            validate_sensor_data(data)

    def test_invalid_value_type(self):
        data = {
            "sensor_id": "sensor_001",
            "timestamp": "2024-01-01T12:00:00Z",
            "value": "not_a_number",
            "unit": "celsius"
        }
        with pytest.raises(IngestionError, match="Value must be numeric"):
            validate_sensor_data(data)


class TestProcessSensorBatch:
    def test_process_batch_success(self, handler, mock_db, mock_redis):
        batch = [
            {"sensor_id": "s1", "timestamp": "2024-01-01T12:00:00Z", "value": 10.0, "unit": "celsius"},
            {"sensor_id": "s2", "timestamp": "2024-01-01T12:01:00Z", "value": 20.0, "unit": "fahrenheit"}
        ]
        result = process_sensor_batch(handler, batch)
        assert result["processed"] == 2
        assert result["errors"] == []
        assert mock_db.insert_many.called
        assert mock_redis.set.call_count == 2

    def test_process_batch_with_invalid_entries(self, handler, mock_db, mock_redis):
        batch = [
            {"sensor_id": "s1", "timestamp": "2024-01-01T12:00:00Z", "value": 10.0, "unit": "celsius"},
            {"sensor_id": "s2", "timestamp": "invalid", "value": 20.0, "unit": "fahrenheit"}
        ]
        result = process_sensor_batch(handler, batch)
        assert result["processed"] == 1
        assert len(result["errors"]) == 1
        assert "Invalid timestamp" in result["errors"][0]

    def test_process_empty_batch(self, handler):
        result = process_sensor_batch(handler, [])
        assert result["processed"] == 0
        assert result["errors"] == []


class TestStoreSensorData:
    def test_store_single_data_success(self, handler, mock_db, mock_redis):
        data = {"sensor_id": "s1", "timestamp": "2024-01-01T12:00:00Z", "value": 10.0, "unit": "celsius"}
        store_sensor_data(handler, data)
        mock_db.insert_one.assert_called_once_with(data)
        mock_redis.set.assert_called_once()

    def test_store_data_validation_fails(self, handler, mock_db, mock_redis):
        data = {"sensor_id": "s1", "timestamp": "invalid", "value": 10.0, "unit": "celsius"}
        with pytest.raises(IngestionError):
            store_sensor_data(handler, data)
        assert not mock_db.insert_one.called


class TestNotifyIngestionComplete:
    def test_notify_success(self, handler, mock_redis):
        mock_redis.publish.return_value = 1
        notify_ingestion_complete(handler, {"total": 100, "processed": 100})
        mock_redis.publish.assert_called_once()
        args = mock_redis.publish.call_args[0]
        assert "ingestion_complete" in args[0]

    def test_notify_redis_failure(self, handler, mock_redis):
        mock_redis.publish.side_effect = Exception("Redis unavailable")
        with pytest.raises(IngestionError, match="Failed to send notification"):
            notify_ingestion_complete(handler, {"total": 100, "processed": 99})


class TestIngestionHandler:
    def test_handler_initialization(self, mock_db, mock_redis):
        handler = IngestionHandler(db_client=mock_db, redis_client=mock_redis)
        assert handler.db_client is mock_db
        assert handler.redis_client is mock_redis

    def test_ingest_pipeline(self, handler, mock_db, mock_redis):
        batch = [
            {"sensor_id": "s1", "timestamp": "2024-01-01T12:00:00Z", "value": 10.0, "unit": "celsius"},
            {"sensor_id": "s2", "timestamp": "2024-01-01T12:01:00Z", "value": 20.0, "unit": "fahrenheit"}
        ]
        result = handler.ingest(batch)
        assert result["processed"] == 2
        assert mock_db.insert_many.called
        assert mock_redis.set.call_count == 2
        mock_redis.publish.assert_called_once()