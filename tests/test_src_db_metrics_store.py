import pytest
from unittest.mock import MagicMock, patch
from src.db.metrics_store import MetricsStore, MetricsStoreError


@pytest.fixture
def mock_db():
    with patch("src.db.metrics_store.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.db.metrics_store.redis") as mock:
        yield mock


@pytest.fixture
def metrics_store(mock_db, mock_redis):
    return MetricsStore()


class TestMetricsStore:
    def test_init_initializes_db_and_redis(self, mock_db, mock_redis):
        store = MetricsStore()
        assert store._db is not None
        assert store._redis is not None

    def test_record_metric_success(self, metrics_store, mock_db, mock_redis):
        mock_db.execute.return_value = None
        mock_redis.setex.return_value = True

        metrics_store.record_metric("cpu_usage", 42.5, timestamp=1700000000)

        mock_db.execute.assert_called_once()
        mock_redis.setex.assert_called_once()

    def test_record_metric_db_failure_raises_error(self, metrics_store, mock_db, mock_redis):
        mock_db.execute.side_effect = Exception("DB error")
        mock_redis.setex.return_value = True

        with pytest.raises(MetricsStoreError) as exc_info:
            metrics_store.record_metric("memory_usage", 80.1)

        assert "Failed to record metric" in str(exc_info.value)

    def test_record_metric_redis_failure_raises_error(self, metrics_store, mock_db, mock_redis):
        mock_db.execute.return_value = None
        mock_redis.setex.side_effect = Exception("Redis error")

        with pytest.raises(MetricsStoreError) as exc_info:
            metrics_store.record_metric("disk_io", 1024)

        assert "Failed to cache metric" in str(exc_info.value)

    def test_get_metric_success(self, metrics_store, mock_db, mock_redis):
        mock_redis.get.return_value = b'{"value": 55.3, "timestamp": 1700000000}'
        mock_db.execute.return_value = [{"value": 55.3, "timestamp": 1700000000}]

        result = metrics_store.get_metric("cpu_usage")

        assert result["value"] == 55.3
        assert result["timestamp"] == 1700000000
        mock_redis.get.assert_called_once()
        mock_db.execute.assert_not_called()

    def test_get_metric_redis_miss_uses_db(self, metrics_store, mock_redis, mock_db):
        mock_redis.get.return_value = None
        mock_db.execute.return_value = [{"value": 60.1, "timestamp": 1700000001}]

        result = metrics_store.get_metric("memory_usage")

        assert result["value"] == 60.1
        mock_db.execute.assert_called_once()
        mock_redis.setex.assert_called_once()

    def test_get_metric_db_returns_empty_raises_error(self, metrics_store, mock_redis, mock_db):
        mock_redis.get.return_value = None
        mock_db.execute.return_value = []

        with pytest.raises(MetricsStoreError) as exc_info:
            metrics_store.get_metric("nonexistent_metric")

        assert "Metric not found" in str(exc_info.value)

    def test_get_metric_db_failure_raises_error(self, metrics_store, mock_redis, mock_db):
        mock_redis.get.return_value = None
        mock_db.execute.side_effect = Exception("DB query failed")

        with pytest.raises(MetricsStoreError) as exc_info:
            metrics_store.get_metric("disk_io")

        assert "Failed to fetch metric" in str(exc_info.value)

    def test_get_metric_cache_failure_uses_db(self, metrics_store, mock_redis, mock_db):
        mock_redis.get.side_effect = Exception("Redis unavailable")
        mock_db.execute.return_value = [{"value": 70.0, "timestamp": 1700000002}]

        result = metrics_store.get_metric("cpu_usage")

        assert result["value"] == 70.0
        mock_db.execute.assert_called_once()

    def test_get_metric_range_success(self, metrics_store, mock_db):
        mock_db.execute.return_value = [
            {"value": 10.0, "timestamp": 1700000000},
            {"value": 20.0, "timestamp": 1700000001}
        ]

        result = metrics_store.get_metric_range("cpu_usage", 1700000000, 1700000001)

        assert len(result) == 2
        assert result[0]["value"] == 10.0
        assert result[1]["value"] == 20.0

    def test_get_metric_range_db_failure_raises_error(self, metrics_store, mock_db):
        mock_db.execute.side_effect = Exception("DB error")

        with pytest.raises(MetricsStoreError) as exc_info:
            metrics_store.get_metric_range("memory_usage", 1700000000, 1700000001)

        assert "Failed to fetch metric range" in str(exc_info.value)

    def test_get_metric_range_empty_returns_empty_list(self, metrics_store, mock_db):
        mock_db.execute.return_value = []

        result = metrics_store.get_metric_range("disk_io", 1700000000, 1700000001)

        assert result == []

    def test_delete_metric_success(self, metrics_store, mock_db):
        mock_db.execute.return_value = None

        metrics_store.delete_metric("cpu_usage")

        mock_db.execute.assert_called_once()

    def test_delete_metric_failure_raises_error(self, metrics_store, mock_db):
        mock_db.execute.side_effect = Exception("DB delete failed")

        with pytest.raises(MetricsStoreError) as exc_info:
            metrics_store.delete_metric("memory_usage")

        assert "Failed to delete metric" in str(exc_info.value)