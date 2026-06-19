import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from src.api.routes import router, health_check, get_status, update_model


@pytest.fixture
def mock_db():
    with patch("src.api.routes.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.api.routes.redis_client") as mock:
        yield mock


@pytest.fixture
def mock_model_service():
    with patch("src.api.routes.model_service") as mock:
        yield mock


def test_health_check(mock_db, mock_redis):
    # Mock DB and Redis to return healthy status
    mock_db.ping.return_value = True
    mock_redis.ping.return_value = True

    result = health_check()

    assert result == {"status": "healthy", "db": True, "redis": True}
    mock_db.ping.assert_called_once()
    mock_redis.ping.assert_called_once()


def test_health_check_db_failure(mock_db, mock_redis):
    mock_db.ping.side_effect = Exception("DB connection error")
    mock_redis.ping.return_value = True

    result = health_check()

    assert result == {"status": "unhealthy", "db": False, "redis": True}
    mock_db.ping.assert_called_once()


def test_health_check_redis_failure(mock_db, mock_redis):
    mock_db.ping.return_value = True
    mock_redis.ping.side_effect = Exception("Redis connection error")

    result = health_check()

    assert result == {"status": "unhealthy", "db": True, "redis": False}
    mock_redis.ping.assert_called_once()


def test_get_status(mock_db, mock_redis, mock_model_service):
    mock_db.get_status.return_value = {"db_version": "1.2.3", "uptime": 12345}
    mock_redis.get_status.return_value = {"cache_hits": 1000, "cache_misses": 50}
    mock_model_service.get_model_info.return_value = {"model_id": "v2", "version": "2.1.0"}

    result = get_status()

    assert result == {
        "db": {"db_version": "1.2.3", "uptime": 12345},
        "redis": {"cache_hits": 1000, "cache_misses": 50},
        "model": {"model_id": "v2", "version": "2.1.0"},
    }
    mock_db.get_status.assert_called_once()
    mock_redis.get_status.assert_called_once()
    mock_model_service.get_model_info.assert_called_once()


def test_get_status_db_error(mock_db, mock_redis, mock_model_service):
    mock_db.get_status.side_effect = Exception("DB unavailable")
    mock_redis.get_status.return_value = {"cache_hits": 1000}
    mock_model_service.get_model_info.return_value = {"model_id": "v2"}

    with pytest.raises(HTTPException) as exc_info:
        get_status()

    assert exc_info.value.status_code == 503
    assert "Database unavailable" in str(exc_info.value.detail)


def test_update_model_success(mock_db, mock_redis, mock_model_service):
    mock_model_service.update_model.return_value = {"status": "success", "new_version": "3.0.0"}

    result = update_model()

    assert result == {"status": "success", "new_version": "3.0.0"}
    mock_model_service.update_model.assert_called_once()
    mock_db.log_event.assert_called_once_with("model_update", {"new_version": "3.0.0"})
    mock_redis.invalidate_cache.assert_called_once()


def test_update_model_failure(mock_db, mock_redis, mock_model_service):
    mock_model_service.update_model.side_effect = Exception("Model update failed")

    with pytest.raises(HTTPException) as exc_info:
        update_model()

    assert exc_info.value.status_code == 500
    assert "Model update failed" in str(exc_info.value.detail)
    mock_db.log_event.assert_called_once_with("model_update_error", {"error": "Model update failed"})