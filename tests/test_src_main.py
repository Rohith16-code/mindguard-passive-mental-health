import pytest
from unittest.mock import MagicMock, patch
from src.main import app, startup, shutdown, health_check

@pytest.fixture
def mock_db():
    with patch("src.main.db") as mock:
        yield mock

@pytest.fixture
def mock_redis():
    with patch("src.main.redis_client") as mock:
        yield mock

def test_app_exists():
    assert app is not None

def test_startup_initializes_db(mock_db):
    startup()
    mock_db.connect.assert_called_once()

def test_startup_initializes_redis(mock_redis):
    startup()
    mock_redis.connect.assert_called_once()

def test_shutdown_closes_db(mock_db):
    startup()
    shutdown()
    mock_db.disconnect.assert_called_once()

def test_shutdown_closes_redis(mock_redis):
    startup()
    shutdown()
    mock_redis.disconnect.assert_called_once()

def test_health_check_returns_healthy(mock_db, mock_redis):
    mock_db.is_connected.return_value = True
    mock_redis.is_connected.return_value = True
    
    result = health_check()
    
    assert result["status"] == "healthy"
    assert result["database"] == "ok"
    assert result["redis"] == "ok"

def test_health_check_reports_db_failure(mock_db, mock_redis):
    mock_db.is_connected.return_value = False
    mock_redis.is_connected.return_value = True
    
    result = health_check()
    
    assert result["status"] == "unhealthy"
    assert result["database"] == "error"
    assert result["redis"] == "ok"

def test_health_check_reports_redis_failure(mock_db, mock_redis):
    mock_db.is_connected.return_value = True
    mock_redis.is_connected.return_value = False
    
    result = health_check()
    
    assert result["status"] == "unhealthy"
    assert result["database"] == "ok"
    assert result["redis"] == "error"

def test_health_check_reports_both_failures(mock_db, mock_redis):
    mock_db.is_connected.return_value = False
    mock_redis.is_connected.return_value = False
    
    result = health_check()
    
    assert result["status"] == "unhealthy"
    assert result["database"] == "error"
    assert result["redis"] == "error"