import pytest
from unittest.mock import MagicMock, patch
from src.api.health_check import *

@pytest.fixture
def mock_db_connection():
    with patch('src.api.health_check.db_connection') as mock:
        yield mock

@pytest.fixture
def mock_redis_client():
    with patch('src.api.health_check.redis_client') as mock:
        yield mock

@pytest.fixture
def mock_logger():
    with patch('src.api.health_check.logger') as mock:
        yield mock

def test_health_check_returns_200(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.return_value = True
    mock_redis_client.ping.return_value = True

    response = health_check()

    assert response.status_code == 200
    assert response.body == b'{"status": "healthy"}'
    assert response.headers["Content-Type"] == "application/json"

def test_health_check_db_failure_returns_503(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.side_effect = Exception("DB connection failed")
    mock_redis_client.ping.return_value = True

    response = health_check()

    assert response.status_code == 503
    assert b"unhealthy" in response.body
    assert response.headers["Content-Type"] == "application/json"

def test_health_check_redis_failure_returns_503(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.return_value = True
    mock_redis_client.ping.side_effect = Exception("Redis unavailable")

    response = health_check()

    assert response.status_code == 503
    assert b"unhealthy" in response.body

def test_health_check_both_failures_returns_503(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.side_effect = Exception("DB error")
    mock_redis_client.ping.side_effect = Exception("Redis error")

    response = health_check()

    assert response.status_code == 503
    assert b"unhealthy" in response.body

def test_health_check_logs_success(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.return_value = True
    mock_redis_client.ping.return_value = True

    health_check()

    mock_logger.info.assert_any_call("Health check passed")

def test_health_check_logs_failure(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.side_effect = Exception("DB error")
    mock_redis_client.ping.return_value = True

    health_check()

    mock_logger.error.assert_any_call("Health check failed: DB error")

def test_health_check_handles_generic_exception(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.side_effect = ValueError("Unexpected error")
    mock_redis_client.ping.return_value = True

    response = health_check()

    assert response.status_code == 503
    assert b"unhealthy" in response.body

def test_health_check_handles_redis_exception_during_ping(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.return_value = True
    mock_redis_client.ping.side_effect = ConnectionRefusedError("Connection refused")

    response = health_check()

    assert response.status_code == 503
    assert b"unhealthy" in response.body

def test_health_check_returns_json_content_type(mock_db_connection, mock_redis_client, mock_logger):
    mock_db_connection.execute.return_value = True
    mock_redis_client.ping.return_value = True

    response = health_check()

    assert "application/json" in response.headers["Content-Type"]