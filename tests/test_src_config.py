import pytest
from unittest.mock import MagicMock, patch
from src.config import *

def test_get_config_returns_dict():
    config = get_config()
    assert isinstance(config, dict)

def test_get_config_has_required_keys():
    config = get_config()
    required_keys = {"db_url", "redis_url", "thresholds", "log_level"}
    assert all(key in config for key in required_keys)

def test_get_thresholds_returns_dict():
    thresholds = get_thresholds()
    assert isinstance(thresholds, dict)

def test_get_thresholds_has_expected_keys():
    thresholds = get_thresholds()
    expected_keys = {"cpu_warning", "cpu_critical", "memory_warning", "memory_critical", "latency_warning_ms", "latency_critical_ms"}
    assert all(key in thresholds for key in expected_keys)

def test_get_thresholds_values_are_numeric():
    thresholds = get_thresholds()
    for key, value in thresholds.items():
        assert isinstance(value, (int, float)), f"Threshold '{key}' must be numeric"

def test_validate_thresholds_valid():
    valid_thresholds = {
        "cpu_warning": 70,
        "cpu_critical": 90,
        "memory_warning": 75,
        "memory_critical": 95,
        "latency_warning_ms": 200,
        "latency_critical_ms": 500
    }
    result = validate_thresholds(valid_thresholds)
    assert result is True

def test_validate_thresholds_missing_key():
    invalid_thresholds = {
        "cpu_warning": 70,
        "cpu_critical": 90
    }
    with pytest.raises(ValueError, match="Missing required threshold keys"):
        validate_thresholds(invalid_thresholds)

def test_validate_thresholds_non_numeric():
    invalid_thresholds = {
        "cpu_warning": "seventy",
        "cpu_critical": 90,
        "memory_warning": 75,
        "memory_critical": 95,
        "latency_warning_ms": 200,
        "latency_critical_ms": 500
    }
    with pytest.raises(ValueError, match="must be numeric"):
        validate_thresholds(invalid_thresholds)

def test_validate_thresholds_negative_value():
    invalid_thresholds = {
        "cpu_warning": -10,
        "cpu_critical": 90,
        "memory_warning": 75,
        "memory_critical": 95,
        "latency_warning_ms": 200,
        "latency_critical_ms": 500
    }
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_thresholds(invalid_thresholds)

def test_validate_thresholds_warning_greater_than_critical():
    invalid_thresholds = {
        "cpu_warning": 95,
        "cpu_critical": 90,
        "memory_warning": 75,
        "memory_critical": 95,
        "latency_warning_ms": 200,
        "latency_critical_ms": 500
    }
    with pytest.raises(ValueError, match="warning must be less than or equal to critical"):
        validate_thresholds(invalid_thresholds)

@pytest.fixture
def mock_db_connection():
    with patch("src.config.create_engine") as mock_engine:
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        yield mock_conn

@pytest.fixture
def mock_redis_client():
    with patch("src.config.Redis") as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        yield mock_client

def test_load_config_from_db(mock_db_connection):
    mock_db_connection.execute.return_value.fetchone.return_value = {
        "config_key": "thresholds",
        "config_value": '{"cpu_warning": 80, "cpu_critical": 95, "memory_warning": 80, "memory_critical": 95, "latency_warning_ms": 300, "latency_critical_ms": 600}'
    }
    thresholds = load_thresholds_from_db()
    assert thresholds["cpu_warning"] == 80
    assert thresholds["cpu_critical"] == 95

def test_load_config_from_db_handles_none(mock_db_connection):
    mock_db_connection.execute.return_value.fetchone.return_value = None
    with pytest.raises(RuntimeError, match="Failed to load thresholds from DB"):
        load_thresholds_from_db()

def test_load_config_from_redis(mock_redis_client):
    mock_redis_client.get.return_value = b'{"cpu_warning": 75, "cpu_critical": 92, "memory_warning": 70, "memory_critical": 90, "latency_warning_ms": 150, "latency_critical_ms": 400}'
    thresholds = load_thresholds_from_redis()
    assert thresholds["cpu_warning"] == 75
    assert thresholds["cpu_critical"] == 92

def test_load_config_from_redis_handles_missing(mock_redis_client):
    mock_redis_client.get.return_value = None
    with pytest.raises(RuntimeError, match="Failed to load thresholds from Redis"):
        load_thresholds_from_redis()

def test_get_thresholds_uses_db_when_available(mock_db_connection, mock_redis_client):
    mock_db_connection.execute.return_value.fetchone.return_value = {
        "config_key": "thresholds",
        "config_value": '{"cpu_warning": 60, "cpu_critical": 85, "memory_warning": 65, "memory_critical": 88, "latency_warning_ms": 100, "latency_critical_ms": 300}'
    }
    thresholds = get_thresholds()
    assert thresholds["cpu_warning"] == 60

def test_get_thresholds_falls_back_to_redis(mock_db_connection, mock_redis_client):
    mock_db_connection.execute.return_value.fetchone.return_value = None
    mock_redis_client.get.return_value = b'{"cpu_warning": 55, "cpu_critical": 80, "memory_warning": 60, "memory_critical": 85, "latency_warning_ms": 90, "latency_critical_ms": 250}'
    thresholds = get_thresholds()
    assert thresholds["cpu_warning"] == 55

def test_get_thresholds_falls_back_to_default(mock_db_connection, mock_redis_client):
    mock_db_connection.execute.return_value.fetchone.return_value = None
    mock_redis_client.get.return_value = None
    thresholds = get_thresholds()
    assert thresholds["cpu_warning"] == 70
    assert thresholds["cpu_critical"] == 90

def test_get_config_integrates_thresholds():
    config = get_config()
    assert "thresholds" in config
    assert isinstance(config["thresholds"], dict)