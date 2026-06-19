import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from src.utils.time_utils import *

@pytest.fixture
def mock_db():
    with patch('src.utils.time_utils.db') as mock:
        yield mock

@pytest.fixture
def mock_redis():
    with patch('src.utils.time_utils.redis_client') as mock:
        yield mock

@pytest.fixture
def mock_now():
    with patch('src.utils.time_utils.datetime') as mock_dt:
        mock_dt.now.return_value = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield mock_dt

def test_get_current_utc_timestamp():
    result = get_current_utc_timestamp()
    assert isinstance(result, datetime)
    assert result.tzinfo == timezone.utc
    assert abs((datetime.now(timezone.utc) - result).total_seconds()) < 1

def test_convert_to_utc():
    # Test with timezone-aware datetime
    dt = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone(timedelta(hours=2)))
    result = convert_to_utc(dt)
    assert result.tzinfo == timezone.utc
    assert result.hour == 12

    # Test with timezone-naive datetime (assumed UTC)
    dt_naive = datetime(2024, 6, 15, 10, 0, 0)
    result_naive = convert_to_utc(dt_naive)
    assert result_naive.tzinfo == timezone.utc
    assert result_naive == datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)

def test_parse_iso8601_to_utc():
    # Valid ISO8601 string with timezone
    result = parse_iso8601_to_utc("2024-06-15T14:30:00+02:00")
    assert result.tzinfo == timezone.utc
    assert result.hour == 12

    # Valid ISO8601 string with Z (UTC)
    result_z = parse_iso8601_to_utc("2024-06-15T10:00:00Z")
    assert result_z.tzinfo == timezone.utc
    assert result_z.hour == 10

    # Invalid string raises ValueError
    with pytest.raises(ValueError):
        parse_iso8601_to_utc("not-a-datetime")

def test_get_sleep_window_start(mock_db, mock_redis, mock_now):
    # Mock DB to return user timezone
    mock_db.get_user_timezone.return_value = "America/New_York"
    mock_redis.get.return_value = None  # No override in Redis

    result = get_sleep_window_start("user123", 23, 7)
    # Expected: 23:00 NY time on previous day (since current time is 12:00 UTC = 8:00 NY)
    # 23:00 NY = 03:00 UTC next day
    assert result.tzinfo == timezone.utc
    assert result.hour == 3
    assert result.day == 16  # next day

def test_get_sleep_window_start_with_redis_override(mock_db, mock_redis, mock_now):
    mock_db.get_user_timezone.return_value = "UTC"
    mock_redis.get.return_value = b"02:00"  # 2:00 UTC override

    result = get_sleep_window_start("user456", 23, 7)
    assert result.hour == 2
    assert result.minute == 0

def test_get_sleep_window_start_db_fallback(mock_db, mock_redis, mock_now):
    mock_db.get_user_timezone.side_effect = Exception("DB error")
    mock_redis.get.return_value = None

    with pytest.raises(Exception):
        get_sleep_window_start("user789", 23, 7)

def test_calculate_sleep_duration():
    start = datetime(2024, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 6, 16, 6, 0, 0, tzinfo=timezone.utc)
    result = calculate_sleep_duration(start, end)
    assert result == 8.0

def test_calculate_sleep_duration_next_day():
    start = datetime(2024, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 6, 15, 6, 0, 0, tzinfo=timezone.utc)
    result = calculate_sleep_duration(start, end)
    assert result == 8.0  # wraps around midnight

def test_calculate_sleep_duration_invalid_order():
    start = datetime(2024, 6, 16, 6, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 6, 15, 22, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        calculate_sleep_duration(start, end)