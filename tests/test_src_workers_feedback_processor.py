import pytest
from unittest.mock import MagicMock, patch
from src.workers.feedback_processor import process_feedback, validate_feedback, save_feedback_to_db, publish_to_redis


@pytest.fixture
def mock_db():
    with patch('src.workers.feedback_processor.db') as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch('src.workers.feedback_processor.redis_client') as mock:
        yield mock


@pytest.fixture
def valid_feedback_data():
    return {
        "user_id": "user_123",
        "mood": "happy",
        "timestamp": "2024-06-15T10:30:00Z",
        "note": "Feeling great today!"
    }


@pytest.fixture
def invalid_feedback_data():
    return {
        "user_id": "user_456",
        "mood": "ecstatic",  # invalid mood
        "timestamp": "2024-06-15T10:30:00Z"
    }


def test_validate_feedback_valid(valid_feedback_data):
    result = validate_feedback(valid_feedback_data)
    assert result is True


def test_validate_feedback_missing_fields():
    incomplete = {"user_id": "user_123"}
    result = validate_feedback(incomplete)
    assert result is False


def test_validate_feedback_invalid_mood():
    data = {"user_id": "user_123", "mood": "ecstatic", "timestamp": "2024-06-15T10:30:00Z"}
    result = validate_feedback(data)
    assert result is False


def test_validate_feedback_empty_user_id():
    data = {"user_id": "", "mood": "happy", "timestamp": "2024-06-15T10:30:00Z"}
    result = validate_feedback(data)
    assert result is False


def test_validate_feedback_invalid_timestamp():
    data = {"user_id": "user_123", "mood": "happy", "timestamp": "not-a-timestamp"}
    result = validate_feedback(data)
    assert result is False


def test_save_feedback_to_db_success(mock_db, valid_feedback_data):
    save_feedback_to_db(valid_feedback_data)
    mock_db.insert.assert_called_once_with("feedback", valid_feedback_data)


def test_save_feedback_to_db_failure(mock_db, valid_feedback_data):
    mock_db.insert.side_effect = Exception("DB connection failed")
    with pytest.raises(Exception, match="DB connection failed"):
        save_feedback_to_db(valid_feedback_data)


def test_publish_to_redis_success(mock_redis, valid_feedback_data):
    publish_to_redis(valid_feedback_data)
    mock_redis.publish.assert_called_once_with("feedback_channel", valid_feedback_data)


def test_publish_to_redis_failure(mock_redis, valid_feedback_data):
    mock_redis.publish.side_effect = ConnectionError("Redis unavailable")
    with pytest.raises(ConnectionError, match="Redis unavailable"):
        publish_to_redis(valid_feedback_data)


def test_process_feedback_success(valid_feedback_data, mock_db, mock_redis):
    result = process_feedback(valid_feedback_data)
    assert result["status"] == "success"
    assert result["user_id"] == valid_feedback_data["user_id"]
    mock_db.insert.assert_called_once_with("feedback", valid_feedback_data)
    mock_redis.publish.assert_called_once_with("feedback_channel", valid_feedback_data)


def test_process_feedback_validation_failure(invalid_feedback_data, mock_db, mock_redis):
    result = process_feedback(invalid_feedback_data)
    assert result["status"] == "validation_error"
    assert "mood" in result.get("errors", [])
    mock_db.insert.assert_not_called()
    mock_redis.publish.assert_not_called()


def test_process_feedback_db_failure(valid_feedback_data, mock_db, mock_redis):
    mock_db.insert.side_effect = Exception("DB error")
    result = process_feedback(valid_feedback_data)
    assert result["status"] == "db_error"
    assert "DB error" in result["message"]
    mock_redis.publish.assert_not_called()


def test_process_feedback_redis_failure(valid_feedback_data, mock_db, mock_redis):
    mock_redis.publish.side_effect = ConnectionError("Redis error")
    result = process_feedback(valid_feedback_data)
    assert result["status"] == "redis_error"
    assert "Redis error" in result["message"]
    mock_db.insert.assert_called_once()  # DB still saved