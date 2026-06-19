import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.validator import validate_record, validate_schema, sanitize_string, sanitize_number, sanitize_timestamp


@pytest.fixture
def mock_db():
    with patch('src.ingestion.validator.db') as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch('src.ingestion.validator.redis_client') as mock:
        yield mock


def test_validate_record_valid(mock_db, mock_redis):
    record = {
        "id": "123",
        "name": "Test User",
        "age": 30,
        "created_at": "2023-01-01T12:00:00Z"
    }
    schema = {
        "id": str,
        "name": str,
        "age": int,
        "created_at": str
    }
    result = validate_record(record, schema)
    assert result["is_valid"] is True
    assert result["sanitized_record"] == {
        "id": "123",
        "name": "Test User",
        "age": 30,
        "created_at": "2023-01-01T12:00:00Z"
    }


def test_validate_record_missing_field(mock_db, mock_redis):
    record = {
        "id": "123",
        "name": "Test User"
    }
    schema = {
        "id": str,
        "name": str,
        "age": int
    }
    result = validate_record(record, schema)
    assert result["is_valid"] is False
    assert "age" in result["errors"]


def test_validate_record_type_mismatch(mock_db, mock_redis):
    record = {
        "id": 123,
        "name": "Test User",
        "age": "thirty",
        "created_at": "2023-01-01T12:00:00Z"
    }
    schema = {
        "id": str,
        "name": str,
        "age": int,
        "created_at": str
    }
    result = validate_record(record, schema)
    assert result["is_valid"] is False
    assert "id" in result["errors"]
    assert "age" in result["errors"]


def test_validate_schema_valid(mock_db, mock_redis):
    schema = {
        "id": str,
        "name": str,
        "age": int
    }
    result = validate_schema(schema)
    assert result is True


def test_validate_schema_invalid_type(mock_db, mock_redis):
    schema = {
        "id": str,
        "name": "string",
        "age": int
    }
    result = validate_schema(schema)
    assert result is False


def test_sanitize_string_valid(mock_db, mock_redis):
    result = sanitize_string("  Hello World!  ")
    assert result == "Hello World!"


def test_sanitize_string_none(mock_db, mock_redis):
    result = sanitize_string(None)
    assert result == ""


def test_sanitize_string_non_string(mock_db, mock_redis):
    result = sanitize_string(123)
    assert result == "123"


def test_sanitize_number_valid_int(mock_db, mock_redis):
    result = sanitize_number("42")
    assert result == 42


def test_sanitize_number_valid_float(mock_db, mock_redis):
    result = sanitize_number("3.14")
    assert result == 3.14


def test_sanitize_number_invalid(mock_db, mock_redis):
    result = sanitize_number("not_a_number")
    assert result is None


def test_sanitize_timestamp_valid(mock_db, mock_redis):
    result = sanitize_timestamp("2023-01-01T12:00:00Z")
    assert result == "2023-01-01T12:00:00Z"


def test_sanitize_timestamp_invalid(mock_db, mock_redis):
    result = sanitize_timestamp("not-a-timestamp")
    assert result is None


def test_validate_record_with_sanitization(mock_db, mock_redis):
    record = {
        "id": "  456  ",
        "name": "  John Doe  ",
        "age": "25",
        "created_at": "2023-06-15T08:30:00Z"
    }
    schema = {
        "id": str,
        "name": str,
        "age": int,
        "created_at": str
    }
    result = validate_record(record, schema)
    assert result["is_valid"] is True
    assert result["sanitized_record"]["id"] == "456"
    assert result["sanitized_record"]["name"] == "John Doe"
    assert result["sanitized_record"]["age"] == 25
    assert result["sanitized_record"]["created_at"] == "2023-06-15T08:30:00Z"