import pytest
from unittest.mock import MagicMock, patch
from src.utils.logger import Logger, log_event, log_error, get_logger


@pytest.fixture
def mock_db():
    with patch("src.utils.logger.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.utils.logger.redis") as mock:
        yield mock


@pytest.fixture
def logger_instance(mock_db, mock_redis):
    return Logger("test_service", "test_instance")


class TestLogger:
    def test_logger_initialization(self, mock_db, mock_redis):
        logger = Logger("my_service", "instance_1")
        assert logger.service_name == "my_service"
        assert logger.instance_id == "instance_1"
        assert logger.level == "INFO"

    def test_logger_set_level(self, logger_instance):
        logger_instance.set_level("DEBUG")
        assert logger_instance.level == "DEBUG"

        logger_instance.set_level("WARNING")
        assert logger_instance.level == "WARNING"

    def test_logger_log_event_info_level(self, logger_instance, mock_db, mock_redis):
        logger_instance.set_level("INFO")
        logger_instance.log_event("user_login", user_id=123)

        mock_db.insert.assert_called_once()
        mock_redis.publish.assert_called_once()

    def test_logger_log_event_debug_level_not_logged(self, logger_instance, mock_db, mock_redis):
        logger_instance.set_level("WARNING")
        logger_instance.log_event("user_login", user_id=123)

        mock_db.insert.assert_not_called()
        mock_redis.publish.assert_not_called()

    def test_logger_log_event_with_metadata(self, logger_instance, mock_db, mock_redis):
        logger_instance.log_event("data_sync", status="success", count=42)

        call_args = mock_db.insert.call_args[0][0]
        assert call_args["event_type"] == "data_sync"
        assert call_args["status"] == "success"
        assert call_args["count"] == 42
        assert "timestamp" in call_args
        assert "service" in call_args
        assert "instance" in call_args

    def test_logger_log_error(self, logger_instance, mock_db, mock_redis):
        error = ValueError("Test error")
        logger_instance.log_error(error, context={"operation": "parse_data"})

        call_args = mock_db.insert.call_args[0][0]
        assert call_args["event_type"] == "error"
        assert "Test error" in call_args["message"]
        assert call_args["context"]["operation"] == "parse_data"
        assert "traceback" in call_args

    def test_logger_log_error_without_traceback(self, logger_instance, mock_db, mock_redis):
        error = RuntimeError("No traceback")
        logger_instance.log_error(error, include_traceback=False)

        call_args = mock_db.insert.call_args[0][0]
        assert "traceback" not in call_args


def test_log_event(logger_instance, mock_db, mock_redis):
    with patch("src.utils.logger.get_logger", return_value=logger_instance):
        log_event("test_event", key="value")
        mock_db.insert.assert_called_once()


def test_log_error(logger_instance, mock_db, mock_redis):
    with patch("src.utils.logger.get_logger", return_value=logger_instance):
        error = TypeError("Type mismatch")
        log_error(error, extra_field="data")
        mock_db.insert.assert_called_once()


def test_get_logger_new_instance(mock_db, mock_redis):
    logger = get_logger("new_service", "new_instance", level="ERROR")
    assert logger.service_name == "new_service"
    assert logger.instance_id == "new_instance"
    assert logger.level == "ERROR"


def test_get_logger_returns_singleton(mock_db, mock_redis):
    logger1 = get_logger("singleton_service", "inst1")
    logger2 = get_logger("singleton_service", "inst1")
    assert logger1 is logger2


def test_get_logger_creates_new_instance_if_different_params(mock_db, mock_redis):
    logger1 = get_logger("diff_service", "inst1")
    logger2 = get_logger("diff_service", "inst2")
    assert logger1 is not logger2

    logger3 = get_logger("diff_service", "inst1")
    assert logger3 is logger1