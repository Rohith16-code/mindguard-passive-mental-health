import pytest
from unittest.mock import MagicMock, patch
from src.alerts.notifier import *

@pytest.fixture
def mock_db():
    with patch('src.alerts.notifier.db') as mock:
        yield mock

@pytest.fixture
def mock_redis():
    with patch('src.alerts.notifier.redis_client') as mock:
        yield mock

@pytest.fixture
def mock_logger():
    with patch('src.alerts.notifier.logger') as mock:
        yield mock

def test_send_email_success(mock_logger, mock_db):
    mock_db.get_user_email.return_value = "user@example.com"
    
    result = send_email(user_id=123, subject="Test", body="Hello")
    
    assert result is True
    mock_db.get_user_email.assert_called_once_with(123)
    mock_logger.info.assert_called_once_with("Email sent to user 123: subject='Test'")

def test_send_email_user_not_found(mock_logger, mock_db):
    mock_db.get_user_email.return_value = None
    
    result = send_email(user_id=999, subject="Test", body="Hello")
    
    assert result is False
    mock_db.get_user_email.assert_called_once_with(999)
    mock_logger.warning.assert_called_once_with("User 999 not found for email notification")

def test_send_sms_success(mock_logger, mock_redis):
    mock_redis.get_user_phone.return_value = "+1234567890"
    
    result = send_sms(user_id=456, message="Alert!")
    
    assert result is True
    mock_redis.get_user_phone.assert_called_once_with(456)
    mock_logger.info.assert_called_once_with("SMS sent to user 456: '+1234567890'")

def test_send_sms_user_not_found(mock_logger, mock_redis):
    mock_redis.get_user_phone.return_value = None
    
    result = send_sms(user_id=789, message="Alert!")
    
    assert result is False
    mock_redis.get_user_phone.assert_called_once_with(789)
    mock_logger.warning.assert_called_once_with("User 789 not found for SMS notification")

def test_dispatch_alert_email(mock_logger, mock_db, mock_redis):
    mock_db.get_user_email.return_value = "user@example.com"
    mock_redis.get_user_phone.return_value = "+1234567890"
    
    result = dispatch_alert(user_id=111, channel="email", subject="Alert", body="Critical issue")
    
    assert result is True
    mock_db.get_user_email.assert_called_once_with(111)
    mock_redis.get_user_phone.assert_not_called()
    mock_logger.info.assert_called_once_with("Alert dispatched to user 111 via email")

def test_dispatch_alert_sms(mock_logger, mock_db, mock_redis):
    mock_db.get_user_email.return_value = "user@example.com"
    mock_redis.get_user_phone.return_value = "+1234567890"
    
    result = dispatch_alert(user_id=222, channel="sms", subject="Alert", body="Critical issue")
    
    assert result is True
    mock_db.get_user_email.assert_not_called()
    mock_redis.get_user_phone.assert_called_once_with(222)
    mock_logger.info.assert_called_once_with("Alert dispatched to user 222 via sms")

def test_dispatch_alert_invalid_channel(mock_logger):
    result = dispatch_alert(user_id=333, channel="push", subject="Alert", body="Critical issue")
    
    assert result is False
    mock_logger.error.assert_called_once_with("Invalid channel 'push' for user 333")

def test_dispatch_alert_user_not_found_email(mock_logger, mock_db, mock_redis):
    mock_db.get_user_email.return_value = None
    mock_redis.get_user_phone.return_value = "+1234567890"
    
    result = dispatch_alert(user_id=444, channel="email", subject="Alert", body="Critical issue")
    
    assert result is False
    mock_db.get_user_email.assert_called_once_with(444)
    mock_logger.warning.assert_called_once_with("User 444 not found for email notification")

def test_dispatch_alert_user_not_found_sms(mock_logger, mock_db, mock_redis):
    mock_db.get_user_email.return_value = "user@example.com"
    mock_redis.get_user_phone.return_value = None
    
    result = dispatch_alert(user_id=555, channel="sms", subject="Alert", body="Critical issue")
    
    assert result is False
    mock_redis.get_user_phone.assert_called_once_with(555)
    mock_logger.warning.assert_called_once_with("User 555 not found for SMS notification")