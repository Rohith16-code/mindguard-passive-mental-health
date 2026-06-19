import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from src.consent.audit_log import (
    AuditLog,
    LogEntry,
    ConsentAction,
    Action,
    _get_utc_now,
    _serialize_entry,
    _deserialize_entry,
)


@pytest.fixture
def mock_db():
    with patch("src.consent.audit_log.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.consent.audit_log.redis") as mock:
        yield mock


@pytest.fixture
def audit_log(mock_db, mock_redis):
    return AuditLog()


def test_get_utc_now_returns_aware_utc_datetime():
    now = _get_utc_now()
    assert now.tzinfo is not None
    assert now.tzinfo.utcoffset(now) == timezone.utc.utcoffset(now)


def test_log_entry_serialization_deserialization():
    entry = LogEntry(
        user_id="user123",
        action=ConsentAction.CONSENT_GIVEN,
        details={"consent_version": "2.0"},
        timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        session_id="sess_abc",
        ip_address="192.168.1.1",
    )
    serialized = _serialize_entry(entry)
    assert isinstance(serialized, str)
    deserialized = _deserialize_entry(serialized)
    assert deserialized.user_id == entry.user_id
    assert deserialized.action == entry.action
    assert deserialized.details == entry.details
    assert deserialized.timestamp == entry.timestamp
    assert deserialized.session_id == entry.session_id
    assert deserialized.ip_address == entry.ip_address


def test_log_entry_serialization_handles_none_session_ip():
    entry = LogEntry(
        user_id="user456",
        action=Action.UPDATE_PREFERENCES,
        details={},
        timestamp=datetime.now(timezone.utc),
        session_id=None,
        ip_address=None,
    )
    serialized = _serialize_entry(entry)
    deserialized = _deserialize_entry(serialized)
    assert deserialized.session_id is None
    assert deserialized.ip_address is None


def test_audit_log_record_consent_given(audit_log, mock_db, mock_redis):
    mock_db.execute.return_value = None
    mock_redis.setex.return_value = True

    result = audit_log.record_consent_given(
        user_id="user789",
        consent_version="3.1",
        session_id="sess_xyz",
        ip_address="10.0.0.1",
    )

    assert result is True
    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args[0][0]
    assert "INSERT INTO audit_log" in call_args
    assert "user789" in call_args
    assert "CONSENT_GIVEN" in call_args
    assert '"consent_version": "3.1"' in call_args


def test_audit_log_record_consent_withdrawn(audit_log, mock_db, mock_redis):
    mock_db.execute.return_value = None
    mock_redis.setex.return_value = True

    result = audit_log.record_consent_withdrawn(
        user_id="user789",
        reason="no_longer_wish_to_share",
        session_id="sess_xyz",
        ip_address="10.0.0.1",
    )

    assert result is True
    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args[0][0]
    assert "CONSENT_WITHDRAWN" in call_args
    assert '"reason": "no_longer_wish_to_share"' in call_args


def test_audit_log_record_action(audit_log, mock_db, mock_redis):
    mock_db.execute.return_value = None
    mock_redis.setex.return_value = True

    result = audit_log.record_action(
        user_id="user789",
        action=Action.DATA_EXPORT_REQUESTED,
        details={"format": "json"},
        session_id="sess_xyz",
        ip_address="10.0.0.1",
    )

    assert result is True
    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args[0][0]
    assert "DATA_EXPORT_REQUESTED" in call_args
    assert '"format": "json"' in call_args


def test_audit_log_record_action_invalid_action_type(audit_log, mock_db, mock_redis):
    with pytest.raises(ValueError, match="Unsupported action type"):
        audit_log.record_action(
            user_id="user789",
            action="INVALID_ACTION",
            details={},
        )


def test_audit_log_get_user_audit_history(audit_log, mock_db, mock_redis):
    mock_db.fetchall.return_value = [
        '{"user_id": "user789", "action": "CONSENT_GIVEN", "details": {"v": "1.0"}, "timestamp": "2024-06-01T12:00:00+00:00", "session_id": "s1", "ip_address": "1.2.3.4"}',
        '{"user_id": "user789", "action": "DATA_EXPORT_REQUESTED", "details": {}, "timestamp": "2024-06-02T14:30:00+00:00", "session_id": "s2", "ip_address": "1.2.3.4"}',
    ]

    entries = audit_log.get_user_audit_history(user_id="user789", limit=10)

    assert len(entries) == 2
    assert entries[0].action == ConsentAction.CONSENT_GIVEN
    assert entries[1].action == Action.DATA_EXPORT_REQUESTED
    assert entries[0].details == {"v": "1.0"}
    mock_db.fetchall.assert_called_once()


def test_audit_log_get_user_audit_history_empty(audit_log, mock_db, mock_redis):
    mock_db.fetchall.return_value = []

    entries = audit_log.get_user_audit_history(user_id="nonexistent_user")

    assert entries == []
    mock_db.fetchall.assert_called_once()


def test_audit_log_get_user_audit_history_db_error(audit_log, mock_db, mock_redis):
    mock_db.fetchall.side_effect = Exception("DB connection failed")

    with pytest.raises(Exception, match="DB connection failed"):
        audit_log.get_user_audit_history(user_id="user789")


def test_audit_log_get_user_audit_history_redis_caching(audit_log, mock_db, mock_redis):
    mock_redis.get.return_value = None
    mock_db.fetchall.return_value = [
        '{"user_id": "user789", "action": "CONSENT_GIVEN", "details": {}, "timestamp": "2024-06-01T12:00:00+00:00", "session_id": null, "ip_address": null}'
    ]
    mock_redis.setex.return_value = True

    entries = audit_log.get_user_audit_history(user_id="user789", use_cache=True)

    assert len(entries) == 1
    mock_redis.get.assert_called_once_with("audit:user789")
    mock_redis.setex.assert_called_once()


def test_audit_log_get_user_audit_history_redis_hit(audit_log, mock_db, mock_redis):
    mock_redis.get.return_value = (
        '{"user_id": "user789", "action": "CONSENT_GIVEN", "details": {}, "timestamp": "2024-06-01T12:00:00+00:00", "session_id": null, "ip_address": null}'
    )
    mock_db.fetchall.return_value = []

    entries = audit_log.get_user_audit_history(user_id="user789", use_cache=True)

    assert len(entries) == 1
    assert entries[0].action == ConsentAction.CONSENT_GIVEN
    mock_redis.get.assert_called_once()
    mock_db.fetchall.assert_not_called()