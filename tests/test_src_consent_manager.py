import pytest
from unittest.mock import MagicMock, patch
from src.consent.manager import ConsentManager, ConsentStatus, ConsentRecord


@pytest.fixture
def mock_db():
    with patch("src.consent.manager.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.consent.manager.redis") as mock:
        yield mock


@pytest.fixture
def consent_manager(mock_db, mock_redis):
    return ConsentManager()


class TestConsentManager:
    def test_init_initializes_db_and_redis(self, mock_db, mock_redis):
        ConsentManager()
        mock_db.connect.assert_called_once()
        mock_redis.connect.assert_called_once()

    def test_record_consent_creates_record(self, consent_manager, mock_db):
        user_id = "user_123"
        consent_type = "marketing"
        timestamp = 1234567890

        consent_manager.record_consent(user_id, consent_type, timestamp)

        mock_db.insert.assert_called_once()
        call_args = mock_db.insert.call_args
        assert call_args[0][0]["user_id"] == user_id
        assert call_args[0][0]["consent_type"] == consent_type
        assert call_args[0][0]["status"] == ConsentStatus.GRANTED
        assert call_args[0][0]["timestamp"] == timestamp

    def test_record_consent_invalid_type_raises(self, consent_manager):
        with pytest.raises(ValueError, match="Invalid consent type"):
            consent_manager.record_consent("user_123", "invalid_type", 1234567890)

    def test_revoke_consent_updates_record(self, consent_manager, mock_db):
        user_id = "user_123"
        consent_type = "analytics"
        mock_db.select.return_value = [
            {"user_id": user_id, "consent_type": consent_type, "status": ConsentStatus.GRANTED}
        ]

        consent_manager.revoke_consent(user_id, consent_type)

        mock_db.update.assert_called_once()
        call_args = mock_db.update.call_args
        assert call_args[0][0]["status"] == ConsentStatus.REVOKED

    def test_revoke_consent_not_found_raises(self, consent_manager, mock_db):
        user_id = "user_123"
        consent_type = "analytics"
        mock_db.select.return_value = []

        with pytest.raises(ValueError, match="No consent record found"):
            consent_manager.revoke_consent(user_id, consent_type)

    def test_get_consent_status_returns_granted(self, consent_manager, mock_db):
        user_id = "user_123"
        consent_type = "marketing"
        mock_db.select.return_value = [{"status": ConsentStatus.GRANTED}]

        status = consent_manager.get_consent_status(user_id, consent_type)

        assert status == ConsentStatus.GRANTED

    def test_get_consent_status_returns_revoked(self, consent_manager, mock_db):
        user_id = "user_123"
        consent_type = "marketing"
        mock_db.select.return_value = [{"status": ConsentStatus.REVOKED}]

        status = consent_manager.get_consent_status(user_id, consent_type)

        assert status == ConsentStatus.REVOKED

    def test_get_consent_status_not_found_returns_none(self, consent_manager, mock_db):
        user_id = "user_123"
        consent_type = "marketing"
        mock_db.select.return_value = []

        status = consent_manager.get_consent_status(user_id, consent_type)

        assert status is None

    def test_get_user_consent_records_caches_in_redis(self, consent_manager, mock_db, mock_redis):
        user_id = "user_123"
        mock_db.select.return_value = [
            {"user_id": user_id, "consent_type": "marketing", "status": ConsentStatus.GRANTED}
        ]
        mock_redis.get.return_value = None  # no cache hit

        records = consent_manager.get_user_consent_records(user_id)

        mock_redis.set.assert_called_once()
        mock_db.select.assert_called_once()
        assert len(records) == 1
        assert records[0]["consent_type"] == "marketing"

    def test_get_user_consent_records_uses_cache(self, consent_manager, mock_db, mock_redis):
        user_id = "user_123"
        cached_records = [{"consent_type": "marketing", "status": "granted"}]
        mock_redis.get.return_value = cached_records

        records = consent_manager.get_user_consent_records(user_id)

        assert records == cached_records
        mock_db.select.assert_not_called()

    def test_bulk_update_consent(self, consent_manager, mock_db):
        updates = [
            {"user_id": "user_1", "consent_type": "marketing", "status": ConsentStatus.GRANTED},
            {"user_id": "user_2", "consent_type": "analytics", "status": ConsentStatus.REVOKED},
        ]

        consent_manager.bulk_update_consent(updates)

        assert mock_db.insert.call_count == 1
        assert mock_db.update.call_count == 1
        insert_call = mock_db.insert.call_args_list[0]
        update_call = mock_db.update.call_args_list[0]
        assert insert_call[0][0]["user_id"] == "user_1"
        assert update_call[0][0]["user_id"] == "user_2"

    def test_bulk_update_consent_empty_list(self, consent_manager, mock_db):
        consent_manager.bulk_update_consent([])

        mock_db.insert.assert_not_called()
        mock_db.update.assert_not_called()

    def test_is_consent_valid_returns_true_for_granted(self, consent_manager):
        record = ConsentRecord(
            user_id="user_123",
            consent_type="marketing",
            status=ConsentStatus.GRANTED,
            timestamp=1234567890
        )
        assert consent_manager.is_consent_valid(record) is True

    def test_is_consent_valid_returns_false_for_revoked(self, consent_manager):
        record = ConsentRecord(
            user_id="user_123",
            consent_type="marketing",
            status=ConsentStatus.REVOKED,
            timestamp=1234567890
        )
        assert consent_manager.is_consent_valid(record) is False

    def test_is_consent_valid_returns_false_for_none(self, consent_manager):
        assert consent_manager.is_consent_valid(None) is False

    def test_get_consent_history(self, consent_manager, mock_db):
        user_id = "user_123"
        mock_db.select.return_value = [
            {"user_id": user_id, "consent_type": "marketing", "status": ConsentStatus.GRANTED, "timestamp": 1000},
            {"user_id": user_id, "consent_type": "marketing", "status": ConsentStatus.REVOKED, "timestamp": 2000},
        ]

        history = consent_manager.get_consent_history(user_id)

        assert len(history) == 2
        assert history[0]["timestamp"] == 1000
        assert history[1]["timestamp"] == 2000
        mock_db.select.assert_called_once_with(
            table="consent_history",
            filters={"user_id": user_id},
            order_by="timestamp",
            descending=True
        )