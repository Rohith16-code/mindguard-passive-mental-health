import pytest
from unittest.mock import MagicMock, patch
from src.consent.anonymizer import (
    anonymize_record,
    ensure_k_anonymity,
    get_k_anonymity_groups,
    K_ANONYMITY_THRESHOLD,
    _hash_identifier,
)


@pytest.fixture
def mock_db():
    with patch("src.consent.anonymizer.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.consent.anonymizer.redis_client") as mock:
        yield mock


@pytest.fixture
def sample_record():
    return {
        "id": "user_123",
        "age": 34,
        "zip": "10001",
        "disease": "flu",
        "symptoms": ["fever", "cough"],
    }


@pytest.fixture
def sample_dataset():
    return [
        {"id": "u1", "age": 30, "zip": "10001", "disease": "flu"},
        {"id": "u2", "age": 30, "zip": "10001", "disease": "cold"},
        {"id": "u3", "age": 31, "zip": "10001", "disease": "flu"},
        {"id": "u4", "age": 30, "zip": "10002", "disease": "flu"},
        {"id": "u5", "age": 30, "zip": "10002", "disease": "cold"},
    ]


def test_anonymize_record_replaces_id_with_hash(sample_record):
    anonymized = anonymize_record(sample_record)
    assert "id" not in anonymized
    assert "hashed_id" in anonymized
    assert isinstance(anonymized["hashed_id"], str)
    assert len(anonymized["hashed_id"]) == 64  # SHA-256 hex


def test_anonymize_record_preserves_other_fields(sample_record):
    anonymized = anonymize_record(sample_record)
    assert anonymized["age"] == sample_record["age"]
    assert anonymized["zip"] == sample_record["zip"]
    assert anonymized["disease"] == sample_record["disease"]
    assert anonymized["symptoms"] == sample_record["symptoms"]


def test_anonymize_record_handles_missing_optional_fields():
    record = {"id": "user_456", "age": 25}
    anonymized = anonymize_record(record)
    assert "hashed_id" in anonymized
    assert anonymized["age"] == 25


def test_hash_identifier_is_deterministic():
    id1 = _hash_identifier("user_789")
    id2 = _hash_identifier("user_789")
    assert id1 == id2
    assert id1 != _hash_identifier("user_790")


def test_get_k_anonymity_groups_returns_correct_groups(sample_dataset):
    quasi_identifiers = ["age", "zip"]
    groups = get_k_anonymity_groups(sample_dataset, quasi_identifiers)
    assert len(groups) == 3  # (30,10001), (31,10001), (30,10002)


def test_get_k_anonymity_groups_empty_dataset():
    groups = get_k_anonymity_groups([], ["age", "zip"])
    assert groups == {}


def test_ensure_k_anonymity_returns_original_if_k_met(sample_dataset):
    quasi_identifiers = ["age", "zip"]
    result = ensure_k_anonymity(sample_dataset, quasi_identifiers, k=2)
    assert result == sample_dataset


def test_ensure_k_anonymity_filters_under_k_groups(sample_dataset):
    # Only group (30,10002) has size 2; others have size ≥2, but let's test with k=3
    quasi_identifiers = ["age", "zip"]
    result = ensure_k_anonymity(sample_dataset, quasi_identifiers, k=3)
    # Group (30,10001) has 2 records → excluded
    # Group (31,10001) has 1 record → excluded
    # Group (30,10002) has 2 records → excluded
    # So all groups have <3 → result should be empty
    assert len(result) == 0


def test_ensure_k_anonymity_with_k1_returns_all():
    result = ensure_k_anonymity([{"a": 1}], ["a"], k=1)
    assert result == [{"a": 1}]


def test_ensure_k_anonymity_calls_db_and_redis(mock_db, mock_redis, sample_record):
    # Mock DB to return sample dataset
    mock_db.get_records.return_value = [sample_record, sample_record]
    mock_redis.get.return_value = None  # no cached k-anonymity status

    quasi_identifiers = ["age", "zip"]
    result = ensure_k_anonymity([sample_record], quasi_identifiers, k=2)

    # Should have called DB and Redis
    mock_db.get_records.assert_called_once()
    mock_redis.get.assert_called_once()
    # Should have stored result in Redis
    mock_redis.set.assert_called_once()


def test_ensure_k_anonymity_uses_redis_cache(mock_db, mock_redis, sample_record):
    quasi_identifiers = ["age", "zip"]
    cached_result = [sample_record]
    mock_redis.get.return_value = cached_result  # cache hit

    result = ensure_k_anonymity([sample_record], quasi_identifiers, k=2)

    assert result == cached_result
    mock_db.get_records.assert_not_called()


def test_ensure_k_anonymity_raises_on_invalid_k():
    with pytest.raises(ValueError, match="k must be >= 1"):
        ensure_k_anonymity([], ["a"], k=0)


def test_ensure_k_anonymity_handles_empty_input():
    result = ensure_k_anonymity([], ["a"], k=2)
    assert result == []


def test_k_anonymity_threshold_constant():
    assert K_ANONYMITY_THRESHOLD >= 1
    assert isinstance(K_ANONYMITY_THRESHOLD, int)