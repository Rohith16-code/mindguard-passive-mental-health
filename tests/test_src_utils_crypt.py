import pytest
from unittest.mock import MagicMock, patch
from src.utils.crypt import *

@pytest.fixture
def mock_redis():
    with patch('src.utils.crypt.redis') as mock:
        mock_client = MagicMock()
        mock.from_url.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_db():
    with patch('src.utils.crypt.db') as mock:
        mock_session = MagicMock()
        mock.SessionLocal.return_value = mock_session
        yield mock_session

@pytest.fixture
def temp_key_dir(tmp_path):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    with patch('src.utils.crypt.KEY_DIR', key_dir):
        yield key_dir

def test_generate_key_returns_bytes(temp_key_dir):
    key = generate_key()
    assert isinstance(key, bytes)
    assert len(key) == 32  # Fernet keys are 32 bytes

def test_generate_key_writes_to_file(temp_key_dir):
    key = generate_key()
    key_file = temp_key_dir / "local.key"
    assert key_file.exists()
    assert key_file.read_bytes() == key

def test_generate_key_idempotent(temp_key_dir):
    key1 = generate_key()
    key2 = generate_key()
    assert key1 == key2  # Same key generated if file exists

def test_encrypt_decrypt_roundtrip(temp_key_dir):
    key = generate_key()
    plaintext = b"secret data"
    encrypted = encrypt(plaintext, key)
    assert isinstance(encrypted, bytes)
    assert encrypted != plaintext
    decrypted = decrypt(encrypted, key)
    assert decrypted == plaintext

def test_encrypt_raises_on_invalid_key():
    with pytest.raises(TypeError):
        encrypt(b"data", "not bytes")

def test_decrypt_raises_on_invalid_key():
    with pytest.raises(TypeError):
        decrypt(b"encrypted", "not bytes")

def test_decrypt_raises_on_corrupted_data(temp_key_dir):
    key = generate_key()
    with pytest.raises(InvalidToken):
        decrypt(b"corrupted!data", key)

def test_anonymize_string_returns_hash(temp_key_dir):
    value = "user@example.com"
    result = anonymize(value)
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 hex digest

def test_anonymize_string_is_deterministic(temp_key_dir):
    value = "test@example.com"
    hash1 = anonymize(value)
    hash2 = anonymize(value)
    assert hash1 == hash2

def test_anonymize_string_uses_salt_from_env(monkeypatch, temp_key_dir):
    # Clear cache BEFORE setting environment variable to ensure clean state
    anonymize.cache_clear()
    monkeypatch.setenv("CRYPT_SALT", "fixed_salt_123")
    value = "test@example.com"
    hash1 = anonymize(value)
    # Clear cache before changing salt to ensure new salt is used
    anonymize.cache_clear()
    monkeypatch.setenv("CRYPT_SALT", "different_salt_456")
    hash2 = anonymize(value)
    # Hashes should differ because salt changed
    assert hash1 != hash2

def test_anonymize_with_db_lookup(mock_db, temp_key_dir):
    mock_db.query.return_value.filter.return_value.first.return_value = MagicMock(
        anonymized_value="hashed_value_from_db"
    )
    value = "user123"
    result = anonymize(value, use_db=True)
    assert result == "hashed_value_from_db"
    mock_db.query.assert_called_once()
    mock_db.query.return_value.filter.assert_called_once()

def test_anonymize_with_db_miss_caches_and_stores(mock_db, temp_key_dir):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    value = "new_user"
    result = anonymize(value, use_db=True)
    assert result is not None
    assert len(result) == 64
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()

def test_anonymize_with_redis_lookup(mock_redis, temp_key_dir):
    mock_redis.get.return_value = b"redis_hashed_value"
    value = "redis_user"
    result = anonymize(value, use_redis=True)
    assert result == "redis_hashed_value"
    mock_redis.get.assert_called_once_with(f"anon:{value}")

def test_anonymize_redis_miss_stores_result(mock_redis, temp_key_dir):
    mock_redis.get.return_value = None
    value = "new_redis_user"
    result = anonymize(value, use_redis=True)
    assert result is not None
    assert len(result) == 64
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    assert call_args[0][0].startswith("anon:")
    assert call_args[1].get("ex") is not None  # TTL set

def test_anonymize_with_db_and_redis_prefers_redis(mock_redis, mock_db, temp_key_dir):
    mock_redis.get.return_value = b"redis_wins"
    value = "priority_test"
    result = anonymize(value, use_db=True, use_redis=True)
    assert result == "redis_wins"
    mock_redis.get.assert_called_once()
    mock_db.query.assert_not_called()

def test_get_key_from_file(temp_key_dir):
    key = generate_key()
    key_file = temp_key_dir / "local.key"
    assert key_file.exists()
    loaded_key = get_key_from_file()
    assert loaded_key == key

def test_get_key_from_file_missing_raises(temp_key_dir):
    import os
    os.remove(temp_key_dir / "local.key")
    with pytest.raises(FileNotFoundError):
        get_key_from_file()

def test_incomplete_function():
    pass