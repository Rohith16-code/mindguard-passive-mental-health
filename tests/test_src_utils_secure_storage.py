import pytest
from unittest.mock import MagicMock, patch
from src.utils.secure_storage import SecureStorage, SecureStorageError

@pytest.fixture
def mock_db():
    with patch('src.utils.secure_storage.db') as mock:
        yield mock

@pytest.fixture
def mock_redis():
    with patch('src.utils.secure_storage.redis') as mock:
        yield mock

@pytest.fixture
def secure_storage(mock_db, mock_redis):
    return SecureStorage()

def test_secure_storage_initialization(secure_storage):
    assert secure_storage is not None

def test_store_model_success(secure_storage, mock_db):
    model_id = "model_123"
    model_data = {"name": "test_model", "type": "classification"}
    encrypted_data = b"encrypted_payload"
    
    with patch('src.utils.secure_storage.encrypt_data', return_value=encrypted_data):
        secure_storage.store_model(model_id, model_data)
    
    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args
    assert call_args[0][0] == "INSERT INTO models (id, data) VALUES (?, ?)"
    assert call_args[0][1][0] == model_id
    assert call_args[0][1][1] == encrypted_data

def test_store_model_db_error(secure_storage, mock_db):
    model_id = "model_456"
    model_data = {"name": "failing_model"}
    
    mock_db.execute.side_effect = Exception("DB connection failed")
    
    with pytest.raises(SecureStorageError, match="Failed to store model"):
        secure_storage.store_model(model_id, model_data)

def test_retrieve_model_success(secure_storage, mock_db):
    model_id = "model_789"
    encrypted_data = b"encrypted_payload"
    decrypted_data = {"name": "retrieved_model"}
    
    mock_db.execute.return_value.fetchone.return_value = (encrypted_data,)
    
    with patch('src.utils.secure_storage.decrypt_data', return_value=decrypted_data):
        result = secure_storage.retrieve_model(model_id)
    
    assert result == decrypted_data
    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args
    assert call_args[0][0] == "SELECT data FROM models WHERE id = ?"
    assert call_args[0][1][0] == model_id

def test_retrieve_model_not_found(secure_storage, mock_db):
    model_id = "nonexistent_model"
    mock_db.execute.return_value.fetchone.return_value = None
    
    with pytest.raises(SecureStorageError, match="Model not found"):
        secure_storage.retrieve_model(model_id)

def test_retrieve_model_decrypt_error(secure_storage, mock_db):
    model_id = "model_decrypt_fail"
    encrypted_data = b"corrupted_payload"
    
    mock_db.execute.return_value.fetchone.return_value = (encrypted_data,)
    
    with patch('src.utils.secure_storage.decrypt_data', side_effect=ValueError("Decryption failed")):
        with pytest.raises(SecureStorageError, match="Failed to decrypt model"):
            secure_storage.retrieve_model(model_id)

def test_store_key_success(secure_storage, mock_db):
    key_id = "key_abc"
    key_data = {"key": "secret_key_value", "algorithm": "AES-256"}
    encrypted_data = b"key_encrypted_payload"
    
    with patch('src.utils.secure_storage.encrypt_data', return_value=encrypted_data):
        secure_storage.store_key(key_id, key_data)
    
    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args
    assert call_args[0][0] == "INSERT INTO keys (id, data) VALUES (?, ?)"
    assert call_args[0][1][0] == key_id
    assert call_args[0][1][1] == encrypted_data

def test_retrieve_key_success(secure_storage, mock_db):
    key_id = "key_xyz"
    encrypted_data = b"key_encrypted_payload"
    decrypted_data = {"key": "secret_key_value", "algorithm": "AES-256"}
    
    mock_db.execute.return_value.fetchone.return_value = (encrypted_data,)
    
    with patch('src.utils.secure_storage.decrypt_data', return_value=decrypted_data):
        result = secure_storage.retrieve_key(key_id)
    
    assert result == decrypted_data
    mock_db.execute.assert_called_once()

def test_retrieve_key_not_found(secure_storage, mock_db):
    key_id = "nonexistent_key"
    mock_db.execute.return_value.fetchone.return_value = None
    
    with pytest.raises(SecureStorageError, match="Key not found"):
        secure_storage.retrieve_key(key_id)

def test_store_model_caches_in_redis(secure_storage, mock_db, mock_redis):
    model_id = "model_cached"
    model_data = {"name": "cached_model"}
    encrypted_data = b"cached_payload"
    
    mock_redis.setex.return_value = True
    
    with patch('src.utils.secure_storage.encrypt_data', return_value=encrypted_data):
        secure_storage.store_model(model_id, model_data)
    
    mock_redis.setex.assert_called_once_with(
        f"model:{model_id}",
        3600,
        encrypted_data
    )

def test_retrieve_model_uses_redis_cache_first(secure_storage, mock_db, mock_redis):
    model_id = "model_redis_hit"
    encrypted_data = b"cached_payload"
    decrypted_data = {"name": "cached_model"}
    
    mock_redis.get.return_value = encrypted_data
    
    with patch('src.utils.secure_storage.decrypt_data', return_value=decrypted_data):
        result = secure_storage.retrieve_model(model_id)
    
    assert result == decrypted_data
    mock_redis.get.assert_called_once_with(f"model:{model_id}")
    mock_db.execute.assert_not_called()

def test_retrieve_key_uses_redis_cache_first(secure_storage, mock_db, mock_redis):
    key_id = "key_redis_hit"
    encrypted_data = b"key_cached_payload"
    decrypted_data = {"key": "cached_secret", "algorithm": "AES-256"}
    
    mock_redis.get.return_value = encrypted_data
    
    with patch('src.utils.secure_storage.decrypt_data', return_value=decrypted_data):
        result = secure_storage.retrieve_key(key_id)
    
    assert result == decrypted_data
    mock_redis.get.assert_called_once_with(f"key:{key_id}")
    mock_db.execute.assert_not_called()

def test_secure_storage_error_message():
    error = SecureStorageError("Test error message")
    assert str(error) == "Test error message"