import pytest
from unittest.mock import MagicMock, patch
from src.db.cache import *

@pytest.fixture
def mock_redis():
    with patch('src.db.cache.redis') as mock:
        mock_client = MagicMock()
        mock.Redis.return_value = mock_client
        yield mock_client

@pytest.fixture
def lru_cache(mock_redis):
    cache = LRUCache(max_size=3)
    return cache

def test_lru_cache_init(lru_cache):
    assert lru_cache.max_size == 3
    assert len(lru_cache.cache) == 0
    assert lru_cache.order == []

def test_lru_cache_get_hit(lru_cache):
    lru_cache.cache = {'key1': 'value1', 'key2': 'value2'}
    lru_cache.order = ['key1', 'key2']
    result = lru_cache.get('key1')
    assert result == 'value1'
    assert lru_cache.order == ['key2', 'key1']

def test_lru_cache_get_miss(lru_cache):
    lru_cache.cache = {'key1': 'value1'}
    lru_cache.order = ['key1']
    result = lru_cache.get('nonexistent')
    assert result is None
    assert lru_cache.order == ['key1']

def test_lru_cache_put_new_key(lru_cache):
    lru_cache.put('key1', 'value1')
    assert lru_cache.cache['key1'] == 'value1'
    assert lru_cache.order == ['key1']

def test_lru_cache_put_update_key(lru_cache):
    lru_cache.cache = {'key1': 'old_value'}
    lru_cache.order = ['key1']
    lru_cache.put('key1', 'new_value')
    assert lru_cache.cache['key1'] == 'new_value'
    assert lru_cache.order == ['key1']

def test_lru_cache_put_eviction(lru_cache):
    lru_cache.cache = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
    lru_cache.order = ['key1', 'key2', 'key3']
    lru_cache.put('key4', 'val4')
    assert 'key1' not in lru_cache.cache
    assert 'key1' not in lru_cache.order
    assert lru_cache.order == ['key2', 'key3', 'key4']

def test_lru_cache_put_eviction_updates_order(lru_cache):
    lru_cache.cache = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
    lru_cache.order = ['key1', 'key2', 'key3']
    lru_cache.get('key2')  # move key2 to end
    lru_cache.put('key4', 'val4')  # evict key1 (oldest)
    assert 'key1' not in lru_cache.cache
    assert lru_cache.order == ['key3', 'key2', 'key4']

def test_lru_cache_delete(lru_cache):
    lru_cache.cache = {'key1': 'value1', 'key2': 'value2'}
    lru_cache.order = ['key1', 'key2']
    lru_cache.delete('key1')
    assert 'key1' not in lru_cache.cache
    assert 'key1' not in lru_cache.order
    assert lru_cache.order == ['key2']

def test_lru_cache_delete_missing_key(lru_cache):
    lru_cache.cache = {'key1': 'value1'}
    lru_cache.order = ['key1']
    lru_cache.delete('nonexistent')
    assert lru_cache.cache == {'key1'}
    assert lru_cache.order == ['key1']

def test_lru_cache_clear(lru_cache):
    lru_cache.cache = {'key1': 'value1', 'key2': 'value2'}
    lru_cache.order = ['key1', 'key2']
    lru_cache.clear()
    assert lru_cache.cache == {}
    assert lru_cache.order == []

def test_lru_cache_len(lru_cache):
    lru_cache.cache = {'key1': 'value1', 'key2': 'value2'}
    assert len(lru_cache) == 2

def test_lru_cache_contains(lru_cache):
    lru_cache.cache = {'key1': 'value1'}
    assert 'key1' in lru_cache
    assert 'nonexistent' not in lru_cache

def test_lru_cache_get_from_redis_on_miss(mock_redis, lru_cache):
    mock_redis.get.return_value = b'redis_value'
    lru_cache.cache = {}
    lru_cache.order = []
    result = lru_cache.get('redis_key')
    assert result == 'redis_value'
    assert lru_cache.cache['redis_key'] == 'redis_value'
    assert lru_cache.order == ['redis_key']
    mock_redis.get.assert_called_once_with('redis_key')

def test_lru_cache_put_to_redis(mock_redis, lru_cache):
    lru_cache.cache = {}
    lru_cache.order = []
    lru_cache.put('redis_key', 'redis_value')
    mock_redis.set.assert_called_once_with('redis_key', b'redis_value')

def test_lru_cache_delete_removes_from_redis(mock_redis, lru_cache):
    lru_cache.cache = {'redis_key': 'value'}
    lru_cache.order = ['redis_key']
    lru_cache.delete('redis_key')
    mock_redis.delete.assert_called_once_with('redis_key')
    assert 'redis_key' not in lru_cache.cache

def test_lru_cache_redis_error_handling(mock_redis, lru_cache):
    mock_redis.get.side_effect = Exception("Redis down")
    lru_cache.cache = {}
    lru_cache.order = []
    result = lru_cache.get('error_key')
    assert result is None
    assert 'error_key' not in lru_cache.cache

def test_lru_cache_redis_set_error_handling(mock_redis, lru_cache):
    mock_redis.set.side_effect = Exception("Redis write failed")
    lru_cache.cache = {}
    lru_cache.order = []
    lru_cache.put('error_key', 'value')
    assert 'error_key' in lru_cache.cache
    assert lru_cache.order == ['error_key']