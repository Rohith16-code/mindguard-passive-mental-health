import pytest
from unittest.mock import MagicMock, patch
from src.features.normalizer import *

@pytest.fixture
def mock_db():
    with patch('src.features.normalizer.db') as mock:
        yield mock

@pytest.fixture
def mock_redis():
    with patch('src.features.normalizer.redis_client') as mock:
        yield mock

@pytest.fixture
def sample_user_data():
    return {
        'user_id': 'user_123',
        'baseline_values': [100.0, 102.0, 98.0, 101.0],
        'current_values': [105.0, 99.0, 103.0, 100.0]
    }

@pytest.fixture
def empty_user_data():
    return {
        'user_id': 'user_456',
        'baseline_values': [],
        'current_values': []
    }

def test_calculate_baseline_mean(mock_db, sample_user_data):
    mock_db.get_baseline.return_value = sample_user_data['baseline_values']
    result = calculate_baseline_mean(sample_user_data['user_id'])
    expected = sum(sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])
    assert result == expected

def test_calculate_baseline_mean_empty(mock_db):
    mock_db.get_baseline.return_value = []
    result = calculate_baseline_mean('nonexistent_user')
    assert result == 0.0

def test_calculate_baseline_std(mock_db, sample_user_data):
    import math
    mock_db.get_baseline.return_value = sample_user_data['baseline_values']
    result = calculate_baseline_std(sample_user_data['user_id'])
    mean = sum(sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])
    variance = sum((x - mean) ** 2 for x in sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])
    expected = math.sqrt(variance)
    assert result == expected

def test_calculate_baseline_std_single_value(mock_db):
    mock_db.get_baseline.return_value = [100.0]
    result = calculate_baseline_std('user_single')
    assert result == 0.0

def test_calculate_baseline_std_empty(mock_db):
    mock_db.get_baseline.return_value = []
    result = calculate_baseline_std('user_empty')
    assert result == 0.0

def test_normalize_values(mock_db, sample_user_data):
    mock_db.get_baseline.return_value = sample_user_data['baseline_values']
    result = normalize_values(sample_user_data['user_id'], sample_user_data['current_values'])
    mean = sum(sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])
    std = (sum((x - mean) ** 2 for x in sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])) ** 0.5
    expected = [(x - mean) / std if std != 0 else 0.0 for x in sample_user_data['current_values']]
    assert result == expected

def test_normalize_values_zero_std(mock_db):
    mock_db.get_baseline.return_value = [100.0, 100.0, 100.0]
    result = normalize_values('user_zero_std', [100.0, 101.0, 99.0])
    assert result == [0.0, 0.0, 0.0]

def test_normalize_values_empty_current(mock_db, sample_user_data):
    mock_db.get_baseline.return_value = sample_user_data['baseline_values']
    result = normalize_values(sample_user_data['user_id'], [])
    assert result == []

def test_get_normalized_profile(mock_db, mock_redis, sample_user_data):
    mock_db.get_baseline.return_value = sample_user_data['baseline_values']
    mock_redis.get.return_value = None  # no cached profile
    result = get_normalized_profile(sample_user_data['user_id'], sample_user_data['current_values'])
    mean = sum(sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])
    std = (sum((x - mean) ** 2 for x in sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])) ** 0.5
    expected = [(x - mean) / std if std != 0 else 0.0 for x in sample_user_data['current_values']]
    assert result == expected
    mock_redis.set.assert_called_once()

def test_get_normalized_profile_cached(mock_db, mock_redis, sample_user_data):
    import json
    cached = json.dumps({'profile': [0.5, -0.5, 1.0, 0.0]})
    mock_redis.get.return_value = cached
    result = get_normalized_profile(sample_user_data['user_id'], sample_user_data['current_values'])
    assert result == [0.5, -0.5, 1.0, 0.0]
    mock_redis.get.assert_called_once()
    mock_db.get_baseline.assert_not_called()

def test_get_normalized_profile_db_error(mock_db, mock_redis):
    mock_db.get_baseline.side_effect = Exception("DB connection failed")
    mock_redis.get.return_value = None
    with pytest.raises(Exception, match="DB connection failed"):
        get_normalized_profile('user_err', [100.0])

def test_calculate_zscore():
    result = calculate_zscore(110.0, 100.0, 5.0)
    assert result == 2.0

def test_calculate_zscore_zero_std():
    result = calculate_zscore(110.0, 100.0, 0.0)
    assert result == 0.0

def test_calculate_zscore_none_std():
    result = calculate_zscore(110.0, 100.0, None)
    assert result == 0.0

def test_calculate_zscore_none_mean():
    result = calculate_zscore(110.0, None, 5.0)
    assert result == 0.0

def test_calculate_zscore_none_value():
    result = calculate_zscore(None, 100.0, 5.0)
    assert result == 0.0

def test_calculate_zscore_all_none():
    result = calculate_zscore(None, None, None)
    assert result == 0.0

def test_calculate_zscore_vectorized(mock_db, sample_user_data):
    mock_db.get_baseline.return_value = sample_user_data['baseline_values']
    result = calculate_zscore_vectorized(sample_user_data['user_id'], sample_user_data['current_values'])
    mean = sum(sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])
    std = (sum((x - mean) ** 2 for x in sample_user_data['baseline_values']) / len(sample_user_data['baseline_values'])) ** 0.5
    import numpy as np
    expected = np.array(sample_user_data['current_values'])
    expected = (expected - mean) / std if std != 0 else np.zeros_like(expected)
    assert np.allclose(result, expected)

def test_calculate_zscore_vectorized_empty(mock_db):
    mock_db.get_baseline.return_value = []
    result = calculate_zscore_vectorized('user_empty', [])
    assert len(result) == 0

def test_calculate_zscore_vectorized_single_value(mock_db):
    mock_db.get_baseline.return_value = [100.0]
    result = calculate_zscore_vectorized('user_single', [102.0])
    assert result == [0.0]  # std=0 => zscore=0

def test_calculate_zscore_vectorized_nonnumeric(mock_db):
    mock_db.get_baseline.return_value = [100.0]
    with pytest.raises(TypeError):
        calculate_zscore_vectorized('user_err', ['a', 'b'])