import pytest
from unittest.mock import MagicMock, patch
from src.ml.data_preprocessor import *

@pytest.fixture
def mock_db():
    with patch('src.ml.data_preprocessor.db') as mock:
        yield mock

@pytest.fixture
def mock_redis():
    with patch('src.ml.data_preprocessor.redis_client') as mock:
        yield mock

@pytest.fixture
def sample_data():
    return [
        {'feature_a': 1.0, 'feature_b': 2.0, 'feature_c': None},
        {'feature_a': 3.0, 'feature_b': None, 'feature_c': 5.0},
        {'feature_a': None, 'feature_b': 4.0, 'feature_c': 6.0}
    ]

@pytest.fixture
def valid_schema():
    return {
        'feature_a': {'type': 'float', 'min': 0.0, 'max': 10.0},
        'feature_b': {'type': 'float', 'min': -5.0, 'max': 5.0},
        'feature_c': {'type': 'float', 'min': 0.0, 'max': 10.0}
    }

def test_normalize_data(sample_data, valid_schema):
    result = normalize_data(sample_data, valid_schema)
    assert len(result) == 3
    assert all('feature_a' in item and 'feature_b' in item and 'feature_c' in item for item in result)
    for item in result:
        for feature in ['feature_a', 'feature_b', 'feature_c']:
            assert 0.0 <= item[feature] <= 1.0

def test_normalize_data_missing_feature(sample_data, valid_schema):
    sample_data[0]['feature_d'] = 10.0
    result = normalize_data(sample_data, valid_schema)
    assert 'feature_d' not in result[0]

def test_normalize_data_out_of_bounds(sample_data, valid_schema):
    sample_data[0]['feature_a'] = 15.0
    result = normalize_data(sample_data, valid_schema)
    assert result[0]['feature_a'] == 1.0

def test_handle_missing_values(sample_data):
    result = handle_missing_values(sample_data, strategy='mean')
    assert all(v is not None for item in result for v in item.values())

def test_handle_missing_values_with_zero(sample_data):
    result = handle_missing_values(sample_data, strategy='zero')
    assert result[0]['feature_c'] == 0.0
    assert result[1]['feature_b'] == 0.0
    assert result[2]['feature_a'] == 0.0

def test_handle_missing_values_invalid_strategy(sample_data):
    with pytest.raises(ValueError):
        handle_missing_values(sample_data, strategy='invalid')

def test_fetch_data_from_db(mock_db):
    mock_db.query.return_value = [{'id': 1, 'value': 100}]
    result = fetch_data_from_db("SELECT * FROM data")
    assert result == [{'id': 1, 'value': 100}]
    mock_db.query.assert_called_once_with("SELECT * FROM data")

def test_fetch_data_from_db_empty(mock_db):
    mock_db.query.return_value = []
    result = fetch_data_from_db("SELECT * FROM empty_table")
    assert result == []

def test_cache_preprocessed_data(mock_redis):
    data = [{'a': 1.0}]
    cache_key = "preprocess:batch1"
    result = cache_preprocessed_data(data, cache_key)
    assert result is True
    mock_redis.set.assert_called_once_with(cache_key, str(data))

def test_cache_preprocessed_data_redis_error(mock_redis):
    mock_redis.set.side_effect = Exception("Redis down")
    with pytest.raises(Exception):
        cache_preprocessed_data([{'a': 1.0}], "key")

def test_preprocess_pipeline(sample_data, valid_schema, mock_db, mock_redis):
    mock_db.query.return_value = sample_data
    with patch('src.ml.data_preprocessor.cache_preprocessed_data', return_value=True):
        result = preprocess_pipeline("SELECT * FROM raw_data", valid_schema)
        assert len(result) == 3
        assert all(0.0 <= v <= 1.0 for item in result for v in item.values())

def test_preprocess_pipeline_no_data(mock_db, valid_schema):
    mock_db.query.return_value = []
    with patch('src.ml.data_preprocessor.cache_preprocessed_data', return_value=True):
        result = preprocess_pipeline("SELECT * FROM empty", valid_schema)
        assert result == []

def test_preprocess_pipeline_invalid_schema(sample_data, mock_db):
    mock_db.query.return_value = sample_data
    with pytest.raises(KeyError):
        preprocess_pipeline("SELECT * FROM raw_data", {})

def test_preprocess_pipeline_missing_feature_in_schema(sample_data, mock_db):
    mock_db.query.return_value = sample_data
    schema = {'feature_a': {'type': 'float', 'min': 0.0, 'max': 10.0}}
    with pytest.raises(KeyError):
        preprocess_pipeline("SELECT * FROM raw_data", schema)