import pytest
from unittest.mock import MagicMock, patch
from src.features.extractor import SignalExtractor, extract_features, normalize_signal


@pytest.fixture
def mock_db():
    with patch("src.features.extractor.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.features.extractor.redis_client") as mock:
        yield mock


@pytest.fixture
def sample_signal():
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def extractor(mock_db, mock_redis):
    return SignalExtractor()


def test_signal_extractor_initialization(extractor):
    assert extractor is not None


def test_extract_features_with_valid_signal(sample_signal):
    result = extract_features(sample_signal)
    assert "mean" in result
    assert "std" in result
    assert "min" in result
    assert "max" in result
    assert "sum" in result
    assert result["mean"] == 3.0
    assert abs(result["std"] - 1.41421356) < 1e-6


def test_extract_features_with_empty_signal():
    result = extract_features([])
    assert result["mean"] is None
    assert result["std"] is None
    assert result["min"] is None
    assert result["max"] is None
    assert result["sum"] == 0.0


def test_extract_features_with_single_element():
    result = extract_features([5.0])
    assert result["mean"] == 5.0
    assert result["std"] == 0.0
    assert result["min"] == 5.0
    assert result["max"] == 5.0
    assert result["sum"] == 5.0


def test_normalize_signal(sample_signal):
    result = normalize_signal(sample_signal)
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i, val in enumerate(result):
        assert abs(val - expected[i]) < 1e-6


def test_normalize_signal_with_constant_values():
    result = normalize_signal([7.0, 7.0, 7.0])
    assert all(abs(val - 0.0) < 1e-6 for val in result)


def test_normalize_signal_with_empty_list():
    result = normalize_signal([])
    assert result == []


def test_normalize_signal_with_single_element():
    result = normalize_signal([42.0])
    assert result == [0.0]


def test_SignalExtractor_fetch_signal_from_db(extractor, mock_db):
    mock_db.query.return_value = [1.0, 2.0, 3.0]
    signal = extractor.fetch_signal_from_db("sensor_1", "2023-01-01", "2023-01-02")
    assert signal == [1.0, 2.0, 3.0]
    mock_db.query.assert_called_once_with("sensor_1", "2023-01-01", "2023-01-02")


def test_SignalExtractor_fetch_signal_from_redis(extractor, mock_redis):
    mock_redis.get.return_value = b"[1.0,2.0,3.0]"
    signal = extractor.fetch_signal_from_redis("sensor_2")
    assert signal == [1.0, 2.0, 3.0]
    mock_redis.get.assert_called_once_with("sensor_2")


def test_SignalExtractor_process_signal(extractor, sample_signal):
    mock_redis.get.return_value = None
    mock_db.query.return_value = sample_signal
    result = extractor.process_signal("sensor_3")
    assert result["mean"] == 3.0
    assert result["max"] == 5.0


def test_SignalExtractor_process_signal_from_redis(extractor, mock_redis):
    mock_redis.get.return_value = b"[10.0,20.0,30.0]"
    result = extractor.process_signal("sensor_4")
    assert result["mean"] == 20.0
    assert result["max"] == 30.0


def test_SignalExtractor_process_signal_with_invalid_redis_data(extractor, mock_redis):
    mock_redis.get.return_value = b"invalid"
    mock_db.query.return_value = [1.0, 2.0]
    result = extractor.process_signal("sensor_5")
    assert result["mean"] == 1.5
    assert result["max"] == 2.0


def test_SignalExtractor_process_signal_fallback_to_db_on_redis_error(extractor, mock_redis, mock_db):
    mock_redis.get.side_effect = Exception("Redis down")
    mock_db.query.return_value = [100.0, 200.0]
    result = extractor.process_signal("sensor_6")
    assert result["mean"] == 150.0
    mock_db.query.assert_called_once()