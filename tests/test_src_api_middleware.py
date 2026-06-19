import pytest
from unittest.mock import MagicMock, patch
from src.api.middleware import *

@pytest.fixture
def mock_redis():
    with patch('src.api.middleware.redis') as mock:
        mock_client = MagicMock()
        mock.from_url.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_db_session():
    with patch('src.api.middleware.Session') as mock:
        session = MagicMock()
        mock.return_value = session
        yield session

@pytest.fixture
def mock_time():
    with patch('src.api.middleware.time') as mock:
        mock.time.return_value = 1000.0
        yield mock

@pytest.fixture
def mock_logger():
    with patch('src.api.middleware.logger') as mock:
        yield mock

def test_rate_limit_middleware_allows_under_limit(mock_redis, mock_time):
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.ttl.return_value = 60

    request = MagicMock()
    request.client.host = "192.168.1.1"
    request.url.path = "/api/data"
    request.method = "POST"

    response = rate_limit_middleware(request, lambda req: MagicMock(status_code=200))

    assert response.status_code == 200
    mock_redis.get.assert_called_once_with("ratelimit:192.168.1.1:/api/data")
    mock_redis.setex.assert_called_once_with(
        "ratelimit:192.168.1.1:/api/data", 60, 1
    )

def test_rate_limit_middleware_blocks_over_limit(mock_redis, mock_time):
    mock_redis.get.return_value = b"5"
    mock_redis.setex.return_value = True

    request = MagicMock()
    request.client.host = "192.168.1.1"
    request.url.path = "/api/data"
    request.method = "POST"

    response = rate_limit_middleware(request, lambda req: MagicMock(status_code=200))

    assert response.status_code == 429
    assert "Rate limit exceeded" in response.body.decode()

def test_rate_limit_middleware_handles_redis_error(mock_redis, mock_logger):
    mock_redis.get.side_effect = Exception("Redis unavailable")

    request = MagicMock()
    request.client.host = "192.168.1.1"
    request.url.path = "/api/data"
    request.method = "POST"

    response = rate_limit_middleware(request, lambda req: MagicMock(status_code=200))

    assert response.status_code == 200
    mock_logger.warning.assert_called_once()

def test_anomaly_detector_detects_high_latency(mock_db_session, mock_time):
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    detector = AnomalyDetector(window_size=10, threshold=2.0)
    detector.add_sample(1.0)
    detector.add_sample(1.1)
    detector.add_sample(0.9)
    detector.add_sample(1.05)
    detector.add_sample(1.02)
    detector.add_sample(0.98)
    detector.add_sample(1.01)
    detector.add_sample(0.99)
    detector.add_sample(1.03)
    detector.add_sample(5.0)

    assert detector.is_anomalous() is True

def test_anomaly_detector_no_anomaly_under_threshold(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=5, threshold=2.0)
    for _ in range(5):
        detector.add_sample(1.0)

    assert detector.is_anomalous() is False

def test_anomaly_detector_insufficient_samples(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=5, threshold=2.0)
    detector.add_sample(1.0)
    detector.add_sample(2.0)

    assert detector.is_anomalous() is False

def test_anomaly_detector_saves_to_db(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=2, threshold=2.0)
    detector.add_sample(1.0)
    detector.add_sample(10.0)
    detector.is_anomalous()

    assert mock_db_session.add.called
    assert mock_db_session.commit.called
    record = mock_db_session.add.call_args[0][0]
    assert record.metric_type == "latency"
    assert record.value == 10.0
    assert record.is_anomaly is True

def test_anomaly_detector_clears_window_after_check(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=3, threshold=2.0)
    detector.add_sample(1.0)
    detector.add_sample(1.1)
    detector.add_sample(1.2)
    detector.is_anomalous()
    assert len(detector.samples) == 0

def test_anomaly_detector_rejects_negative_samples(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=5, threshold=2.0)
    with pytest.raises(ValueError):
        detector.add_sample(-1.0)

def test_anomaly_detector_rejects_nan(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=5, threshold=2.0)
    with pytest.raises(ValueError):
        detector.add_sample(float('nan'))

def test_anomaly_detector_rejects_inf(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=5, threshold=2.0)
    with pytest.raises(ValueError):
        detector.add_sample(float('inf'))

def test_anomaly_detector_rejects_none(mock_db_session, mock_time):
    detector = AnomalyDetector(window_size=5, threshold=2.0)
    with pytest.raises(ValueError):
        detector.add_sample(None)