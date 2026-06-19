import pytest
from unittest.mock import MagicMock, patch
from src.utils.metrics import LatencyTracker, AccuracyTracker, compute_accuracy, record_latency, get_latency_stats, record_accuracy


@pytest.fixture
def mock_redis():
    with patch('src.utils.metrics.redis.Redis') as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_db():
    with patch('src.utils.metrics.db') as mock:
        yield mock


def test_latency_tracker_init():
    tracker = LatencyTracker()
    assert tracker.latencies == []
    assert tracker.window_size == 100


def test_latency_tracker_record():
    tracker = LatencyTracker(window_size=3)
    tracker.record(0.1)
    tracker.record(0.2)
    tracker.record(0.3)
    assert tracker.latencies == [0.1, 0.2, 0.3]
    tracker.record(0.4)
    assert tracker.latencies == [0.2, 0.3, 0.4]


def test_latency_tracker_get_stats_empty():
    tracker = LatencyTracker()
    stats = tracker.get_stats()
    assert stats == {'count': 0, 'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0}


def test_latency_tracker_get_stats():
    tracker = LatencyTracker(window_size=5)
    for val in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tracker.record(val)
    stats = tracker.get_stats()
    assert stats['count'] == 5
    assert stats['mean'] == 0.3
    assert stats['p50'] == 0.3
    assert stats['p95'] == 0.45
    assert stats['p99'] == 0.495


def test_accuracy_tracker_init():
    tracker = AccuracyTracker()
    assert tracker.correct == 0
    assert tracker.total == 0


def test_accuracy_tracker_record():
    tracker = AccuracyTracker()
    tracker.record(True)
    tracker.record(False)
    tracker.record(True)
    assert tracker.correct == 2
    assert tracker.total == 3


def test_accuracy_tracker_get_accuracy():
    tracker = AccuracyTracker()
    tracker.record(True)
    tracker.record(False)
    tracker.record(True)
    assert tracker.get_accuracy() == pytest.approx(0.6666666666666666)


def test_compute_accuracy():
    predictions = [1, 0, 1, 1, 0]
    targets = [1, 0, 0, 1, 1]
    acc = compute_accuracy(predictions, targets)
    assert acc == 0.6


def test_compute_accuracy_empty():
    acc = compute_accuracy([], [])
    assert acc == 0.0


@patch('src.utils.metrics.LatencyTracker')
def test_record_latency(mock_tracker_class, mock_redis):
    mock_tracker = MagicMock()
    mock_tracker_class.return_value = mock_tracker
    mock_redis.get.return_value = None

    record_latency(0.123)

    mock_tracker.record.assert_called_once_with(0.123)
    mock_redis.set.assert_called_once_with('latency_tracker', mock_tracker)


@patch('src.utils.metrics.LatencyTracker')
def test_record_latency_load_from_redis(mock_tracker_class, mock_redis):
    mock_tracker = MagicMock()
    mock_tracker_class.from_dict.return_value = mock_tracker
    mock_redis.get.return_value = b'{"latencies":[0.1],"window_size":100}'

    record_latency(0.2)

    mock_tracker_class.from_dict.assert_called_once()
    mock_tracker.record.assert_called_once_with(0.2)
    mock_redis.set.assert_called_once()


def test_get_latency_stats(mock_redis):
    mock_tracker = MagicMock()
    mock_tracker.get_stats.return_value = {'count': 1, 'mean': 0.1}
    mock_redis.get.return_value = None

    with patch('src.utils.metrics.LatencyTracker', return_value=mock_tracker):
        stats = get_latency_stats()

    assert stats == {'count': 1, 'mean': 0.1}


def test_record_accuracy(mock_db, mock_redis):
    mock_db.execute = MagicMock()
    record_accuracy(True)

    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args[0][0]
    assert 'INSERT INTO accuracy' in call_args
    assert True in call_args[1] or call_args[1][0] is True


def test_record_accuracy_false(mock_db, mock_redis):
    mock_db.execute = MagicMock()
    record_accuracy(False)

    call_args = mock_db.execute.call_args[0][0]
    assert False in call_args[1] or call_args[1][0] is False