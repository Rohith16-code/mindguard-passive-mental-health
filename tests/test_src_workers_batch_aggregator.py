import pytest
from unittest.mock import MagicMock, patch
from src.workers.batch_aggregator import BatchAggregator, aggregate_features


@pytest.fixture
def mock_db():
    with patch("src.workers.batch_aggregator.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.workers.batch_aggregator.redis") as mock:
        yield mock


@pytest.fixture
def mock_logger():
    with patch("src.workers.batch_aggregator.logger") as mock:
        yield mock


@pytest.fixture
def aggregator(mock_db, mock_redis, mock_logger):
    return BatchAggregator(window_size=5, aggregation_interval=10)


def test_aggregator_initialization(aggregator):
    assert aggregator.window_size == 5
    assert aggregator.aggregation_interval == 10
    assert len(aggregator._window) == 0


def test_aggregator_add_feature(aggregator):
    feature = {"id": "f1", "value": 42.0, "timestamp": 1000}
    aggregator.add_feature(feature)
    assert len(aggregator._window) == 1
    assert aggregator._window[0] == feature


def test_aggregator_add_feature_evicts_old_features(aggregator):
    # Set up window with 5 features
    for i in range(5):
        aggregator.add_feature({"id": f"f{i}", "value": float(i), "timestamp": 1000 + i})
    assert len(aggregator._window) == 5

    # Add a new feature with timestamp beyond window
    aggregator.add_feature({"id": "f6", "value": 100.0, "timestamp": 1015})
    assert len(aggregator._window) == 5  # still 5, but oldest evicted
    assert aggregator._window[0]["id"] == "f2"  # f0 and f1 evicted


def test_aggregator_aggregate_empty_window(aggregator, mock_db, mock_redis):
    result = aggregator.aggregate()
    assert result is None


def test_aggregator_aggregate_single_feature(aggregator, mock_db, mock_redis):
    feature = {"id": "f1", "value": 10.0, "timestamp": 1000}
    aggregator.add_feature(feature)
    result = aggregator.aggregate()

    assert result is not None
    assert result["feature_count"] == 1
    assert result["avg_value"] == 10.0
    assert result["min_value"] == 10.0
    assert result["max_value"] == 10.0
    assert result["sum_value"] == 10.0


def test_aggregator_aggregate_multiple_features(aggregator, mock_db, mock_redis):
    features = [
        {"id": f"f{i}", "value": float(i), "timestamp": 1000 + i}
        for i in range(1, 6)
    ]
    for f in features:
        aggregator.add_feature(f)

    result = aggregator.aggregate()

    assert result is not None
    assert result["feature_count"] == 5
    assert result["avg_value"] == 3.0  # (1+2+3+4+5)/5
    assert result["min_value"] == 1.0
    assert result["max_value"] == 5.0
    assert result["sum_value"] == 15.0


def test_aggregator_aggregate_clears_window(aggregator, mock_db, mock_redis):
    aggregator.add_feature({"id": "f1", "value": 1.0, "timestamp": 1000})
    aggregator.aggregate()
    assert len(aggregator._window) == 0


def test_aggregator_aggregate_saves_to_db(aggregator, mock_db, mock_redis):
    aggregator.add_feature({"id": "f1", "value": 1.0, "timestamp": 1000})
    aggregator.aggregate()

    mock_db.insert_aggregation.assert_called_once()
    call_args = mock_db.insert_aggregation.call_args[0][0]
    assert call_args["feature_count"] == 1


def test_aggregator_aggregate_publishes_to_redis(aggregator, mock_db, mock_redis):
    aggregator.add_feature({"id": "f1", "value": 1.0, "timestamp": 1000})
    aggregator.aggregate()

    mock_redis.publish.assert_called_once()
    channel, message = mock_redis.publish.call_args[0]
    assert channel == "aggregations"
    assert "feature_count" in message


def test_aggregator_aggregate_logs_result(aggregator, mock_db, mock_redis, mock_logger):
    aggregator.add_feature({"id": "f1", "value": 1.0, "timestamp": 1000})
    aggregator.aggregate()

    mock_logger.info.assert_called()
    log_call = mock_logger.info.call_args[0][0]
    assert "feature_count" in log_call


def test_aggregate_features_function(aggregator, mock_db, mock_redis, mock_logger):
    with patch("src.workers.batch_aggregator.BatchAggregator", return_value=aggregator):
        features = [
            {"id": "f1", "value": 5.0, "timestamp": 1000},
            {"id": "f2", "value": 15.0, "timestamp": 1001}
        ]
        result = aggregate_features(features)

        assert result is not None
        assert result["feature_count"] == 2
        assert result["avg_value"] == 10.0


def test_aggregate_features_function_handles_empty_input(aggregator, mock_db, mock_redis, mock_logger):
    with patch("src.workers.batch_aggregator.BatchAggregator", return_value=aggregator):
        result = aggregate_features([])

        assert result is None