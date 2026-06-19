import pytest
from unittest.mock import MagicMock, patch
from src.ml.federated_aggregator import FederatedAggregator, aggregate_updates, get_cohort_updates


@pytest.fixture
def mock_db():
    with patch("src.ml.federated_aggregator.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.ml.federated_aggregator.redis_client") as mock:
        yield mock


@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.update_weights.return_value = None
    return mock


@pytest.fixture
def aggregator(mock_db, mock_redis):
    return FederatedAggregator()


def test_aggregator_initialization(aggregator):
    assert aggregator is not None


def test_aggregator_aggregate_updates(aggregator, mock_model):
    updates = [
        {"weights": [1.0, 2.0], "count": 10},
        {"weights": [3.0, 4.0], "count": 20},
    ]
    result = aggregator.aggregate_updates(updates, mock_model)
    mock_model.update_weights.assert_called_once()
    assert result is None


def test_aggregate_updates_empty_updates():
    result = aggregate_updates([])
    assert result == []


def test_aggregate_updates_single_update():
    updates = [{"weights": [1.0], "count": 1}]
    result = aggregate_updates(updates)
    assert result == [{"weights": [1.0], "count": 1}]


def test_aggregate_updates_weighted_average():
    updates = [
        {"weights": [0.0, 0.0], "count": 10},
        {"weights": [2.0, 4.0], "count": 10},
    ]
    result = aggregate_updates(updates)
    expected = [{"weights": [1.0, 2.0], "count": 20}]
    assert len(result) == 1
    assert result[0]["weights"] == expected[0]["weights"]
    assert result[0]["count"] == expected[0]["count"]


def test_get_cohort_updates_success(mock_db, mock_redis):
    mock_db.query.return_value = [
        {"client_id": "c1", "weights": [1.0, 2.0], "count": 5},
        {"client_id": "c2", "weights": [3.0, 4.0], "count": 15},
    ]
    mock_redis.get.return_value = b'{"cohort_id": "c1", "timestamp": 12345}'

    result = get_cohort_updates("cohort_1")
    assert len(result) == 2
    assert result[0]["weights"] == [1.0, 2.0]
    assert result[1]["count"] == 15


def test_get_cohort_updates_redis_miss(mock_db, mock_redis):
    mock_redis.get.return_value = None
    mock_db.query.return_value = []

    result = get_cohort_updates("unknown_cohort")
    assert result == []


def test_get_cohort_updates_db_error(mock_db, mock_redis):
    mock_db.query.side_effect = Exception("DB connection failed")
    result = get_cohort_updates("cohort_1")
    assert result == []


def test_federated_aggregator_roundtrip(aggregator, mock_model):
    updates = [
        {"weights": [1.0, 2.0, 3.0], "count": 100},
        {"weights": [2.0, 4.0, 6.0], "count": 200},
    ]
    aggregator.aggregate_updates(updates, mock_model)
    mock_model.update_weights.assert_called_once()
    call_args = mock_model.update_weights.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["weights"] == [5.0/3, 10.0/3, 5.0]  # weighted avg: (1*100 + 2*200)/300 etc.
    assert call_args[0]["count"] == 300