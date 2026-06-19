import pytest
from unittest.mock import MagicMock, patch
from src.ml.training_data_generator import (
    generate_synthetic_events,
    apply_smote,
    filter_rare_events,
    TrainingDataGenerator
)


@pytest.fixture
def mock_db():
    with patch('src.ml.training_data_generator.DBClient') as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch('src.ml.training_data_generator.RedisClient') as mock:
        yield mock


@pytest.fixture
def sample_data():
    return [
        {'feature1': 0.1, 'feature2': 0.2, 'label': 0},
        {'feature1': 0.3, 'feature2': 0.4, 'label': 0},
        {'feature1': 0.9, 'feature2': 0.8, 'label': 1},
    ]


@pytest.fixture
def rare_events_data():
    return [
        {'feature1': 0.1, 'feature2': 0.2, 'label': 1},
        {'feature1': 0.15, 'feature2': 0.25, 'label': 1},
    ]


def test_generate_synthetic_events_basic(mock_db, sample_data):
    mock_db_instance = MagicMock()
    mock_db_instance.fetch_events.return_value = sample_data
    mock_db.return_value = mock_db_instance

    result = generate_synthetic_events(min_events=1, max_events=3, noise_std=0.05)

    assert isinstance(result, list)
    assert len(result) >= 1
    assert all('feature1' in item and 'feature2' in item and 'label' in item for item in result)


def test_generate_synthetic_events_no_events(mock_db):
    mock_db_instance = MagicMock()
    mock_db_instance.fetch_events.return_value = []
    mock_db.return_value = mock_db_instance

    result = generate_synthetic_events(min_events=0, max_events=0, noise_std=0.0)

    assert result == []


def test_generate_synthetic_events_with_rare(mock_db, rare_events_data):
    mock_db_instance = MagicMock()
    mock_db_instance.fetch_events.return_value = rare_events_data
    mock_db.return_value = mock_db_instance

    result = generate_synthetic_events(min_events=2, max_events=5, noise_std=0.1)

    assert len(result) >= 2
    assert any(item['label'] == 1 for item in result)


def test_apply_smote_basic(sample_data):
    X = [[d['feature1'], d['feature2']] for d in sample_data]
    y = [d['label'] for d in sample_data]

    X_res, y_res = apply_smote(X, y, k=1, sampling_strategy=0.5)

    assert len(X_res) == len(y_res)
    assert len(y_res) >= len(y)
    assert sum(y_res) >= sum(y)


def test_apply_smote_no_minority():
    X = [[0.1, 0.2], [0.3, 0.4]]
    y = [0, 0]

    X_res, y_res = apply_smote(X, y, k=1, sampling_strategy=0.5)

    assert X_res == X
    assert y_res == y


def test_filter_rare_events_basic(sample_data):
    result = filter_rare_events(sample_data, label=1, min_count=1)

    assert len(result) == 1
    assert result[0]['label'] == 1


def test_filter_rare_events_no_match(sample_data):
    result = filter_rare_events(sample_data, label=2, min_count=1)

    assert result == []


def test_filter_rare_events_insufficient(sample_data):
    result = filter_rare_events(sample_data, label=1, min_count=5)

    assert result == []


class TestTrainingDataGenerator:
    def test_init(self):
        gen = TrainingDataGenerator(db_host='localhost', redis_host='localhost')
        assert gen.db_host == 'localhost'
        assert gen.redis_host == 'localhost'

    @patch('src.ml.training_data_generator.DBClient')
    @patch('src.ml.training_data_generator.RedisClient')
    def test_generate(self, mock_redis_client, mock_db_client, sample_data):
        mock_db_instance = MagicMock()
        mock_db_instance.fetch_events.return_value = sample_data
        mock_db_client.return_value = mock_db_instance

        mock_redis_instance = MagicMock()
        mock_redis_instance.get.return_value = None
        mock_redis_client.return_value = mock_redis_instance

        gen = TrainingDataGenerator()
        result = gen.generate(min_events=1, max_events=3, noise_std=0.05)

        assert isinstance(result, list)
        assert len(result) >= 1

    @patch('src.ml.training_data_generator.DBClient')
    @patch('src.ml.training_data_generator.RedisClient')
    def test_generate_caching(self, mock_redis_client, mock_db_client, sample_data):
        mock_db_instance = MagicMock()
        mock_db_instance.fetch_events.return_value = sample_data
        mock_db_client.return_value = mock_db_instance

        mock_redis_instance = MagicMock()
        mock_redis_instance.get.return_value = b'cached'
        mock_redis_client.return_value = mock_redis_instance

        gen = TrainingDataGenerator()
        result = gen.generate(min_events=1, max_events=3, noise_std=0.05)

        assert result == 'cached'
        mock_db_instance.fetch_events.assert_not_called()