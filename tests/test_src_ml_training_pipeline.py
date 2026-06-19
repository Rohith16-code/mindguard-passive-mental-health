import pytest
from unittest.mock import MagicMock, patch
from src.ml.training_pipeline import TrainingPipeline, train_model, run_pipeline


@pytest.fixture
def mock_db():
    with patch("src.ml.training_pipeline.DBClient") as mock:
        yield mock.return_value


@pytest.fixture
def mock_redis():
    with patch("src.ml.training_pipeline.RedisClient") as mock:
        yield mock.return_value


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.train.return_value = {"loss": 0.123, "accuracy": 0.95}
    model.save.return_value = "/models/model_v123.pkl"
    return model


@pytest.fixture
def mock_data_loader():
    with patch("src.ml.training_pipeline.DataLoader") as mock:
        mock.return_value.load.return_value = (["x1", "x2"], ["y1", "y2"])
        yield mock.return_value


@pytest.fixture
def mock_training_config():
    return {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "model_type": "CNN",
        "data_path": "/data/train"
    }


def test_training_pipeline_initialization(mock_db, mock_redis):
    pipeline = TrainingPipeline(db_client=mock_db, redis_client=mock_redis)
    assert pipeline.db_client == mock_db
    assert pipeline.redis_client == mock_redis


def test_training_pipeline_prepare_data(mock_db, mock_redis, mock_data_loader):
    pipeline = TrainingPipeline(db_client=mock_db, redis_client=mock_redis)
    with patch("src.ml.training_pipeline.DataLoader", return_value=mock_data_loader):
        features, labels = pipeline.prepare_data("/data/train")
        assert features == ["x1", "x2"]
        assert labels == ["y1", "y2"]
        mock_data_loader.load.assert_called_once_with("/data/train")


def test_training_pipeline_train(mock_db, mock_redis, mock_model, mock_training_config):
    pipeline = TrainingPipeline(db_client=mock_db, redis_client=mock_redis)
    with patch("src.ml.training_pipeline.get_model", return_value=mock_model):
        result = pipeline.train(mock_training_config)
        assert result["model_path"] == "/models/model_v123.pkl"
        assert result["metrics"]["loss"] == 0.123
        assert result["metrics"]["accuracy"] == 0.95
        mock_model.train.assert_called_once_with(["x1", "x2"], ["y1", "y2"], epochs=10, batch_size=32, learning_rate=0.001)
        mock_model.save.assert_called_once_with("/models/model_v123.pkl")


def test_training_pipeline_save_model(mock_db, mock_redis, mock_model):
    pipeline = TrainingPipeline(db_client=mock_db, redis_client=mock_redis)
    model_path = pipeline.save_model(mock_model, "model_v456")
    assert model_path == "/models/model_v456.pkl"
    mock_model.save.assert_called_once_with("/models/model_v456.pkl")


def test_training_pipeline_publish_status(mock_db, mock_redis):
    pipeline = TrainingPipeline(db_client=mock_db, redis_client=mock_redis)
    pipeline.publish_status("training", "running", {"epoch": 1})
    mock_redis.publish.assert_called_once_with("ml:status:training", '{"status": "running", "details": {"epoch": 1}}')


def test_train_model(mock_db, mock_redis, mock_model, mock_training_config):
    with patch("src.ml.training_pipeline.TrainingPipeline") as MockPipeline:
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.train.return_value = {
            "model_path": "/models/model_v789.pkl",
            "metrics": {"loss": 0.05, "accuracy": 0.98}
        }
        result = train_model(mock_training_config)
        assert result["model_path"] == "/models/model_v789.pkl"
        MockPipeline.assert_called_once_with(db_client=mock_db, redis_client=mock_redis)


def test_run_pipeline(mock_db, mock_redis, mock_training_config):
    with patch("src.ml.training_pipeline.TrainingPipeline") as MockPipeline:
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.prepare_data.return_value = (["x1", "x2"], ["y1", "y2"])
        mock_pipeline_instance.train.return_value = {
            "model_path": "/models/model_final.pkl",
            "metrics": {"loss": 0.01, "accuracy": 0.99}
        }
        mock_pipeline_instance.save_model.return_value = "/models/model_final.pkl"
        mock_pipeline_instance.publish_status.side_effect = lambda *args: None

        result = run_pipeline(mock_training_config)

        assert result["status"] == "completed"
        assert result["model_path"] == "/models/model_final.pkl"
        assert result["metrics"]["accuracy"] == 0.99
        mock_pipeline_instance.publish_status.assert_any_call("training", "running", {"stage": "data_preparation"})
        mock_pipeline_instance.prepare_data.assert_called_once_with("/data/train")
        mock_pipeline_instance.train.assert_called_once_with(mock_training_config)