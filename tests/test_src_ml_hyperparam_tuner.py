import pytest
from unittest.mock import MagicMock, patch
from src.ml.hyperparam_tuner import HyperparamTuner, run_grid_search


@pytest.fixture
def mock_db():
    with patch("src.ml.hyperparam_tuner.DBClient") as mock:
        yield mock.return_value


@pytest.fixture
def mock_redis():
    with patch("src.ml.hyperparam_tuner.RedisClient") as mock:
        yield mock.return_value


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.fit = MagicMock()
    model.predict = MagicMock(return_value=[0.1, 0.9])
    model.score = MagicMock(return_value=0.85)
    return model


@pytest.fixture
def sample_data():
    return {
        "user_id": "user_123",
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "y": [0, 1, 1],
        "calibration_params": {"lr": [0.01, 0.1], "reg": [0.001, 0.01]}
    }


def test_hyperparam_tuner_initialization():
    tuner = HyperparamTuner()
    assert tuner is not None
    assert tuner.best_params is None
    assert tuner.best_score is None


def test_hyperparam_tuner_run_grid_search(mock_model, sample_data, mock_db, mock_redis):
    tuner = HyperparamTuner()
    tuner._build_model = MagicMock(return_value=mock_model)
    
    tuner.run_grid_search(sample_data)
    
    assert tuner.best_params is not None
    assert "lr" in tuner.best_params
    assert "reg" in tuner.best_params
    assert tuner.best_score > 0


def test_hyperparam_tuner_run_grid_search_no_improvement(mock_model, sample_data, mock_db, mock_redis):
    mock_model.score = MagicMock(return_value=0.5)
    tuner = HyperparamTuner()
    tuner._build_model = MagicMock(return_value=mock_model)
    
    tuner.run_grid_search(sample_data)
    
    assert tuner.best_params is not None
    assert tuner.best_score == 0.5


def test_hyperparam_tuner_build_model():
    tuner = HyperparamTuner()
    params = {"lr": 0.01, "reg": 0.001}
    model = tuner._build_model(params)
    
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_run_grid_search_function(sample_data, mock_db, mock_redis):
    with patch("src.ml.hyperparam_tuner.HyperparamTuner") as MockTuner:
        mock_instance = MagicMock()
        mock_instance.run_grid_search = MagicMock()
        MockTuner.return_value = mock_instance
        
        result = run_grid_search(sample_data)
        
        MockTuner.assert_called_once()
        mock_instance.run_grid_search.assert_called_once_with(sample_data)
        assert result is not None


def test_run_grid_search_with_db_save(mock_db, mock_redis, sample_data):
    with patch("src.ml.hyperparam_tuner.HyperparamTuner") as MockTuner:
        mock_instance = MagicMock()
        mock_instance.run_grid_search = MagicMock()
        mock_instance.best_params = {"lr": 0.1, "reg": 0.01}
        mock_instance.best_score = 0.92
        MockTuner.return_value = mock_instance
        
        run_grid_search(sample_data)
        
        mock_db.save_calibration_result.assert_called_once_with(
            sample_data["user_id"],
            {"lr": 0.1, "reg": 0.01},
            0.92
        )
        mock_redis.invalidate_user_cache.assert_called_once_with(sample_data["user_id"])


def test_hyperparam_tuner_run_grid_search_empty_params():
    tuner = HyperparamTuner()
    data = {
        "user_id": "user_456",
        "X": [[1.0, 2.0]],
        "y": [0],
        "calibration_params": {}
    }
    
    with pytest.raises(ValueError):
        tuner.run_grid_search(data)