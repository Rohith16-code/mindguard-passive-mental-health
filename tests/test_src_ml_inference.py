import pytest
from unittest.mock import MagicMock, patch
from src.ml.inference import load_model, run_inference, compute_wellness_index

@pytest.fixture
def mock_tflite_interpreter():
    with patch('src.ml.inference.tf.lite.Interpreter') as mock_interp:
        mock_instance = MagicMock()
        mock_interp.return_value = mock_instance
        mock_instance.allocate_tensors = MagicMock()
        mock_instance.set_tensor = MagicMock()
        mock_instance.get_tensor = MagicMock()
        mock_instance.get_tensor.return_value = 0.75
        yield mock_instance

@pytest.fixture
def mock_redis_client():
    with patch('src.ml.inference.redis.Redis') as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_db_session():
    with patch('src.ml.inference.SessionLocal') as mock_session:
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        yield mock_db

def test_load_model_success(mock_tflite_interpreter):
    model_path = "model.tflite"
    interpreter = load_model(model_path)
    assert interpreter is not None
    mock_tflite_interpreter.allocate_tensors.assert_called_once()

def test_load_model_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent.tflite")

def test_run_inference_success(mock_tflite_interpreter):
    input_data = [1.0, 2.0, 3.0]
    model_path = "model.tflite"
    interpreter = load_model(model_path)
    result = run_inference(interpreter, input_data)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_run_inference_wrong_input_shape(mock_tflite_interpreter):
    interpreter = mock_tflite_interpreter
    interpreter.get_tensor.side_effect = lambda tensor_id, *args: (
        [1.0, 2.0] if tensor_id == interpreter.get_input_details.return_value[0]['index'] else 0.75
    )
    interpreter.get_input_details.return_value = [{'index': 0, 'shape': [1, 2]}]
    with pytest.raises(ValueError):
        run_inference(interpreter, [1.0, 2.0, 3.0])

def test_compute_wellness_index_success(mock_redis_client, mock_db_session):
    inference_result = 0.6
    user_id = "user123"
    wellness = compute_wellness_index(user_id, inference_result)
    assert isinstance(wellness, float)
    assert wellness > 0
    mock_redis_client.set.assert_called_once()
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

def test_compute_wellness_index_redis_failure(mock_redis_client, mock_db_session):
    mock_redis_client.set.side_effect = Exception("Redis unavailable")
    inference_result = 0.8
    user_id = "user456"
    wellness = compute_wellness_index(user_id, inference_result)
    assert isinstance(wellness, float)
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

def test_compute_wellness_index_db_failure(mock_redis_client, mock_db_session):
    mock_db_session.commit.side_effect = Exception("DB commit failed")
    inference_result = 0.9
    user_id = "user789"
    wellness = compute_wellness_index(user_id, inference_result)
    assert isinstance(wellness, float)
    mock_redis_client.set.assert_called_once()