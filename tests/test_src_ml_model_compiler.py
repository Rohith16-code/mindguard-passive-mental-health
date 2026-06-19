import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
import tempfile
import onnx
import tensorflow as tf
from tensorflow.lite.python import interpreter as tflite_interpreter
from src.ml.model_compiler import (
    load_onnx_model,
    convert_onnx_to_tflite,
    validate_tflite_model,
    save_tflite_model,
    compile_model_pipeline
)


@pytest.fixture
def mock_onnx_model():
    """Create a minimal mock ONNX model"""
    model = MagicMock(spec=onnx.ModelProto)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1000
    return model


@pytest.fixture
def mock_tflite_model():
    """Create a minimal mock TFLite model"""
    model = MagicMock(spec=tf.lite.OptimizedModel)
    model.get_tensor_details.return_value = [
        {"name": "input", "shape": [1, 3, 224, 224], "dtype": tf.float32},
        {"name": "output", "shape": [1, 1000], "dtype": tf.float32}
    ]
    return model


@pytest.fixture
def mock_tflite_interpreter():
    """Mock TFLite interpreter"""
    interpreter = MagicMock(spec=tflite_interpreter.Interpreter)
    interpreter.allocate_tensors.return_value = None
    interpreter.get_input_details.return_value = [{"index": 0, "shape": [1, 3, 224, 224], "dtype": tf.float32}]
    interpreter.get_output_details.return_value = [{"index": 1, "shape": [1, 1000], "dtype": tf.float32}]
    return interpreter


def test_load_onnx_model_success(mock_onnx_model):
    with patch('onnx.load', return_value=mock_onnx_model):
        model_path = "/fake/path/model.onnx"
        model = load_onnx_model(model_path)
        assert model is not None
        assert isinstance(model, MagicMock)


def test_load_onnx_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_onnx_model("/nonexistent/path/model.onnx")


def test_convert_onnx_to_tflite_success(mock_onnx_model, mock_tflite_model):
    with patch('tf.lite.TFLiteConverter.from_frozen_graph') as mock_converter:
        mock_instance = MagicMock()
        mock_instance.convert.return_value = b'\x00\x01\x02\x03'
        mock_converter.return_value = mock_instance
        tflite_bytes = convert_onnx_to_tflite(mock_onnx_model)
        assert isinstance(tflite_bytes, bytes)
        assert len(tflite_bytes) > 0


def test_convert_onnx_to_tflite_conversion_error(mock_onnx_model):
    with patch('tf.lite.TFLiteConverter.from_frozen_graph') as mock_converter:
        mock_converter.side_effect = Exception("Conversion failed")
        with pytest.raises(Exception, match="Conversion failed"):
            convert_onnx_to_tflite(mock_onnx_model)


def test_validate_tflite_model_success(mock_tflite_interpreter):
    with patch('tflite_interpreter.Interpreter', return_value=mock_tflite_interpreter):
        tflite_bytes = b'\x00\x01\x02\x03'
        is_valid = validate_tflite_model(tflite_bytes)
        assert is_valid is True


def test_validate_tflite_model_invalid_bytes():
    with patch('tflite_interpreter.Interpreter') as mock_interpreter:
        mock_interpreter.side_effect = ValueError("Invalid TFLite model")
        is_valid = validate_tflite_model(b"invalid_bytes")
        assert is_valid is False


def test_save_tflite_model_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tflite")
        tflite_bytes = b'\x00\x01\x02\x03'
        result_path = save_tflite_model(tflite_bytes, model_path)
        assert result_path == model_path
        assert os.path.exists(model_path)
        with open(model_path, 'rb') as f:
            assert f.read() == tflite_bytes


def test_save_tflite_model_directory_not_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "subdir", "model.tflite")
        tflite_bytes = b'\x00\x01\x02\x03'
        result_path = save_tflite_model(tflite_bytes, model_path)
        assert os.path.exists(os.path.dirname(model_path))
        assert result_path == model_path


def test_compile_model_pipeline_success(mock_onnx_model, mock_tflite_interpreter):
    with patch('onnx.load', return_value=mock_onnx_model), \
         patch('tf.lite.TFLiteConverter.from_frozen_graph') as mock_converter, \
         patch('tflite_interpreter.Interpreter', return_value=mock_tflite_interpreter), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('builtins.open', mock_open()) as m:
        mock_instance = MagicMock()
        mock_instance.convert.return_value = b'\x00\x01\x02\x03'
        mock_converter.return_value = mock_instance

        onnx_path = "/fake/model.onnx"
        output_path = "/fake/model.tflite"

        result = compile_model_pipeline(onnx_path, output_path)

        assert result == output_path
        assert os.path.exists(output_path) is True


def test_compile_model_pipeline_failure_onnx_load():
    with patch('onnx.load', side_effect=FileNotFoundError("ONNX model not found")):
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            compile_model_pipeline("/nonexistent.onnx", "/out.tflite")


def test_compile_model_pipeline_failure_conversion():
    with patch('onnx.load', return_value=MagicMock(spec=onnx.ModelProto)), \
         patch('tf.lite.TFLiteConverter.from_frozen_graph') as mock_converter:
        mock_converter.side_effect = Exception("Conversion error")
        with pytest.raises(Exception, match="Conversion error"):
            compile_model_pipeline("/fake.onnx", "/out.tflite")