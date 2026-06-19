"""ONNX to TensorFlow Lite conversion pipeline for mental health crisis detection models."""
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import onnx
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import interpreter as tflite_interpreter
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.convert import to_lite_model


def load_onnx_model(model_path: str) -> onnx.ModelProto:
    """Load an ONNX model from disk.
    
    Args:
        model_path: Path to the ONNX model file.
        
    Returns:
        Loaded ONNX model.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model is invalid or cannot be loaded.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load or validate ONNX model: {e}")


def convert_onnx_to_tflite(
    onnx_model: onnx.ModelProto,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    quantize: bool = True
) -> tf.lite.TFLiteConverter:
    """Convert an ONNX model to a TensorFlow Lite converter.
    
    Args:
        onnx_model: Loaded ONNX model.
        input_shapes: Optional dict mapping input names to shapes.
        quantize: Whether to apply post-training quantization.
        
    Returns:
        Configured TensorFlow Lite converter.
        
    Raises:
        RuntimeError: If conversion setup fails.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            onnx.save(onnx_model, onnx_path)
            
            converter = tf.lite.TFLiteConverter.from_saved_model(onnx_path)
            
            if input_shapes:
                converter.allow_custom_ops = True
                for name, shape in input_shapes.items():
                    converter._set_input_shapes({name: shape})
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
            
            return converter
    except Exception as e:
        raise RuntimeError(f"Failed to configure TFLite conversion: {e}")


def validate_tflite_model(tflite_model: bytes) -> bool:
    """Validate a TensorFlow Lite model.
    
    Args:
        tflite_model: Serialized TFLite model bytes.
        
    Returns:
        True if model is valid.
        
    Raises:
        ValueError: If model validation fails.
    """
    try:
        interpreter = tflite_interpreter.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        return True
    except Exception as e:
        raise ValueError(f"TFLite model validation failed: {e}")


def save_tflite_model(tflite_model: bytes, output_path: str) -> None:
    """Save a TensorFlow Lite model to disk.
    
    Args:
        tflite_model: Serialized TFLite model bytes.
        output_path: Path to save the TFLite model.
        
    Raises:
        IOError: If file cannot be written.
    """
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    except Exception as e:
        raise IOError(f"Failed to save TFLite model to {output_path}: {e}")


def compile_model_pipeline(
    onnx_path: str,
    output_path: str,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    quantize: bool = True,
    validate: bool = True
) -> bytes:
    """Execute the full ONNX to TFLite conversion pipeline.
    
    Args:
        onnx_path: Path to input ONNX model.
        output_path: Path to save output TFLite model.
        input_shapes: Optional dict mapping input names to shapes.
        quantize: Whether to apply post-training quantization.
        validate: Whether to validate the resulting TFLite model.
        
    Returns:
        Serialized TFLite model bytes.
        
    Raises:
        RuntimeError: If any pipeline step fails.
    """
    try:
        onnx_model = load_onnx_model(onnx_path)
        converter = convert_onnx_to_tflite(onnx_model, input_shapes, quantize)
        tflite_model = converter.convert()
        
        if validate:
            validate_tflite_model(tflite_model)
        
        save_tflite_model(tflite_model, output_path)
        return tflite_model
    except Exception as e:
        raise RuntimeError(f"Model compilation pipeline failed: {e}")