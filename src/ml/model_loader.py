"""Model loader module for dynamic model versioning and fallback."""
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from src.db import db
from src.cache import redis_client
from src.ml.exceptions import ModelNotFoundError, ModelVersionMismatchError
from src.ml.models import ModelRegistry

logger = logging.getLogger(__name__)

FALLBACK_MODEL_VERSION = "fallback-v1"
MODEL_CACHE_TTL = 3600  # 1 hour


def get_latest_model_version() -> str:
    """Get the latest registered model version from the database."""
    try:
        latest = db.get_latest_model()
        if latest is None:
            raise ModelNotFoundError("No models registered in the database")
        return latest.version
    except Exception as e:
        logger.warning(f"Failed to get latest model version: {e}")
        return FALLBACK_MODEL_VERSION


def get_model_version(version: str) -> str:
    """Validate and return the requested model version."""
    try:
        model = db.get_model(version)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {version}")
        if model.version != version:
            raise ModelVersionMismatchError(
                f"Requested version {version} but got {model.version}"
            )
        return model.version
    except (ModelNotFoundError, ModelVersionMismatchError):
        raise
    except Exception as e:
        logger.warning(f"Error validating model version {version}: {e}")
        raise ModelNotFoundError(f"Model not found: {version}")


def get_fallback_model() -> Any:
    """Load and return the fallback model."""
    try:
        cache_key = f"model:fallback:{FALLBACK_MODEL_VERSION}"
        cached_model = redis_client.get(cache_key)
        if cached_model:
            logger.info("Loaded fallback model from cache")
            return cached_model

        fallback_path = Path(__file__).parent / "fallback_model.tflite"
        if not fallback_path.exists():
            fallback_path = Path(__file__).parent / "fallback_model.pt"
        if not fallback_path.exists():
            raise FileNotFoundError("No fallback model file found")

        fallback_model = load_from_path(str(fallback_path))
        fallback_model.version = FALLBACK_MODEL_VERSION
        redis_client.set(cache_key, fallback_model, ex=MODEL_CACHE_TTL)
        logger.info("Loaded fallback model from disk")
        return fallback_model
    except Exception as e:
        logger.error(f"Failed to load fallback model: {e}")
        raise ModelNotFoundError("Fallback model unavailable")


def load_model(version: str) -> Any:
    """Load model by version with fallback on error."""
    try:
        model = db.get_model(version)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {version}")

        cache_key = f"model:{version}"
        cached_model = redis_client.get(cache_key)
        if cached_model:
            logger.info(f"Loaded model {version} from cache")
            return cached_model

        loaded_model = load_from_path(model.path)
        loaded_model.version = model.version
        redis_client.set(cache_key, loaded_model, ex=MODEL_CACHE_TTL)
        logger.info(f"Loaded model {version} from {model.path}")
        return loaded_model
    except ModelNotFoundError:
        raise
    except Exception as e:
        logger.warning(f"Failed to load model {version}: {e}")
        fallback_model = get_fallback_model()
        logger.info("Falling back to fallback model")
        return fallback_model


def load_from_path(path: str) -> Any:
    """Load model from file path using appropriate loader."""
    path = Path(path)
    if path.suffix == ".tflite":
        try:
            import tensorflow as tf
            return tf.lite.Interpreter(model_path=str(path))
        except ImportError:
            raise RuntimeError("TensorFlow not installed but .tflite model requested")
    elif path.suffix in (".pt", ".pth"):
        try:
            import torch
            return torch.load(path, map_location="cpu", weights_only=True)
        except ImportError:
            raise RuntimeError("PyTorch not installed but .pt model requested")
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")