"""Model registry module for versioned model storage and selection."""
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import json
import hashlib
from datetime import datetime

from src.db import db
from src.cache import redis_client
from src.ml.exceptions import ModelNotFoundError, ModelVersionMismatchError
from src.ml.models import ModelRegistry

logger = logging.getLogger(__name__)

FALLBACK_MODEL_VERSION = "fallback-v1"
MODEL_CACHE_TTL = 3600  # 1 hour
MODEL_REGISTRY_DIR = Path(__file__).parent / "model_registry"


def ensure_registry_dir():
    """Ensure model registry directory exists."""
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _generate_model_checksum(model_path: Path) -> str:
    """Generate SHA256 checksum for model file."""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def register_model(
    model_path: str,
    version: str,
    metadata: Optional[Dict[str, Any]] = None,
    checksum: Optional[str] = None,
) -> ModelRegistry:
    """Register a model in the registry."""
    ensure_registry_dir()
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise ModelNotFoundError(f"Model file not found: {model_path}")
    
    if checksum is None:
        checksum = _generate_model_checksum(model_file)
    
    if metadata is None:
        metadata = {}
    
    registry_entry = ModelRegistry(
        version=version,
        path=str(model_file),
        checksum=checksum,
        metadata=json.dumps(metadata),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    try:
        db.add_model(registry_entry)
        logger.info(f"Registered model version {version}")
        return registry_entry
    except Exception as e:
        logger.error(f"Failed to register model {version}: {e}")
        raise


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

        fallback_path = MODEL_REGISTRY_DIR / "fallback_model.tflite"
        if not fallback_path.exists():
            fallback_path = Path(__file__).parent / "fallback_model.tflite"
        if not fallback_path.exists():
            raise ModelNotFoundError("Fallback model file not found")

        import tensorflow as tf
        model = tf.lite.TFLiteConverter.from_file(str(fallback_path))
        model = model.convert()
        
        redis_client.setex(cache_key, MODEL_CACHE_TTL, model)
        logger.info("Loaded and cached fallback model")
        return model
    except Exception as e:
        logger.error(f"Failed to load fallback model: {e}")
        raise ModelNotFoundError(f"Fallback model unavailable: {e}")


def list_registered_models() -> List[ModelRegistry]:
    """List all registered models."""
    try:
        return db.get_all_models()
    except Exception as e:
        logger.error(f"Failed to list registered models: {e}")
        return []


def delete_model_version(version: str) -> bool:
    """Delete a model version from the registry."""
    try:
        model = db.get_model(version)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {version}")
        
        db.delete_model(version)
        cache_key = f"model:{version}"
        redis_client.delete(cache_key)
        logger.info(f"Deleted model version {version}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete model {version}: {e}")
        return False


def get_model_by_version(version: str) -> ModelRegistry:
    """Get model registry entry by version."""
    try:
        model = db.get_model(version)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {version}")
        return model
    except Exception as e:
        logger.error(f"Failed to get model {version}: {e}")
        raise ModelNotFoundError(f"Model not found: {version}")