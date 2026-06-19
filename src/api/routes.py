"""Routes module for health, status, and model update endpoints."""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import logging
import os
import json
import hashlib
import shutil
import time

from src.utils.logger import get_logger
from src.utils.crypt import encrypt_file, decrypt_file, generate_key
from src.ml.model_arch import build_tflite_model
from src.workers.scheduler import schedule_model_refresh
from src.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["system"])


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_version: Optional[str] = None
    disk_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None


class ModelUpdateRequest(BaseModel):
    model_path: str
    model_hash: str


class ModelUpdateResponse(BaseModel):
    success: bool
    message: str
    model_version: Optional[str] = None


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        model_path = Path(settings.MODEL_PATH)
        model_loaded = model_path.exists() and model_path.is_file()
        model_version = None
        if model_loaded:
            try:
                with open(model_path, "rb") as f:
                    model_version = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                model_version = "unknown"

        disk_usage = None
        memory_usage = None
        try:
            stat = os.statvfs("/")
            disk_usage = round((1 - stat.f_bavail / stat.f_blocks) * 100, 2)
        except Exception:
            pass

        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
                total = int(lines[0].split()[1])
                available = int(lines[2].split()[1])
                memory_usage = round((1 - available / total) * 100, 2)
        except Exception:
            pass

        return HealthCheck(
            status="healthy",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_loaded=model_loaded,
            model_version=model_version,
            disk_usage_percent=disk_usage,
            memory_usage_percent=memory_usage,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/status", response_model=Dict[str, Any])
async def system_status():
    """System status endpoint."""
    try:
        status = {
            "system": "running",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": {
                "path": settings.MODEL_PATH,
                "loaded": Path(settings.MODEL_PATH).exists(),
                "version": None,
            },
            "data": {
                "db_path": str(settings.DB_PATH),
                "db_exists": Path(settings.DB_PATH).exists(),
            },
            "workers": {
                "scheduler": "scheduled",
                "anomaly_detector": "active",
                "feedback_processor": "active",
                "health_monitor": "active",
            },
            "resources": {
                "disk_usage_percent": None,
                "memory_usage_percent": None,
            },
        }

        model_path = Path(settings.MODEL_PATH)
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    status["model"]["version"] = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass

        try:
            stat = os.statvfs("/")
            status["resources"]["disk_usage_percent"] = round(
                (1 - stat.f_bavail / stat.f_blocks) * 100, 2
            )
        except Exception:
            pass

        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
                total = int(lines[0].split()[1])
                available = int(lines[2].split()[1])
                status["resources"]["memory_usage_percent"] = round(
                    (1 - available / total) * 100, 2
                )
        except Exception:
            pass

        return status
    except Exception as e:
        logger.error(f"Status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Status endpoint failed")


@router.post("/model/update", response_model=ModelUpdateResponse)
async def update_model(
    request: ModelUpdateRequest,
    background_tasks: BackgroundTasks,
):
    """Update the ML model."""
    try:
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=400, detail="Model file not found")

        with open(model_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != request.model_hash:
            raise HTTPException(status_code=400, detail="Model hash mismatch")

        encrypted_path = Path(settings.MODEL_PATH)
        temp_path = Path(settings.MODEL_PATH + ".tmp")

        try:
            shutil.copy2(model_path, temp_path)

            if settings.ENCRYPT_MODELS:
                key = generate_key()
                encrypt_file(temp_path, key)
                temp_path.rename(encrypted_path)
            else:
                temp_path.rename(encrypted_path)

            model_version = hashlib.sha256(
                open(encrypted_path, "rb").read()
            ).hexdigest()[:16]

            background_tasks.add_task(schedule_model_refresh)

            logger.info(f"Model updated successfully: {model_version}")
            return ModelUpdateResponse(
                success=True,
                message="Model updated successfully",
                model_version=model_version,
            )
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Model update failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model update failed: {e}")
        raise HTTPException(status_code=500, detail="Model update failed")


@router.get("/model/version", response_model=Dict[str, str])
async def get_model_version():
    """Get current model version."""
    try:
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        with open(model_path, "rb") as f:
            version = hashlib.sha256(f.read()).hexdigest()[:16]

        return {"version": version}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model version lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Model version lookup failed")