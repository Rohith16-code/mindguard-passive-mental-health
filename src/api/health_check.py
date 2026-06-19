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
            memory_usage_percent=memory_usage
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_loaded=False,
            model_version=None,
            disk_usage_percent=None,
            memory_usage_percent=None
        )


@router.get("/ready", response_model=HealthCheck)
async def readiness_check():
    """Readiness check endpoint."""
    try:
        model_path = Path(settings.MODEL_PATH)
        model_loaded = model_path.exists() and model_path.is_file()
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return HealthCheck(
            status="ready",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_loaded=True,
            model_version=hashlib.sha256(model_path.read_bytes()).hexdigest()[:16],
            disk_usage_percent=None,
            memory_usage_percent=None
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Not ready")


@router.post("/model/update", response_model=ModelUpdateResponse)
async def update_model(
    request: ModelUpdateRequest,
    background_tasks: BackgroundTasks
):
    """Update model endpoint."""
    try:
        model_path = Path(settings.MODEL_PATH)
        new_model_path = Path(request.model_path)

        if not new_model_path.exists():
            raise HTTPException(status_code=400, detail="Model file not found")

        with open(new_model_path, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()
        if computed_hash != request.model_hash:
            raise HTTPException(status_code=400, detail="Model hash mismatch")

        backup_path = model_path.with_suffix(".bak")
        if model_path.exists():
            shutil.copy2(model_path, backup_path)

        shutil.copy2(new_model_path, model_path)

        background_tasks.add_task(schedule_model_refresh)

        return ModelUpdateResponse(
            success=True,
            message="Model updated successfully",
            model_version=computed_hash[:16]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model update failed: {e}")
        raise HTTPException(status_code=500, detail="Model update failed")