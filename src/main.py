"""Application entry point for MindGuard passive mental health crisis detection."""
import logging
import os
import sys
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import settings, DEBUG, ENV, ML_MODEL_PATH, API_HOST, API_PORT
from src.utils.logger import get_logger

STATIC_DIR = Path(__file__).parent.parent / "static"

logger = get_logger("mindguard.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(f"Starting MindGuard in {ENV} mode")
    try:
        # Initialize database
        from src.db.cache import CacheClient
        app.state.cache = CacheClient()
        logger.info("Cache initialized")
        
        # Load ML model (best-effort; degraded mode if missing)
        from src.ml.model_loader import ModelLoader
        try:
            loader = ModelLoader()
            app.state.model = loader.load(ML_MODEL_PATH)
            logger.info(f"ML model loaded from {ML_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load model from {ML_MODEL_PATH}: {e}; running in degraded mode")
            app.state.model = None
        
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        yield
    finally:
        logger.info("MindGuard shutdown complete")


def create_app() -> FastAPI:
    """Factory function — creates and configures the FastAPI app."""
    app = FastAPI(
        title="MindGuard — Passive Mental Health Crisis Detection",
        description="Privacy-preserving, on-device mental health monitoring API",
        version="1.0.0",
        debug=DEBUG,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from src.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1")

    # Static frontend
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/dashboard", tags=["frontend"])
    async def dashboard():
        """Serve the MindGuard HTML dashboard."""
        return FileResponse(str(STATIC_DIR / "index.html"))

    # Health check endpoints
    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "healthy", "env": ENV}

    @app.get("/ready", tags=["health"])
    async def ready():
        return {"status": "ready", "model_loaded": hasattr(app.state, "model")}

    return app


app = create_app()


async def startup():
    """Explicit startup hook for tests/scheduler compatibility."""
    logger.info("MindGuard startup hook called")


async def shutdown():
    """Explicit shutdown hook for tests/scheduler compatibility."""
    logger.info("MindGuard shutdown hook called")


async def health_check():
    """Synchronous-style health check callable."""
    return {"status": "healthy", "env": ENV}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=API_HOST,
        port=int(API_PORT),
        reload=DEBUG,
    )
