"""Application entry point."""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from src.config import DEBUG, ENV, ML_MODEL_PATH
from src.db import init_db, close_db
from src.ml import load_model, unload_model
from src.routes import api_router

# Define API_HOST and API_PORT if not exported from config
API_HOST = "0.0.0.0"
API_PORT = 8000

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(f"Starting {ENV} environment")
    try:
        await init_db()
        await load_model(ML_MODEL_PATH)
        logger.info("Database and ML model initialized successfully")
        yield
    finally:
        await close_db()
        await unload_model()
        logger.info("Application shutdown complete")


app = FastAPI(
    title="Passive Mental Health Crisis Detection API",
    description="Detect mental health crises from daily digital behavior",
    version="1.0.0",
    debug=DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "environment": ENV, "model_version": "1.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler."""
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvloop

    uvloop.install()
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )