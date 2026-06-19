"""Application entry point for mindguard-ondevice."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Startup: initialize DB, load ML model, etc.
    yield
    # Shutdown: cleanup connections

def create_app() -> FastAPI:
    """Factory function — creates and configures the FastAPI app."""
    app = FastAPI(
        title="mindguard-ondevice",
        description="Passive, privacy-preserving mental health crisis detection using on-device smartphone behavioral sig",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoints
    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "healthy"}

    @app.get("/ready", tags=["health"])
    async def ready():
        return {"status": "ready"}

    return app

# Global app instance for imports
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
