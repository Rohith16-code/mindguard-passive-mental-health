"""Central configuration management."""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()

# ── Environment ───────────────────────────────────────────────
ENV = os.getenv("APP_ENV", "development")
DEBUG = ENV == "development"

# ── Database ──────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")

# ── Redis ─────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── ML Model ──────────────────────────────────────────────────
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", str(BASE_DIR / "models" / "model.pt"))
ML_VERSION = os.getenv("ML_VERSION", "1.0.0")

# ── API ───────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
