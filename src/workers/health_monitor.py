"""Health monitoring module for on-device resource and battery-aware scheduling."""
try:
    import psutil
except ImportError:
    psutil = None
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.config import settings