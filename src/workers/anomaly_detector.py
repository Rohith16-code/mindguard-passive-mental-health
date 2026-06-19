"""Anomaly detection worker for real-time signal deviation monitoring."""
import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config import settings
from src.db.models import SignalType, WellnessIndex