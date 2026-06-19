"""Middleware module for on-device rate limiting and anomaly detection."""
from typing import Callable, Dict, List, Optional
from time import time
from collections import defaultdict
from functools import wraps
import asyncio

from fastapi import Request, Response, Depends, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings


class TokenBucket:
    """Token bucket rate limiter for per-client throttling."""

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time()

    def consume(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        now = time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class AnomalyDetector:
    """Simple on-device anomaly detector using sliding window statistics."""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._values: List[float] = []
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, value: float) -> Optional[float]:
        """Update detector with new value; returns z-score if anomaly detected."""
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value

        if len(self._values) > self.window_size:
            old = self._values.pop(0)
            self._sum -= old
            self._sum_sq -= old * old

        if len(self._values) < 2:
            return None

        n = len(self._values)
        mean = self._sum / n
        variance = (self._sum_sq / n) - (mean * mean)
        std = variance ** 0.5 if variance > 0 else 1e-6

        z_score = (value - mean) / std
        return z_score if abs(z_score) > self.z_threshold else None


class RateLimiter:
    """Per-client rate limiter with anomaly detection."""

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._anomaly_detectors: Dict[str, AnomalyDetector] = {}
        self._request_counts: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _get_client_id(self, request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _get_bucket(self, client_id: str) -> TokenBucket:
        if client_id not in self._buckets:
            self._buckets[client_id] = TokenBucket(
                capacity=settings.RATE_LIMIT_CAPACITY,
                refill_rate=settings.RATE_LIMIT_REFILL_RATE
            )
        return self._buckets[client_id]

    def _get_anomaly_detector(self, client_id: str) -> AnomalyDetector:
        if client_id not in self._anomaly_detectors:
            self._anomaly_detectors[client_id] = AnomalyDetector(
                window_size=settings.ANOMALY_WINDOW_SIZE,
                z_threshold=settings.ANOMALY_Z_THRESHOLD
            )
        return self._anomaly_detectors[client_id]

    async def check(self, request: Request) -> tuple[bool, Optional[float]]:
        """Check if request should be allowed; returns (allowed, z_score)."""
        client_id = self._get_client_id(request)
        async with self._lock:
            bucket = self._get_bucket(client_id)
            detector = self._get_anomaly_detector(client_id)

            now = time()
            self._request_counts[client_id].append(now)

            # Clean old entries
            cutoff = now - settings.ANOMALY_WINDOW_SIZE
            self._request_counts[client_id] = [
                t for t in self._request_counts[client_id] if t > cutoff
            ]

            # Calculate inter-arrival rate
            counts = self._request_counts[client_id]
            if len(counts) > 1:
                inter_arrival = (counts[-1] - counts[-2]) if len(counts) >= 2 else 0.0
                z_score = detector.update(inter_arrival)
            else:
                z_score = None

            allowed = bucket.consume()
            return allowed, z_score


rate_limiter = RateLimiter()


async def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency for rate limiting."""
    allowed, z_score = await rate_limiter.check(request)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if z_score is not None and z_score > 0:
        request.state.anomaly_detected = True
        request.state.z_score = z_score


class OnDeviceMiddleware(BaseHTTPMiddleware):
    """Middleware for on-device rate limiting and anomaly detection."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        allowed, z_score = await rate_limiter.check(request)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(int(settings.RATE_LIMIT_REFILL_RATE))}
            )

        if z_score is not None and z_score > 0:
            request.state.anomaly_detected = True
            request.state.z_score = z_score

        response = await call_next(request)
        return response


def rate_limit(max_requests: int = 10, window_seconds: int = 60) -> Callable:
    """Decorator for rate limiting specific endpoints."""
    def decorator(func: Callable) -> Callable:
        bucket = TokenBucket(
            capacity=max_requests,
            refill_rate=max_requests / window_seconds
        )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not bucket.consume():
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            return await func(*args, **kwargs)

        return wrapper
    return decorator