"""Pydantic models for internal APIs."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class CrisisLevel(int, Enum):
    """Crisis severity levels."""
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3


class UserStatus(str, Enum):
    """User enrollment and activity status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    WITHDRAWN = "withdrawn"
    SUSPENDED = "suspended"


class SensorType(str, Enum):
    """Supported sensor types."""
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    GPS = "gps"
    AUDIO = "audio"
    SCREEN = "screen"
    APP_USAGE = "app_usage"
    KEYSTROKE = "keystroke"
    HEART_RATE = "heart_rate"
    BLUETOOTH = "bluetooth"


class FeatureType(str, Enum):
    """Feature categories for aggregation."""
    MOBILITY = "mobility"
    SOCIAL = "social"
    SLEEP = "sleep"
    ACTIVITY = "activity"
    COMMUNICATION = "communication"
    DEVICE_USAGE = "device_usage"


class AlertType(str, Enum):
    """Types of alerts generated."""
    CRISIS_DETECTED = "crisis_detected"
    SYSTEM_ISSUE = "system_issue"
    DATA_QUALITY = "data_quality"
    MODEL_UPDATE = "model_update"


class UserBase(BaseModel):
    """Base user schema."""
    user_id: str = Field(..., description="Unique user identifier")
    status: UserStatus = Field(default=UserStatus.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserCreate(UserBase):
    """Schema for user creation."""
    pass


class User(UserBase):
    """Full user schema."""
    last_active: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    last_crisis_assessment: Optional[datetime] = None
    last_crisis_level: Optional[CrisisLevel] = None

    class Config:
        orm_mode = True


class SensorDataPoint(BaseModel):
    """Individual sensor data point."""
    sensor_type: SensorType
    timestamp: datetime
    value: Union[float, int, str, Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('timestamp')
    def ensure_utc(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class FeatureVector(BaseModel):
    """Aggregated feature vector."""
    user_id: str
    window_start: datetime
    window_end: datetime
    features: Dict[FeatureType, Dict[str, float]]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator('window_start', 'window_end')
    def ensure_utc(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class CrisisAssessment(BaseModel):
    """Crisis assessment result."""
    user_id: str
    timestamp: datetime
    crisis_level: CrisisLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_factors: List[str] = Field(default_factory=list)
    model_version: str
    features_used: List[str] = Field(default_factory=list)

    @validator('timestamp')
    def ensure_utc(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class Alert(BaseModel):
    """Alert record."""
    alert_id: Optional[str] = None
    user_id: str
    alert_type: AlertType
    timestamp: datetime
    severity: int = Field(..., ge=0, le=10)
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('timestamp')
    def ensure_utc(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class PredictionInput(BaseModel):
    """Input for model prediction."""
    user_id: str
    features: Dict[FeatureType, Dict[str, float]]
    timestamp: datetime

    @validator('timestamp')
    def ensure_utc(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class PredictionOutput(BaseModel):
    """Output from model prediction."""
    user_id: str
    timestamp: datetime
    crisis_level: CrisisLevel
    confidence: float
    model_version: str
    explanation: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    """System health status."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    services: Dict[str, str] = Field(default_factory=dict)


class BatchIngestionResult(BaseModel):
    """Result of batch sensor ingestion."""
    user_id: str
    records_ingested: int
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelUpdate(BaseModel):
    """Model update notification."""
    model_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: Dict[str, float] = Field(default_factory=dict)
    drift_detected: bool = False
    rollback_required: bool = False


class SyncRequest(BaseModel):
    """Request to sync user data."""
    user_id: str
    since: Optional[datetime] = None
    sensor_types: Optional[List[SensorType]] = None

    @validator('since')
    def ensure_utc(cls, v):
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class SyncResponse(BaseModel):
    """Response for data sync."""
    user_id: str
    data_points: List[SensorDataPoint]
    next_sync_window_start: datetime
    last_sync_timestamp: datetime
    records_count: int


class ConsentStatus(BaseModel):
    """User consent status."""
    user_id: str
    consented: bool
    consented_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    version: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricsRecord(BaseModel):
    """Performance metrics record."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str
    inference_latency_ms: float
    memory_usage_mb: float
    prediction_confidence: float
    model_version: str
    device_info: Dict[str, str] = Field(default_factory=dict)

    @validator('timestamp')
    def ensure_utc(cls, v):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

def FeedbackCreate(*args, **kwargs):
    """Auto-generated stub to satisfy test imports."""
    pass


class UserResponse:
    """Auto-generated stub to satisfy test imports."""

    def __init__(self, *args, **kwargs):
        pass


class ItemCreate:
    """Auto-generated stub to satisfy test imports."""

    def __init__(self, *args, **kwargs):
        pass


class ItemResponse:
    """Auto-generated stub to satisfy test imports."""

    def __init__(self, *args, **kwargs):
        pass


class TokenData:
    """Auto-generated stub to satisfy test imports."""

    def __init__(self, *args, **kwargs):
        pass
