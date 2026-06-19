"""Database models for the MindGuard on-device mental health crisis detection system."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from aerich import Model
from tortoise import fields
from tortoise.models import Model as TortoiseModel


class RiskLevel(str, Enum):
    """Risk level classification for mental health crisis detection."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SensorType(str, Enum):
    """Supported sensor data types."""
    KEYSTROKE = "keystroke"
    MOUSE = "mouse"
    TOUCH = "touch"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    GPS = "gps"
    AUDIO = "audio"
    SCREEN = "screen"
    APP_USAGE = "app_usage"
    NOTIFICATION = "notification"


class BaseModelMixin(TortoiseModel):
    """Base model with common fields."""
    id = fields.IntField(pk=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        abstract = True


class User(BaseModelMixin):
    """User account model."""
    user_id = fields.CharField(max_length=64, unique=True)
    device_id = fields.CharField(max_length=64)
    enrollment_date = fields.DatetimeField(default=datetime.utcnow)
    active = fields.BooleanField(default=True)
    metadata = fields.JSONField(default=dict)

    class Meta:
        table = "users"


class SensorData(BaseModelMixin):
    """Raw sensor data record."""
    user = fields.ForeignKeyField("models.User", related_name="sensor_data")
    sensor_type = fields.CharField(max_length=32, enum=SensorType)
    timestamp = fields.DatetimeField()
    data = fields.JSONField()
    metadata = fields.JSONField(default=dict)

    class Meta:
        table = "sensor_data"
        indexes = (
            ("user_id", "sensor_type", "timestamp"),
            ("timestamp",),
        )


class FeatureRecord(BaseModelMixin):
    """Engineered feature record."""
    user = fields.ForeignKeyField("models.User", related_name="feature_records")
    timestamp = fields.DatetimeField()
    feature_type = fields.CharField(max_length=64)
    values = fields.JSONField()
    metadata = fields.JSONField(default=dict)

    class Meta:
        table = "feature_records"
        indexes = (
            ("user_id", "feature_type", "timestamp"),
            ("timestamp",),
        )


class InferenceResult(BaseModelMixin):
    """Model inference result record."""
    user = fields.ForeignKeyField("models.User", related_name="inference_results")
    timestamp = fields.DatetimeField()
    risk_level = fields.CharField(max_length=16, enum=RiskLevel)
    wellness_index = fields.FloatField()
    model_version = fields.CharField(max_length=32)
    features_used = fields.JSONField()
    metadata = fields.JSONField(default=dict)

    class Meta:
        table = "inference_results"
        indexes = (
            ("user_id", "timestamp"),
            ("risk_level",),
        )


class Alert(BaseModelMixin):
    """Crisis alert record."""
    user = fields.ForeignKeyField("models.User", related_name="alerts")
    timestamp = fields.DatetimeField()
    risk_level = fields.CharField(max_length=16, enum=RiskLevel)
    confidence = fields.FloatField()
    triggered_features = fields.JSONField()
    action_taken = fields.CharField(max_length=64, default="none")
    resolved = fields.BooleanField(default=False)
    resolved_at = fields.DatetimeField(null=True)
    metadata = fields.JSONField(default=dict)

    class Meta:
        table = "alerts"
        indexes = (
            ("user_id", "timestamp"),
            ("risk_level", "resolved"),
        )


class ModelVersion(BaseModelMixin):
    """Model version tracking."""
    version = fields.CharField(max_length=32, unique=True)
    model_type = fields.CharField(max_length=32)
    path = fields.CharField(max_length=255)
    hash = fields.CharField(max_length=64)
    metadata = fields.JSONField(default=dict)
    active = fields.BooleanField(default=False)
    deployed_at = fields.DatetimeField(null=True)

    class Meta:
        table = "model_versions"


class CalibrationRecord(BaseModelMixin):
    """Per-user baseline calibration record."""
    user = fields.ForeignKeyField("models.User", related_name="calibration_records")
    feature_type = fields.CharField(max_length=64)
    timestamp = fields.DatetimeField()
    baseline_mean = fields.JSONField()
    baseline_std = fields.JSONField()
    calibration_window_start = fields.DatetimeField()
    calibration_window_end = fields.DatetimeField()
    metadata = fields.JSONField(default=dict)

    class Meta:
        table = "calibration_records"
        indexes = (
            ("user_id", "feature_type"),
        )


class MigrationHistory(Model):
    """Aerich migration history."""
    version = fields.CharField(max_length=255)
    app = fields.CharField(max_length=20)
    name = fields.CharField(max_length=255)
    applied = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "aerich"