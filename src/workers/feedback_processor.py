"""Feedback processor module for handling user-reported mood feedback."""
from datetime import datetime
from typing import Dict, Any, Optional
import re

from src.config import settings
from src.db.db import db
from src.redis import redis_client
from src.schemas.feedback import FeedbackIn


VALID_MOODS = {"happy", "sad", "anxious", "neutral", "excited", "calm", "tired", "frustrated"}


def validate_timestamp(timestamp_str: str) -> bool:
    """Validate ISO 8601 timestamp format."""
    try:
        datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return True
    except (ValueError, AttributeError):
        return False


def validate_feedback(data: Dict[str, Any]) -> bool:
    """Validate feedback data structure and content."""
    required_fields = {"user_id", "mood", "timestamp"}
    
    if not isinstance(data, dict):
        return False
    
    if not required_fields.issubset(data.keys()):
        return False
    
    user_id = data.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        return False
    
    mood = data.get("mood")
    if not isinstance(mood, str) or mood.lower() not in VALID_MOODS:
        return False
    
    timestamp = data.get("timestamp")
    if not isinstance(timestamp, str) or not validate_timestamp(timestamp):
        return False
    
    return True


def save_feedback_to_db(data: Dict[str, Any]) -> bool:
    """Save validated feedback to database."""
    try:
        feedback_data = {
            "user_id": data["user_id"],
            "mood": data["mood"].lower(),
            "timestamp": data["timestamp"],
            "note": data.get("note", "")
        }
        db.insert("feedback", feedback_data)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to save feedback to database: {e}")


def publish_to_redis(data: Dict[str, Any]) -> bool:
    """Publish feedback to Redis for real-time processing."""
    try:
        key = f"feedback:{data['user_id']}:{data['timestamp']}"
        redis_client.setex(key, settings.FEEDBACK_TTL, str(data))
        redis_client.publish("feedback_channel", str(data))
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to publish feedback to Redis: {e}")


def process_feedback(feedback_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process user-reported mood feedback."""
    if not validate_feedback(feedback_data):
        raise ValueError("Invalid feedback data")
    
    try:
        save_feedback_to_db(feedback_data)
        publish_to_redis(feedback_data)
        return {"status": "success", "message": "Feedback processed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}