"""Per-user baseline calibration for mental health crisis detection."""
from typing import List, Union
import math
import statistics

from src.database import db
from src.cache import redis_client


def calculate_baseline_mean(user_id: str) -> float:
    """Calculate the mean of baseline values for a user.
    
    Args:
        user_id: Unique identifier for the user.
        
    Returns:
        Mean of baseline values, or 0.0 if no baseline data exists.
    """
    baseline_values = db.get_baseline(user_id)
    if not baseline_values:
        return 0.0
    return statistics.mean(baseline_values)


def calculate_baseline_std(user_id: str) -> float:
    """Calculate the standard deviation of baseline values for a user.
    
    Args:
        user_id: Unique identifier for the user.
        
    Returns:
        Standard deviation of baseline values, or 0.0 if no baseline data exists
        or only one value exists.
    """
    baseline_values = db.get_baseline(user_id)
    if len(baseline_values) < 2:
        return 0.0
    mean = statistics.mean(baseline_values)
    variance = sum((x - mean) ** 2 for x in baseline_values) / len(baseline_values)
    return math.sqrt(variance)


def normalize_value(value: float, mean: float, std: float) -> float:
    """Normalize a single value using z-score normalization.
    
    Args:
        value: Raw value to normalize.
        mean: Baseline mean for the user.
        std: Baseline standard deviation for the user.
        
    Returns:
        Z-score normalized value. If std is 0, returns 0.0.
    """
    if std == 0.0:
        return 0.0
    return (value - mean) / std


def normalize_values(
    values: List[float], user_id: str
) -> List[float]:
    """Normalize a list of values using user-specific baseline statistics.
    
    Args:
        values: List of raw values to normalize.
        user_id: Unique identifier for the user.
        
    Returns:
        List of z-score normalized values.
    """
    mean = calculate_baseline_mean(user_id)
    std = calculate_baseline_std(user_id)
    return [normalize_value(value, mean, std) for value in values]


def get_normalized_current_values(user_id: str) -> List[float]:
    """Get normalized current values for a user using their baseline statistics.
    
    Args:
        user_id: Unique identifier for the user.
        
    Returns:
        List of normalized current values.
    """
    current_values = db.get_current_values(user_id)
    return normalize_values(current_values, user_id)


def update_baseline(user_id: str, new_values: List[float]) -> bool:
    """Update the baseline for a user with new values.
    
    Args:
        user_id: Unique identifier for the user.
        new_values: List of new baseline values to add.
        
    Returns:
        True if update was successful, False otherwise.
    """
    try:
        db.update_baseline(user_id, new_values)
        redis_client.delete(f"baseline_mean:{user_id}")
        redis_client.delete(f"baseline_std:{user_id}")
        return True
    except Exception:
        return False


def get_or_calculate_baseline_stats(user_id: str) -> tuple:
    """Get or calculate baseline statistics for a user.
    
    Args:
        user_id: Unique identifier for the user.
        
    Returns:
        Tuple of (mean, std) for the user's baseline.
    """
    mean = calculate_baseline_mean(user_id)
    std = calculate_baseline_std(user_id)
    return mean, std


def get_normalized_values_with_stats(
    values: List[float], user_id: str
) -> tuple:
    """Normalize values and return along with statistics used.
    
    Args:
        values: List of raw values to normalize.
        user_id: Unique identifier for the user.
        
    Returns:
        Tuple of (normalized_values, mean, std).
    """
    mean, std = get_or_calculate_baseline_stats(user_id)
    normalized = [normalize_value(v, mean, std) for v in values]
    return normalized, mean, std