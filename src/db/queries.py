"""Database query helpers for SQLite operations."""
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.config import settings


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection with appropriate settings."""
    path = db_path or settings.DB_PATH
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def execute_query(query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return results as list of dicts."""
    with get_db_connection() as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def execute_insert(query: str, params: Tuple = ()) -> int:
    """Execute an INSERT/UPDATE/DELETE query and return affected row count."""
    with get_db_connection() as conn:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.rowcount


def execute_many(query: str, params_list: List[Tuple]) -> int:
    """Execute the same query with multiple parameter sets."""
    with get_db_connection() as conn:
        cursor = conn.executemany(query, params_list)
        conn.commit()
        return cursor.rowcount


def get_user_data_points(
    user_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Get sensor data points for a user within a time range."""
    query = """
        SELECT id, timestamp, sensor_type, data, metadata, created_at
        FROM sensor_data
        WHERE user_id = ?
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    end = end_time or datetime.utcnow()
    start = start_time or (end - timedelta(days=7))
    return execute_query(query, (user_id, start.isoformat(), end.isoformat(), limit))


def get_user_features(
    user_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Get extracted features for a user within a time range."""
    query = """
        SELECT id, timestamp, feature_type, value, source_data_id, created_at
        FROM features
        WHERE user_id = ?
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp DESC
    """
    end = end_time or datetime.utcnow()
    start = start_time or (end - timedelta(days=7))
    return execute_query(query, (user_id, start.isoformat(), end.isoformat()))


def get_latest_wellness_index(user_id: int) -> Optional[Dict[str, Any]]:
    """Get the most recent wellness index for a user."""
    query = """
        SELECT id, timestamp, wellness_index, risk_level, confidence, model_version, created_at
        FROM wellness_indices
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """
    results = execute_query(query, (user_id,))
    return results[0] if results else None


def get_wellness_history(
    user_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get wellness index history for a user."""
    query = """
        SELECT id, timestamp, wellness_index, risk_level, confidence, model_version, created_at
        FROM wellness_indices
        WHERE user_id = ?
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    end = end_time or datetime.utcnow()
    start = start_time or (end - timedelta(days=30))
    return execute_query(query, (user_id, start.isoformat(), end.isoformat(), limit))


def insert_sensor_data(
    user_id: int,
    timestamp: datetime,
    sensor_type: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a sensor data point."""
    query = """
        INSERT INTO sensor_data (user_id, timestamp, sensor_type, data, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    params = (
        user_id,
        timestamp.isoformat(),
        sensor_type,
        str(data),
        str(metadata or {}),
        datetime.utcnow().isoformat(),
    )
    return execute_insert(query, params)


def insert_features(
    user_id: int,
    timestamp: datetime,
    feature_type: str,
    value: float,
    source_data_id: int,
) -> int:
    """Insert a feature vector."""
    query = """
        INSERT INTO features (user_id, timestamp, feature_type, value, source_data_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    params = (
        user_id,
        timestamp.isoformat(),
        feature_type,
        value,
        source_data_id,
        datetime.utcnow().isoformat(),
    )
    return execute_insert(query, params)


def insert_wellness_index(
    user_id: int,
    timestamp: datetime,
    wellness_index: float,
    risk_level: str,
    confidence: float,
    model_version: str,
) -> int:
    """Insert a wellness index record."""
    query = """
        INSERT INTO wellness_indices (user_id, timestamp, wellness_index, risk_level, confidence, model_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        user_id,
        timestamp.isoformat(),
        wellness_index,
        risk_level,
        confidence,
        model_version,
        datetime.utcnow().isoformat(),
    )
    return execute_insert(query, params)


def get_sensor_types_for_user(user_id: int) -> List[str]:
    """Get unique sensor types observed for a user."""
    query = """
        SELECT DISTINCT sensor_type
        FROM sensor_data
        WHERE user_id = ?
    """
    results = execute_query(query, (user_id,))
    return [row["sensor_type"] for row in results]


def get_feature_types_for_user(user_id: int) -> List[str]:
    """Get unique feature types computed for a user."""
    query = """
        SELECT DISTINCT feature_type
        FROM features
        WHERE user_id = ?
    """
    results = execute_query(query, (user_id,))
    return [row["feature_type"] for row in results]


def get_data_points_count(
    user_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> int:
    """Get count of sensor data points for a user."""
    query = """
        SELECT COUNT(*) as count
        FROM sensor_data
        WHERE user_id = ?
        AND timestamp >= ?
        AND timestamp <= ?
    """
    end = end_time or datetime.utcnow()
    start = start_time or (end - timedelta(days=7))
    results = execute_query(query, (user_id, start.isoformat(), end.isoformat()))
    return results[0]["count"] if results else 0


def get_wellness_risk_counts(
    user_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Dict[str, int]:
    """Get count of wellness indices by risk level."""
    query = """
        SELECT risk_level, COUNT(*) as count
        FROM wellness_indices
        WHERE user_id = ?
        AND timestamp >= ?
        AND timestamp <= ?
        GROUP BY risk_level
    """
    end = end_time or datetime.utcnow()
    start = start_time or (end - timedelta(days=7))
    results = execute_query(query, (user_id, start.isoformat(), end.isoformat()))
    return {row["risk_level"]: row["count"] for row in results}


def get_recent_crisis_alerts(
    user_id: int,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Get recent crisis alerts (high risk level) for a user."""
    query = """
        SELECT id, timestamp, wellness_index, risk_level, confidence, model_version, created_at
        FROM wellness_indices
        WHERE user_id = ?
        AND risk_level IN ('high', 'critical')
        ORDER BY timestamp DESC
        LIMIT ?
    """
    return execute_query(query, (user_id, limit))


def get_baseline_features(
    user_id: int,
    feature_type: str,
    days: int = 30,
) -> List[float]:
    """Get baseline feature values for computing per-user normalization."""
    query = """
        SELECT value
        FROM features
        WHERE user_id = ?
        AND feature_type = ?
        AND timestamp >= ?
        ORDER BY timestamp DESC
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    results = execute_query(query, (user_id, feature_type, start.isoformat()))
    return [row["value"] for row in results]