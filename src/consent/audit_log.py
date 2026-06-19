"""Immutable audit log for consent and system actions."""
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import sqlite3
import hashlib
import threading

from pydantic import BaseModel, Field, validator


class ActionType(Enum):
    """Types of actions logged in the audit trail."""
    CONSENT_GRANT = "consent_grant"
    CONSENT_REVOKE = "consent_revoke"
    DATA_ACCESS = "data_access"
    MODEL_INFERENCE = "model_inference"
    ALERT_GENERATED = "alert_generated"
    SYSTEM_CONFIG = "system_config"
    USER_PROFILE_UPDATE = "user_profile_update"


class AuditRecord(BaseModel):
    """Represents a single audit log entry."""
    action_type: ActionType
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    version: str = "1.0"

    @validator("timestamp", pre=True, always=True)
    def set_timestamp(cls, v):
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class AuditLog:
    """Immutable audit log for consent and system actions."""

    SCHEMA_VERSION = "1.0"
    DB_NAME = "audit_log.db"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.db_path = self.data_dir / self.DB_NAME
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure database schema exists and is up to date."""
        with self._lock:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT NOT NULL,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    version TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    prev_hash TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_records(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_records(timestamp)
            """)
            self._conn.commit()

    def _compute_hash(self, record: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for a record."""
        record_str = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(record_str.encode()).hexdigest()

    def _get_prev_hash(self) -> str:
        """Get the hash of the last record (for chain integrity)."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT hash FROM audit_records ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row["hash"] if row else ""

    def log_action(
        self,
        action_type: ActionType,
        user_id: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditRecord:
        """Log an action with immutable record."""
        with self._lock:
            timestamp = datetime.now(timezone.utc)
            record_dict = {
                "action_type": action_type.value,
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "details": details or {},
                "session_id": session_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "version": self.SCHEMA_VERSION,
            }

            prev_hash = self._get_prev_hash()
            record_dict["prev_hash"] = prev_hash
            record_hash = self._compute_hash(record_dict)
            record_dict["hash"] = record_hash

            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_records (
                    action_type, user_id, timestamp, details, session_id,
                    ip_address, user_agent, version, hash, prev_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_dict["action_type"],
                    record_dict["user_id"],
                    record_dict["timestamp"],
                    json.dumps(record_dict["details"]),
                    record_dict["session_id"],
                    record_dict["ip_address"],
                    record_dict["user_agent"],
                    record_dict["version"],
                    record_dict["hash"],
                    record_dict["prev_hash"],
                ),
            )
            self._conn.commit()

            return AuditRecord(**record_dict)

    def get_records(
        self,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action_types: Optional[List[ActionType]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditRecord]:
        """Retrieve audit records with optional filters."""
        with self._lock:
            cursor = self._conn.cursor()
            query = "SELECT * FROM audit_records WHERE 1=1"
            params: List[Any] = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            if action_types:
                placeholders = ", ".join("?" * len(action_types))
                query += f" AND action_type IN ({placeholders})"
                params.extend([t.value for t in action_types])

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                AuditRecord(
                    action_type=ActionType(row["action_type"]),
                    user_id=row["user_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    details=json.loads(row["details"]),
                    session_id=row["session_id"],
                    ip_address=row["ip_address"],
                    user_agent=row["user_agent"],
                    version=row["version"],
                )
                for row in rows
            ]

    def verify_integrity(self) -> bool:
        """Verify the integrity of the audit chain."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT id, hash, prev_hash FROM audit_records ORDER BY id")
            rows = cursor.fetchall()

            if not rows:
                return True

            prev_hash = ""
            for row in rows:
                if row["prev_hash"] != prev_hash:
                    return False
                prev_hash = row["hash"]
            return True

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None