"""User consent tracking & opt-in flows module."""
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import sqlite3
import hashlib

from pydantic import BaseModel, Field, validator


class ConsentStatus(Enum):
    """Consent status enumeration."""
    NOT_CONSENTED = "not_consented"
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"


class ConsentType(Enum):
    """Types of consent tracked."""
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    MODEL_TRAINING = "model_training"
    ALERTS = "alerts"


class ConsentRecord(BaseModel):
    """Represents a single consent record."""
    user_id: str
    consent_type: ConsentType
    status: ConsentStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("timestamp", pre=True, always=True)
    def set_timestamp(cls, v):
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class ConsentManager:
    """Manages user consent tracking and opt-in flows."""

    SCHEMA_VERSION = "1.0"
    DB_NAME = "consent.db"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.db_path = self.data_dir / self.DB_NAME
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure database schema exists and is up to date."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consent_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                consent_type TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                version TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(user_id, consent_type)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                version TEXT NOT NULL
            )
        """)
        cursor.execute("INSERT OR IGNORE INTO schema_info (version) VALUES (?)",
                       (self.SCHEMA_VERSION,))
        self._conn.commit()

    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy-preserving storage."""
        return hashlib.sha256(user_id.encode()).hexdigest()

    def _row_to_record(self, row: sqlite3.Row) -> ConsentRecord:
        """Convert database row to ConsentRecord."""
        return ConsentRecord(
            user_id=row["user_id"],
            consent_type=ConsentType(row["consent_type"]),
            status=ConsentStatus(row["status"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            version=row["version"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    def get_consent(self, user_id: str, consent_type: ConsentType) -> Optional[ConsentRecord]:
        """Get current consent status for a user and consent type."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM consent_records WHERE user_id = ? AND consent_type = ?",
            (self._hash_user_id(user_id), consent_type.value)
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def set_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        status: ConsentStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsentRecord:
        """Set or update consent status for a user and consent type."""
        cursor = self._conn.cursor()
        record = ConsentRecord(
            user_id=self._hash_user_id(user_id),
            consent_type=consent_type,
            status=status,
            metadata=metadata or {}
        )
        cursor.execute(
            """
            INSERT INTO consent_records (user_id, consent_type, status, timestamp, version, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, consent_type) DO UPDATE SET
                status = excluded.status,
                timestamp = excluded.timestamp,
                version = excluded.version,
                metadata = excluded.metadata
            """,
            (
                record.user_id,
                record.consent_type.value,
                record.status.value,
                record.timestamp.isoformat(),
                record.version,
                json.dumps(record.metadata)
            )
        )
        self._conn.commit()
        return record

    def get_all_consent(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for a user."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM consent_records WHERE user_id = ?",
            (self._hash_user_id(user_id),)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def has_active_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if user has active (granted) consent for a type."""
        record = self.get_consent(user_id, consent_type)
        return record is not None and record.status == ConsentStatus.GRANTED

    def get_pending_consent(self, user_id: str) -> List[ConsentRecord]:
        """Get all pending consent requests for a user."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM consent_records WHERE user_id = ? AND status = ?",
            (self._hash_user_id(user_id), ConsentStatus.PENDING.value)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def revoke_all(self, user_id: str) -> int:
        """Revoke all consent for a user. Returns number of records revoked."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE consent_records SET status = ? WHERE user_id = ?",
            (ConsentStatus.REVOKED.value, self._hash_user_id(user_id))
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()