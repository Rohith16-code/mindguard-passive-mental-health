"""On-device sensor data ingestion pipeline."""
import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.db.models import IngestionEvent, SensorData, UserSession
from src.ingestion.validator import validate_sensor_event
from src.utils.crypt import anonymize_device_id, encrypt_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IngestionHandler:
    """Handles on-device sensor data ingestion with privacy preservation."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._batch_buffer: List[Dict[str, Any]] = []
        self._batch_size = settings.INGESTION_BATCH_SIZE
        self._max_buffer_size = settings.INGESTION_MAX_BUFFER_SIZE
        self._last_flush_time = time.time()
        self._flush_interval = settings.INGESTION_FLUSH_INTERVAL_SECONDS

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self) -> None:
        """Establish database connection."""
        try:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            logger.info("Database connection established")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def disconnect(self) -> None:
        """Close database connection and flush remaining data."""
        if self._conn:
            try:
                await self._flush_buffer()
            finally:
                self._conn.close()
                self._conn = None

    async def ingest_event(self, event: Dict[str, Any]) -> Tuple[bool, str]:
        """Ingest a single sensor event with validation and anonymization."""
        try:
            if not validate_sensor_event(event):
                return False, "Invalid event format"

            anonymized_device_id = anonymize_device_id(event.get("device_id", ""))
            encrypted_payload = encrypt_data(json.dumps(event.get("payload", {})))

            ingestion_record = {
                "device_id_hash": anonymized_device_id,
                "sensor_type": event.get("sensor_type", "unknown"),
                "timestamp": event.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "payload_encrypted": encrypted_payload,
                "ingestion_time": datetime.now(timezone.utc).isoformat(),
                "raw_size": len(json.dumps(event)),
                "validated": True,
            }

            self._batch_buffer.append(ingestion_record)

            if len(self._batch_buffer) >= self._max_buffer_size:
                await self._flush_buffer()

            return True, "Event queued"

        except Exception as e:
            logger.error(f"Error ingesting event: {e}")
            return False, str(e)

    async def ingest_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of sensor events."""
        results = {"success": 0, "failed": 0, "errors": []}

        for event in events:
            success, message = await self.ingest_event(event)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(message)

        return results

    async def _flush_buffer(self) -> None:
        """Flush buffered events to database."""
        if not self._batch_buffer:
            return

        try:
            cursor = self._conn.cursor()

            for record in self._batch_buffer:
                cursor.execute(
                    """
                    INSERT INTO sensor_data (
                        device_id_hash, sensor_type, timestamp, payload_encrypted,
                        ingestion_time, raw_size, validated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["device_id_hash"],
                        record["sensor_type"],
                        record["timestamp"],
                        record["payload_encrypted"],
                        record["ingestion_time"],
                        record["raw_size"],
                        record["validated"],
                    ),
                )

            self._conn.commit()
            logger.info(f"Flushed {len(self._batch_buffer)} events to database")

        except sqlite3.Error as e:
            logger.error(f"Database error during flush: {e}")
            raise
        finally:
            self._batch_buffer = []
            self._last_flush_time = time.time()

    async def _periodic_flush(self) -> None:
        """Flush buffer if interval exceeded."""
        if time.time() - self._last_flush_time >= self._flush_interval:
            await self._flush_buffer()

    async def create_session(self, user_id: str, session_type: str) -> int:
        """Create a new user session."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_sessions (user_id_hash, session_type, start_time, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                anonymize_device_id(user_id),
                session_type,
                datetime.now(timezone.utc).isoformat(),
                "active",
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    async def end_session(self, session_id: int) -> bool:
        """End a user session."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE user_sessions SET end_time = ?, status = ?
            WHERE id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), "completed", session_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    async def get_recent_events(
        self, sensor_type: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve recent sensor events."""
        cursor = self._conn.cursor()
        if sensor_type:
            cursor.execute(
                """
                SELECT * FROM sensor_data
                WHERE sensor_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (sensor_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM sensor_data
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_session_events(
        self, session_id: int, sensor_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get events associated with a specific session."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT sd.* FROM sensor_data sd
            JOIN ingestion_events ie ON sd.id = ie.sensor_data_id
            WHERE ie.session_id = ?
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    async def link_event_to_session(self, event_id: int, session_id: int) -> bool:
        """Link an event to a session."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO ingestion_events (sensor_data_id, session_id, link_time)
                VALUES (?, ?, ?)
                """,
                (event_id, session_id, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to link event to session: {e}")
            return False

    async def process_stream(
        self, stream: Any, callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Process a stream of sensor data."""
        results = {"processed": 0, "errors": []}

        async for event in stream:
            try:
                success, message = await self.ingest_event(event)
                if success and callback:
                    await callback(event)
                results["processed"] += 1
            except Exception as e:
                results["errors"].append(str(e))
                logger.error(f"Stream processing error: {e}")

        await self._flush_buffer()
        return results

    async def get_buffer_status(self) -> Dict[str, Any]:
        """Get current ingestion buffer status."""
        return {
            "buffer_size": len(self._batch_buffer),
            "max_buffer_size": self._max_buffer_size,
            "batch_size": self._batch_size,
            "last_flush_seconds_ago": time.time() - self._last_flush_time,
        }