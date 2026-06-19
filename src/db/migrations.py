"""Database migrations module for schema versioning and management."""
from pathlib import Path
from typing import List, Optional, Tuple
import sqlite3
import asyncio
import logging

from aerich import Command
from aerich.const import MIGRATE_FILE_HEADER
from aerich.db import SQLiteDatabase
from aerich.models import Aerich

from src.config import settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MIGRATIONS_DIR.mkdir(exist_ok=True)


class MigrationManager:
    """Manages database schema migrations using Aerich."""

    def __init__(self, db_url: str = settings.DATABASE_URL):
        self.db_url = db_url
        self.db = SQLiteDatabase(db_url)
        self.command = Command(
            db_url=db_url,
            location=str(MIGRATIONS_DIR),
            app="models",
            models="src.db.models",
        )

    async def init(self) -> None:
        """Initialize Aerich and ensure migrations directory exists."""
        if not MIGRATIONS_DIR.exists():
            MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
        await self.command.init()

    async def upgrade(self) -> bool:
        """Run all pending migrations. Returns True if successful."""
        try:
            await self.command.upgrade()
            logger.info("Database migrations applied successfully")
            return True
        except Exception as e:
            logger.error(f"Migration upgrade failed: {e}")
            raise

    async def downgrade(self, version: Optional[str] = None) -> bool:
        """Downgrade database schema to specified version or latest."""
        try:
            await self.command.downgrade(version)
            logger.info("Database downgrade completed")
            return True
        except Exception as e:
            logger.error(f"Migration downgrade failed: {e}")
            raise

    async def current_version(self) -> Optional[str]:
        """Get current database schema version."""
        try:
            aerich = await Aerich.filter(app="models").first()
            return aerich.version if aerich else None
        except Exception as e:
            logger.warning(f"Failed to get current version: {e}")
            return None

    async def history(self) -> List[str]:
        """Get list of applied migration versions in order."""
        try:
            aeriches = await Aerich.filter(app="models").order_by("-date").all()
            return [a.version for a in aeriches]
        except Exception as e:
            logger.warning(f"Failed to get migration history: {e}")
            return []

    async def needs_migration(self) -> bool:
        """Check if database requires migration."""
        try:
            current = await self.current_version()
            if current is None:
                return True
            migrations = list(MIGRATIONS_DIR.glob("*.py"))
            if not migrations:
                return False
            latest = max(
                (int(f.stem.split("_")[0]) for f in migrations if f.stem[0].isdigit()),
                default=0,
            )
            return int(current.split("_")[0]) < latest
        except Exception as e:
            logger.error(f"Migration check failed: {e}")
            return True

    async def create_migration(self, message: str = "auto") -> Optional[str]:
        """Create a new migration file."""
        try:
            result = await self.command.migrate(message)
            if result:
                logger.info(f"Migration created: {result}")
                return result
            return None
        except Exception as e:
            logger.error(f"Migration creation failed: {e}")
            raise

    async def validate_schema(self) -> Tuple[bool, List[str]]:
        """Validate current database schema against models."""
        errors = []
        try:
            conn = sqlite3.connect(settings.DATABASE_URL.split("://")[1])
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            existing_tables = {row[0] for row in cursor.fetchall()}

            from src.db.models import User, SensorData, FeatureVector, CrisisSignal

            expected_tables = {
                "user",
                "sensordata",
                "featurevector",
                "crisissignal",
                "aerich",
            }

            missing = expected_tables - existing_tables
            if missing:
                errors.append(f"Missing tables: {', '.join(missing)}")

            conn.close()
            return len(errors) == 0, errors
        except Exception as e:
            errors.append(f"Schema validation error: {e}")
            return False, errors

    async def ensure_schema(self) -> bool:
        """Ensure database schema is up-to-date, applying migrations if needed."""
        if await self.needs_migration():
            logger.info("Pending migrations detected. Applying...")
            await self.upgrade()
        return await self.validate_schema()[0]


async def run_migrations() -> bool:
    """Run migrations on application startup."""
    manager = MigrationManager()
    await manager.init()
    return await manager.ensure_schema()


def run_migrations_sync() -> bool:
    """Synchronous wrapper for migration execution."""
    return asyncio.run(run_migrations())