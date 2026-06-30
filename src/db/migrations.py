"""Database migration utilities using sqlite3."""
import sqlite3
from pathlib import Path
from typing import List
from src.config import DATABASE_URL, DB_PATH
from src.utils.logger import get_logger

logger = get_logger("mindguard.migrations")


def get_db_path() -> Path:
    """Extract file path from DATABASE_URL."""
    if "sqlite:///" in DATABASE_URL:
        return Path(DATABASE_URL.replace("sqlite:///", ""))
    return Path(DB_PATH)


def run_migrations() -> List[str]:
    """Run all pending migrations."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    applied = []
    
    # Create migrations table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Define migrations
    migrations = [
        ("001_initial", """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                device_id TEXT NOT NULL,
                enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT 1,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
            CREATE TABLE IF NOT EXISTS wellness_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mwi_score REAL NOT NULL,
                risk_level TEXT DEFAULT 'low',
                contributing_factors TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                severity TEXT NOT NULL,
                mwi_score REAL NOT NULL,
                notification_sent BOOLEAN DEFAULT 0,
                acknowledged BOOLEAN DEFAULT 0
            );
        """),
    ]
    
    for name, sql in migrations:
        try:
            cursor.execute("SELECT 1 FROM _migrations WHERE name = ?", (name,))
            if cursor.fetchone():
                continue
            
            cursor.executescript(sql)
            cursor.execute("INSERT INTO _migrations (name) VALUES (?)", (name,))
            applied.append(name)
            logger.info(f"Applied migration: {name}")
        except Exception as e:
            logger.error(f"Migration {name} failed: {e}")
    
    conn.commit()
    conn.close()
    
    return applied
