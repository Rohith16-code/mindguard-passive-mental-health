import pytest
from unittest.mock import MagicMock, patch
from src.db.queries import (
    get_user_by_id,
    create_user,
    get_active_sessions,
    invalidate_session,
    get_cached_user_profile,
    set_cached_user_profile,
    delete_cached_user_profile,
    get_user_activity_stats,
    execute_with_retry
)


@pytest.fixture
def mock_db_connection():
    with patch('src.db.queries.sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def mock_redis_client():
    with patch('src.db.queries.redis.Redis') as mock_redis:
        client = MagicMock()
        mock_redis.return_value = client
        yield client


class TestGetUserById:
    def test_get_user_by_id_success(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.fetchone.return_value = (1, "alice@example.com", "Alice", 1)

        result = get_user_by_id(1)

        assert result == {"id": 1, "email": "alice@example.com", "name": "Alice", "is_active": 1}
        mock_cursor.execute.assert_called_once_with(
            "SELECT id, email, name, is_active FROM users WHERE id = ?", (1,)
        )

    def test_get_user_by_id_not_found(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.fetchone.return_value = None

        result = get_user_by_id(999)

        assert result is None

    def test_get_user_by_id_db_error(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.execute.side_effect = sqlite3.Error("DB error")

        with pytest.raises(sqlite3.Error):
            get_user_by_id(1)


class TestCreateUser:
    def test_create_user_success(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.lastrowid = 42

        result = create_user("bob@example.com", "Bob", True)

        assert result == 42
        mock_cursor.execute.assert_called_once_with(
            "INSERT INTO users (email, name, is_active) VALUES (?, ?, ?)",
            ("bob@example.com", "Bob", 1)
        )

    def test_create_user_db_error(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.execute.side_effect = sqlite3.IntegrityError("Duplicate email")

        with pytest.raises(sqlite3.IntegrityError):
            create_user("alice@example.com", "Alice", True)


class TestGetActiveSessions:
    def test_get_active_sessions_success(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.fetchall.return_value = [
            (1, "abc123", 1700000000, 1700003600),
            (2, "def456", 1700001000, 1700004600)
        ]

        result = get_active_sessions()

        assert len(result) == 2
        assert result[0] == {"session_id": 1, "token": "abc123", "created_at": 1700000000, "expires_at": 1700003600}
        mock_cursor.execute.assert_called_once_with(
            "SELECT id, token, created_at, expires_at FROM sessions WHERE expires_at > ? AND invalidated = 0",
            (int(pytest.time.time()),)
        )

    def test_get_active_sessions_empty(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.fetchall.return_value = []

        result = get_active_sessions()

        assert result == []


class TestInvalidateSession:
    def test_invalidate_session_success(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.rowcount = 1

        result = invalidate_session("abc123")

        assert result is True
        mock_cursor.execute.assert_called_once_with(
            "UPDATE sessions SET invalidated = 1 WHERE token = ?",
            ("abc123",)
        )

    def test_invalidate_session_no_rows(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.rowcount = 0

        result = invalidate_session("nonexistent_token")

        assert result is False


class TestGetCachedUserProfile:
    def test_get_cached_user_profile_hit(self, mock_redis_client):
        mock_redis_client.get.return_value = b'{"id": 1, "name": "Alice"}'

        result = get_cached_user_profile(1)

        assert result == {"id": 1, "name": "Alice"}
        mock_redis_client.get.assert_called_once_with("user:profile:1")

    def test_get_cached_user_profile_miss(self, mock_redis_client):
        mock_redis_client.get.return_value = None

        result = get_cached_user_profile(1)

        assert result is None


class TestSetCachedUserProfile:
    def test_set_cached_user_profile_success(self, mock_redis_client):
        profile = {"id": 1, "name": "Alice", "email": "alice@example.com"}

        set_cached_user_profile(1, profile, ttl=3600)

        mock_redis_client.setex.assert_called_once_with(
            "user:profile:1", 3600, '{"id": 1, "name": "Alice", "email": "alice@example.com"}'
        )

    def test_set_cached_user_profile_no_ttl(self, mock_redis_client):
        profile = {"id": 2, "name": "Bob"}

        set_cached_user_profile(2, profile)

        mock_redis_client.setex.assert_called_once_with(
            "user:profile:2", 300, '{"id": 2, "name": "Bob"}'
        )


class TestDeleteCachedUserProfile:
    def test_delete_cached_user_profile_success(self, mock_redis_client):
        delete_cached_user_profile(1)

        mock_redis_client.delete.assert_called_once_with("user:profile:1")


class TestGetUserActivityStats:
    def test_get_user_activity_stats_success(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.fetchone.return_value = (5, 12, 3)

        result = get_user_activity_stats(1)

        assert result == {"login_count": 5, "post_count": 12, "comment_count": 3}
        mock_cursor.execute.assert_called_once_with(
            """
            SELECT 
                (SELECT COUNT(*) FROM sessions WHERE user_id = ?),
                (SELECT COUNT(*) FROM posts WHERE author_id = ?),
                (SELECT COUNT(*) FROM comments WHERE author_id = ?)
            """,
            (1, 1, 1)
        )

    def test_get_user_activity_stats_nulls(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.fetchone.return_value = (None, None, None)

        result = get_user_activity_stats(1)

        assert result == {"login_count": 0, "post_count": 0, "comment_count": 0}


class TestExecuteWithRetry:
    def test_execute_with_retry_success_first_try(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.execute.return_value = None

        result = execute_with_retry("SELECT 1")

        assert result is None
        assert mock_cursor.execute.call_count == 1

    def test_execute_with_retry_success_after_retry(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.execute.side_effect = [sqlite3.OperationalError("disk full"), None]

        result = execute_with_retry("SELECT 1", max_retries=2)

        assert result is None
        assert mock_cursor.execute.call_count == 2

    def test_execute_with_retry_exhaust_retries(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.execute.side_effect = sqlite3.OperationalError("disk full")

        with pytest.raises(sqlite3.OperationalError):
            execute_with_retry("SELECT 1", max_retries=2)

        assert mock_cursor.execute.call_count == 2

    def test_execute_with_retry_non_retryable_error(self, mock_db_connection):
        mock_cursor = mock_db_connection.cursor.return_value
        mock_cursor.execute.side_effect = sqlite3.IntegrityError("Unique constraint")

        with pytest.raises(sqlite3.IntegrityError):
            execute_with_retry("INSERT ...", max_retries=3)

        assert mock_cursor.execute.call_count == 1


# Import sqlite3 for exception types in tests
import sqlite3