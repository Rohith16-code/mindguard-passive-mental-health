import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.buffer import *

@pytest.fixture
def buffer_instance():
    """Create a ring buffer instance with capacity 5 for testing."""
    with patch("src.ingestion.buffer.redis.Redis") as mock_redis:
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        buffer = RingBuffer(capacity=5, redis_key="test:buffer")
        buffer._redis = mock_redis_client  # Inject mock
        yield buffer

@pytest.fixture
def buffer_with_data(buffer_instance):
    """Create a ring buffer with some pre-filled data."""
    buffer = buffer_instance
    for i in range(3):
        buffer.push(i)
    yield buffer

class TestRingBuffer:
    def test_init_defaults(self):
        """Test initialization with default values."""
        with patch("src.ingestion.buffer.redis.Redis"):
            buf = RingBuffer()
            assert buf.capacity == 1000
            assert buf._data == []
            assert buf._head == 0
            assert buf._size == 0

    def test_init_custom_capacity(self):
        """Test initialization with custom capacity."""
        with patch("src.ingestion.buffer.redis.Redis"):
            buf = RingBuffer(capacity=10)
            assert buf.capacity == 10

    def test_push_under_capacity(self, buffer_instance):
        """Test pushing elements when buffer is not full."""
        buffer = buffer_instance
        buffer.push(1)
        buffer.push(2)
        assert buffer._data == [1, 2]
        assert buffer._size == 2

    def test_push_over_capacity_overwrites_oldest(self, buffer_instance):
        """Test that pushing beyond capacity overwrites oldest element."""
        buffer = buffer_instance
        buffer.push(1)
        buffer.push(2)
        buffer.push(3)
        buffer.push(4)
        buffer.push(5)
        buffer.push(6)  # Should overwrite 1
        assert buffer._data == [6, 2, 3, 4, 5]
        assert buffer._size == 5

    def test_push_with_redis_sync(self, buffer_instance):
        """Test that push syncs to Redis when enabled."""
        buffer = buffer_instance
        buffer._redis_sync_enabled = True
        buffer.push(42)
        buffer._redis.lpush.assert_called_once_with("test:buffer", 42)
        buffer._redis.ltrim.assert_called_once_with("test:buffer", 0, buffer.capacity - 1)

    def test_pop_empty_buffer(self, buffer_instance):
        """Test popping from empty buffer returns None."""
        buffer = buffer_instance
        assert buffer.pop() is None

    def test_pop_nonempty_buffer(self, buffer_with_data):
        """Test popping from non-empty buffer."""
        buffer = buffer_with_data
        assert buffer.pop() == 1
        assert buffer.pop() == 2
        assert buffer.pop() == 3
        assert buffer.pop() is None

    def test_peek_empty_buffer(self, buffer_instance):
        """Test peeking empty buffer returns None."""
        buffer = buffer_instance
        assert buffer.peek() is None

    def test_peek_nonempty_buffer(self, buffer_with_data):
        """Test peeking non-empty buffer returns oldest element."""
        buffer = buffer_with_data
        assert buffer.peek() == 1
        assert buffer._size == 3  # Unchanged

    def test_len(self, buffer_instance):
        """Test __len__ returns correct count."""
        buffer = buffer_instance
        assert len(buffer) == 0
        buffer.push(1)
        assert len(buffer) == 1
        buffer.push(2)
        assert len(buffer) == 2

    def test_is_empty(self, buffer_instance):
        """Test is_empty property."""
        buffer = buffer_instance
        assert buffer.is_empty is True
        buffer.push(1)
        assert buffer.is_empty is False

    def test_is_full(self, buffer_instance):
        """Test is_full property."""
        buffer = buffer_instance
        assert buffer.is_full is False
        for i in range(5):
            buffer.push(i)
        assert buffer.is_full is True

    def test_clear(self, buffer_with_data):
        """Test clearing buffer."""
        buffer = buffer_with_data
        buffer.clear()
        assert buffer._data == []
        assert buffer._size == 0
        assert buffer._head == 0

    def test_clear_with_redis(self, buffer_with_data):
        """Test clearing buffer also clears Redis."""
        buffer = buffer_with_data
        buffer._redis_sync_enabled = True
        buffer.clear()
        buffer._redis.delete.assert_called_once_with("test:buffer")

    def test_to_list(self, buffer_with_data):
        """Test conversion to list preserves order."""
        buffer = buffer_with_data
        result = buffer.to_list()
        assert result == [1, 2, 3]

    def test_to_list_empty(self, buffer_instance):
        """Test to_list on empty buffer."""
        buffer = buffer_instance
        assert buffer.to_list() == []

    def test_to_list_with_wraparound(self, buffer_instance):
        """Test to_list when data wraps around in buffer."""
        buffer = buffer_instance
        # Fill and overwrite to create wraparound
        for i in range(7):
            buffer.push(i)
        # Buffer now: [5, 6, 2, 3, 4] (head=2, size=5)
        result = buffer.to_list()
        assert result == [2, 3, 4, 5, 6]

    def test_from_redis_empty(self, buffer_instance):
        """Test loading from Redis when empty."""
        buffer = buffer_instance
        buffer._redis.lrange.return_value = []
        buffer._load_from_redis()
        assert buffer._data == []
        assert buffer._size == 0

    def test_from_redis_with_data(self, buffer_instance):
        """Test loading from Redis with data."""
        buffer = buffer_instance
        buffer._redis.lrange.return_value = [b"3", b"2", b"1"]
        buffer._load_from_redis()
        assert buffer._data == [1, 2, 3]
        assert buffer._size == 3

    def test_from_redis_with_non_integers(self, buffer_instance):
        """Test loading from Redis with string data."""
        buffer = buffer_instance
        buffer._redis.lrange.return_value = [b"hello", b"world"]
        buffer._load_from_redis()
        assert buffer._data == ["world", "hello"]
        assert buffer._size == 2

    def test_redis_sync_disabled_by_default(self, buffer_instance):
        """Test Redis sync is disabled by default."""
        buffer = buffer_instance
        assert buffer._redis_sync_enabled is False
        buffer.push(1)
        buffer._redis.lpush.assert_not_called()

    def test_enable_redis_sync(self, buffer_instance):
        """Test enabling Redis sync."""
        buffer = buffer_instance
        buffer.enable_redis_sync()
        assert buffer._redis_sync_enabled is True

    def test_disable_redis_sync(self, buffer_instance):
        """Test disabling Redis sync."""
        buffer = buffer_instance
        buffer.enable_redis_sync()
        buffer.disable_redis_sync()
        assert buffer._redis_sync_enabled is False

    def test_push_with_redis_sync_disabled(self, buffer_instance):
        """Test push does not sync to Redis when disabled."""
        buffer = buffer_instance
        buffer._redis_sync_enabled = False
        buffer.push(1)
        buffer._redis.lpush.assert_not_called()

    def test_push_with_redis_sync_enabled(self, buffer_instance):
        """Test push syncs to Redis when enabled."""
        buffer = buffer_instance
        buffer.enable_redis_sync()
        buffer.push(1)
        buffer._redis.lpush.assert_called_once_with("test:buffer", 1)
        buffer._redis.ltrim.assert_called_once_with("test:buffer", 0, buffer.capacity - 1)

    def test_push_with_redis_sync_multiple(self, buffer_instance):
        """Test multiple pushes sync correctly."""
        buffer = buffer_instance
        buffer.enable_redis_sync()
        buffer.push(1)
        buffer.push(2)
        assert buffer._redis.lpush.call_count == 2
        assert buffer._redis.ltrim.call_count == 2

    def test_load_from_redis_on_init(self, buffer_instance):
        """Test loading from Redis during initialization."""
        buffer = buffer_instance
        buffer._redis.lrange.return_value = [b"10", b"20"]
        buffer._load_from_redis()
        assert buffer._data == [20, 10]
        assert buffer._size == 2

    def test_load_from_redis_truncates_to_capacity(self, buffer_instance):
        """Test loading from Redis truncates to capacity."""
        buffer = buffer_instance
        buffer._redis.lrange.return_value = [b"1", b"2", b"3", b"4", b"5", b"6"]
        buffer._load_from_redis()
        assert buffer._data == [6, 5, 4, 3, 2]
        assert buffer._size == 5

    def test_load_from_redis_handles_empty_list(self, buffer_instance):
        """Test loading from Redis with empty list."""
        buffer = buffer_instance
        buffer._redis.lrange.return_value = []
        buffer._load_from_redis()
        assert buffer._data == []
        assert buffer._size == 0

    def test_load_from_redis_handles_none(self, buffer_instance):
        """Test loading from Redis when lrange returns None."""
        buffer = buffer_instance
        buffer._redis.lrange.return_value = None
        buffer._load_from_redis()
        assert buffer._data == []
        assert buffer._size == 0

    def test_to_list_preserves_order_after_pop(self, buffer_with_data):
        """Test to_list reflects state after pop operations."""
        buffer = buffer_with_data
        buffer.pop()
        result = buffer.to_list()
        assert result == [2, 3]

    def test_to_list_preserves_order_after_push(self, buffer_with_data):
        """Test to_list reflects state after push operations."""
        buffer = buffer_with_data
        buffer.push(4)
        buffer.push(5)
        result = buffer.to_list()
        assert result == [1, 2, 3, 4, 5]

    def test_to_list_after_wraparound(self, buffer_instance):
        """Test to_list after multiple wraps."""
        buffer = buffer_instance
        for i in range(12):
            buffer.push(i)
        result = buffer.to_list()
        assert result == [7, 8, 9, 10, 11]