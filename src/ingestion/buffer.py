"""Ring buffer module for high-frequency signal ingestion."""
from typing import Any, List, Optional
import redis


class RingBuffer:
    """A thread-safe ring buffer implementation for high-frequency signal data."""

    def __init__(self, capacity: int = 1000, redis_key: Optional[str] = None):
        """
        Initialize the ring buffer.

        Args:
            capacity: Maximum number of elements the buffer can hold.
            redis_key: Optional Redis key for persistence (if Redis is used).
        """
        self.capacity = capacity
        self._data: List[Any] = []
        self._head = 0
        self._size = 0
        self._redis_key = redis_key
        self._redis = None

        if redis_key:
            try:
                self._redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            except redis.ConnectionError:
                self._redis = None

    def push(self, item: Any) -> None:
        """
        Add an item to the buffer.

        If the buffer is full, the oldest item is overwritten.

        Args:
            item: The item to add to the buffer.
        """
        if self._size < self.capacity:
            self._data.append(item)
            self._size += 1
        else:
            self._data[self._head] = item
            self._head = (self._head + 1) % self.capacity

        if self._redis and self._redis_key:
            try:
                self._redis.lpush(self._redis_key, item)
                self._redis.ltrim(self._redis_key, 0, self.capacity - 1)
            except redis.RedisError:
                pass

    def pop(self) -> Any:
        """
        Remove and return the oldest item from the buffer.

        Returns:
            The oldest item, or None if the buffer is empty.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("pop from empty buffer")
        item = self._data[self._head]
        self._head = (self._head + 1) % self.capacity
        self._size -= 1
        return item

    def peek(self) -> Any:
        """
        Return the oldest item without removing it.

        Returns:
            The oldest item, or None if the buffer is empty.
        """
        if self._size == 0:
            return None
        return self._data[self._head]

    def __len__(self) -> int:
        """Return the current number of items in the buffer."""
        return self._size

    def __getitem__(self, index: int) -> Any:
        """
        Access items by index (0-based from oldest to newest).

        Args:
            index: Index of the item to retrieve.

        Returns:
            The item at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError("buffer index out of range")
        actual_index = (self._head + index) % self.capacity
        return self._data[actual_index]

    def to_list(self) -> List[Any]:
        """
        Return a list of items in chronological order (oldest first).

        Returns:
            A list of all items in the buffer in order.
        """
        if self._size == 0:
            return []
        result = []
        idx = self._head
        for _ in range(self._size):
            result.append(self._data[idx])
            idx = (idx + 1) % self.capacity
        return result

    def clear(self) -> None:
        """Clear all items from the buffer."""
        self._data = []
        self._head = 0
        self._size = 0
        if self._redis and self._redis_key:
            try:
                self._redis.delete(self._redis_key)
            except redis.RedisError:
                pass