"""Secure local storage utilities for encrypted model and key management."""
import os
import secrets
import struct
from pathlib import Path
from typing import Union, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .secure_storage import KEY_DIR, KEY_FILE, generate_key, encrypt, decrypt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class EncryptedStorage:
    """Secure encrypted storage for models and sensitive data."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or DATA_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._key = generate_key()

    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password)

    def _get_key_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.key"

    def _get_data_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.enc"

    def store(self, name: str, data: bytes, password: Optional[bytes] = None) -> None:
        """Store encrypted data with optional password-based key derivation."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        if password is None:
            key = self._key
            encrypted_data = encrypt(data, key)
        else:
            salt = secrets.token_bytes(16)
            derived_key = self._derive_key(password, salt)
            encrypted_data = encrypt(data, derived_key)
            salt_path = self._get_key_path(name)
            salt_path.write_bytes(salt)

        data_path = self._get_data_path(name)
        data_path.write_bytes(encrypted_data)

    def load(self, name: str, password: Optional[bytes] = None) -> bytes:
        """Load and decrypt stored data."""
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        data_path = self._get_data_path(name)
        if not data_path.exists():
            raise FileNotFoundError(f"No stored data found for {name}")

        encrypted_data = data_path.read_bytes()

        if password is None:
            key = self._key
        else:
            salt_path = self._get_key_path(name)
            if not salt_path.exists():
                raise FileNotFoundError(f"No salt found for {name}")
            salt = salt_path.read_bytes()
            key = self._derive_key(password, salt)

        try:
            return decrypt(encrypted_data, key)
        except InvalidToken as exc:
            raise ValueError("Decryption failed: invalid password or corrupted data") from exc

    def delete(self, name: str) -> None:
        """Delete stored data and associated key material."""
        data_path = self._get_data_path(name)
        key_path = self._get_key_path(name)

        if data_path.exists():
            data_path.unlink()
        if key_path.exists():
            key_path.unlink()

    def exists(self, name: str) -> bool:
        """Check if stored data exists."""
        return self._get_data_path(name).exists()

    def list_stored(self) -> list:
        """List all stored data names."""
        return [
            f.stem
            for f in self.base_dir.glob("*.enc")
            if f.is_file()
        ]

    def clear_all(self) -> None:
        """Remove all stored data and keys."""
        for f in self.base_dir.glob("*.enc"):
            f.unlink()
        for f in self.base_dir.glob("*.key"):
            f.unlink()


def store_model(name: str, model_data: bytes, password: Optional[bytes] = None) -> None:
    """Convenience function to store a model."""
    storage = EncryptedStorage()
    storage.store(name, model_data, password)


def load_model(name: str, password: Optional[bytes] = None) -> bytes:
    """Convenience function to load a model."""
    storage = EncryptedStorage()
    return storage.load(name, password)


def delete_model(name: str) -> None:
    """Convenience function to delete a model."""
    storage = EncryptedStorage()
    storage.delete(name)