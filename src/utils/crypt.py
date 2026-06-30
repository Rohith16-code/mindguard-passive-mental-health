"""Cryptographic utilities for local key management and data anonymization."""
import hashlib
import os
import secrets
from pathlib import Path
from typing import Union

from cryptography.fernet import Fernet, InvalidToken

KEY_DIR = Path(__file__).resolve().parent.parent / "keys"
KEY_FILE = KEY_DIR / "local.key"


def generate_key() -> bytes:
    """Generate or load a 32-byte Fernet key from persistent storage."""
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = secrets.token_bytes(32)
    KEY_FILE.write_bytes(key)
    return key


def encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt data using Fernet with the provided key."""
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    fernet = Fernet(key)
    return fernet.encrypt(data)


def decrypt(token: bytes, key: bytes) -> bytes:
    """Decrypt data using Fernet with the provided key."""
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if not isinstance(token, bytes):
        raise TypeError("token must be bytes")
    fernet = Fernet(key)
    return fernet.decrypt(token)


def anonymize(value: str) -> str:
    """Anonymize a string value using SHA-256 hashing."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def anonymize_email(email: str) -> str:
    """Anonymize an email address by hashing the local part and domain separately."""
    if "@" not in email:
        raise ValueError("Invalid email format")
    local, domain = email.rsplit("@", 1)
    return f"{anonymize(local)}@{anonymize(domain)}"


def anonymize_user_id(user_id: Union[int, str]) -> str:
    """Anonymize a user identifier using a deterministic hash."""
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:12]


def anonymize_device_id(device_id: str) -> str:
    """Hash a device ID for privacy-safe storage."""
    return hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:16]


def encrypt_data(data: bytes, key: bytes = None) -> bytes:
    """Encrypt raw bytes using Fernet symmetric encryption."""
    if key is None:
        key = generate_key()
    f = Fernet(key)
    return f.encrypt(data)


def decrypt_data(encrypted: bytes, key: bytes) -> bytes:
    """Decrypt data encrypted with encrypt_data."""
    f = Fernet(key)
    return f.decrypt(encrypted)


def encrypt_file(filepath: str, key: bytes = None) -> str:
    """Encrypt a file and return the path to the encrypted file."""
    if key is None:
        key = generate_key()
    with open(filepath, "rb") as f:
        data = f.read()
    encrypted = encrypt(data, key)
    enc_path = filepath + ".enc"
    with open(enc_path, "wb") as f:
        f.write(encrypted)
    return enc_path


def decrypt_file(filepath: str, key: bytes) -> bytes:
    """Decrypt a file and return the decrypted contents."""
    with open(filepath, "rb") as f:
        encrypted = f.read()
    return decrypt(encrypted, key)