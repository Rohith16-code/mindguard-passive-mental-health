"""Shared pytest fixtures."""

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_db():
    """Return a mock database session."""
    return MagicMock()

@pytest.fixture
def mock_redis():
    """Return a mock Redis client."""
    return MagicMock()

@pytest.fixture
def sample_patient_data():
    """Return sample patient data for tests."""
    return {
        "vitals": {"heart_rate": 72, "bp_systolic": 120, "bp_diastolic": 80},
        "symptoms": {"fever": "yes", "chest_pain": "no"},
        "history": {"diabetes": True, "smoker": False}
    }
