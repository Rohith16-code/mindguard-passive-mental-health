import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, date
from pydantic import ValidationError
from src.api.schemas import UserCreate, UserResponse, ItemCreate, ItemResponse, TokenData


@pytest.fixture
def valid_user_create_data():
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecurePass123!",
        "full_name": "Test User"
    }


@pytest.fixture
def valid_user_response_data():
    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "is_active": True,
        "created_at": "2023-01-01T12:00:00"
    }


@pytest.fixture
def valid_item_create_data():
    return {
        "name": "Widget",
        "description": "A useful widget",
        "price": 19.99,
        "tax": 1.50,
        "owner_id": 1
    }


@pytest.fixture
def valid_item_response_data():
    return {
        "id": 42,
        "name": "Widget",
        "description": "A useful widget",
        "price": 19.99,
        "tax": 1.50,
        "owner_id": 1,
        "created_at": "2023-06-15T08:30:00"
    }


@pytest.fixture
def valid_token_data():
    return {
        "sub": "user123",
        "scopes": ["read", "write"]
    }


def test_user_create_valid(valid_user_create_data):
    user = UserCreate(**valid_user_create_data)
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"
    assert user.password == "SecurePass123!"


def test_user_create_minimal():
    user = UserCreate(username="minimal", email="minimal@example.com", password="Pass123!")
    assert user.username == "minimal"
    assert user.email == "minimal@example.com"
    assert user.full_name is None


def test_user_create_invalid_email():
    with pytest.raises(ValidationError) as exc_info:
        UserCreate(username="test", email="not-an-email", password="Pass123!")
    assert "email" in str(exc_info.value)


def test_user_create_short_password():
    with pytest.raises(ValidationError) as exc_info:
        UserCreate(username="test", email="test@example.com", password="short")
    assert "password" in str(exc_info.value)


def test_user_response_valid(valid_user_response_data):
    user = UserResponse(**valid_user_response_data)
    assert user.id == 1
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)


def test_user_response_created_at_string():
    data = valid_user_response_data()
    data["created_at"] = "2023-01-01T12:00:00"
    user = UserResponse(**data)
    assert isinstance(user.created_at, datetime)


def test_item_create_valid(valid_item_create_data):
    item = ItemCreate(**valid_item_create_data)
    assert item.name == "Widget"
    assert item.description == "A useful widget"
    assert item.price == 19.99
    assert item.tax == 1.50
    assert item.owner_id == 1


def test_item_create_optional_fields():
    item = ItemCreate(name="Minimal", price=9.99)
    assert item.name == "Minimal"
    assert item.description is None
    assert item.tax == 0.0
    assert item.owner_id is None


def test_item_response_valid(valid_item_response_data):
    item = ItemResponse(**valid_item_response_data)
    assert item.id == 42
    assert item.name == "Widget"
    assert item.price == 19.99
    assert isinstance(item.created_at, datetime)


def test_item_response_created_at_string():
    data = valid_item_response_data()
    data["created_at"] = "2023-06-15T08:30:00"
    item = ItemResponse(**data)
    assert isinstance(item.created_at, datetime)


def test_token_data_valid(valid_token_data):
    token = TokenData(**valid_token_data)
    assert token.sub == "user123"
    assert token.scopes == ["read", "write"]


def test_token_data_empty_scopes():
    token = TokenData(sub="user123")
    assert token.sub == "user123"
    assert token.scopes == []


def test_token_data_missing_sub():
    with pytest.raises(ValidationError) as exc_info:
        TokenData(scopes=["read"])
    assert "sub" in str(exc_info.value)


@patch("src.api.schemas.datetime")
def test_user_response_created_at_now(mock_datetime):
    mock_now = datetime(2023, 7, 4, 10, 0, 0)
    mock_datetime.now.return_value = mock_now
    mock_datetime.fromisoformat.side_effect = lambda x: datetime.fromisoformat(x)

    data = valid_user_response_data()
    data["created_at"] = None
    user = UserResponse(**data)
    assert user.created_at == mock_now


@patch("src.api.schemas.datetime")
def test_item_response_created_at_now(mock_datetime):
    mock_now = datetime(2023, 7, 4, 10, 0, 0)
    mock_datetime.now.return_value = mock_now
    mock_datetime.fromisoformat.side_effect = lambda x: datetime.fromisoformat(x)

    data = valid_item_response_data()
    data["created_at"] = None
    item = ItemResponse(**data)
    assert item.created_at == mock_now


def test_user_response_extra_fields_forbidden():
    data = valid_user_response_data()
    data["extra_field"] = "should fail"
    with pytest.raises(ValidationError):
        UserResponse(**data)


def test_item_response_extra_fields_forbidden():
    data = valid_item_response_data()
    data["extra_field"] = "should fail"
    with pytest.raises(ValidationError):
        ItemResponse(**data)