import pytest
from unittest.mock import MagicMock, patch
from src.db.models import Base, User, Session, engine, init_db, get_session


@pytest.fixture
def mock_engine():
    with patch("src.db.models.create_engine") as mock:
        yield mock


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.query.return_value.filter_by.return_value.first.return_value = None
    yield session


@pytest.fixture
def mock_db_connection(mock_engine):
    mock_connection = MagicMock()
    mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
    return mock_connection


def test_base_exists():
    assert Base is not None
    assert hasattr(Base, "metadata")


def test_user_model_has_required_fields():
    user = User(username="testuser", email="test@example.com")
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert hasattr(user, "id")
    assert hasattr(user, "created_at")


def test_user_model_str_repr():
    user = User(username="alice", email="alice@example.com")
    assert str(user) == "<User(id=None, username='alice', email='alice@example.com')>"


def test_engine_created(mock_engine):
    init_db()
    mock_engine.assert_called_once_with("sqlite:///app.db")


def test_init_db_creates_tables(mock_engine, mock_db_connection):
    with patch("src.db.models.Base.metadata.create_all") as mock_create_all:
        init_db()
        mock_create_all.assert_called_once_with(bind=mock_engine.return_value)


def test_get_session_yields_session(mock_engine, mock_session):
    with patch("src.db.models.Session", return_value=mock_session):
        with get_session() as session:
            assert session is mock_session
        mock_session.close.assert_called_once()


def test_get_session_closes_on_exception(mock_engine, mock_session):
    with patch("src.db.models.Session", return_value=mock_session):
        with pytest.raises(ValueError):
            with get_session() as session:
                raise ValueError("test error")
        mock_session.close.assert_called_once()