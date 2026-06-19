import pytest
from unittest.mock import MagicMock, patch
from src.ml.model_loader import (
    load_model,
    get_model_version,
    get_latest_model_version,
    get_fallback_model,
    ModelNotFoundError,
    ModelVersionMismatchError,
)


@pytest.fixture
def mock_db():
    with patch("src.ml.model_loader.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.ml.model_loader.redis_client") as mock:
        yield mock


@pytest.fixture
def mock_model_registry():
    with patch("src.ml.model_loader.ModelRegistry") as mock:
        yield mock


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.version = "v1.2.3"
    model.path = "/models/v1.2.3"
    return model


def test_load_model_success(mock_db, mock_model):
    mock_db.get_model.return_value = mock_model
    with patch("src.ml.model_loader.load_from_path") as mock_load:
        mock_load.return_value = mock_model
        result = load_model("v1.2.3")
        assert result is mock_model
        assert result.version == "v1.2.3"
        mock_db.get_model.assert_called_once_with("v1.2.3")


def test_load_model_not_found(mock_db):
    mock_db.get_model.return_value = None
    with pytest.raises(ModelNotFoundError) as exc_info:
        load_model("v9.9.9")
    assert "Model not found: v9.9.9" in str(exc_info.value)


def test_load_model_fallback_on_error(mock_db, mock_model):
    mock_db.get_model.side_effect = Exception("DB connection failed")
    fallback_model = MagicMock()
    fallback_model.version = "fallback"
    with patch("src.ml.model_loader.get_fallback_model", return_value=fallback_model):
        result = load_model("v1.2.3")
        assert result is fallback_model
        assert result.version == "fallback"


def test_get_model_version_success(mock_db, mock_model):
    mock_db.get_model.return_value = mock_model
    version = get_model_version("v1.2.3")
    assert version == "v1.2.3"
    mock_db.get_model.assert_called_once_with("v1.2.3")


def test_get_model_version_not_found(mock_db):
    mock_db.get_model.return_value = None
    with pytest.raises(ModelNotFoundError):
        get_model_version("v0.0.0")


def test_get_latest_model_version_success(mock_db, mock_model):
    mock_db.get_latest_model.return_value = mock_model
    version = get_latest_model_version()
    assert version == "v1.2.3"
    mock_db.get_latest_model.assert_called_once()


def test_get_latest_model_version_empty(mock_db):
    mock_db.get_latest_model.return_value = None
    with pytest.raises(ModelNotFoundError):
        get_latest_model_version()


def test_get_fallback_model_success(mock_redis, mock_model):
    mock_redis.get.return_value = b"/models/fallback"
    with patch("src.ml.model_loader.load_from_path") as mock_load:
        mock_load.return_value = mock_model
        fallback = get_fallback_model()
        assert fallback is mock_model
        mock_redis.get.assert_called_once_with("ml:fallback_model_path")


def test_get_fallback_model_redis_not_found(mock_redis):
    mock_redis.get.return_value = None
    with pytest.raises(ModelNotFoundError):
        get_fallback_model()


def test_get_fallback_model_load_failure(mock_redis):
    mock_redis.get.return_value = b"/models/fallback"
    with patch("src.ml.model_loader.load_from_path") as mock_load:
        mock_load.side_effect = Exception("Corrupt model file")
        with pytest.raises(ModelNotFoundError):
            get_fallback_model()


def test_load_model_version_mismatch(mock_db, mock_model):
    mock_db.get_model.return_value = MagicMock(version="v1.2.3")
    with patch("src.ml.model_loader.load_from_path") as mock_load:
        mismatch_model = MagicMock(version="v1.2.4")
        mock_load.return_value = mismatch_model
        with pytest.raises(ModelVersionMismatchError) as exc_info:
            load_model("v1.2.3")
        assert "Expected version v1.2.3, got v1.2.4" in str(exc_info.value)


def test_get_fallback_model_redis_no(mock_redis):
    mock_redis.get.return_value = None
    with pytest.raises(ModelNotFoundError):
        get_fallback_model()