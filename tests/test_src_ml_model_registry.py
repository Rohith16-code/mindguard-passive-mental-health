import pytest
from unittest.mock import MagicMock, patch
from src.ml.model_registry import ModelRegistry, ModelVersion, ModelNotFoundError, InvalidVersionError


@pytest.fixture
def mock_db():
    with patch('src.ml.model_registry.Database') as MockDB:
        db_instance = MagicMock()
        MockDB.return_value = db_instance
        yield db_instance


@pytest.fixture
def mock_redis():
    with patch('src.ml.model_registry.Redis') as MockRedis:
        redis_instance = MagicMock()
        MockRedis.return_value = redis_instance
        yield redis_instance


@pytest.fixture
def registry(mock_db, mock_redis):
    return ModelRegistry()


class TestModelVersion:
    def test_model_version_creation(self):
        version = ModelVersion(version="1.0.0", path="/models/v1", metadata={"accuracy": 0.95})
        assert version.version == "1.0.0"
        assert version.path == "/models/v1"
        assert version.metadata == {"accuracy": 0.95}

    def test_model_version_equality(self):
        v1 = ModelVersion("1.0.0", "/models/v1", {"acc": 0.9})
        v2 = ModelVersion("1.0.0", "/models/v1", {"acc": 0.9})
        v3 = ModelVersion("1.0.1", "/models/v1", {"acc": 0.9})
        assert v1 == v2
        assert v1 != v3

    def test_model_version_hash(self):
        v1 = ModelVersion("1.0.0", "/models/v1", {"acc": 0.9})
        v2 = ModelVersion("1.0.0", "/models/v1", {"acc": 0.9})
        s = {v1, v2}
        assert len(s) == 1


class TestModelRegistry:
    def test_init_initializes_db_and_redis(self, mock_db, mock_redis):
        registry = ModelRegistry()
        mock_db.assert_called_once()
        mock_redis.assert_called_once()

    def test_register_model_success(self, registry, mock_db, mock_redis):
        registry.register_model("model_a", "1.0.0", "/models/model_a_v1", {"acc": 0.95})
        mock_db.save_model_version.assert_called_once_with(
            "model_a", ModelVersion("1.0.0", "/models/model_a_v1", {"acc": 0.95})
        )
        mock_redis.set.assert_called_once_with(
            "model_registry:latest:model_a", "1.0.0"
        )

    def test_register_model_invalid_version(self, registry):
        with pytest.raises(InvalidVersionError):
            registry.register_model("model_a", "invalid", "/path", {})

    def test_get_model_version_success(self, registry, mock_db):
        expected = ModelVersion("1.0.0", "/models/model_a_v1", {"acc": 0.95})
        mock_db.get_model_version.return_value = expected

        result = registry.get_model_version("model_a", "1.0.0")
        assert result == expected
        mock_db.get_model_version.assert_called_once_with("model_a", "1.0.0")

    def test_get_model_version_not_found(self, registry, mock_db):
        mock_db.get_model_version.return_value = None
        with pytest.raises(ModelNotFoundError):
            registry.get_model_version("model_a", "1.0.0")

    def test_get_latest_version_success(self, registry, mock_db, mock_redis):
        mock_redis.get.return_value = b"2.0.0"
        expected = ModelVersion("2.0.0", "/models/model_a_v2", {"acc": 0.97})
        mock_db.get_model_version.return_value = expected

        result = registry.get_latest_version("model_a")
        assert result == expected
        mock_redis.get.assert_called_once_with("model_registry:latest:model_a")
        mock_db.get_model_version.assert_called_once_with("model_a", "2.0.0")

    def test_get_latest_version_redis_miss(self, registry, mock_db):
        mock_redis.get.return_value = None
        mock_db.get_latest_version.return_value = ModelVersion("1.5.0", "/models/model_a_v1.5", {})
        result = registry.get_latest_version("model_a")
        assert result.version == "1.5.0"
        mock_db.get_latest_version.assert_called_once_with("model_a")

    def test_get_latest_version_not_found(self, registry, mock_db):
        mock_db.get_latest_version.return_value = None
        with pytest.raises(ModelNotFoundError):
            registry.get_latest_version("model_a")

    def test_list_versions(self, registry, mock_db):
        versions = [
            ModelVersion("1.0.0", "/models/model_a_v1", {}),
            ModelVersion("1.1.0", "/models/model_a_v1.1", {}),
        ]
        mock_db.list_versions.return_value = versions

        result = registry.list_versions("model_a")
        assert result == versions
        mock_db.list_versions.assert_called_once_with("model_a")

    def test_delete_version_success(self, registry, mock_db, mock_redis):
        registry.delete_version("model_a", "1.0.0")
        mock_db.delete_version.assert_called_once_with("model_a", "1.0.0")
        mock_redis.delete.assert_called_once_with("model_registry:latest:model_a")

    def test_delete_latest_version_clears_cache(self, registry, mock_db, mock_redis):
        mock_db.get_latest_version.return_value = ModelVersion("1.0.0", "/path", {})
        registry.delete_version("model_a", "1.0.0")
        mock_redis.delete.assert_called_once_with("model_registry:latest:model_a")

    def test_select_model_success(self, registry, mock_db, mock_redis):
        mock_redis.get.return_value = b"2.0.0"
        mock_db.get_model_version.return_value = ModelVersion("2.0.0", "/models/model_a_v2", {})
        result = registry.select_model("model_a")
        assert result.version == "2.0.0"
        assert result.path == "/models/model_a_v2"

    def test_select_model_no_latest(self, registry, mock_db):
        mock_redis.get.return_value = None
        mock_db.get_latest_version.return_value = None
        with pytest.raises(ModelNotFoundError):
            registry.select_model("model_a")