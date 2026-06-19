import pytest
from unittest.mock import MagicMock, patch
from src.ml.model_validator import ModelValidator, ModelDriftDetected, RollbackFailed


@pytest.fixture
def mock_db():
    with patch('src.ml.model_validator.DBClient') as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_redis():
    with patch('src.ml.model_validator.RedisClient') as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def validator(mock_db, mock_redis):
    return ModelValidator(db_client=mock_db, redis_client=mock_redis)


class TestModelValidator:
    def test_init_success(self, mock_db, mock_redis):
        validator = ModelValidator(db_client=mock_db, redis_client=mock_redis)
        assert validator.db_client is mock_db
        assert validator.redis_client is mock_redis

    def test_validate_model_no_drift(self, validator, mock_db, mock_redis):
        mock_db.get_model_metrics.return_value = {'accuracy': 0.92, 'latency_ms': 45}
        mock_db.get_baseline_metrics.return_value = {'accuracy': 0.90, 'latency_ms': 50}
        mock_redis.get_model_version.return_value = 'v1.2.3'

        result = validator.validate_model('model_v1')

        assert result['status'] == 'ok'
        assert result['drift_detected'] is False
        assert result['model_version'] == 'v1.2.3'

    def test_validate_model_drift_detected_accuracy(self, validator, mock_db, mock_redis):
        mock_db.get_model_metrics.return_value = {'accuracy': 0.85, 'latency_ms': 45}
        mock_db.get_baseline_metrics.return_value = {'accuracy': 0.90, 'latency_ms': 50}
        mock_redis.get_model_version.return_value = 'v1.2.3'

        with pytest.raises(ModelDriftDetected) as exc_info:
            validator.validate_model('model_v1')

        assert exc_info.value.args[0] == 'Model accuracy dropped below threshold (0.85 < 0.90)'
        assert exc_info.value.drift_type == 'accuracy'

    def test_validate_model_drift_detected_latency(self, validator, mock_db, mock_redis):
        mock_db.get_model_metrics.return_value = {'accuracy': 0.92, 'latency_ms': 70}
        mock_db.get_baseline_metrics.return_value = {'accuracy': 0.90, 'latency_ms': 50}
        mock_redis.get_model_version.return_value = 'v1.2.3'

        with pytest.raises(ModelDriftDetected) as exc_info:
            validator.validate_model('model_v1')

        assert exc_info.value.drift_type == 'latency'

    def test_validate_model_missing_baseline(self, validator, mock_db, mock_redis):
        mock_db.get_model_metrics.return_value = {'accuracy': 0.92}
        mock_db.get_baseline_metrics.return_value = None

        with pytest.raises(ModelDriftDetected) as exc_info:
            validator.validate_model('model_v1')

        assert exc_info.value.drift_type == 'baseline_missing'

    def test_rollback_model_success(self, validator, mock_db, mock_redis):
        mock_db.get_model_version_history.return_value = ['v1.2.2', 'v1.2.1']
        mock_db.load_model.return_value = MagicMock()
        mock_redis.set_model_version.return_value = True

        result = validator.rollback_model('model_v1', target_version='v1.2.2')

        assert result['status'] == 'success'
        assert result['rolled_back_to'] == 'v1.2.2'
        mock_db.load_model.assert_called_once_with('model_v1', 'v1.2.2')
        mock_redis.set_model_version.assert_called_once_with('model_v1', 'v1.2.2')

    def test_rollback_model_no_history(self, validator, mock_db, mock_redis):
        mock_db.get_model_version_history.return_value = []

        with pytest.raises(RollbackFailed) as exc_info:
            validator.rollback_model('model_v1')

        assert exc_info.value.args[0] == 'No version history available for rollback'

    def test_rollback_model_db_load_failure(self, validator, mock_db, mock_redis):
        mock_db.get_model_version_history.return_value = ['v1.2.2']
        mock_db.load_model.side_effect = Exception('Model file corrupted')

        with pytest.raises(RollbackFailed) as exc_info:
            validator.rollback_model('model_v1', target_version='v1.2.2')

        assert 'Failed to load model' in exc_info.value.args[0]

    def test_rollback_model_redis_update_failure(self, validator, mock_db, mock_redis):
        mock_db.get_model_version_history.return_value = ['v1.2.2']
        mock_db.load_model.return_value = MagicMock()
        mock_redis.set_model_version.return_value = False

        with pytest.raises(RollbackFailed) as exc_info:
            validator.rollback_model('model_v1', target_version='v1.2.2')

        assert 'Redis update failed' in exc_info.value.args[0]

    def test_get_model_status(self, validator, mock_db, mock_redis):
        mock_db.get_model_metrics.return_value = {'accuracy': 0.92}
        mock_db.get_baseline_metrics.return_value = {'accuracy': 0.90}
        mock_redis.get_model_version.return_value = 'v1.2.3'

        status = validator.get_model_status('model_v1')

        assert status['model_version'] == 'v1.2.3'
        assert status['current_metrics'] == {'accuracy': 0.92}
        assert status['baseline_metrics'] == {'accuracy': 0.90}
        assert status['drift_status'] == 'ok'

    def test_get_model_status_no_baseline(self, validator, mock_db, mock_redis):
        mock_db.get_model_metrics.return_value = {'accuracy': 0.92}
        mock_db.get_baseline_metrics.return_value = None
        mock_redis.get_model_version.return_value = 'v1.2.3'

        status = validator.get_model_status('model_v1')

        assert status['drift_status'] == 'baseline_missing'