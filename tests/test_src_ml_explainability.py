import pytest
from unittest.mock import MagicMock, patch
from src.ml.explainability import *

@pytest.fixture
def mock_redis_client():
    with patch('src.ml.explainability.redis.Redis') as mock_redis:
        yield mock_redis.return_value

@pytest.fixture
def mock_db_session():
    with patch('src.ml.explainability.Session') as mock_session:
        yield mock_session.return_value

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0.8]
    model.feature_importances_ = [0.3, 0.5, 0.2]
    return model

@pytest.fixture
def sample_features():
    return [10.0, 20.0, 5.0]

@pytest.fixture
def sample_alert():
    return {
        'alert_id': 'alert_123',
        'features': [10.0, 20.0, 5.0],
        'timestamp': '2024-01-01T00:00:00Z',
        'model_id': 'model_v1'
    }

def test_compute_shap_values_returns_valid_attribution(mock_model, sample_features):
    with patch('src.ml.explainability.shap.Explainer') as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = [[0.1, 0.6, 0.3]]
        result = compute_shap_values(mock_model, sample_features)
        assert len(result) == len(sample_features)
        assert all(isinstance(v, float) for v in result)
        assert sum(result) > 0

def test_compute_shap_values_handles_empty_features(mock_model):
    with patch('src.ml.explainability.shap.Explainer') as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = [[]]
        result = compute_shap_values(mock_model, [])
        assert result == []

def test_store_attribution_saves_to_db_and_redis(mock_db_session, mock_redis_client, sample_alert):
    attribution = [0.1, 0.6, 0.3]
    with patch('src.ml.explainability.datetime') as mock_dt:
        mock_dt.utcnow.return_value.isoformat.return_value = '2024-01-01T00:00:00Z'
        store_attribution(sample_alert, attribution)
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_redis_client.set.assert_called_once()

def test_get_top_features_returns_sorted_indices():
    attribution = [0.1, 0.6, 0.3]
    result = get_top_features(attribution, top_k=2)
    assert result == [1, 2]

def test_get_top_features_handles_ties():
    attribution = [0.5, 0.5, 0.1]
    result = get_top_features(attribution, top_k=2)
    assert len(result) == 2
    assert set(result) == {0, 1}

def test_get_top_features_with_zero_attribution():
    attribution = [0.0, 0.0, 0.0]
    result = get_top_features(attribution, top_k=2)
    assert result == [0, 1]

def test_get_top_features_k_larger_than_attribution_length():
    attribution = [0.1, 0.2]
    result = get_top_features(attribution, top_k=5)
    assert result == [1, 0]

def test_get_top_features_negative_attribution():
    attribution = [-0.1, 0.5, -0.3]
    result = get_top_features(attribution, top_k=2)
    assert result == [1, 0]

def test_generate_explanation_returns_dict(mock_model, sample_features):
    with patch('src.ml.explainability.compute_shap_values') as mock_shap:
        mock_shap.return_value = [0.1, 0.6, 0.3]
        explanation = generate_explanation(mock_model, sample_features)
        assert isinstance(explanation, dict)
        assert 'attribution' in explanation
        assert 'top_features' in explanation
        assert explanation['top_features'] == [1, 2]

def test_generate_explanation_handles_model_without_importances():
    model = MagicMock()
    model.predict.return_value = [0.8]
    model.feature_importances_ = None
    with patch('src.ml.explainability.compute_shap_values') as mock_shap:
        mock_shap.return_value = [0.2, 0.5, 0.3]
        explanation = generate_explanation(model, [1.0, 2.0, 3.0])
        assert 'top_features' in explanation

def test_process_alert_calls_all_components(mock_model, mock_db_session, mock_redis_client, sample_alert):
    with patch('src.ml.explainability.load_model') as mock_load_model, \
         patch('src.ml.explainability.generate_explanation') as mock_gen_exp:
        mock_load_model.return_value = mock_model
        mock_gen_exp.return_value = {'attribution': [0.1, 0.6, 0.3], 'top_features': [1, 2]}
        result = process_alert(sample_alert)
        mock_load_model.assert_called_once_with(sample_alert['model_id'])
        mock_gen_exp.assert_called_once_with(mock_model, sample_alert['features'])
        assert result['attribution'] == [0.1, 0.6, 0.3]
        assert result['top_features'] == [1, 2]

def test_process_alert_handles_model_load_failure(sample_alert):
    with patch('src.ml.explainability.load_model') as mock_load_model:
        mock_load_model.side_effect = FileNotFoundError("Model not found")
        with pytest.raises(FileNotFoundError):
            process_alert(sample_alert)

def test_process_alert_handles_empty_features(sample_alert):
    sample_alert['features'] = []
    with patch('src.ml.explainability.load_model') as mock_load_model:
        mock_load_model.return_value = MagicMock()
        with pytest.raises(ValueError):
            process_alert(sample_alert)