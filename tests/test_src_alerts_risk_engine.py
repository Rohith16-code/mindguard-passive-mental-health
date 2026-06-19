import pytest
from unittest.mock import MagicMock, patch
from src.alerts.risk_engine import (
    evaluate_threshold,
    trigger_soft_check,
    RiskEngine,
    ThresholdConfig,
    SoftCheckConfig,
)


@pytest.fixture
def mock_db():
    with patch("src.alerts.risk_engine.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.alerts.risk_engine.redis") as mock:
        yield mock


@pytest.fixture
def threshold_config():
    return ThresholdConfig(
        metric="latency_ms",
        threshold=100.0,
        op="gt",
        window_seconds=60,
    )


@pytest.fixture
def soft_check_config():
    return SoftCheckConfig(
        metric="error_rate",
        soft_threshold=0.05,
        hard_threshold=0.1,
        window_seconds=300,
    )


class TestEvaluateThreshold:
    def test_evaluate_threshold_gt_exceeded(self, threshold_config, mock_db):
        threshold_config.op = "gt"
        threshold_config.threshold = 100.0
        mock_db.get_metric_avg.return_value = 120.5
        result = evaluate_threshold(threshold_config)
        assert result is True

    def test_evaluate_threshold_gt_not_exceeded(self, threshold_config, mock_db):
        threshold_config.op = "gt"
        threshold_config.threshold = 100.0
        mock_db.get_metric_avg.return_value = 95.0
        result = evaluate_threshold(threshold_config)
        assert result is False

    def test_evaluate_threshold_lt_exceeded(self, threshold_config, mock_db):
        threshold_config.op = "lt"
        threshold_config.threshold = 50.0
        mock_db.get_metric_avg.return_value = 30.0
        result = evaluate_threshold(threshold_config)
        assert result is True

    def test_evaluate_threshold_eq_exceeded(self, threshold_config, mock_db):
        threshold_config.op = "eq"
        threshold_config.threshold = 100.0
        mock_db.get_metric_avg.return_value = 100.0
        result = evaluate_threshold(threshold_config)
        assert result is True

    def test_evaluate_threshold_invalid_op(self, threshold_config, mock_db):
        threshold_config.op = "invalid_op"
        mock_db.get_metric_avg.return_value = 100.0
        with pytest.raises(ValueError, match="Unsupported operator"):
            evaluate_threshold(threshold_config)

    def test_evaluate_threshold_db_error(self, threshold_config, mock_db):
        mock_db.get_metric_avg.side_effect = Exception("DB connection failed")
        with pytest.raises(Exception, match="DB connection failed"):
            evaluate_threshold(threshold_config)


class TestTriggerSoftCheck:
    def test_trigger_soft_check_soft_only(self, soft_check_config, mock_db, mock_redis):
        mock_db.get_metric_avg.return_value = 0.07
        mock_redis.get.return_value = None
        result = trigger_soft_check(soft_check_config)
        assert result == "soft"
        mock_redis.set.assert_called_once()

    def test_trigger_soft_check_hard(self, soft_check_config, mock_db, mock_redis):
        mock_db.get_metric_avg.return_value = 0.15
        mock_redis.get.return_value = None
        result = trigger_soft_check(soft_check_config)
        assert result == "hard"
        mock_redis.set.assert_called_once()

    def test_trigger_soft_check_normal(self, soft_check_config, mock_db, mock_redis):
        mock_db.get_metric_avg.return_value = 0.03
        mock_redis.get.return_value = None
        result = trigger_soft_check(soft_check_config)
        assert result == "normal"
        mock_redis.set.assert_not_called()

    def test_trigger_soft_check_redis_error(self, soft_check_config, mock_db, mock_redis):
        mock_db.get_metric_avg.return_value = 0.07
        mock_redis.get.side_effect = Exception("Redis unavailable")
        with pytest.raises(Exception, match="Redis unavailable"):
            trigger_soft_check(soft_check_config)

    def test_trigger_soft_check_db_error(self, soft_check_config, mock_db, mock_redis):
        mock_db.get_metric_avg.side_effect = Exception("DB error")
        with pytest.raises(Exception, match="DB error"):
            trigger_soft_check(soft_check_config)


class TestRiskEngine:
    def test_risk_engine_init(self):
        engine = RiskEngine(
            thresholds=[ThresholdConfig(metric="cpu", threshold=80.0, op="gt")],
            soft_checks=[SoftCheckConfig(metric="error_rate", soft_threshold=0.05, hard_threshold=0.1)],
        )
        assert engine.thresholds is not None
        assert engine.soft_checks is not None

    @patch("src.alerts.risk_engine.evaluate_threshold")
    @patch("src.alerts.risk_engine.trigger_soft_check")
    def test_risk_engine_run_all_checks(
        self, mock_trigger_soft, mock_eval_thresh, mock_db, mock_redis
    ):
        mock_eval_thresh.return_value = True
        mock_trigger_soft.return_value = "soft"
        engine = RiskEngine(
            thresholds=[ThresholdConfig(metric="cpu", threshold=80.0, op="gt")],
            soft_checks=[SoftCheckConfig(metric="error_rate", soft_threshold=0.05, hard_threshold=0.1)],
        )
        results = engine.run_all_checks()
        assert results["thresholds"]["cpu"] is True
        assert results["soft_checks"]["error_rate"] == "soft"
        mock_eval_thresh.assert_called_once()
        mock_trigger_soft.assert_called_once()

    def test_risk_engine_run_all_checks_empty(self):
        engine = RiskEngine(thresholds=[], soft_checks=[])
        results = engine.run_all_checks()
        assert results["thresholds"] == {}
        assert results["soft_checks"] == {}

    def test_risk_engine_run_all_checks_db_error(self, mock_db, mock_redis):
        mock_db.get_metric_avg.side_effect = Exception("DB error")
        engine = RiskEngine(
            thresholds=[ThresholdConfig(metric="cpu", threshold=80.0, op="gt")],
            soft_checks=[],
        )
        with pytest.raises(Exception, match="DB error"):
            engine.run_all_checks()