import pytest
from unittest.mock import MagicMock, patch
from src.alerts.crisis_protocol import (
    evaluate_crisis_rules,
    escalate_alert,
    get_active_rules,
    CrisisRule,
    AlertPriority,
    AlertStatus,
    Alert,
    RuleEvaluationResult,
)


@pytest.fixture
def mock_db():
    with patch("src.alerts.crisis_protocol.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.alerts.crisis_protocol.redis_client") as mock:
        yield mock


@pytest.fixture
def sample_rule():
    return CrisisRule(
        id="rule_001",
        name="Critical BP Drop",
        condition={"vital_sign": "bp_systolic", "operator": "<", "threshold": 80},
        priority=AlertPriority.HIGH,
        escalation_path=["nurse", "physician"],
        active=True,
    )


@pytest.fixture
def sample_alert():
    return Alert(
        id="alert_123",
        patient_id="patient_456",
        timestamp="2024-05-10T12:00:00Z",
        vitals={"bp_systolic": 75, "heart_rate": 110},
        status=AlertStatus.PENDING,
    )


@pytest.fixture
def mock_rule_engine():
    with patch("src.alerts.crisis_protocol.RuleEvaluator") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


def test_evaluate_crisis_rules_no_active_rules(mock_db, mock_rule_engine):
    mock_db.get_active_rules.return_value = []
    result = evaluate_crisis_rules("patient_789")
    assert result == []


def test_evaluate_crisis_rules_single_rule_match(mock_db, mock_rule_engine, sample_rule, sample_alert):
    mock_db.get_active_rules.return_value = [sample_rule]
    mock_rule_engine.evaluate.return_value = RuleEvaluationResult(match=True, priority=AlertPriority.HIGH)

    result = evaluate_crisis_rules(sample_alert.patient_id, sample_alert)

    assert len(result) == 1
    assert result[0].match is True
    assert result[0].priority == AlertPriority.HIGH
    mock_rule_engine.evaluate.assert_called_once_with(sample_rule, sample_alert)


def test_evaluate_crisis_rules_multiple_rules_mixed_match(mock_db, mock_rule_engine, sample_alert):
    rule1 = CrisisRule(
        id="rule_001",
        name="Critical BP Drop",
        condition={"vital_sign": "bp_systolic", "operator": "<", "threshold": 80},
        priority=AlertPriority.HIGH,
        escalation_path=["nurse"],
        active=True,
    )
    rule2 = CrisisRule(
        id="rule_002",
        name="High Heart Rate",
        condition={"vital_sign": "heart_rate", "operator": ">", "threshold": 130},
        priority=AlertPriority.CRITICAL,
        escalation_path=["physician", "code_team"],
        active=True,
    )

    mock_db.get_active_rules.return_value = [rule1, rule2]
    mock_rule_engine.evaluate.side_effect = [
        RuleEvaluationResult(match=True, priority=AlertPriority.HIGH),
        RuleEvaluationResult(match=False, priority=None),
    ]

    result = evaluate_crisis_rules(sample_alert.patient_id, sample_alert)

    assert len(result) == 2
    assert result[0].match is True
    assert result[1].match is False


def test_evaluate_crisis_rules_no_vitals_provided(mock_db, mock_rule_engine, sample_rule):
    mock_db.get_active_rules.return_value = [sample_rule]
    alert = Alert(
        id="alert_999",
        patient_id="patient_456",
        timestamp="2024-05-10T12:00:00Z",
        vitals={},
        status=AlertStatus.PENDING,
    )
    result = evaluate_crisis_rules(alert.patient_id, alert)
    assert len(result) == 1
    assert result[0].match is False


def test_escalate_alert_success(mock_db, mock_redis, sample_rule, sample_alert):
    mock_db.get_active_rules.return_value = [sample_rule]
    mock_redis.setex.return_value = True

    result = escalate_alert(sample_alert, sample_rule)

    assert result is True
    mock_redis.setex.assert_called_once()
    call_args = mock_redis.setex.call_args
    assert "alert_123" in call_args[0][0]
    assert "nurse" in call_args[0][1] or "physician" in call_args[0][1]


def test_escalate_alert_no_escalation_path(mock_db, sample_alert):
    rule = CrisisRule(
        id="rule_003",
        name="No Escalation",
        condition={"vital_sign": "bp_systolic", "operator": "<", "threshold": 90},
        priority=AlertPriority.LOW,
        escalation_path=[],
        active=True,
    )
    result = escalate_alert(sample_alert, rule)
    assert result is False


def test_get_active_rules(mock_db, sample_rule):
    mock_db.get_active_rules.return_value = [sample_rule]
    rules = get_active_rules()
    assert len(rules) == 1
    assert rules[0].id == "rule_001"
    assert rules[0].active is True


def test_get_active_rules_empty(mock_db):
    mock_db.get_active_rules.return_value = []
    rules = get_active_rules()
    assert rules == []


def test_escalate_alert_redis_failure(mock_db, mock_redis, sample_rule, sample_alert):
    mock_db.get_active_rules.return_value = [sample_rule]
    mock_redis.setex.side_effect = Exception("Redis unavailable")

    with pytest.raises(Exception, match="Redis unavailable"):
        escalate_alert(sample_alert, sample_rule)