import pytest
from unittest.mock import MagicMock, patch
from src.consent.protocols import (
    ConsentProtocol,
    ConsentStage,
    ConsentFlow,
    validate_consent_stage,
    get_next_stage,
    record_consent,
    get_consent_status,
    IRB_COMPLIANT_STAGES,
)


@pytest.fixture
def mock_db():
    with patch("src.consent.protocols.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.consent.protocols.redis_client") as mock:
        yield mock


@pytest.fixture
def sample_consent_flow():
    return ConsentFlow(
        id="flow_001",
        name="Standard Consent Flow",
        stages=[
            ConsentStage(id="init", name="Initialization", required=True),
            ConsentStage(id="disclosure", name="Information Disclosure", required=True),
            ConsentStage(id="agreement", name="Agreement", required=True),
            ConsentStage(id="confirmation", name="Confirmation", required=True),
        ],
    )


@pytest.fixture
def sample_consent_protocol(sample_consent_flow):
    return ConsentProtocol(
        id="protocol_001",
        name="Basic Research Consent",
        flow=sample_consent_flow,
        version="1.0",
        irb_approved=True,
    )


def test_consent_stage_creation():
    stage = ConsentStage(id="test_stage", name="Test Stage", required=True)
    assert stage.id == "test_stage"
    assert stage.name == "Test Stage"
    assert stage.required is True
    assert stage.order is None


def test_consent_stage_with_order():
    stage = ConsentStage(id="stage1", name="First", required=True, order=1)
    assert stage.order == 1


def test_consent_flow_creation(sample_consent_flow):
    flow = sample_consent_flow
    assert flow.id == "flow_001"
    assert flow.name == "Standard Consent Flow"
    assert len(flow.stages) == 4
    assert all(isinstance(s, ConsentStage) for s in flow.stages)


def test_consent_flow_stages_ordered_by_id_by_default(sample_consent_flow):
    flow = ConsentFlow(
        id="flow_002",
        name="Unordered Flow",
        stages=[
            ConsentStage(id="zeta", name="Zeta"),
            ConsentStage(id="alpha", name="Alpha"),
        ],
    )
    stages = flow.get_stages()
    assert stages[0].id == "alpha"
    assert stages[1].id == "zeta"


def test_consent_flow_get_stage_by_id(sample_consent_flow):
    stage = sample_consent_flow.get_stage("disclosure")
    assert stage.id == "disclosure"
    assert stage.name == "Information Disclosure"


def test_consent_flow_get_stage_missing(sample_consent_flow):
    stage = sample_consent_flow.get_stage("nonexistent")
    assert stage is None


def test_consent_protocol_creation(sample_consent_protocol):
    protocol = sample_consent_protocol
    assert protocol.id == "protocol_001"
    assert protocol.name == "Basic Research Consent"
    assert protocol.version == "1.0"
    assert protocol.irb_approved is True
    assert isinstance(protocol.flow, ConsentFlow)


def test_validate_consent_stage_valid():
    stage = ConsentStage(id="agreement", name="Agreement", required=True)
    result = validate_consent_stage(stage)
    assert result is True


def test_validate_consent_stage_invalid_id():
    stage = ConsentStage(id="invalid_stage", name="Invalid", required=True)
    with pytest.raises(ValueError, match="Invalid stage ID"):
        validate_consent_stage(stage)


def test_validate_consent_stage_missing_required():
    stage = ConsentStage(id="agreement", name="Agreement", required=True)
    stage.required = False
    with pytest.raises(ValueError, match="Required stage 'agreement' must be marked required"):
        validate_consent_stage(stage)


def test_get_next_stage_current_first(sample_consent_flow):
    current = sample_consent_flow.get_stage("init")
    next_stage = get_next_stage(sample_consent_flow, current)
    assert next_stage.id == "disclosure"


def test_get_next_stage_current_middle(sample_consent_flow):
    current = sample_consent_flow.get_stage("disclosure")
    next_stage = get_next_stage(sample_consent_flow, current)
    assert next_stage.id == "agreement"


def test_get_next_stage_current_last(sample_consent_flow):
    current = sample_consent_flow.get_stage("confirmation")
    next_stage = get_next_stage(sample_consent_flow, current)
    assert next_stage is None


def test_get_next_stage_invalid_current(sample_consent_flow):
    current = ConsentStage(id="nonexistent", name="Nonexistent")
    with pytest.raises(ValueError, match="Current stage not found in flow"):
        get_next_stage(sample_consent_flow, current)


def test_record_consent_success(mock_db, mock_redis):
    mock_db.execute.return_value = None
    mock_redis.setex.return_value = True

    result = record_consent(
        protocol_id="protocol_001",
        user_id="user_123",
        stage_id="agreement",
        timestamp="2024-01-01T12:00:00Z",
    )

    assert result is True
    mock_db.execute.assert_called_once()
    mock_redis.setex.assert_called_once()


def test_record_consent_db_failure(mock_db, mock_redis):
    mock_db.execute.side_effect = Exception("DB error")

    with pytest.raises(Exception, match="DB error"):
        record_consent(
            protocol_id="protocol_001",
            user_id="user_123",
            stage_id="agreement",
            timestamp="2024-01-01T12:00:00Z",
        )


def test_get_consent_status_complete(mock_db, mock_redis):
    mock_redis.get.return_value = b"1"
    mock_db.fetchone.return_value = ("user_123", "protocol_001", "agreement", "2024-01-01T12:00:00Z")

    status = get_consent_status(protocol_id="protocol_001", user_id="user_123")

    assert status["completed"] is True
    assert status["current_stage"] == "agreement"
    mock_redis.get.assert_called()
    mock_db.fetchone.assert_called()


def test_get_consent_status_incomplete(mock_db, mock_redis):
    mock_redis.get.return_value = None
    mock_db.fetchone.return_value = None

    status = get_consent_status(protocol_id="protocol_001", user_id="user_123")

    assert status["completed"] is False
    assert status["current_stage"] is None
    mock_redis.get.assert_called()
    mock_db.fetchone.assert_called()


def test_irb_compliant_stages():
    assert isinstance(IRB_COMPLIANT_STAGES, list)
    assert len(IRB_COMPLIANT_STAGES) > 0
    assert "agreement" in IRB_COMPLIANT_STAGES
    assert "disclosure" in IRB_COMPLIANT_STAGES