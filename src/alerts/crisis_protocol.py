"""Crisis protocol module for evaluating and escalating mental health alerts."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from src.database import db
from src.cache import redis_client
from src.models.alerts import Alert, AlertStatus, AlertPriority
from src.rules.engine import RuleEvaluator

logger = logging.getLogger(__name__)


class RuleEvaluationResult:
    """Result of evaluating a crisis rule against an alert."""

    def __init__(self, match: bool, priority: AlertPriority, details: Dict[str, Any] = None):
        self.match = match
        self.priority = priority
        self.details = details or {}


@dataclass
class CrisisRule:
    """Represents a clinician-defined crisis detection rule."""
    id: str
    name: str
    condition: Dict[str, Any]
    priority: AlertPriority
    escalation_path: List[str]
    active: bool = True
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


def get_active_rules() -> List[CrisisRule]:
    """Retrieve all active crisis rules from the database."""
    try:
        rules_data = db.get_active_rules()
        return [CrisisRule(**rule) for rule in rules_data]
    except Exception as e:
        logger.error(f"Failed to retrieve active rules: {e}")
        return []


def evaluate_crisis_rules(patient_id: str, alert: Optional[Alert] = None) -> List[RuleEvaluationResult]:
    """Evaluate all active crisis rules against patient data."""
    results = []
    rules = get_active_rules()

    if not rules:
        return results

    evaluator = RuleEvaluator()

    if alert is None:
        alert = db.get_latest_alert(patient_id)

    if alert is None:
        logger.warning(f"No alert data found for patient {patient_id}")
        return results

    for rule in rules:
        if not rule.active:
            continue

        try:
            evaluation = evaluator.evaluate(rule, alert)
            results.append(RuleEvaluationResult(
                match=evaluation.match,
                priority=evaluation.priority,
                details=evaluation.details
            ))
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.id}: {e}")
            continue

    return results


def escalate_alert(alert: Alert, escalation_level: int = 0) -> bool:
    """Escalate an alert through the defined escalation path."""
    try:
        rule = db.get_rule_by_id(alert.rule_id)
        if not rule or not rule.escalation_path:
            logger.warning(f"No escalation path defined for rule {alert.rule_id}")
            return False

        if escalation_level >= len(rule.escalation_path):
            logger.error(f"Escalation level {escalation_level} exceeds path length {len(rule.escalation_path)}")
            return False

        recipient = rule.escalation_path[escalation_level]
        notification = {
            "alert_id": alert.id,
            "patient_id": alert.patient_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "recipient": recipient,
            "level": escalation_level,
            "priority": alert.priority.value
        }

        redis_client.publish("crisis_escalation", notification)

        if escalation_level < len(rule.escalation_path) - 1:
            db.update_alert_status(alert.id, AlertStatus.ESCALATED)
            db.record_escalation(alert.id, recipient, escalation_level)
            return True
        else:
            db.update_alert_status(alert.id, AlertStatus.FINAL_ESCALATION)
            db.record_escalation(alert.id, recipient, escalation_level)
            return True

    except Exception as e:
        logger.error(f"Failed to escalate alert {alert.id}: {e}")
        return False


def process_alert(alert: Alert) -> List[RuleEvaluationResult]:
    """Process a new alert through the crisis protocol."""
    results = evaluate_crisis_rules(alert.patient_id, alert)

    for result in results:
        if result.match:
            alert.priority = max(alert.priority, result.priority)
            db.update_alert_status(alert.id, AlertStatus.ACTIVE)
            escalate_alert(alert)

    return results


def get_rule_by_id(rule_id: str) -> Optional[CrisisRule]:
    """Retrieve a specific crisis rule by ID."""
    try:
        rule_data = db.get_rule_by_id(rule_id)
        if rule_data:
            return CrisisRule(**rule_data)
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve rule {rule_id}: {e}")
        return None


def activate_rule(rule_id: str) -> bool:
    """Activate a previously inactive crisis rule."""
    try:
        db.activate_rule(rule_id)
        redis_client.delete("crisis_rules:active")
        return True
    except Exception as e:
        logger.error(f"Failed to activate rule {rule_id}: {e}")
        return False


def deactivate_rule(rule_id: str) -> bool:
    """Deactivate an active crisis rule."""
    try:
        db.deactivate_rule(rule_id)
        redis_client.delete("crisis_rules:active")
        return True
    except Exception as e:
        logger.error(f"Failed to deactivate rule {rule_id}: {e}")
        return False