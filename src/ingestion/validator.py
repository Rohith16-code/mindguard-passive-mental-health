"""Data sanitization and format validation utilities for mental health crisis detection."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from dateutil import parser as date_parser
from dateutil.parser import ParserError


def sanitize_string(value: Any, max_length: int = 255) -> Optional[str]:
    """Sanitize string values by stripping whitespace and enforcing length limits."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    sanitized = value.strip()
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def sanitize_number(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Optional[float]:
    """Sanitize numeric values by converting to float and enforcing bounds."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(',', '').strip()
        numeric_value = float(value)
        if min_val is not None and numeric_value < min_val:
            numeric_value = min_val
        if max_val is not None and numeric_value > max_val:
            numeric_value = max_val
        return numeric_value
    except (ValueError, TypeError):
        return None


def sanitize_timestamp(value: Any) -> Optional[str]:
    """Sanitize timestamp values by parsing to ISO 8601 format."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            dt = date_parser.parse(value)
            return dt.isoformat()
        except (ParserError, TypeError):
            return None
    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(value)
            return dt.isoformat()
        except (OSError, OverflowError, TypeError):
            return None
    return None


def validate_schema(schema: Dict[str, Type]) -> bool:
    """Validate that schema contains only supported type definitions."""
    supported_types = {str, int, float, bool, list, dict, type(None)}
    for field_name, field_type in schema.items():
        if field_type not in supported_types:
            return False
        if not isinstance(field_name, str) or not field_name.strip():
            return False
    return True


def validate_record(record: Dict[str, Any], schema: Dict[str, Type]) -> Dict[str, Any]:
    """Validate and sanitize a record against a schema."""
    result = {
        "is_valid": True,
        "sanitized_record": {},
        "errors": []
    }

    if not isinstance(record, dict):
        result["is_valid"] = False
        result["errors"].append("Record must be a dictionary")
        return result

    if not validate_schema(schema):
        result["is_valid"] = False
        result["errors"].append("Invalid schema definition")
        return result

    for field, expected_type in schema.items():
        if field not in record:
            result["errors"].append(f"Missing required field: {field}")
            result["is_valid"] = False
            continue

        value = record[field]

        if expected_type == str:
            sanitized = sanitize_string(value)
            if sanitized is None:
                result["errors"].append(f"Field '{field}' must be a valid string")
                result["is_valid"] = False
            else:
                result["sanitized_record"][field] = sanitized

        elif expected_type == int:
            try:
                if isinstance(value, str):
                    value = value.replace(',', '').strip()
                int_val = int(float(value))
                result["sanitized_record"][field] = int_val
            except (ValueError, TypeError):
                result["errors"].append(f"Field '{field}' must be an integer")
                result["is_valid"] = False

        elif expected_type == float:
            sanitized = sanitize_number(value)
            if sanitized is None:
                result["errors"].append(f"Field '{field}' must be a valid number")
                result["is_valid"] = False
            else:
                result["sanitized_record"][field] = sanitized

        elif expected_type == bool:
            if isinstance(value, bool):
                result["sanitized_record"][field] = value
            else:
                result["errors"].append(f"Field '{field}' must be a boolean")
                result["is_valid"] = False

        elif expected_type == list:
            if isinstance(value, list):
                result["sanitized_record"][field] = value
            else:
                result["errors"].append(f"Field '{field}' must be a list")
                result["is_valid"] = False

        elif expected_type == dict:
            if isinstance(value, dict):
                result["sanitized_record"][field] = value
            else:
                result["errors"].append(f"Field '{field}' must be a dictionary")
                result["is_valid"] = False

        elif expected_type == type(None):
            if value is None:
                result["sanitized_record"][field] = None
            else:
                result["errors"].append(f"Field '{field}' must be null")
                result["is_valid"] = False

        else:
            result["errors"].append(f"Unsupported type for field '{field}'")
            result["is_valid"] = False

    return result