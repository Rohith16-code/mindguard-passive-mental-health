"""Time utilities for timezone-aware sleep inference."""
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any
import pytz


def infer_sleep_window(
    activity_data: Dict[str, Any],
    user_timezone: str,
    default_offset_hours: Tuple[int, int] = (-2, 2)
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Infer sleep onset and offset times from activity data in a timezone-aware manner.

    Args:
        activity_data: Dictionary containing activity metrics with keys:
            - 'last_movement_time': ISO string of last movement detected
            - 'screen_off_time': ISO string of last screen off
            - 'first_movement_time': ISO string of first movement after suspected wake
        user_timezone: IANA timezone string (e.g., 'America/New_York')
        default_offset_hours: Tuple of (min_hours_before, max_hours_after) relative to last movement

    Returns:
        Tuple of (sleep_onset, sleep_offset) as timezone-aware datetimes or None if insufficient data
    """
    if not activity_data:
        return None, None

    try:
        tz = pytz.timezone(user_timezone)
    except pytz.UnknownTimeZoneError:
        tz = pytz.UTC

    def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(tz)
        except (ValueError, TypeError):
            return None

    last_movement = _parse_iso(activity_data.get('last_movement_time'))
    screen_off = _parse_iso(activity_data.get('screen_off_time'))
    first_movement = _parse_iso(activity_data.get('first_movement_time'))

    if not last_movement:
        return None, None

    if screen_off and last_movement and screen_off < last_movement:
        sleep_onset = screen_off
    elif last_movement:
        sleep_onset = last_movement
    else:
        return None, None

    if first_movement:
        sleep_offset = first_movement
    else:
        min_offset, max_offset = default_offset_hours
        sleep_offset = sleep_onset + timedelta(hours=min_offset)
        if sleep_offset < sleep_onset:
            sleep_offset = sleep_onset + timedelta(hours=max_offset)

    if sleep_offset <= sleep_onset:
        sleep_offset = sleep_onset + timedelta(hours=1)

    return sleep_onset, sleep_offset


def get_localized_now(tz_name: str) -> datetime:
    """Get current time in specified timezone."""
    try:
        tz = pytz.timezone(tz_name)
        return datetime.now(tz)
    except pytz.UnknownTimeZoneError:
        return datetime.now(timezone.utc)


def calculate_sleep_duration(
    onset: Optional[datetime],
    offset: Optional[datetime]
) -> Optional[timedelta]:
    """Calculate sleep duration between onset and offset."""
    if not onset or not offset:
        return None
    if offset <= onset:
        return timedelta(0)
    return offset - onset


def is_within_sleep_window(
    timestamp: datetime,
    sleep_onset: datetime,
    sleep_offset: datetime,
    buffer_minutes: int = 15
) -> bool:
    """Check if timestamp falls within sleep window with optional buffer."""
    buffer = timedelta(minutes=buffer_minutes)
    return (sleep_onset - buffer) <= timestamp <= (sleep_offset + buffer)