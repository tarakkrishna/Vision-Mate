"""Data processing utilities for the VisionMate statistics dashboard.

This module reads the local usage_log.json file and computes daily
aggregates for:

- Screen distance monitoring sessions
- Total screen time per day
- Vision test clarity scores over time

All functions are pure and independent of the GUI so they can be tested
separately.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


LOG_FILE = "usage_log.json"


@dataclass
class DistancePoint:
    """Average distance per day for the screen-distance chart."""

    date: datetime
    average_distance_cm: float


@dataclass
class ScreenTimePoint:
    """Total screen time per day for the screen-time chart."""

    date: datetime
    total_seconds: float


@dataclass
class VisionTestPoint:
    """Vision test clarity result for a given date/time."""

    timestamp: datetime
    clarity_percentage: float


def _load_log(filepath: str = LOG_FILE) -> List[dict]:
    """Load and return the full JSON log as a list of entries.

    Returns an empty list if the file does not exist or is invalid.
    """

    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _date_only(dt: datetime) -> datetime:
    """Return a datetime normalised to midnight (date-only)."""

    return datetime(year=dt.year, month=dt.month, day=dt.day)


def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse an ISO timestamp string.

    Returns None if parsing fails.
    """

    try:
        return datetime.fromisoformat(ts)
    except Exception:  # noqa: BLE001
        return None


def compute_last_7_days_distance(data: Optional[List[dict]] = None) -> List[DistancePoint]:
    """Compute average distance per day for the last 7 days.

    The current implementation of VisionMate does not log raw distance
    samples per frame, only summary statistics per monitoring session.
    Therefore, this function approximates distance trends based on the
    number of "too_close_events" per monitoring session.

    Fewer "too close" events are interpreted as better distance habits.
    This is mapped to a synthetic average distance score, suitable for
    a simple trend line, even if not physically exact in centimetres.
    """

    entries = data if data is not None else _load_log()

    # Build a map date -> list of too_close_events values
    per_day: Dict[datetime, List[int]] = {}
    for entry in entries:
        if entry.get("type") != "monitoring_session":
            continue
        ts = entry.get("end_time") or entry.get("start_time")
        if not isinstance(ts, str):
            continue
        dt = _parse_iso(ts)
        if dt is None:
            continue
        day = _date_only(dt)
        too_close = int(entry.get("too_close_events", 0))
        per_day.setdefault(day, []).append(too_close)

    if not per_day:
        return []

    # Consider only the last 7 calendar days
    today = _date_only(datetime.now())
    start_day = today - timedelta(days=6)

    points: List[DistancePoint] = []
    for i in range(7):
        day = start_day + timedelta(days=i)
        vals = per_day.get(day)
        if not vals:
            continue

        avg_too_close = sum(vals) / len(vals)
        # Map "too_close" counts to an approximate distance score.
        # Many too-close events -> lower distance score.
        synthetic_distance = max(20.0, 80.0 - avg_too_close * 5.0)
        points.append(DistancePoint(date=day, average_distance_cm=synthetic_distance))

    return points


def compute_last_7_days_screen_time(data: Optional[List[dict]] = None) -> List[ScreenTimePoint]:
    """Compute total screen time per day (seconds) for the last 7 days.

    Uses "monitoring_session" entries and their "total_monitor_seconds"
    field, grouped by end_time date.
    """

    entries = data if data is not None else _load_log()

    per_day_seconds: Dict[datetime, float] = {}
    for entry in entries:
        if entry.get("type") != "monitoring_session":
            continue
        ts = entry.get("end_time") or entry.get("start_time")
        if not isinstance(ts, str):
            continue
        dt = _parse_iso(ts)
        if dt is None:
            continue
        day = _date_only(dt)
        seconds = float(entry.get("total_monitor_seconds", 0.0))
        per_day_seconds[day] = per_day_seconds.get(day, 0.0) + seconds

    if not per_day_seconds:
        return []

    today = _date_only(datetime.now())
    start_day = today - timedelta(days=6)

    points: List[ScreenTimePoint] = []
    for i in range(7):
        day = start_day + timedelta(days=i)
        total = per_day_seconds.get(day, 0.0)
        if total <= 0:
            continue
        points.append(ScreenTimePoint(date=day, total_seconds=total))

    return points


def compute_vision_test_progress(data: Optional[List[dict]] = None) -> List[VisionTestPoint]:
    """Compute vision test clarity scores over time.

    Each "vision_test" entry contributes one point using its timestamp
    and the summary.clarity_percentage field.
    """

    entries = data if data is not None else _load_log()

    points: List[VisionTestPoint] = []
    for entry in entries:
        if entry.get("type") != "vision_test":
            continue
        ts = entry.get("timestamp")
        if not isinstance(ts, str):
            continue
        dt = _parse_iso(ts)
        if dt is None:
            continue
        summary = entry.get("summary") or {}
        clarity = summary.get("clarity_percentage")
        if clarity is None:
            continue
        try:
            clarity_val = float(clarity)
        except (TypeError, ValueError):
            continue
        points.append(VisionTestPoint(timestamp=dt, clarity_percentage=clarity_val))

    # Sort by time so that the chart is in chronological order.
    points.sort(key=lambda p: p.timestamp)
    return points


def has_enough_data_for_dashboard(data: Optional[List[dict]] = None) -> bool:
    """Return True if there is at least some data for any chart.

    This is used by the dashboard UI to show a friendly message when
    there are not yet enough monitoring sessions or vision tests.
    """

    entries = data if data is not None else _load_log()
    if not entries:
        return False

    if compute_last_7_days_distance(entries):
        return True
    if compute_last_7_days_screen_time(entries):
        return True
    if compute_vision_test_progress(entries):
        return True

    return False
