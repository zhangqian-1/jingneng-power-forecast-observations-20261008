"""Shared result vocabulary; no model or third-party imports."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from uuid import uuid4

EVENT_KEY = "JNH.Fluxcast.Compute"
FORECAST = "totalPowerForecast"
ACTUAL = "totalPowerActual"
DEVIATION = "totalPowerDeviation"
BATCH_ID = "forecastBatchId"
GENERATED_AT = "forecastGeneratedAt"
FORECAST_NUMBERS = {FORECAST, "forecastPointCount", "forecastIntervalMinutes"}
STATION_SUFFIXES = {
    "powerActual", "powerShare", "dataStatus", "validPointCount",
    "requiredPointCount", "missingPoints", "reason",
}
UTC_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?P<separator>[ T])[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?P<fraction>\.[0-9]{1,9})?(?P<suffix>Z|\+00:00|\+0000)?"
)


def utc_epoch(value: str) -> float:
    if not isinstance(value, str) or UTC_PATTERN.fullmatch(value) is None:
        raise ValueError("Expected a full UTC timestamp")
    stamp = datetime.strptime(value[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    fraction = UTC_PATTERN.fullmatch(value)["fraction"]
    return stamp.replace(tzinfo=timezone.utc).timestamp() + (float(fraction) if fraction else 0.0)


def utc_label(epoch: float, sample: str) -> str:
    """Preserve separator, fractional precision and UTC suffix at second precision."""
    match = UTC_PATTERN.fullmatch(sample)
    if match is None:
        raise ValueError("Expected a full UTC timestamp")
    return (datetime.fromtimestamp(epoch, timezone.utc).strftime(
        "%Y-%m-%d" + match["separator"] + "%H:%M:%S")
        + ("." + "0" * (len(match["fraction"]) - 1) if match["fraction"] else "")
        + (match["suffix"] or ""))


def point(name: str, timestamp: str, value: float | str) -> dict:
    return {"varname": name, "timestamp": timestamp, "value": value}


def station_var(code: str, suffix: str) -> str:
    if suffix not in STATION_SUFFIXES:
        raise ValueError("Unknown station result suffix")
    return f"{code}_{suffix}"


def forecast_response(points: list[dict], generated_at: datetime | None = None,
                      batch_id: str | None = None) -> dict:
    """Decorate an unchanged UTC curve after a successful model computation."""
    generated_at = generated_at or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        raise ValueError("Generation time must be timezone aware")
    times = [utc_epoch(p["timestamp"]) for p in points]
    if len(times) != 96 or any(b - a != 900 for a, b in zip(times, times[1:])):
        raise ValueError("Expected 96 ordered quarter-hour forecast targets")
    stamp = utc_label(generated_at.timestamp(), points[0]["timestamp"])
    identifier = batch_id or "dayahead-" + uuid4().hex
    return {
        "result_point": [*points, point("forecastPointCount", stamp, 96.0),
                         point("forecastIntervalMinutes", stamp, 15.0)],
        "extra_info": [point(BATCH_ID, stamp, identifier), point(GENERATED_AT, stamp, stamp),
                       point("forecastStartAt", stamp, points[0]["timestamp"]),
                       point("forecastEndAt", stamp, points[-1]["timestamp"])],
        "event_key": EVENT_KEY,
    }


def validate_envelope(payload: dict) -> None:
    """Validate types and semantic UTC keys, including alternate time spellings."""
    if not isinstance(payload, dict) or payload.get("event_key") != EVENT_KEY:
        raise ValueError("Unexpected result event_key")
    seen = set()
    for group in ("result_point", "extra_info"):
        entries = payload.get(group, [] if group == "extra_info" else None)
        if not isinstance(entries, list):
            raise ValueError(f"{group} must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"varname", "timestamp", "value"}:
                raise ValueError("Result entries require varname, timestamp and value")
            name = entry["varname"]
            if not isinstance(name, str) or not name:
                raise ValueError("Result varname must be a nonempty string")
            key = (name, utc_epoch(entry["timestamp"]))
            if key in seen:
                raise ValueError("Duplicate result variable and UTC timestamp")
            seen.add(key)
            value = entry["value"]
            if group == "extra_info":
                if not isinstance(value, str):
                    raise ValueError("extra_info values must be strings")
            else:
                try:
                    valid = not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError("result_point values must be finite numbers")
