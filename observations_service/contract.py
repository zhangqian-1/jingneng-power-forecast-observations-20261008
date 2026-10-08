"""Measurement and UTC contracts without importing the prediction runtime."""
from __future__ import annotations

import math
import re
import reprlib
from datetime import datetime, timezone

from app.result_contract import (
    ACTUAL, BATCH_ID, DEVIATION, EVENT_KEY, FORECAST, FORECAST_NUMBERS, GENERATED_AT,
    point, station_var, utc_epoch, validate_envelope,
)

STATION_POINTS = {
    "GARD": ("GARD_11MBY0100000BJ01XQ01", "GARD_12MBY0100000BJ01XQ01", "GARD_13MKA01CE903BJ01XQ01"),
    "JXRD": ("JXRD_11MBY0100000BJ01XQ01", "JXRD_12MBY0100000BJ01XQ01",
             "JXRD_13MKA01GA001BJ02XQ01", "JXRD_14MBY0100000BJ01XQ01", "JXRD_15MKA01GA001BJ02XQ01"),
    "JYRD": ("JYRD_LOADCTL:GTMWSEL1_1.OUT", "JYRD_LOADCTL:GTMWSEL1_2.OUT", "JYRD_30DCS01:FU101.PNT"),
    "JQRD": ("JQRD_10CBA00FA107XQ93", "JQRD_10CBA00FA108XQ93", "JQRD_10CBA00FA109XQ93"),
    "JFRD": ("JFRD_11MKA01GA001BJ40XQ01",),
    "WLRD": ("WLRD_13MKA0100000BJ01XQ01", "WLRD_11MBY10CE901XQ01"),
    "SZRD": ("SZRD_10DCS02FA133", "SZRD_10DCS02FA134"),
}
POINTS = tuple(point for points in STATION_POINTS.values() for point in points)
UTC_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?P<fraction>\.[0-9]{1,9})?(?:Z|\+00:00|\+0000)?"
)


def utc_slot(value: str) -> int:
    """Use UTC instants as keys, retaining the original spelling separately."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be a UTC string")
    match = UTC_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("Expected YYYY-MM-DD[ T]HH:mm:ss with an optional UTC suffix")
    if any(digit != "0" for digit in (match["fraction"] or ".")[1:]):
        raise ValueError("timestamp must be exactly aligned to 15 minutes")
    stamp = datetime.strptime(value[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    if stamp.minute % 15 or stamp.second:
        raise ValueError("timestamp must be exactly aligned to 15 minutes")
    return int(stamp.replace(tzinfo=timezone.utc).timestamp())


def numeric(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _value_issue(frame: dict, point: str) -> str | None:
    """Distinguish missing values from values that cannot be used as power."""
    if point not in frame:
        return "missing_field"
    value = frame[point]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "invalid_type(bool)"
    if isinstance(value, (int, float)):
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            return "numeric_overflow"
        return None if math.isfinite(converted) else "nonfinite"
    return f"invalid_type({type(value).__name__})"


def _value_preview(value) -> str:
    try:
        return reprlib.repr(value)
    except ValueError:
        return f"<{type(value).__name__} too large to display>"


def _finite_sum(values) -> float | None:
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError):
        return None
    return total if math.isfinite(total) else None


def measurement(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"point_table", "frames"}:
        raise ValueError("Request must contain only point_table and frames")
    table = payload["point_table"]
    if (not isinstance(table, list) or not all(isinstance(p, str) for p in table)
            or len(table) != len(POINTS) or set(table) != set(POINTS)):
        raise ValueError("point_table must list all 19 unique power point codes")
    frames = payload["frames"]
    if not isinstance(frames, list) or len(frames) != 1 or not isinstance(frames[0], dict):
        raise ValueError("frames must contain exactly one object")
    frame = frames[0]
    if set(frame) - set(POINTS) - {"timestamp"}:
        raise ValueError("frame contains unknown point codes")
    stamp = frame.get("timestamp")
    slot = utc_slot(stamp)
    issues = {point: {"reason": issue, "value": _value_preview(frame[point])
                     if point in frame else "<absent>"} for point in POINTS
              if (issue := _value_issue(frame, point)) is not None}
    values = {point: numeric(frame.get(point)) for point in POINTS}
    values = {point: max(0.0, value) if value is not None else None for point, value in values.items()}
    missing = [point for point, value in values.items() if value is None]
    status = "complete" if not missing else ("missing" if len(missing) == len(POINTS) else "incomplete")
    total = _finite_sum(values.values()) if status == "complete" else None
    aggregate_issue = "power_sum_out_of_range" if status == "complete" and total is None else None
    return {"timestamp": stamp, "slot": slot, "status": status, "values": values,
            "missing": missing, "total": total, "issues": issues,
            "aggregate_issue": aggregate_issue}


def forecast_curve(payload: dict) -> list[tuple[int, float]]:
    validate_envelope(payload)
    if any(p["varname"] not in FORECAST_NUMBERS for p in payload["result_point"]):
        raise ValueError("Unexpected forecast numeric variable")
    points = [p for p in payload["result_point"] if p["varname"] == FORECAST]
    if len(points) != 96:
        raise ValueError("Forecast must contain 96 points")
    curve = []
    for point in points:
        if not isinstance(point, dict) or point.get("varname") != "totalPowerForecast":
            raise ValueError("Unexpected forecast varname")
        value = numeric(point.get("value"))
        if value is None:
            raise ValueError("Forecast values must be finite numbers")
        curve.append((utc_slot(point.get("timestamp")), value))
    curve.sort()
    if any(right[0] - left[0] != 900 for left, right in zip(curve, curve[1:])):
        raise ValueError("Forecast timestamps must be unique and continuous")
    return curve


def forecast_metadata(payload: dict) -> tuple[str | None, float | None]:
    entries = payload.get("extra_info", [])
    metadata = {}
    for entry in entries:
        if entry["varname"] in {BATCH_ID, GENERATED_AT}:
            if entry["varname"] in metadata:
                raise ValueError("Duplicate forecast batch metadata")
            metadata[entry["varname"]] = entry["value"]
    if not metadata:
        return None, None
    identifier = metadata.get(BATCH_ID)
    if (not isinstance(identifier, str) or re.fullmatch(r"dayahead-[0-9a-f]{32}", identifier) is None
            or GENERATED_AT not in metadata):
        raise ValueError("Expected paired forecastBatchId and forecastGeneratedAt")
    return identifier, utc_epoch(metadata[GENERATED_AT])


def report_with_diagnostics(item: dict, prediction: float | None,
                            batch_id: str | None = None) -> tuple[dict, dict]:
    stamp = item["timestamp"]
    def entry(name, value):
        return point(name, stamp, value)

    numbers = []
    info = [entry("dataStatus", item["status"])]
    diagnostics = {"timestamp": stamp, "causes": []}
    shares = [station_var(code, "powerShare") for code in STATION_POINTS]
    total_dependents = [ACTUAL, DEVIATION, *shares]

    def unavailable(reason, inputs, omitted):
        diagnostics["causes"].append({"reason": reason, "inputs": inputs, "omitted": omitted})

    def power_inputs(points):
        return {name: item["values"][name] for name in points}

    total = item["total"]
    if item["missing"]:
        info.append(entry("missingPoints", ",".join(item["missing"])))
    elif total is None:
        unavailable("power_sum_out_of_range", power_inputs(POINTS), total_dependents)

    if total is not None:
        numbers.append(entry(ACTUAL, float(total)))
        if prediction is None:
            unavailable("no_matching_forecast", {}, [DEVIATION])
        else:
            deviation = prediction - total
            if math.isfinite(deviation):
                numbers.append(entry(DEVIATION, float(deviation)))
                if batch_id is not None:
                    info.append(entry(BATCH_ID, batch_id))
            else:
                unavailable("deviation_out_of_range", {FORECAST: prediction, ACTUAL: total}, [DEVIATION])
        if total == 0:
            unavailable("total_power_zero", {ACTUAL: total}, shares)

    for code, points in STATION_POINTS.items():
        missing = [p for p in points if item["values"][p] is None]
        valid = len(points) - len(missing)
        status = "complete" if not missing else ("missing" if not valid else "incomplete")
        numbers.extend([entry(station_var(code, "validPointCount"), float(valid)),
                        entry(station_var(code, "requiredPointCount"), float(len(points)))])
        info.append(entry(station_var(code, "dataStatus"), status))
        if missing:
            info.append(entry(station_var(code, "missingPoints"), ",".join(missing)))
            unavailable("missing_power_points", {p: item["issues"][p] for p in missing},
                        [station_var(code, "powerActual"), *total_dependents])
            continue
        power = _finite_sum(item["values"][p] for p in points)
        if power is None:
            info.append(entry(station_var(code, "reason"), "power_sum_out_of_range"))
            unavailable("power_sum_out_of_range", power_inputs(points),
                        [station_var(code, "powerActual"), *total_dependents])
            continue
        numbers.append(entry(station_var(code, "powerActual"), float(power)))
        if total is not None and total > 0:
            numbers.append(entry(station_var(code, "powerShare"), float(power / total * 100.0)))

    if diagnostics["causes"]:
        reason = "missing_power_points" if item["missing"] else diagnostics["causes"][0]["reason"]
        info.insert(1, entry("reason", reason))
        messages = []
        for problem in diagnostics["causes"]:
            inputs = []
            for name, value in problem["inputs"].items():
                detail = f"{value['reason']} (value={value['value']})" if isinstance(value, dict) else str(value)
                inputs.append(f"{name}={detail}")
            detail = ", ".join(inputs) or "no eligible forecast at the same UTC timestamp"
            messages.append(f"{problem['reason']}: {detail}; omitted={','.join(problem['omitted'])}")
        info.append(entry("message", " | ".join(messages)))
    return {"result_point": numbers, "extra_info": info, "event_key": EVENT_KEY}, diagnostics


def report(item: dict, prediction: float | None, batch_id: str | None = None) -> dict:
    return report_with_diagnostics(item, prediction, batch_id)[0]
