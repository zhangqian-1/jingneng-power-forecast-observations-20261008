"""UTC at the API boundary; Beijing clock labels for existing model features."""
from __future__ import annotations

import re

import pandas as pd


MODEL_TIMEZONE = "Asia/Shanghai"
TIME_POLICY_ID = "utc_to_asia_shanghai_v1"
TIMEZONE_BASIS = "user_confirmed_asia_shanghai"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?P<separator>[ T])"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?P<fraction>\.[0-9]{1,9})?"
    r"(?P<suffix>Z|\+00:00|\+0000)?"
)


def request_timestamp_format(value: str) -> str:
    """Retain the spelling of a validated quarter-hour UTC timestamp."""
    match = UTC_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("Expected YYYY-MM-DD[ T]HH:mm:ss with optional fractional seconds and UTC suffix")
    return ("%Y-%m-%d" + match["separator"] + "%H:%M:%S"
            + (match["fraction"] or "") + (match["suffix"] or ""))


def utc_to_model_clock(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    elif timestamp.utcoffset() != pd.Timedelta(0):
        raise ValueError("Expected a UTC timestamp")
    return timestamp.tz_convert(MODEL_TIMEZONE).tz_localize(None)


def model_clock_to_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        raise ValueError("Expected a naive model-clock timestamp")
    return timestamp.tz_localize(MODEL_TIMEZONE).tz_convert("UTC").tz_localize(None)
