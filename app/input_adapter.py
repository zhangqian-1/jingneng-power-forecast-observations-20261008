"""Validate external JSON and build model features from real measurements only."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.utils import time_feature_frame


POINTS_PER_DAY = 96
INTERVAL_MINUTES = 15
POWER_POLICY_ID = "point_negative_to_zero_v1"

STATION_FEATURES = {
    "高安屯热电": "高安屯热电",
    "京西热电": "京西热电",
    "京阳热电": "京阳热电",
    "京桥热电": "京桥热电",
    "京丰燃气": "京丰燃气",
    "未来热电": "未来热电",
    "上庄热电": "上庄热电",
}

STATION_LOAD_POINTS = {
    "高安屯热电": {
        "GARD_11MBY0100000BJ01XQ01",
        "GARD_12MBY0100000BJ01XQ01",
        "GARD_13MKA01CE903BJ01XQ01",
    },
    "京西热电": {
        "JXRD_11MBY0100000BJ01XQ01",
        "JXRD_12MBY0100000BJ01XQ01",
        "JXRD_13MKA01GA001BJ02XQ01",
        "JXRD_14MBY0100000BJ01XQ01",
        "JXRD_15MKA01GA001BJ02XQ01",
    },
    "京阳热电": {
        "JYRD_LOADCTL:GTMWSEL1_1.OUT",
        "JYRD_LOADCTL:GTMWSEL1_2.OUT",
        "JYRD_30DCS01:FU101.PNT",
    },
    "京桥热电": {
        "JQRD_10CBA00FA107XQ93",
        "JQRD_10CBA00FA108XQ93",
        "JQRD_10CBA00FA109XQ93",
    },
    "京丰燃气": {"JFRD_11MKA01GA001BJ40XQ01"},
    "未来热电": {"WLRD_13MKA0100000BJ01XQ01", "WLRD_11MBY10CE901XQ01"},
    "上庄热电": {"SZRD_10DCS02FA133", "SZRD_10DCS02FA134"},
}

STATION_WEATHER_POINTS = {
    "高安屯热电": {"GARD_11MBL3100000BT01XQ01", "GARD_11MBL3100000BM01XQ01"},
    "京西热电": {"JXRD_11MBL3100000BT01XQ01", "JXRD_11MBL3100000BM01XQ01"},
    "京阳热电": {"JYRD_QIXIANGYI:RIN7.MEAS", "JYRD_11MBL3100000BM01XQ01"},
    "京桥热电": {
        "JQRD_11MBL11CT010XQ01",
        "JQRD_12MBL11CT010XQ01",
        "JQRD_11MBL11CM001XQ01",
        "JQRD_12MBL11CM001XQ01",
    },
    "京丰燃气": {"JFRD_11MBL0100000BT02XQ01", "JFRD_11MBL0100000BM01XQ01"},
    "未来热电": {"WLRD_11MBL01WP001BT01XQ01", "WLRD_11MBL01WP001BM01XQ01"},
    "上庄热电": {"SZRD_11MBL10CT901ZQ01", "SZRD_11MBL10CM901ZQ01"},
}

# The point IDs are taken from the approved input format in the handover document.
WEATHER_POINT_KIND = {
    "GARD_11MBL3100000BT01XQ01": "temperature",
    "GARD_11MBL3100000BM01XQ01": "humidity",
    "JXRD_11MBL3100000BT01XQ01": "temperature",
    "JXRD_11MBL3100000BM01XQ01": "humidity",
    "JYRD_QIXIANGYI:RIN7.MEAS": "temperature",
    "JYRD_11MBL3100000BM01XQ01": "humidity",
    "JQRD_11MBL11CT010XQ01": "temperature",
    "JQRD_12MBL11CT010XQ01": "temperature",
    "JQRD_11MBL11CM001XQ01": "humidity",
    "JQRD_12MBL11CM001XQ01": "humidity",
    "JFRD_11MBL0100000BT02XQ01": "temperature",
    "JFRD_11MBL0100000BM01XQ01": "humidity",
    "WLRD_11MBL01WP001BT01XQ01": "temperature",
    "WLRD_11MBL01WP001BM01XQ01": "humidity",
    "SZRD_11MBL10CT901ZQ01": "temperature",
    "SZRD_11MBL10CM901ZQ01": "humidity",
}

FAHRENHEIT_POINT_IDS = {
    "SZYH_11MBL01GQ001BT01XQ02",
    "SZYH_14MBC01GQ001GQ02XJ01",
}


class InputValidationError(ValueError):
    """Raised when submitted measurements do not match the production contract."""


@dataclass(frozen=True)
class ParsedInput:
    batch_time: str
    frame: pd.DataFrame
    input_quality: dict[str, Any]
    point_columns: list[str]

    @property
    def last_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame["ts"].iloc[-1])


def _measurement_or_nan(value: Any) -> float:
    """Convert a point measurement, treating unusable values as missing."""
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, str) and not value.strip():
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


class InputAdapter:
    """Convert the documented seven-station JSON payload into model features."""

    def parse_json(self, payload: Mapping[str, Any]) -> ParsedInput:
        if not isinstance(payload, Mapping):
            raise InputValidationError("请求体必须是 JSON 对象")

        interval = payload.get("intervalMinutes", INTERVAL_MINUTES)
        if interval != INTERVAL_MINUTES:
            raise InputValidationError("intervalMinutes 必须为 15")

        history_days = payload.get("historyDays", 1)
        if history_days != 1:
            raise InputValidationError("historyDays 必须为 1，每次提交最近一天真实数据")

        records = payload.get("data")
        if not isinstance(records, list):
            raise InputValidationError("data 必须是数组")
        if len(records) != POINTS_PER_DAY:
            raise InputValidationError(f"data 必须正好包含 {POINTS_PER_DAY} 个15分钟点，实际为 {len(records)}")

        rows = [self._parse_record(record, index) for index, record in enumerate(records)]
        frame = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        self._validate_timeline(frame["ts"])

        load_columns = [
            self._point_column("load", station_name, point_id)
            for station_name, point_ids in STATION_LOAD_POINTS.items()
            for point_id in point_ids
        ]
        weather_columns = [
            self._point_column("weather", station_name, point_id)
            for station_name, point_ids in STATION_WEATHER_POINTS.items()
            for point_id in point_ids
        ]
        input_quality = {
            "missingValuePolicy": "power_zero_weather_causal_forward_fill",
            "zeroValuePolicy": "zero_is_valid_measurement",
            "submittedMissingLoadValues": int(frame[load_columns].isna().sum().sum()),
            "submittedMissingWeatherValues": int(frame[weather_columns].isna().sum().sum()),
        }

        batch_time = str(payload.get("batchTime") or frame["ts"].iloc[-1].strftime("%Y%m%d%H%M"))
        return ParsedInput(
            batch_time=batch_time,
            frame=frame,
            input_quality=input_quality,
            point_columns=load_columns + weather_columns,
        )

    def build_model_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Calculate station totals and model features from a filled point-level frame."""
        frame = frame.copy()
        station_total_columns: list[str] = []
        station_valid_columns: list[str] = []
        station_temperatures: list[pd.Series] = []
        station_humidities: list[pd.Series] = []
        for station_name, feature_prefix in STATION_FEATURES.items():
            station_load_columns = [
                self._point_column("load", station_name, point_id)
                for point_id in STATION_LOAD_POINTS[station_name]
            ]
            station_total_column = f"{feature_prefix}_total_power"
            station_valid_column = f"{feature_prefix}_station_valid"
            frame[station_total_column] = frame[station_load_columns].sum(axis=1)
            frame[station_valid_column] = frame[f"__{feature_prefix}_source_valid"].astype(float)
            station_total_columns.append(station_total_column)
            station_valid_columns.append(station_valid_column)

            temperature_columns = [
                self._point_column("weather", station_name, point_id)
                for point_id in STATION_WEATHER_POINTS[station_name]
                if WEATHER_POINT_KIND[point_id] == "temperature"
            ]
            humidity_columns = [
                self._point_column("weather", station_name, point_id)
                for point_id in STATION_WEATHER_POINTS[station_name]
                if WEATHER_POINT_KIND[point_id] == "humidity"
            ]
            station_temperatures.append(frame[temperature_columns].mean(axis=1))
            station_humidities.append(frame[humidity_columns].mean(axis=1))

        frame["total_power"] = frame[station_total_columns].sum(axis=1)
        frame["temperature"] = pd.concat(station_temperatures, axis=1).mean(axis=1)
        frame["humidity"] = pd.concat(station_humidities, axis=1).mean(axis=1)
        frame["source_valid_station_count"] = frame[station_valid_columns].sum(axis=1)
        frame["available_station_count_after_fill"] = float(len(STATION_FEATURES))
        frame["is_all_station_available_after_fill"] = 1.0

        return frame

    def _parse_record(self, record: Any, index: int) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise InputValidationError(f"data[{index}] 必须是对象")

        raw_ts = record.get("ts")
        if raw_ts is None:
            raise InputValidationError(f"data[{index}].ts 缺失")
        try:
            timestamp = pd.Timestamp(raw_ts)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"data[{index}].ts 格式错误: {raw_ts!r}") from exc
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)

        stations = record.get("stations") or {}
        if not isinstance(stations, Mapping):
            raise InputValidationError(f"data[{index}].stations 必须是对象")
        extra_stations = sorted(set(stations) - set(STATION_FEATURES))
        if extra_stations:
            raise InputValidationError(f"data[{index}].stations 包含未知场站: {extra_stations}")

        row: dict[str, Any] = {"ts": timestamp}
        for station_name, feature_prefix in STATION_FEATURES.items():
            station = stations.get(station_name) or {}
            if not isinstance(station, Mapping):
                raise InputValidationError(f"data[{index}].stations.{station_name} 必须是对象")

            load_points = station.get("load_points") or {}
            if not isinstance(load_points, Mapping):
                raise InputValidationError(f"data[{index}].stations.{station_name}.load_points 必须是对象")
            extra_load_points = sorted(set(load_points) - STATION_LOAD_POINTS[station_name])
            if extra_load_points:
                raise InputValidationError(
                    f"data[{index}].stations.{station_name}.load_points 包含未知测点: {extra_load_points}"
                )

            source_load_values: list[float] = []
            for point_id in STATION_LOAD_POINTS[station_name]:
                value = load_points.get(point_id)
                column = self._point_column("load", station_name, point_id)
                number = _measurement_or_nan(value)
                row[column] = number
                if not np.isnan(number):
                    source_load_values.append(number)
            row[f"__{feature_prefix}_source_valid"] = float(
                len(source_load_values) == len(STATION_LOAD_POINTS[station_name])
            )

            weather_points = station.get("weather_points") or {}
            if not isinstance(weather_points, Mapping):
                raise InputValidationError(f"data[{index}].stations.{station_name}.weather_points 必须是对象")
            extra_weather_points = sorted(set(weather_points) - STATION_WEATHER_POINTS[station_name])
            if extra_weather_points:
                raise InputValidationError(
                    f"data[{index}].stations.{station_name}.weather_points 包含未知测点: {extra_weather_points}"
                )

            for point_id in STATION_WEATHER_POINTS[station_name]:
                value = weather_points.get(point_id)
                column = self._point_column("weather", station_name, point_id)
                number = _measurement_or_nan(value)
                if not np.isnan(number) and point_id in FAHRENHEIT_POINT_IDS:
                    number = (number - 32.0) * 5.0 / 9.0
                row[column] = number
        return row

    @staticmethod
    def _point_column(kind: str, station_name: str, point_id: str) -> str:
        return f"__{kind}::{station_name}::{point_id}"

    @staticmethod
    def fill_missing_points(
        frame: pd.DataFrame,
        load_columns: list[str],
        weather_columns: list[str],
        allow_leading_backfill: bool = False,
    ) -> dict[str, Any]:
        if allow_leading_backfill:
            raise InputValidationError("当前模型禁止使用未来测量值回填历史")
        original = frame[load_columns + weather_columns].copy()
        frame[load_columns] = original[load_columns].fillna(0.0).clip(lower=0.0)
        frame[weather_columns] = original[weather_columns].ffill()
        filled = original[weather_columns].isna() & frame[weather_columns].notna()
        return {
            "missingValuePolicy": "power_zero_weather_causal_forward_fill",
            "zeroValuePolicy": "zero_is_valid_measurement",
            "negativeValuePolicy": POWER_POLICY_ID,
            "negativePowerValuesClipped": int(original[load_columns].lt(0).sum().sum()),
            "missingLoadValues": int(original[load_columns].isna().sum().sum()),
            "missingWeatherValues": int(original[weather_columns].isna().sum().sum()),
            "zeroFilledLoadValues": int(original[load_columns].isna().sum().sum()),
            "forwardFilledWeatherValues": int(filled.sum().sum()),
            "leadingValuesInitializedFromFirstRealValue": 0,
        }

    @staticmethod
    def point_columns() -> tuple[list[str], list[str]]:
        load_columns = [
            InputAdapter._point_column("load", station_name, point_id)
            for station_name, point_ids in STATION_LOAD_POINTS.items()
            for point_id in point_ids
        ]
        weather_columns = [
            InputAdapter._point_column("weather", station_name, point_id)
            for station_name, point_ids in STATION_WEATHER_POINTS.items()
            for point_id in point_ids
        ]
        return load_columns, weather_columns

    @staticmethod
    def _validate_timeline(ts: pd.Series) -> None:
        if ts.isna().any() or not ts.eq(ts.dt.floor("15min")).all():
            raise InputValidationError("时间戳必须有效且对齐15分钟整刻")
        if ts.duplicated().any():
            duplicates = ts[ts.duplicated()].astype(str).tolist()
            raise InputValidationError(f"时间戳重复: {duplicates[:3]}")

        intervals = ts.diff().dropna()
        expected = pd.Timedelta(minutes=INTERVAL_MINUTES)
        invalid = intervals[intervals != expected]
        if not invalid.empty:
            raise InputValidationError("96个时间点必须严格连续，间隔统一为15分钟")
