"""Persistent rolling cache of real point-level measurements."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from input_adapter import InputAdapter, InputValidationError, ParsedInput, POWER_POLICY_ID
from models.utils import time_feature_frame
from models.causal_state import add_causal_states, seed_columns


class HistoryNotReadyError(ValueError):
    def __init__(self, status: dict):
        self.status = status
        super().__init__(
            f"真实历史数据不足：当前连续 {status['continuousPoints']} 点，"
            f"模型需要 {status['requiredPoints']} 点"
        )


class RealHistoryCache:
    """Merge overlapping one-day requests and retain recent real history."""

    def __init__(self, cache_path: Path, required_points: int = 672, retention_points: int = 768,
                 state_centers: dict | None = None, model_name: str = "seven_station_2025"):
        self.cache_path = Path(cache_path)
        self.required_points = required_points
        self.retention_points = max(retention_points, required_points)
        self.adapter = InputAdapter()
        self.state_centers = state_centers or {}
        self.model_name = model_name

    def merge(self, parsed: ParsedInput) -> tuple[pd.DataFrame, dict]:
        incoming = parsed.frame[["ts"] + parsed.point_columns + self._source_valid_columns()].copy()
        incoming["ts"] = pd.to_datetime(incoming["ts"])

        seed = None
        cached = None
        if self.cache_path.exists():
            cached = pd.read_csv(self.cache_path)
            cached["ts"] = pd.to_datetime(cached["ts"])
            if "__power_policy" not in cached or not cached["__power_policy"].eq(POWER_POLICY_ID).all():
                raise InputValidationError("历史缓存的功率处理规则不匹配，请使用新的预测缓存目录并补传7天历史")
            if "__model" not in cached or not cached["__model"].eq(self.model_name).all():
                raise InputValidationError("历史缓存的模型或时间规则不匹配，请为本版本使用独立的空缓存目录")
            if incoming["ts"].iloc[-1] < cached["ts"].iloc[-1]:
                raise InputValidationError("不接受早于缓存末端的旧批次，避免使用未来历史预测过去")
            if incoming["ts"].iloc[0] < cached["ts"].iloc[0]:
                raise InputValidationError("请求早于保留的缓存范围，请使用独立缓存进行历史回放")
            if self.state_centers:
                required = [c for station in self.state_centers for c in seed_columns(station)]
                if any(c not in cached for c in required):
                    raise InputValidationError("缓存缺少当前版本状态上下文，请使用独立的空缓存目录")
                seed = cached.iloc[0]
            combined = pd.concat([cached, incoming], ignore_index=True, sort=False)
        else:
            combined = incoming

        source_columns = self._source_valid_columns()
        load_columns, weather_columns = self.adapter.point_columns()
        weather_carry_columns = [f"__weather_before::{column}" for column in weather_columns]
        weather_seed = pd.Series(float("nan"), index=weather_columns)
        if cached is not None:
            if any(column not in cached for column in weather_carry_columns):
                raise InputValidationError("缓存缺少温湿度上下文，请使用本版本独立的空缓存目录")
            weather_seed[:] = cached.iloc[0][weather_carry_columns].to_numpy(dtype=float)
        combined = combined.sort_values("ts", kind="stable").reset_index(drop=True)
        # Each request is a complete seven-station snapshot. A newer missing
        # power measurement means zero, never yesterday's or a cached power.
        latest = combined.drop_duplicates("ts", keep="last")[["ts"] + load_columns + source_columns]
        weather = combined.groupby("ts", as_index=False)[weather_columns].last()
        combined = pd.merge(latest, weather, on="ts", how="left").sort_values("ts").reset_index(drop=True)
        filled = combined.copy()
        filled.loc[0, weather_columns] = filled.loc[0, weather_columns].astype(float).fillna(weather_seed)
        fill_quality = self.adapter.fill_missing_points(
            filled,
            load_columns,
            weather_columns,
            allow_leading_backfill=False,
        )
        fill_quality["missingWeatherValues"] = int(combined[weather_columns].isna().sum().sum())
        fill_quality["forwardFilledWeatherValues"] = int((combined[weather_columns].isna() & filled[weather_columns].notna()).sum().sum())
        # Persist observations separately from fills so a corrected past weather
        # value also updates the later missing points, including after restart.
        prior_weather = filled[weather_columns].shift(1)
        prior_weather.iloc[0] = weather_seed.to_numpy()
        combined[weather_carry_columns] = prior_weather.to_numpy()
        model_frame = self.adapter.build_model_frame(filled)
        if self.state_centers:
            model_frame = add_causal_states(model_frame, self.state_centers, seed=seed)
            carry_columns = [c for station in self.state_centers for c in seed_columns(station)]
            combined[carry_columns] = model_frame[carry_columns]
        combined["__model"] = self.model_name
        combined["__power_policy"] = POWER_POLICY_ID
        combined = combined.tail(self.retention_points).reset_index(drop=True)
        model_frame = model_frame.tail(self.retention_points).reset_index(drop=True)
        filled = filled.tail(self.retention_points).reset_index(drop=True)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        combined.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(self.cache_path)

        history, status = self._latest_continuous_history(filled)
        # Unknown leading weather is not backfilled from future measurements.
        # Only the continuous suffix with all weather inputs can warm the model.
        weather_missing = history[weather_columns].isna().any(axis=1)
        if weather_missing.any():
            status["missingWeatherPoints"] = sorted(
                column.rsplit("::", 1)[-1]
                for column in weather_columns if history[column].isna().any()
            )
            history = history.loc[history.index > weather_missing[weather_missing].index[-1]]
            status["continuousPoints"] = int(len(history))
            status["ready"] = len(history) >= self.required_points
            status["continuousStart"] = None if history.empty else str(history["ts"].iloc[0])
            status["waitingForWeatherHistory"] = True
        status.update(
            {
                "cacheFile": str(self.cache_path),
                "cachedPoints": int(len(combined)),
                "cacheStart": combined["ts"].iloc[0].isoformat(sep=" "),
                "cacheEnd": combined["ts"].iloc[-1].isoformat(sep=" "),
                "cacheFillQuality": fill_quality,
            }
        )
        if len(history) < self.required_points:
            raise HistoryNotReadyError(status)

        model_frame = model_frame.loc[history.index].reset_index(drop=True)
        time_features = time_feature_frame(model_frame["ts"])
        for column in time_features.columns:
            model_frame[column] = time_features[column].to_numpy(dtype=float)
        return model_frame, status

    def _latest_continuous_history(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if frame.empty:
            return frame, {"ready": False, "continuousPoints": 0, "requiredPoints": self.required_points}
        breaks = frame["ts"].diff().ne(pd.Timedelta(minutes=15))
        segment = breaks.cumsum()
        latest_segment = segment.iloc[-1]
        continuous = frame[segment == latest_segment].copy()
        return continuous, {
            "ready": len(continuous) >= self.required_points,
            "continuousPoints": int(len(continuous)),
            "requiredPoints": self.required_points,
            "continuousStart": continuous["ts"].iloc[0].isoformat(sep=" "),
            "continuousEnd": continuous["ts"].iloc[-1].isoformat(sep=" "),
        }

    @staticmethod
    def _source_valid_columns() -> list[str]:
        from input_adapter import STATION_FEATURES

        return [f"__{prefix}_source_valid" for prefix in STATION_FEATURES.values()]
