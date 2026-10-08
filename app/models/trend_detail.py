"""Inference helpers for the deployable TrendDetail model chain."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def blend(level: np.ndarray, shape: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * level + (1.0 - alpha) * shape


def level_shape_fusion(
    level_prediction: np.ndarray,
    shape_prediction: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    level_mean = level_prediction.mean(axis=1, keepdims=True)
    shape_mean = shape_prediction.mean(axis=1, keepdims=True)
    level = alpha * level_mean + (1.0 - alpha) * shape_mean
    shape = shape_prediction - shape_mean
    return level + beta * shape


def apply_calibration(
    raw_prediction: np.ndarray,
    baseline: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    method = str(config["method"])
    alpha = float(config["alpha"])
    beta = float(config.get("beta", 1.0))
    if method == "blend":
        return blend(baseline, raw_prediction, alpha)
    if method == "level_shape":
        return level_shape_fusion(baseline, raw_prediction, alpha, beta)
    raise ValueError(f"Unknown calibration method: {method}")


def smooth_rows(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    kernel = np.ones(window, dtype=float) / window
    smoothed = []
    for row in values:
        padded = np.pad(row, (left, right), mode="edge")
        smoothed.append(np.convolve(padded, kernel, mode="valid"))
    return np.vstack(smoothed)


def trend_detail_fusion(
    mae_prediction: np.ndarray,
    hf_prediction: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    trend_method = config.get("trend_method")
    detail_method = config.get("detail_method")
    if trend_method or detail_method:
        trend = _build_trend(
            mae_prediction,
            hf_prediction,
            str(trend_method or "mae_raw"),
        )
        detail = _build_detail(
            mae_prediction,
            hf_prediction,
            str(detail_method or "hf_shape_minus_mae_shape"),
        )
        return trend + float(config.get("beta", 1.0)) * detail

    trend_window = int(config["trend_window"])
    trend_gamma = float(config["trend_gamma"])
    detail_window = int(config["detail_window"])
    detail_beta = float(config["detail_beta"])

    mae_smooth = smooth_rows(mae_prediction, trend_window)
    hf_smooth = smooth_rows(hf_prediction, trend_window)
    trend = mae_prediction + trend_gamma * (hf_smooth - mae_smooth)

    detail = hf_prediction - smooth_rows(hf_prediction, detail_window)
    detail = detail - detail.mean(axis=1, keepdims=True)
    return trend + detail_beta * detail


def _candidate_window(name: str, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name)
    return int(match.group(1)) if match else None


def _build_trend(
    mae_prediction: np.ndarray,
    hf_prediction: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "mae_raw":
        return mae_prediction.copy()
    window = _candidate_window(method, "mae_smooth_w")
    if window is not None:
        return smooth_rows(mae_prediction, window)
    window = _candidate_window(method, "hf_smooth_w")
    if window is not None:
        return smooth_rows(hf_prediction, window)
    match = re.fullmatch(r"mae_plus_hf_lowfreq_w(\d+)_g(\d+\.\d+)", method)
    if match:
        window = int(match.group(1))
        gamma = float(match.group(2))
        mae_smooth = smooth_rows(mae_prediction, window)
        hf_smooth = smooth_rows(hf_prediction, window)
        return mae_prediction + gamma * (hf_smooth - mae_smooth)
    raise ValueError(f"Unknown TrendDetail trend method: {method}")


def _build_detail(
    mae_prediction: np.ndarray,
    hf_prediction: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "hf_shape_minus_mae_shape":
        return (hf_prediction - hf_prediction.mean(axis=1, keepdims=True)) - (
            mae_prediction - mae_prediction.mean(axis=1, keepdims=True)
        )
    for prefix, values in (
        ("hf_highpass_w", hf_prediction),
        ("hf_minus_smooth_mae_w", hf_prediction - mae_prediction),
        ("hf_minus_smooth_hf_w", hf_prediction),
    ):
        window = _candidate_window(method, prefix)
        if window is not None:
            detail = values - smooth_rows(values, window)
            return detail - detail.mean(axis=1, keepdims=True)
    raise ValueError(f"Unknown TrendDetail detail method: {method}")


class NeuralForecastComponent:
    """Load one saved NeuralForecast model and predict from total power history."""

    def __init__(self, model_dir: Path, alias: str, device: str) -> None:
        try:
            from neuralforecast import NeuralForecast
        except ImportError as exc:
            raise RuntimeError(
                "TrendDetail requires neuralforecast; install production requirements first"
            ) from exc

        self.alias = alias
        self.model = NeuralForecast.load(str(model_dir), map_location=device)
        accelerator = "gpu" if device == "cuda" else "cpu"
        for fitted_model in self.model.models:
            fitted_model.trainer_kwargs["accelerator"] = accelerator
            fitted_model.trainer_kwargs["devices"] = 1

    def predict(self, frame: pd.DataFrame, input_points: int) -> np.ndarray:
        history = frame[["ts", "total_power"]].tail(input_points).copy()
        if len(history) != input_points:
            raise ValueError(
                f"{self.alias} requires {input_points} history points, got {len(history)}"
            )
        history = history.rename(columns={"ts": "ds", "total_power": "y"})
        history.insert(0, "unique_id", "total_power")
        prediction = self.model.predict(df=history)
        value_columns = [
            column for column in prediction.columns if column not in {"unique_id", "ds"}
        ]
        if len(value_columns) != 1:
            raise ValueError(
                f"{self.alias} returned unexpected prediction columns: {value_columns}"
            )
        return prediction[value_columns[0]].to_numpy(dtype=float).reshape(1, -1)
