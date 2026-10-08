"""Run the production API over the complete offline test window.

The script submits one 96-point request per day, exactly as the production
caller does.  It scores a response only after the corresponding target time
has been reached in the historical files, so the reported metrics are actual
rolling API metrics rather than the response's historical accuracy estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from platform_test_utils import PLATFORM_PATH, validate_not_ready, validate_platform_prediction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PACKAGE_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from input_adapter import (  # noqa: E402
    POWER_POLICY_ID,
    STATION_FEATURES,
    STATION_LOAD_POINTS,
    STATION_WEATHER_POINTS,
)
from time_policy import MODEL_TIMEZONE, TIME_POLICY_ID, TIMEZONE_BASIS, TIMESTAMP_FORMAT, model_clock_to_utc, utc_to_model_clock


STATION_FILES = {
    "高安屯热电": "高安屯热电.csv",
    "京西热电": "京西热电.csv",
    "京阳热电": "京阳热电.csv",
    "京桥热电": "京桥热电.csv",
    "京丰燃气": "京丰燃气.csv",
    "未来热电": "未来热电.csv",
    "上庄热电": "上庄热电.csv",
}

POINTS_PER_DAY = 96
INTERVAL = pd.Timedelta(minutes=15)
DAY = pd.Timedelta(days=1)


def request_json(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"code": exc.code, "msg": raw}
        return exc.code, result


def load_station_frames(raw_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for station, filename in STATION_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing station CSV: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if "ts" not in frame.columns:
            raise ValueError(f"{path} has no ts column")
        frame["ts"] = pd.to_datetime(frame["ts"])
        if frame["ts"].duplicated().any():
            raise ValueError(f"{station}: duplicate timestamps")
        frame = frame.set_index("ts").sort_index()
        expected = set(STATION_LOAD_POINTS[station]) | set(
            STATION_WEATHER_POINTS[station]
        )
        missing = sorted(expected - set(frame.columns))
        if missing:
            raise ValueError(f"{station} is missing point columns: {missing}")
        frames[station] = frame
    return frames


def clean_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def make_payload(
    frames: dict[str, pd.DataFrame], start: pd.Timestamp
) -> dict[str, Any]:
    timeline = pd.date_range(start=start, periods=POINTS_PER_DAY, freq="15min")
    records: list[dict[str, Any]] = []
    for timestamp in timeline:
        record = {"timestamp": model_clock_to_utc(timestamp).strftime(TIMESTAMP_FORMAT)}
        for station in STATION_FEATURES:
            source = frames[station]
            row = source.loc[timestamp] if timestamp in source.index else None
            load_points = {
                point: clean_value(row.get(point)) if row is not None else None
                for point in STATION_LOAD_POINTS[station]
            }
            weather_points = {
                point: clean_value(row.get(point)) if row is not None else None
                for point in STATION_WEATHER_POINTS[station]
            }
            record.update(load_points)
            record.update(weather_points)
        records.append(record)
    return {
        "point_table": sorted(key for key in records[0] if key != "timestamp"),
        "frames": records,
    }


def build_actual_total(
    frames: dict[str, pd.DataFrame], timeline: pd.DatetimeIndex
) -> pd.Series:
    """Build the scored total with the same point-level fill rule as the API."""
    point_series: list[pd.Series] = []
    for station, frame in frames.items():
        for point in STATION_LOAD_POINTS[station]:
            values = pd.to_numeric(frame[point], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).reindex(timeline).fillna(0.0).clip(lower=0.0)
            point_series.append(values.rename(point))
    return pd.concat(point_series, axis=1).sum(axis=1)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - actual
    absolute = np.abs(error)
    nonzero = np.abs(actual) > 1e-8
    symmetric_denominator = np.abs(actual) + np.abs(predicted)
    smape_mask = symmetric_denominator > 1e-8
    ape = absolute[nonzero] / np.abs(actual[nonzero]) * 100.0
    smape = (
        2.0 * absolute[smape_mask] / symmetric_denominator[smape_mask] * 100.0
    )
    actual_mean = float(np.mean(actual))
    total_sum_squares = float(np.sum((actual - actual_mean) ** 2))
    return {
        "points": int(len(actual)),
        "mae_mw": float(np.mean(absolute)),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(ape)) if len(ape) else float("nan"),
        "smape_percent": float(np.mean(smape)) if len(smape) else float("nan"),
        "r2": float(1.0 - np.sum(error**2) / total_sum_squares)
        if total_sum_squares > 0
        else float("nan"),
        "actual_mean_mw": float(np.mean(actual)),
        "predicted_mean_mw": float(np.mean(predicted)),
    }


def write_plot(result: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 6), dpi=160)
    axis.plot(result["ts"], result["actual_total_power"], label="Actual", linewidth=1.2)
    axis.plot(
        result["ts"],
        result["predicted_power"],
        label="Rolling API prediction",
        linewidth=1.0,
    )
    axis.set_title("Production API rolling forecast vs actual total power")
    axis.set_xlabel("Timestamp (Asia/Shanghai, source CSV clock)")
    axis.set_ylabel("Total power (MW)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_outputs(
    result: pd.DataFrame, output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = result.sort_values("ts").reset_index(drop=True)
    result.to_csv(output_dir / "rolling_predictions.csv", index=False, encoding="utf-8-sig")

    daily_rows: list[dict[str, Any]] = []
    for target_date, group in result.groupby("target_date", sort=True):
        actual = group["actual_total_power"].to_numpy(dtype=float)
        predicted = group["predicted_power"].to_numpy(dtype=float)
        daily_rows.append({"target_date": str(target_date), **calculate_metrics(actual, predicted)})
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")

    actual = result["actual_total_power"].to_numpy(dtype=float)
    predicted = result["predicted_power"].to_numpy(dtype=float)
    summary = {
        "test_type": "rolling_production_api_accuracy",
        "request_contract": {
            "points_per_request": POINTS_PER_DAY,
            "interval_minutes": 15,
            "warmup_requests": config["warmup_days"],
            "successful_prediction_windows": config["target_windows"],
        },
        "target_start": config["target_start"],
        "target_end": config["target_end"],
        "raw_dir": str(config["raw_dir"]),
        "api_url": config["api_url"],
        "metrics": calculate_metrics(actual, predicted),
        "http_status_counts": config["http_status_counts"],
        "response_accuracy_is_not_used_for_scoring": True,
        "contract": "fluxcast_v1",
        "time_basis": {"api": "UTC", "source_csv_and_ts_column": MODEL_TIMEZONE,
                       "time_policy": TIME_POLICY_ID, "timezone_basis": TIMEZONE_BASIS},
        "target_start_utc": str(model_clock_to_utc(config["target_start"])),
        "target_end_utc": str(model_clock_to_utc(config["target_end"])),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_plot(result, output_dir / "rolling_accuracy_plot.png")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the production HTTP API on the complete rolling test window."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--raw-dir", type=Path, default=PACKAGE_ROOT / "tests" / "real_data_raw")
    parser.add_argument("--target-start", default="2025-10-20 00:00:00",
                        help="First target in source CSV Asia/Shanghai time; API traffic uses UTC")
    parser.add_argument("--target-windows", type=int, default=73)
    parser.add_argument("--warmup-days", type=int, default=14)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "tests" / "results" / "migration_2025" / "rolling_accuracy",
    )
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.target_windows <= 0 or args.warmup_days <= 0:
        raise ValueError("target-windows and warmup-days must be positive")
    target_start = pd.Timestamp(args.target_start)
    if target_start.tzinfo is not None:
        raise ValueError("target-start must use the source CSV's naive Asia/Shanghai clock")
    if target_start.minute % 15 or target_start.second or target_start.microsecond:
        raise ValueError("target-start must be aligned to a 15-minute boundary")

    frames = load_station_frames(args.raw_dir)
    first_request_end = target_start - (args.warmup_days - 1) * DAY - INTERVAL
    first_request_start = first_request_end - (POINTS_PER_DAY - 1) * INTERVAL
    target_end = target_start + (args.target_windows - 1) * DAY + (POINTS_PER_DAY - 1) * INTERVAL
    score_timeline = pd.date_range(
        start=target_start, end=target_end, freq="15min"
    )
    actual_total = build_actual_total(frames, score_timeline)
    actual_lookup = actual_total.to_dict()

    api_url = args.base_url.rstrip("/") + PLATFORM_PATH
    total_requests = args.warmup_days + args.target_windows - 1
    status_counts: dict[str, int] = {}
    scored_rows: list[dict[str, Any]] = []
    print(
        f"Rolling API test: {total_requests} requests, "
        f"{args.warmup_days} warmup + {args.target_windows} prediction windows"
    )
    print(f"Target range ({MODEL_TIMEZONE}): {target_start} -> {target_end}; API traffic: UTC")

    for request_index in range(total_requests):
        request_start = first_request_start + request_index * DAY
        request_end = request_start + (POINTS_PER_DAY - 1) * INTERVAL
        expected_target_start = request_end + INTERVAL
        payload = make_payload(frames, request_start)
        started = time.perf_counter()
        status, response = request_json(api_url, payload, args.request_timeout)
        elapsed = time.perf_counter() - started
        status_key = str(status)
        status_counts[status_key] = status_counts.get(status_key, 0) + 1

        expected_status = {200}
        if status not in expected_status:
            raise RuntimeError(
                f"request {request_index + 1}/{total_requests} "
                f"({request_start} -> {request_end}) expected HTTP {expected_status}, "
                f"got {status}: {response}"
            )

        if request_index in {0, args.warmup_days - 1}:
            sample_dir = args.output_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_name = "not_ready" if not response.get("result_point") else "success"
            for kind, value in (("input", payload), ("output", response)):
                (sample_dir / f"{kind}_{sample_name}.json").write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
                )

        print(
            f"[{request_index + 1:02d}/{total_requests}] HTTP {status} "
            f"{request_start:%Y-%m-%d %H:%M} -> {expected_target_start:%Y-%m-%d %H:%M} "
            f"({elapsed:.2f}s)"
        )

        if not response.get("result_point"):
            validate_not_ready(response)
            if expected_target_start >= target_start:
                raise RuntimeError("Platform returned no forecast during the scoring interval")
            continue
        validate_platform_prediction(response, payload)
        if expected_target_start < target_start:
            continue
        forecasts = [row for row in response["result_point"] if row["varname"] == "totalPowerForecast"]
        for index, row in enumerate(forecasts, start=1):
            predicted_time = utc_to_model_clock(row["timestamp"])
            if predicted_time not in actual_lookup:
                raise RuntimeError(f"Response timestamp cannot be aligned with actual data: {row}")
            scored_rows.append({
                "request_index": request_index + 1,
                "request_start": request_start,
                "request_end": request_end,
                "target_date": predicted_time.strftime("%Y-%m-%d"),
                "horizon_step": index,
                "ts": predicted_time,
                "timestamp_utc": row["timestamp"],
                "actual_total_power": float(actual_lookup[predicted_time]),
                "predicted_power": row["value"],
            })
        if args.request_delay:
            time.sleep(args.request_delay)

    result = pd.DataFrame(scored_rows)
    if len(result) != args.target_windows * POINTS_PER_DAY:
        raise RuntimeError(
            f"expected {args.target_windows * POINTS_PER_DAY} scored points, got {len(result)}"
        )
    result["error"] = result["predicted_power"] - result["actual_total_power"]
    result["abs_error"] = result["error"].abs()
    result["ape_percent"] = np.where(
        result["actual_total_power"].abs() > 1e-8,
        result["abs_error"] / result["actual_total_power"].abs() * 100.0,
        np.nan,
    )
    result["smape_percent"] = np.where(
        (result["actual_total_power"].abs() + result["predicted_power"].abs()) > 1e-8,
        2.0
        * result["abs_error"]
        / (result["actual_total_power"].abs() + result["predicted_power"].abs())
        * 100.0,
        np.nan,
    )
    result["ts"] = pd.to_datetime(result["ts"])
    config = {
        "warmup_days": args.warmup_days,
        "target_windows": args.target_windows,
        "target_start": str(target_start),
        "target_end": str(target_end),
        "raw_dir": str(args.raw_dir.resolve()),
        "api_url": api_url,
        "http_status_counts": status_counts,
        "power_policy": POWER_POLICY_ID,
        "scoring_actual_policy": "missing_power_zero_and_point_negative_to_zero",
    }
    summary = write_outputs(result, args.output_dir, config)
    metrics = summary["metrics"]
    print("\nRolling API test completed.")
    print(f"Actual scored points: {metrics['points']}")
    print(f"MAPE: {metrics['mape_percent']:.4f}%")
    print(f"RMSE: {metrics['rmse_mw']:.4f} MW")
    print(f"MAE: {metrics['mae_mw']:.4f} MW")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
