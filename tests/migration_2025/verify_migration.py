"""Historical training-parity diagnostic, not acceptance for the new clipping rule.

The original offline features retain negatives; differences are expected after
point_negative_to_zero_v1. Use run_platform_verification.py for current acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "tests")]
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from api import ForecastHandler
from input_adapter import STATION_LOAD_POINTS, STATION_WEATHER_POINTS
from models.utils import time_feature_frame
from predict import PowerPredictor
from platform_adapter import PLATFORM_PATH, PlatformForecastService
from time_policy import MODEL_TIMEZONE, TIME_POLICY_ID, TIMEZONE_BASIS
import build_real_test_payloads as fixture_builder
import run_api_test as smoke
import run_rolling_accuracy_test as rolling


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    started = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests/results/migration_2025")
    parser.add_argument("--update-examples", action="store_true", help="Replace handover examples only when explicitly requested")
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Protect the selected offline run including checkpoints and predictions.
    protected = {p: sha(p) for p in run.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    for name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    torch.set_num_threads(4)
    data_manifest = json.loads((run / "data/manifest.json").read_text(encoding="utf-8"))
    active = json.loads((ROOT / "models/active_model.json").read_text(encoding="utf-8"))
    model_dir = ROOT / "models" / active["model_dir"]
    manifest = json.loads((model_dir / active["manifest"]).read_text(encoding="utf-8"))
    artifact_hashes = {}
    for relative, expected_hash in manifest["artifacts"].items():
        if relative == "station_attention.pt":
            original = run / "models/StationAttention/checkpoint.pt"
        else:
            part = Path(relative)
            alias = {"nhits": "NHITS", "patchtst": "PatchTST"}[part.parts[0]]
            original = run / "models" / alias / "checkpoint" / part.name
        actual_hash = sha(model_dir / relative)
        assert actual_hash == sha(original) == expected_hash, relative
        artifact_hashes[relative] = actual_hash
    fusion = json.loads((run / "fusion_parameters.json").read_text(encoding="utf-8"))
    assert manifest["low_frequency"]["fusion"] == fusion["neural_fusion"]
    assert manifest["station_attention"]["mae_calibration"] == fusion["attention_mae"]
    assert manifest["station_attention"]["hf_calibration"] == fusion["attention_hf"]
    assert manifest["second_stage"]["mae_branch"] == fusion["station_fusion_mae"]
    assert manifest["second_stage"]["hf_branch"] == fusion["station_fusion_hf"]
    assert manifest["trend_detail"] == fusion["trend_detail"]
    for station in data_manifest["stations"]:
        raw_path = args.raw_dir / f"{station['station']}.csv"
        original_hashes = data_manifest["raw_files_sha256"]
        matches = [digest for path, digest in original_hashes.items() if Path(path).name == raw_path.name]
        assert len(matches) == 1 and sha(raw_path) == matches[0], raw_path
    for station in data_manifest["stations"]:
        assert set(station["power_points"]) == STATION_LOAD_POINTS[station["station"]]
        assert set(station["temperature_points"] + station["humidity_points"]) == STATION_WEATHER_POINTS[station["station"]]
    reference = pd.read_csv(run / "data/station_detail.csv", parse_dates=["ts"])
    state = pd.read_csv(run / "data/state_features.csv", parse_dates=["ts"]).drop(columns="total_power")
    reference = reference.merge(state, on="ts").set_index("ts")
    calendar = time_feature_frame(reference.index)
    for column in calendar:
        reference[column] = calendar[column].to_numpy()
    predictions = pd.read_csv(run / "test_predictions.csv", parse_dates=["ts", "cutoff"])
    max_errors = {"features": 0.0, "NHITS": 0.0, "PatchTST": 0.0, "StationAttention_raw": 0.0, "TrendDetail": 0.0}
    calls = []
    with tempfile.TemporaryDirectory(prefix="jingneng-migration-") as temp:
        predictor = PowerPredictor(device="cpu", history_cache_path=Path(temp) / "history.csv")
        assert len(predictor.backend.station.feature_columns) == 44
        original_predict = predictor.backend.predict

        def checked_predict(frame, cutoff):
            target = predictions[predictions.cutoff.eq(cutoff)]
            if target.empty and cutoff < predictions.cutoff.min():
                return original_predict(frame, cutoff)
            assert len(target) == 96
            prepared = predictor.backend.station.prepare_history(frame)
            columns = predictor.backend.station.feature_columns
            actual_features = prepared[columns].to_numpy(dtype=float)
            expected_features = reference.loc[pd.DatetimeIndex(prepared.ts), columns].to_numpy(dtype=float)
            max_errors["features"] = max(max_errors["features"], float(np.max(np.abs(actual_features - expected_features))))
            np.testing.assert_allclose(actual_features, expected_features, atol=1e-8, rtol=1e-10)
            for name, component in (("NHITS", predictor.backend.nhits), ("PatchTST", predictor.backend.patchtst)):
                values = component.predict(prepared, 288).ravel()
                expected = target[name].to_numpy()
                max_errors[name] = max(max_errors[name], float(np.max(np.abs(values - expected))))
                np.testing.assert_allclose(values, expected, atol=0.02, rtol=1e-5)
            raw, _ = predictor.backend.station.predict_raw(prepared, cutoff)
            expected = target.StationAttention_raw.to_numpy()
            max_errors["StationAttention_raw"] = max(max_errors["StationAttention_raw"], float(np.max(np.abs(raw.ravel() - expected))))
            np.testing.assert_allclose(raw.ravel(), expected, atol=0.05, rtol=1e-5)
            values = original_predict(frame, cutoff)
            max_errors["TrendDetail"] = max(max_errors["TrendDetail"], float(np.max(np.abs(values - target.TrendDetail.to_numpy()))))
            np.testing.assert_allclose(values, target.TrendDetail.to_numpy(), atol=0.02, rtol=1e-5)
            calls.append(str(cutoff))
            return values

        predictor.backend.predict = checked_predict
        class Handler(ForecastHandler):
            pass
        Handler.predictor = predictor
        Handler.platform_service = PlatformForecastService(predictor)
        Handler.latest_json = Path(temp) / "latest.json"
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        original_argv = sys.argv
        try:
            sys.argv = ["run_rolling_accuracy_test.py", "--base-url", f"http://127.0.0.1:{server.server_port}",
                        "--raw-dir", str(args.raw_dir.resolve()), "--output-dir", str(out / "rolling_accuracy")]
            rolling.main()
            # Exercise the same fixture generator and HTTP checks used by CI.
            predictor.backend.predict = original_predict
            predictor.history_cache = type(predictor.history_cache)(Path(temp) / "smoke.csv",
                state_centers=predictor.backend.station.state_centers,
                model_name=predictor.model_name + "__" + TIME_POLICY_ID)
            Handler.latest_json = Path(temp) / "smoke_latest.json"
            fixture_dir = Path(temp) / "fixtures"
            fixture_builder.make_payloads(fixture_builder.load_station_frames(ROOT / "tests/real_data_raw"),
                                          fixture_dir, pd.Timestamp("2025-10-12 23:45:00"))
            sys.argv = ["run_api_test.py", "--base-url", f"http://127.0.0.1:{server.server_port}", "--fixture-dir", str(fixture_dir)]
            smoke.main()
            status, before = smoke.request_json(f"http://127.0.0.1:{server.server_port}{PLATFORM_PATH}/latest")
            assert status == 200
            last_payload = json.loads((fixture_dir / "day_07.json").read_text(encoding="utf-8"))
            # New cache object models a service restart without erasing state.
            predictor.history_cache = type(predictor.history_cache)(Path(temp) / "smoke.csv",
                state_centers=predictor.backend.station.state_centers,
                model_name=predictor.model_name + "__" + TIME_POLICY_ID)
            status, after = smoke.request_json(f"http://127.0.0.1:{server.server_port}{PLATFORM_PATH}", method="POST", payload=last_payload)
            assert status == 200 and smoke.forecast_rows(before) == smoke.forecast_rows(after)
            invalid = dict(last_payload, frames=last_payload["frames"][:-1])
            assert smoke.request_json(f"http://127.0.0.1:{server.server_port}{PLATFORM_PATH}", method="POST", payload=invalid)[0] == 400
        finally:
            sys.argv = original_argv
            server.shutdown()
            server.server_close()
            thread.join()
    assert len(calls) == 73
    replay = pd.read_csv(out / "rolling_accuracy/rolling_predictions.csv", parse_dates=["ts"])
    np.testing.assert_array_equal(replay.ts.to_numpy(), predictions.ts.to_numpy())
    np.testing.assert_allclose(replay.actual_total_power, predictions.actual, atol=1e-8, rtol=1e-10)
    assert replay.ts.nunique() == 7008 and replay.predicted_power.notna().all()
    changed = [str(path) for path, digest in protected.items() if sha(path) != digest]
    assert not changed, changed
    report = {"status": "passed", "device": "cpu", "torch": torch.__version__, "test_days": 73,
              "started_at": started_at, "elapsed_seconds": time.monotonic() - started,
              "deployment_root": str(ROOT), "offline_run": str(run), "raw_dir": str(args.raw_dir.resolve()),
              "active_version": active["version"], "artifact_hashes_match_offline": artifact_hashes,
              "fusion_parameters_match_offline": True, "raw_csv_hashes_match_training_sources": True,
              "scoring_targets_match_offline": True, "examples_updated": args.update_examples,
              "api_timezone": "UTC", "training_timezone": MODEL_TIMEZONE,
              "time_policy": TIME_POLICY_ID, "timezone_basis": TIMEZONE_BASIS,
              "test_points": 7008, "max_absolute_difference_vs_offline": max_errors,
              "offline_files_verified_unchanged": len(protected), "retrained": False,
              "container_built": False, "github_uploaded": False,
              "local_ci_fixture_http_test": "passed", "cache_restart_repeated_request": "passed",
              "http_metrics": json.loads((out / "rolling_accuracy/summary.json").read_text(encoding="utf-8"))["metrics"]}
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.update_examples:
        samples = out / "rolling_accuracy/samples"
        for source, destination in (("input_success.json", "platform_input_example.json"), ("output_success.json", "platform_output_example.json"),
                                    ("input_not_ready.json", "platform_input_not_ready.json"),
                                    ("output_not_ready.json", "platform_output_not_ready.json")):
            shutil.copy2(samples / source, ROOT / "examples" / destination)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
