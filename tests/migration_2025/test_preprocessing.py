"""Small unit fixtures for edge cases, not forecast accuracy data."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import pandas as pd

from history_cache import HistoryNotReadyError, RealHistoryCache
from input_adapter import InputAdapter, InputValidationError, STATION_LOAD_POINTS, STATION_WEATHER_POINTS
from models.causal_state import add_causal_states


def payload(start="2025-10-01", count=96):
    return {"data": [{"ts": str(ts), "stations": {
        station: {"load_points": {p: 10.0 for p in points},
                  "weather_points": {p: 20.0 for p in STATION_WEATHER_POINTS[station]}}
        for station, points in STATION_LOAD_POINTS.items()}}
        for ts in pd.date_range(start, periods=count, freq="15min")]}


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "cache.csv"
        self.adapter = InputAdapter()

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, required=96, centers=None):
        return RealHistoryCache(self.path, required_points=required, state_centers=centers)

    def test_power_zero_negative_and_source_validity(self):
        data = payload()
        station = next(iter(STATION_LOAD_POINTS))
        point = next(iter(STATION_LOAD_POINTS[station]))
        for i, value in enumerate((None, "bad", float("inf"), -7.0, 0.0)):
            data["data"][i]["stations"][station]["load_points"][point] = value
        frame, _ = self.cache().merge(self.adapter.parse_json(data))
        np.testing.assert_array_equal(frame.total_power.head(5), [180, 180, 180, 180, 180])
        np.testing.assert_array_equal(frame[f"{station}_station_valid"].head(5), [0, 0, 0, 1, 1])

    def test_duplicate_missing_power_replaces_previous_measurement(self):
        data = payload()
        self.cache().merge(self.adapter.parse_json(data))
        for row in data["data"]:
            for station in row["stations"].values():
                station["load_points"] = {}
        frame, status = self.cache().merge(self.adapter.parse_json(data))
        self.assertTrue(frame.total_power.eq(0).all())
        self.assertEqual(status["cachedPoints"], 96)

    def test_negative_power_clipped_before_sum_but_weather_retained(self):
        data = payload()
        station = next(iter(STATION_LOAD_POINTS))
        load = next(iter(STATION_LOAD_POINTS[station]))
        weather = next(iter(STATION_WEATHER_POINTS[station]))
        data["data"][0]["stations"][station]["load_points"][load] = -10.0
        data["data"][0]["stations"][station]["weather_points"][weather] = -8.0
        frame, status = self.cache().merge(self.adapter.parse_json(data))
        self.assertEqual(frame.iloc[0]["total_power"], 180.0)
        self.assertEqual(frame.iloc[0][self.adapter._point_column("weather", station, weather)], -8.0)
        self.assertEqual(status["cacheFillQuality"]["negativePowerValuesClipped"], 1)

    def test_old_power_policy_cache_rejected_without_modifying_file(self):
        self.cache().merge(self.adapter.parse_json(payload()))
        cached = pd.read_csv(self.path).drop(columns="__power_policy")
        cached.to_csv(self.path, index=False)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(InputValidationError, "功率处理规则"):
            self.cache().merge(self.adapter.parse_json(payload()))
        self.assertEqual(self.path.read_bytes(), before)

    def test_weather_has_no_future_backfill(self):
        data = payload()
        station = next(iter(STATION_WEATHER_POINTS))
        point = next(iter(STATION_WEATHER_POINTS[station]))
        data["data"][0]["stations"][station]["weather_points"][point] = None
        with self.assertRaises(HistoryNotReadyError) as error:
            self.cache().merge(self.adapter.parse_json(data))
        self.assertEqual(error.exception.status["continuousPoints"], 95)

    def test_weather_correction_propagates_after_restart(self):
        data = payload()
        station = next(iter(STATION_WEATHER_POINTS))
        point = next(iter(STATION_WEATHER_POINTS[station]))
        for row in data["data"][1:]:
            row["stations"][station]["weather_points"][point] = None
        self.cache().merge(self.adapter.parse_json(data))
        data["data"][0]["stations"][station]["weather_points"][point] = 33.0
        frame, _ = self.cache().merge(self.adapter.parse_json(data))
        column = self.adapter._point_column("weather", station, point)
        self.assertTrue(frame[column].eq(33).all())
        next_day = payload("2025-10-02")
        for row in next_day["data"]:
            row["stations"][station]["weather_points"][point] = None
        frame, _ = self.cache().merge(self.adapter.parse_json(next_day))
        self.assertTrue(frame[column].eq(33).all())

    def test_state_persistence_across_eviction_and_restart(self):
        station = next(iter(STATION_LOAD_POINTS))
        col = f"{station}_total_power"
        point = next(iter(STATION_LOAD_POINTS[station]))
        centers = {col: [0.0, 100.0]}
        all_frames = []
        for day in range(12):
            data = payload(pd.Timestamp("2025-10-01") + pd.Timedelta(days=day))
            for i, row in enumerate(data["data"]):
                values = row["stations"][station]["load_points"]
                values.update({p: 0.0 for p in values})
                values[point] = 100.0 if day == 0 or i % 2 else 0.0
            parsed = self.adapter.parse_json(data)
            raw = parsed.frame.copy()
            loads, weather = self.adapter.point_columns()
            self.adapter.fill_missing_points(raw, loads, weather)
            all_frames.append(self.adapter.build_model_frame(raw))
            frame, status = self.cache(centers=centers).merge(parsed)
        expected = add_causal_states(pd.concat(all_frames, ignore_index=True), centers).tail(768)
        np.testing.assert_array_equal(frame[col + "_state"], expected[col + "_state"])
        self.assertEqual(status["cachedPoints"], 768)

    def test_reject_stale_batch_without_mutating_cache(self):
        data = payload()
        self.cache().merge(self.adapter.parse_json(data))
        old = self.path.read_bytes()
        with self.assertRaises(InputValidationError):
            self.cache().merge(self.adapter.parse_json(payload("2025-09-30")))
        self.assertEqual(old, self.path.read_bytes())

    def test_grid_and_station_validation(self):
        with self.assertRaises(InputValidationError):
            self.adapter.parse_json(payload("2025-10-01 00:01"))
        data = payload()
        data["data"][0]["stations"]["unknown"] = {}
        with self.assertRaises(InputValidationError):
            self.adapter.parse_json(data)

    def test_every_fifteen_minutes_overlap(self):
        self.cache().merge(self.adapter.parse_json(payload()))
        frame, status = self.cache().merge(self.adapter.parse_json(payload("2025-10-01 00:15")))
        self.assertEqual(status["cachedPoints"], 97)
        self.assertTrue(frame.total_power.eq(190).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
