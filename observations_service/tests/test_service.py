import ast
import copy
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from observations_service.api import ForecastSync, MAX_BODY, ROUTE, make_server
from observations_service.contract import EVENT_KEY, POINTS, STATION_POINTS, forecast_curve, measurement, utc_slot
from observations_service.launcher import supervise
from observations_service.storage import Store
from app.result_contract import forecast_response, validate_envelope

STAMP = "2026-11-25 05:00:00"
TARGET = utc_slot(STAMP)
ROOT = Path(__file__).resolve().parents[2]


def request(stamp=STAMP):
    values = (100, 120, 80, 150, 160, 80, 60, 140, 110, 115, 70, 90, 95, 100, 130, 50, 60, 40, 45)
    return {"point_table": list(POINTS),
            "frames": [{"timestamp": stamp, **dict(zip(POINTS, values))}]}


def curve(value=1850, start=TARGET):
    return {"event_key": EVENT_KEY, "result_point": [
        {"varname": "totalPowerForecast",
         "timestamp": datetime.fromtimestamp(start + i * 900, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "value": float(value)} for i in range(96)]}


def results(response):
    return {point["varname"]: point["value"]
            for points in (response["result_point"], response["extra_info"]) for point in points}


class ContractTests(unittest.TestCase):
    def test_point_mapping_matches_unchanged_predictor(self):
        tree = ast.parse((ROOT / "app/input_adapter.py").read_text(encoding="utf-8"))
        mapping = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "STATION_LOAD_POINTS" for t in node.targets))
        self.assertEqual({frozenset(points) for points in mapping.values()},
                         {frozenset(points) for points in STATION_POINTS.values()})
        self.assertEqual(len(POINTS), 19)

    def test_all_accepted_time_spellings_are_one_utc_instant(self):
        for separator in (" ", "T"):
            for suffix in ("", "Z", "+00:00", "+0000"):
                for fraction in ("", ".0", ".000000000"):
                    with self.subTest(separator=separator, suffix=suffix, fraction=fraction):
                        stamp = STAMP.replace(" ", separator) + fraction + suffix
                        self.assertEqual(measurement(request(stamp))["slot"], TARGET)
                        self.assertEqual(measurement(request(stamp))["timestamp"], stamp)

    def test_invalid_timestamps_are_not_rounded_or_converted(self):
        for stamp in (None, 1, "2026-11-25", "2026-11-25 05:01:00", "2026-11-25 05:00:01",
                      "2026-11-25 05:00:00.000000001Z", "2026-11-25T13:00:00+08:00",
                      "2026-11-31 05:00:00", "2026-11-25 05:00:00Z ", STAMP + "-00:00"):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                utc_slot(stamp)

    def test_complete_sum_and_true_zero_negative(self):
        self.assertEqual(measurement(request())["total"], 1795.0)
        payload = request()
        payload["frames"][0][POINTS[0]] = 0
        payload["frames"][0][POINTS[1]] = -5
        self.assertEqual(measurement(payload)["total"], 1575.0)
        payload["frames"][0].update(dict.fromkeys(POINTS, 0))
        self.assertEqual(measurement(payload)["status"], "complete")
        self.assertEqual(measurement(payload)["total"], 0.0)

    def test_unusable_values_and_omission_are_missing_not_zero(self):
        cases = ((None, "null"), (True, "invalid_type(bool)"), (False, "invalid_type(bool)"),
                 ("100", "invalid_type(str)"), ("bad", "invalid_type(str)"),
                 ([], "invalid_type(list)"), ({}, "invalid_type(dict)"),
                 (math.nan, "nonfinite"), (math.inf, "nonfinite"), (-math.inf, "nonfinite"),
                 (10**500, "numeric_overflow"))
        for value, reason in cases:
            with self.subTest(value=str(value)[:20]):
                payload = request()
                payload["frames"][0][POINTS[0]] = value
                item = measurement(payload)
                self.assertEqual(item["status"], "incomplete")
                self.assertIsNone(item["total"])
                self.assertEqual(item["missing"], [POINTS[0]])
                self.assertEqual(item["issues"][POINTS[0]]["reason"], reason)
                json.dumps(item, allow_nan=False)
        payload = request()
        del payload["frames"][0][POINTS[0]]
        item = measurement(payload)
        self.assertEqual(item["status"], "incomplete")
        self.assertEqual(item["issues"][POINTS[0]], {"reason": "missing_field", "value": "<absent>"})

    def test_all_missing(self):
        payload = request()
        payload["frames"] = [{"timestamp": STAMP}]
        self.assertEqual(measurement(payload)["status"], "missing")

    def test_bad_schema(self):
        malformed = [None, [], {}, {"point_table": []}, {**request(), "unexpected": 1}]
        for frames in ([], [request()["frames"][0]] * 2, [None], {}):
            malformed.append({**request(), "frames": frames})
        for table in ([], list(POINTS[:-1]), list(POINTS) + [POINTS[0]],
                      list(POINTS[:-1]) + ["unknown"], [None] * 19, {}, list(POINTS[:-1]) + [POINTS[0]]):
            malformed.append({**request(), "point_table": table})
        payload = request()
        payload["frames"][0]["unknown"] = 1
        malformed.append(payload)
        for payload in malformed:
            with self.subTest(payload=str(payload)[:90]), self.assertRaises(ValueError):
                measurement(payload)

    def test_power_sum_overflow_is_not_a_request_error(self):
        payload = request()
        payload["frames"][0].update(dict.fromkeys(POINTS, 1e308))
        item = measurement(payload)
        self.assertEqual(item["status"], "complete")
        self.assertIsNone(item["total"])
        self.assertEqual(item["aggregate_issue"], "power_sum_out_of_range")
        self.assertEqual(item["issues"], {})

    def test_forecast_validation(self):
        invalid = []
        payload = curve()
        payload["result_point"].pop()
        invalid.append(payload)
        for name, value in (("value", math.nan), ("value", True), ("value", "1850"),
                            ("varname", "wrong"), ("timestamp", STAMP + "+08:00"),
                            ("timestamp", "2026-11-25 05:01:00")):
            payload = curve()
            payload["result_point"][0][name] = value
            invalid.append(payload)
        payload = curve()
        payload["result_point"][1] = payload["result_point"][0].copy()
        invalid.append(payload)
        invalid.extend([None, {}, {**curve(), "event_key": "wrong"}])
        for payload in invalid:
            with self.subTest(payload=str(payload)[:80]), self.assertRaises(ValueError):
                forecast_curve(payload)
        self.assertEqual(len(forecast_curve(curve())), 96)

    def test_metadata_uses_generation_time_and_preserves_target_curve(self):
        points = curve()["result_point"]
        instant = datetime(2026, 11, 25, 4, 52, 37, tzinfo=timezone.utc)
        result = forecast_response(points, instant)
        validate_envelope(result)
        self.assertEqual(result["result_point"][:96], points)
        self.assertEqual(results(result)["forecastGeneratedAt"], "2026-11-25T04:52:37Z")
        self.assertEqual(forecast_curve(result), forecast_curve(curve()))
        self.assertNotEqual(results(result)["forecastBatchId"],
                            results(forecast_response(points, instant))["forecastBatchId"])

    def test_response_types_and_utc_duplicate_keys_rejected(self):
        for value in ("1.0", True, 10**500):
            bad = curve()
            bad["result_point"][0]["value"] = value
            with self.assertRaises(ValueError):
                validate_envelope(bad)
        bad = curve()
        duplicate = dict(bad["result_point"][0], timestamp=STAMP)
        bad["result_point"].append(duplicate)
        with self.assertRaises(ValueError):
            validate_envelope(bad)
        bad = curve()
        bad["extra_info"] = [{"varname": "reason", "timestamp": STAMP, "value": 1.0}]
        with self.assertRaises(ValueError):
            validate_envelope(bad)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "observations.sqlite3"
        self.store = Store(self.path)

    def test_no_forecast_returns_actual_only(self):
        response = self.store.receive(request())
        values = results(response)
        self.assertEqual(values["totalPowerActual"], 1795.0)
        self.assertEqual(values["dataStatus"], "complete")
        self.assertEqual(values["reason"], "no_matching_forecast")
        self.assertNotIn("totalPowerDeviation", values)
        self.assertNotIn("forecastBatchId", values)

    def test_matching_and_format_preservation(self):
        self.store.archive(curve(), TARGET - 1)
        for stamp in (STAMP, STAMP.replace(" ", "T") + "Z", STAMP + ".000000000+0000"):
            response = self.store.receive(request(stamp))
            values = results(response)
            self.assertEqual(values["totalPowerActual"], 1795.0)
            self.assertEqual(values["totalPowerDeviation"], 55.0)
            self.assertEqual(values["dataStatus"], "complete")
            self.assertEqual(values["latestActualAt"], stamp)
            validate_envelope(response)
            for group in ("result_point", "extra_info"):
                self.assertTrue(all(p["timestamp"] == stamp for p in response[group]))
                self.assertEqual(len(response[group]), len({p["varname"] for p in response[group]}))
            self.assertTrue(all(type(p["value"]) is float for p in response["result_point"]))
        self.assertEqual(self.store.counts()["observations"], 1)

    def test_late_or_same_time_forecasts_are_not_eligible(self):
        for seen in (TARGET, TARGET + 900):
            self.store.archive(curve(value=seen), seen)
        self.assertNotIn("totalPowerDeviation", results(self.store.receive(request())))

    def test_same_batch_repoll_does_not_change_first_seen(self):
        batch = self.store.archive(curve(), TARGET + 1)
        self.assertEqual(batch, self.store.archive(curve(), TARGET - 1))
        self.assertNotIn("totalPowerDeviation", results(self.store.receive(request())))

    def test_equivalent_format_and_order_do_not_create_another_batch(self):
        batch = self.store.archive(curve(), TARGET - 10)
        alternate = curve()
        for point in alternate["result_point"]:
            point["timestamp"] = point["timestamp"].replace("T", " ").removesuffix("Z")
        alternate["result_point"].reverse()
        self.assertEqual(batch, self.store.archive(alternate, TARGET + 10))
        self.assertEqual(self.store.counts()["batches"], 1)
        self.assertEqual(results(self.store.receive(request()))["totalPowerDeviation"], 55.0)

    def test_fixed_association_survives_restart(self):
        self.store.archive(curve(), TARGET - 100)
        self.store.receive(request())
        self.store.archive(curve(2000), TARGET - 50)
        self.assertEqual(results(Store(self.path).receive(request()))["totalPowerDeviation"], 55.0)

    def test_corrections_keep_original_batch_even_after_new_batch(self):
        self.store.archive(curve(), TARGET - 100)
        self.store.receive(request())
        self.store.archive(curve(2000), TARGET - 50)
        payload = request(STAMP + "Z")
        payload["frames"][0][POINTS[0]] = 90
        response = self.store.receive(payload)
        self.assertEqual(results(response)["totalPowerDeviation"], 65.0)
        self.assertEqual(self.store.counts(), {"batches": 2, "observations": 1})

    def test_latest_eligible_batch_selected_on_first_receipt(self):
        self.store.archive(curve(), TARGET - 100)
        self.store.archive(curve(2000), TARGET - 50)
        self.store.archive(curve(9000), TARGET + 10)
        self.assertEqual(results(self.store.receive(request()))["totalPowerDeviation"], 205.0)

    def test_archived_batch_survives_restart_and_new_latest_curve(self):
        self.store.archive(curve(), TARGET - 100)
        self.store.archive(curve(2000, TARGET + 86400), TARGET + 100)
        store = Store(self.path)
        self.assertEqual(results(store.receive(request()))["totalPowerDeviation"], 55.0)

    def test_no_nearest_time_or_eight_hour_shift(self):
        self.store.archive(curve(start=TARGET + 900), TARGET - 100)
        self.assertNotIn("totalPowerDeviation", results(self.store.receive(request())))
        midnight = "2026-11-25 23:45:00"
        self.store.archive(curve(start=utc_slot(midnight)), TARGET - 100)
        self.assertEqual(results(self.store.receive(request("2026-11-26T00:00:00Z")))["totalPowerDeviation"], 55.0)

    def test_incomplete_preserves_complete_stations_but_omits_total_and_shares(self):
        self.store.archive(curve(), TARGET - 10)
        payload = request()
        payload["frames"][0][POINTS[0]] = None
        response = self.store.receive(payload)
        values = results(response)
        self.assertNotIn("totalPowerActual", values)
        self.assertNotIn("totalPowerDeviation", values)
        self.assertNotIn("GARD_powerActual", values)
        self.assertEqual(values["GARD_validPointCount"], 2.0)
        self.assertEqual(values["GARD_requiredPointCount"], 3.0)
        self.assertEqual(values["JXRD_powerActual"], 590.0)
        self.assertFalse(any(name.endswith("powerShare") for name in values))
        self.assertEqual(results(response)["dataStatus"], "incomplete")
        payload["frames"] = [{"timestamp": STAMP}]
        response = self.store.receive(payload)
        self.assertEqual(len(response["result_point"]), 14)
        self.assertTrue(all(results(response)[code + "_validPointCount"] == 0.0 for code in STATION_POINTS))
        self.assertEqual(results(response)["dataStatus"], "missing")
        self.assertEqual(results(self.store.receive(request()))["totalPowerDeviation"], 55.0)

    def test_station_sums_shares_zero_and_negative_normalization(self):
        response = self.store.receive(request())
        values = results(response)
        expected = {"GARD": 300.0, "JXRD": 590.0, "JYRD": 295.0, "JQRD": 285.0,
                    "JFRD": 130.0, "WLRD": 110.0, "SZRD": 85.0}
        for code, value in expected.items():
            self.assertEqual(values[code + "_powerActual"], value)
            self.assertAlmostEqual(values[code + "_powerShare"], value / 1795 * 100)
            self.assertEqual(values[code + "_dataStatus"], "complete")
        self.assertAlmostEqual(sum(values[c + "_powerShare"] for c in expected), 100.0)
        payload = request()
        payload["frames"][0][POINTS[0]] = -10.0
        values = results(self.store.receive(payload))
        self.assertEqual(values["GARD_powerActual"], 200.0)
        self.assertEqual(values["totalPowerActual"], 1695.0)
        payload["frames"][0].update(dict.fromkeys(POINTS, 0.0))
        values = results(self.store.receive(payload))
        self.assertEqual(values["totalPowerActual"], 0.0)
        self.assertFalse(any(name.endswith("powerShare") for name in values))

    def test_each_bad_point_only_omits_dependent_numbers_and_logs_its_cause(self):
        self.store.archive(curve(), TARGET - 10)
        normal = self.store.receive(request())
        expected = {p["varname"] for p in normal["result_point"]}
        shared = {"totalPowerActual", "totalPowerDeviation"} | {c + "_powerShare" for c in STATION_POINTS}
        for code, points in STATION_POINTS.items():
            for name in points:
                with self.subTest(point=name):
                    payload = request()
                    payload["frames"][0][name] = "bad power"
                    with self.assertLogs("observations", level="WARNING") as logs:
                        response = self.store.receive(payload)
                    validate_envelope(response)
                    values = results(response)
                    omitted = expected - {p["varname"] for p in response["result_point"]}
                    self.assertEqual(omitted, shared | {code + "_powerActual"})
                    self.assertEqual(values[code + "_validPointCount"], float(len(points) - 1))
                    self.assertEqual(values["reason"], "missing_power_points")
                    self.assertIn(name, values["message"])
                    self.assertIn("invalid_type(str)", values["message"])
                    diagnostics = json.loads(logs.records[0].getMessage().split(": ", 1)[1])
                    problem = diagnostics["causes"][0]
                    self.assertEqual(diagnostics["timestamp"], STAMP)
                    self.assertEqual(set(problem["omitted"]), omitted)
                    self.assertEqual(problem["inputs"][name],
                                     {"reason": "invalid_type(str)", "value": "'bad power'"})
                    with self.store.connect() as connection:
                        saved = json.loads(connection.execute("SELECT input_json FROM observations").fetchone()[0])
                    self.assertEqual(saved["issues"], problem["inputs"])

    def test_sum_overflow_preserves_other_results_and_reports_all_dependencies(self):
        self.store.archive(curve(), TARGET - 10)
        for bad_station in (False, True):
            with self.subTest(station_sum_overflows=bad_station):
                payload = request()
                payload["frames"][0][POINTS[0]] = 1e308
                payload["frames"][0][POINTS[1 if bad_station else 3]] = 1e308
                with self.assertLogs("observations", level="WARNING") as logs:
                    response = self.store.receive(payload)
                validate_envelope(response)
                values = results(response)
                self.assertEqual(values["dataStatus"], "complete")
                self.assertEqual(values["reason"], "power_sum_out_of_range")
                self.assertNotIn("totalPowerActual", values)
                self.assertNotIn("totalPowerDeviation", values)
                self.assertNotIn("latestActualAt", values)
                self.assertEqual(values["JYRD_powerActual"], 295.0)
                self.assertEqual(values["GARD_validPointCount"], 3.0)
                self.assertFalse(any(name.endswith("powerShare") for name in values))
                if bad_station:
                    self.assertNotIn("GARD_powerActual", values)
                    self.assertEqual(values["GARD_reason"], "power_sum_out_of_range")
                    self.assertEqual(values["JXRD_powerActual"], 590.0)
                else:
                    self.assertEqual(values["GARD_powerActual"], 1e308)
                    self.assertEqual(values["JXRD_powerActual"], 1e308)
                problems = json.loads(logs.records[0].getMessage().split(": ", 1)[1])["causes"]
                self.assertEqual(problems[0]["inputs"][POINTS[0]], 1e308)
                self.assertTrue(all(set(p["omitted"]).isdisjoint(values) for p in problems))
                self.assertTrue(all(all(name in values["message"] for name in p["omitted"]) for p in problems))

    def test_zero_negative_and_unmatched_forecast_have_distinct_explanations(self):
        payload = request()
        payload["frames"][0].update(dict.fromkeys(POINTS, -5.0))
        for matched in (False, True):
            if matched:
                self.store.archive(curve(), TARGET - 1)
            with self.assertLogs("observations", level="INFO") as logs:
                response = self.store.receive(payload)
            values = results(response)
            self.assertEqual(values["dataStatus"], "complete")
            self.assertNotIn("missingPoints", values)
            self.assertEqual(values["totalPowerActual"], 0.0)
            self.assertIn("total_power_zero", values["message"])
            self.assertNotIn("missing_power_points", values["message"])
            self.assertTrue(all(record.levelname == "INFO" for record in logs.records))
            for code, points in STATION_POINTS.items():
                self.assertEqual(values[code + "_powerActual"], 0.0)
                self.assertEqual(values[code + "_validPointCount"], float(len(points)))
                self.assertIn(code + "_powerShare", values["message"])
            if matched:
                self.assertEqual(values["totalPowerDeviation"], 1850.0)
                self.assertEqual(values["reason"], "total_power_zero")
            else:
                self.assertNotIn("totalPowerDeviation", values)
                self.assertIn("no_matching_forecast", values["message"])

    def test_deviation_overflow_preserves_actual_station_powers_and_shares(self):
        self.store.archive(curve(-1e308), TARGET - 1)
        payload = request()
        payload["frames"][0][POINTS[0]] = 1e308
        with self.assertLogs("observations", level="WARNING") as logs:
            response = self.store.receive(payload)
        validate_envelope(response)
        values = results(response)
        self.assertEqual(values["totalPowerActual"], 1e308)
        self.assertNotIn("totalPowerDeviation", values)
        self.assertEqual(values["reason"], "deviation_out_of_range")
        self.assertEqual(values["latestActualAt"], STAMP)
        self.assertEqual(values["JXRD_powerActual"], 590.0)
        self.assertEqual(values["GARD_powerShare"], 100.0)
        problem = json.loads(logs.records[0].getMessage().split(": ", 1)[1])["causes"][0]
        self.assertEqual(problem["omitted"], ["totalPowerDeviation"])
        self.assertEqual(problem["inputs"], {"totalPowerActual": 1e308, "totalPowerForecast": -1e308})

    def test_overflow_correction_and_restart_update_latest_calculable_actual(self):
        newer = "2026-11-25 05:15:00"
        self.store.receive(request())
        self.store.receive(request(newer))
        overflow = request(newer)
        overflow["frames"][0].update(dict.fromkeys(POINTS, 1e308))
        with self.assertLogs("observations", level="WARNING"):
            response = Store(self.path).receive(overflow)
        self.assertEqual(results(response)["latestActualAt"], STAMP)
        corrected = Store(self.path).receive(request(newer))
        self.assertEqual(results(corrected)["latestActualAt"], newer)
        self.assertNotIn("power_sum_out_of_range", results(corrected)["message"])
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("SELECT complete FROM observations WHERE target=?",
                                                (utc_slot(newer),)).fetchone()[0], 1)
        self.assertEqual(self.store.counts()["observations"], 2)

    def test_published_batch_id_restart_and_conflict(self):
        payload = forecast_response(curve()["result_point"], datetime.fromtimestamp(TARGET - 20, timezone.utc))
        batch = self.store.archive(payload, TARGET - 10)
        self.assertTrue(batch.startswith("dayahead-"))
        self.assertEqual(results(self.store.receive(request()))["forecastBatchId"], batch)
        self.assertEqual(Store(self.path).archive(payload, TARGET + 1), batch)
        revised = copy.deepcopy(payload)
        revised["result_point"][0]["value"] += 1.0
        with self.assertRaises(ValueError):
            self.store.archive(revised, TARGET - 1)
        self.assertEqual(self.store.counts()["batches"], 1)
        fresh = forecast_response(curve()["result_point"], datetime.fromtimestamp(TARGET - 5, timezone.utc))
        self.assertNotEqual(self.store.archive(fresh, TARGET - 1), batch)
        self.assertEqual(results(Store(self.path).receive(request()))["forecastBatchId"], batch)

    def test_generation_and_archive_must_both_precede_target(self):
        for generated, seen in ((TARGET, TARGET - 1), (TARGET - 1, TARGET), (TARGET + 1, TARGET - 1)):
            payload = forecast_response(curve()["result_point"], datetime.fromtimestamp(generated, timezone.utc))
            self.store.archive(payload, seen)
        self.assertNotIn("totalPowerDeviation", results(self.store.receive(request())))

    def test_latest_complete_actual_handles_late_and_corrected_measurements(self):
        newer = "2026-11-25 05:15:00"
        self.store.receive(request(newer))
        self.assertEqual(results(self.store.receive(request()))["latestActualAt"], newer)
        incomplete = request(newer)
        incomplete["frames"][0][POINTS[0]] = None
        self.assertEqual(results(Store(self.path).receive(incomplete))["latestActualAt"], STAMP)
        incomplete = request()
        incomplete["frames"] = [{"timestamp": STAMP}]
        self.assertNotIn("latestActualAt", results(self.store.receive(incomplete)))

    def test_legacy_database_migration_preserves_association(self):
        # Construct the previous schema independently of Store's migration.
        old_path = Path(self.temp.name) / "old.sqlite3"
        with closing(sqlite3.connect(old_path)) as connection, connection:
            connection.executescript("""
                CREATE TABLE batches(id TEXT PRIMARY KEY, first_seen REAL NOT NULL, curve_json TEXT NOT NULL);
                CREATE TABLE forecast_points(batch_id TEXT, target INTEGER, value REAL, PRIMARY KEY(batch_id,target));
                CREATE TABLE observations(target INTEGER PRIMARY KEY, batch_id TEXT, input_json TEXT,
                                          response_json TEXT, updated_at REAL);
            """)
            connection.execute("INSERT INTO batches VALUES (?, ?, ?)", ("old", TARGET - 1, "[]"))
            connection.execute("INSERT INTO forecast_points VALUES (?, ?, ?)", ("old", TARGET, 1850.0))
            connection.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?)",
                               (TARGET, "old", "{}", json.dumps({"result_point": [
                                   {"varname": "totalPowerActual", "value": 1795.0}]}), TARGET))
        values = results(Store(old_path).receive(request()))
        self.assertEqual(values["totalPowerDeviation"], 55.0)
        self.assertEqual(values["latestActualAt"], STAMP)
        self.assertNotIn("forecastBatchId", values)

    def test_concurrent_retries_are_idempotent(self):
        self.store.archive(curve(), TARGET - 100)
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: self.store.receive(request()), range(16)))
        self.assertTrue(all(response == responses[0] for response in responses))
        self.assertEqual(self.store.counts()["observations"], 1)

    def test_invalid_curve_does_not_partially_archive(self):
        payload = curve()
        payload["result_point"][-1]["value"] = None
        with self.assertRaises(ValueError):
            self.store.archive(payload, TARGET - 10)
        self.assertEqual(self.store.counts()["batches"], 0)

    def test_example_files_are_consistent_and_explicitly_illustrative(self):
        folder = ROOT / "observations_service/examples"
        copies = {"input_complete.json": "observations_input_example.json",
                  "output_complete.json": "observations_output_example.json",
                  **{f"output_{state}.json": f"observations_output_{state}.json"
                     for state in ("no_forecast", "incomplete", "missing")}}
        for original, exported in copies.items():
            self.assertEqual(json.loads((folder / original).read_text(encoding="utf-8")),
                             json.loads((ROOT / "examples" / exported).read_text(encoding="utf-8")))
        payload = json.loads((folder / "input_complete.json").read_text(encoding="utf-8"))
        self.assertEqual(payload, request())
        self.assertEqual(self.store.receive(payload), json.loads(
            (folder / "output_no_forecast.json").read_text(encoding="utf-8")))
        published = forecast_response(curve()["result_point"], datetime.fromtimestamp(TARGET - 300, timezone.utc),
                                      "dayahead-" + "1" * 32)
        self.store.archive(published, TARGET - 1)
        self.assertEqual(self.store.receive(payload), json.loads(
            (folder / "output_complete.json").read_text(encoding="utf-8")))
        for state in ("incomplete", "missing"):
            payload = json.loads((folder / f"input_{state}.json").read_text(encoding="utf-8"))
            self.assertEqual(self.store.receive(payload), json.loads(
                (folder / f"output_{state}.json").read_text(encoding="utf-8")))


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "db.sqlite3")
        self.upstream_requests = []
        self.upstream_payload = curve()
        self.upstream_status = 200
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.upstream_requests.append(("GET", self.path))
                self.send_response(owner.upstream_status)
                self.end_headers()
                self.wfile.write(json.dumps(owner.upstream_payload).encode())

            def do_POST(self):
                owner.upstream_requests.append(("POST", self.path))
                self.send_response(500)
                self.end_headers()

            def log_message(self, *_):
                pass

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.url = f"http://127.0.0.1:{self.upstream.server_port}/api/v1/fluxcast/compute/latest"
        self.sync = ForecastSync(self.store, self.url, interval=60, timeout=1)
        self.server = make_server("127.0.0.1", 0, self.store, self.sync)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_servers)

    def close_servers(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        if self.sync.thread.is_alive():
            self.sync.close()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join()

    def call(self, method="POST", path=ROUTE, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body,
                               headers=headers or {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_http_sync_is_get_only_and_post_never_runs_prediction(self):
        before = copy.deepcopy(self.upstream_payload)
        with patch("observations_service.api.utc_now", return_value=TARGET - 1):
            self.sync.once()
        count = len(self.upstream_requests)
        code, response = self.call(body=json.dumps(request()))
        self.assertEqual(code, 200)
        self.assertEqual(results(response)["totalPowerDeviation"], 55.0)
        self.assertEqual(len(self.upstream_requests), count)
        self.assertEqual(self.upstream_requests, [("GET", "/api/v1/fluxcast/compute/latest")])
        self.assertEqual(self.upstream_payload, before)

    def test_upstream_failure_retains_archived_predictions(self):
        self.store.archive(curve(), TARGET - 1)
        for status, payload in ((500, {}), (404, {}), (200, {})):
            self.upstream_status, self.upstream_payload = status, payload
            self.sync.once()
            code, response = self.call(body=json.dumps(request()))
            self.assertEqual(code, 200)
            self.assertEqual(results(response)["totalPowerDeviation"], 55.0)

    def test_timeout_does_not_block_actual_request(self):
        with patch.object(self.sync.opener, "open", side_effect=TimeoutError("timeout")):
            self.sync.once()
        code, response = self.call(body=json.dumps(request()))
        self.assertEqual(code, 200)
        self.assertEqual(results(response)["totalPowerActual"], 1795.0)
        self.assertEqual(self.sync.status, "unavailable")

    def test_bad_requests_and_paths(self):
        for body in ('{}', '{', 'NaN', '{"point_table":[],"point_table":[],"frames":[]}',
                     json.dumps(request()) + "garbage"):
            with self.subTest(body=body[:50]):
                self.assertEqual(self.call(body=body)[0], 400)
        self.assertEqual(self.call(body="{}", headers={"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.call(path="/api/v1/fluxcast/compute", body="{}")[0], 404)
        self.assertEqual(self.call("GET", ROUTE)[0], 404)
        self.assertEqual(self.call(body="{}", headers={
            "Content-Type": "application/json", "Content-Length": str(MAX_BODY + 1)})[0], 413)

    def test_unusable_point_values_return_partial_success_and_traceable_reasons(self):
        for value, reason in ((None, "null"), (True, "invalid_type(bool)"),
                              ("bad", "invalid_type(str)"), (10**500, "numeric_overflow")):
            with self.subTest(reason=reason):
                payload = request()
                payload["frames"][0][POINTS[0]] = value
                with self.assertLogs("observations", level="WARNING") as logs:
                    status, response = self.call(body=json.dumps(payload))
                self.assertEqual(status, 200)
                validate_envelope(response)
                values = results(response)
                self.assertEqual(values["JXRD_powerActual"], 590.0)
                self.assertNotIn("totalPowerActual", values)
                self.assertIn(reason, values["message"])
                self.assertIn(POINTS[0], logs.records[0].getMessage())
                self.assertIn("GARD_powerActual", logs.records[0].getMessage())
        payload = request()
        payload["frames"][0][POINTS[0]] = "NUMERIC_TOKEN"
        encoded = json.dumps(payload)
        with self.assertLogs("observations", level="WARNING"):
            status, response = self.call(body=encoded.replace('"NUMERIC_TOKEN"', "1e309"))
        self.assertEqual(status, 200)
        self.assertIn("nonfinite", results(response)["message"])
        for token in ("NaN", "Infinity", "-Infinity"):
            self.assertEqual(self.call(body=encoded.replace('"NUMERIC_TOKEN"', token))[0], 400)

    def test_finite_sum_overflow_does_not_reject_entire_http_request(self):
        payload = request()
        payload["frames"][0][POINTS[0]] = 1e308
        payload["frames"][0][POINTS[1]] = 1e308
        with self.assertLogs("observations", level="WARNING"):
            status, response = self.call(body=json.dumps(payload))
        self.assertEqual(status, 200)
        validate_envelope(response)
        values = results(response)
        self.assertEqual(values["reason"], "power_sum_out_of_range")
        self.assertEqual(values["JXRD_powerActual"], 590.0)
        self.assertEqual(values["GARD_validPointCount"], 3.0)
        self.assertNotIn("totalPowerActual", values)
        self.assertNotIn("latestActualAt", values)
        self.assertEqual(self.upstream_requests, [])

    def test_storage_failure_never_returns_success(self):
        with patch.object(self.store, "receive", side_effect=OSError("database unavailable")):
            with self.assertLogs("observations", level="ERROR"):
                code, response = self.call(body=json.dumps(request()))
            self.assertEqual(code, 500)
            self.assertEqual(response["result_point"], [])
        with patch.object(self.store, "counts", side_effect=OSError("database unavailable")):
            with self.assertLogs("observations", level="ERROR"):
                self.assertEqual(self.call("GET", "/health")[0], 503)

    def test_health_requires_live_archive_worker_but_not_ready_forecast(self):
        self.assertEqual(self.call("GET", "/health")[0], 503)
        self.upstream_status = 404
        self.sync.start()
        self.assertEqual(self.call("GET", "/health")[0], 200)


class LauncherTests(unittest.TestCase):
    def test_child_exit_stops_other_child_and_fails_container(self):
        failed, other = Mock(), Mock()
        failed.pid = 12345
        failed.poll.return_value = 0
        other.poll.return_value = None
        with patch("observations_service.launcher.subprocess.Popen", side_effect=[failed, other]):
            self.assertEqual(supervise([["predict"], ["observe"]], threading.Event()), 1)
        other.terminate.assert_called_once()
        other.wait.assert_called_once()

    def test_stop_shuts_down_both_children(self):
        children = [Mock(), Mock()]
        for child in children:
            child.poll.return_value = None
        stop = threading.Event()
        stop.set()
        with patch("observations_service.launcher.subprocess.Popen", side_effect=children):
            self.assertEqual(supervise([["predict"], ["observe"]], stop), 0)
        for child in children:
            child.terminate.assert_called_once()
            child.wait.assert_called_once()

    def test_failed_second_start_cleans_up_first_child(self):
        child = Mock()
        child.poll.return_value = None
        with patch("observations_service.launcher.subprocess.Popen", side_effect=[child, OSError("failed")]):
            with self.assertRaises(OSError):
                supervise([["predict"], ["observe"]], threading.Event())
        child.terminate.assert_called_once()

    def test_stuck_process_is_killed_after_grace_period(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("test", 20), 0]
        stop = threading.Event()
        stop.set()
        with patch("observations_service.launcher.subprocess.Popen", return_value=child):
            self.assertEqual(supervise([["observe"]], stop), 0)
        child.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
