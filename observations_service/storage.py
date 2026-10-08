"""Own SQLite archive and stable forecast associations, separate from model history."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.result_contract import ACTUAL, point, utc_label
from .contract import forecast_curve, forecast_metadata, measurement, report_with_diagnostics

LOG = logging.getLogger("observations")


def utc_now() -> float:
    return datetime.now(timezone.utc).timestamp()


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, first_seen REAL NOT NULL, curve_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecast_points (
                    batch_id TEXT NOT NULL REFERENCES batches(id), target INTEGER NOT NULL,
                    value REAL NOT NULL, PRIMARY KEY(batch_id, target)
                );
                CREATE INDEX IF NOT EXISTS forecast_target ON forecast_points(target);
                CREATE TABLE IF NOT EXISTS observations (
                    target INTEGER PRIMARY KEY, batch_id TEXT REFERENCES batches(id),
                    input_json TEXT NOT NULL, response_json TEXT NOT NULL, updated_at REAL NOT NULL
                );
            """)
            # Preserve existing archives and associations; legacy generation times stay unknown.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(batches)")}
            if "generated_at" not in columns:
                connection.execute("ALTER TABLE batches ADD COLUMN generated_at REAL")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(observations)")}
            if "complete" not in columns:
                connection.execute("ALTER TABLE observations ADD COLUMN complete INTEGER NOT NULL DEFAULT 0")
                for target, response in connection.execute("SELECT target, response_json FROM observations").fetchall():
                    complete = any(p.get("varname") == ACTUAL for p in json.loads(response).get("result_point", []))
                    connection.execute("UPDATE observations SET complete=? WHERE target=?", (int(complete), target))
            connection.execute("CREATE INDEX IF NOT EXISTS complete_actual ON observations(complete, target)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def archive(self, payload: dict, first_seen: float | None = None) -> str:
        curve = forecast_curve(payload)
        encoded = json.dumps(curve, separators=(",", ":"), allow_nan=False)
        published_id, generated_at = forecast_metadata(payload)
        batch_id = published_id or hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        seen = utc_now() if first_seen is None else first_seen
        if not math.isfinite(seen):
            raise ValueError("first_seen must be finite")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT curve_json, generated_at FROM batches WHERE id=?", (batch_id,)
            ).fetchone()
            if existing is not None and existing != (encoded, generated_at):
                raise ValueError("Forecast batch ID already belongs to a different curve or generation time")
            inserted = connection.execute(
                "INSERT OR IGNORE INTO batches (id, first_seen, curve_json, generated_at) VALUES (?, ?, ?, ?)",
                (batch_id, seen, encoded, generated_at)
            ).rowcount
            if inserted:
                connection.executemany("INSERT INTO forecast_points VALUES (?, ?, ?)",
                                       [(batch_id, target, value) for target, value in curve])
        return batch_id

    def receive(self, payload: dict) -> dict:
        item = measurement(payload)
        target = item["slot"]
        with self.connect() as connection:
            # Serialize retries/corrections so that they cannot select different batches.
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT batch_id FROM observations WHERE target=?", (target,)
            ).fetchone()
            match = None
            if previous and previous[0]:
                match = connection.execute(
                    "SELECT batch_id, value FROM forecast_points WHERE batch_id=? AND target=?",
                    (previous[0], target),
                ).fetchone()
            else:
                match = connection.execute("""
                    SELECT p.batch_id, p.value FROM forecast_points p
                    JOIN batches b ON b.id=p.batch_id
                    WHERE p.target=? AND b.first_seen < ?
                      AND (b.generated_at IS NULL OR b.generated_at < ?)
                    ORDER BY b.first_seen DESC, b.id DESC LIMIT 1
                """, (target, target, target)).fetchone()
            public_id = match[0] if match and match[0].startswith("dayahead-") else None
            response, diagnostics = report_with_diagnostics(item, match[1] if match else None, public_id)
            # A complete frame with a non-finite aggregate is not a calculable total.
            complete = item["total"] is not None
            latest = connection.execute(
                "SELECT MAX(target) FROM observations WHERE complete=1 AND target!=?", (target,)
            ).fetchone()[0]
            if complete:
                latest = target if latest is None else max(target, latest)
            if latest is not None:
                response["extra_info"].append(point("latestActualAt", item["timestamp"],
                                                    utc_label(latest, item["timestamp"])))
            connection.execute("""
                INSERT INTO observations (target, batch_id, input_json, response_json, updated_at, complete)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(target) DO UPDATE SET batch_id=excluded.batch_id,
                    input_json=excluded.input_json, response_json=excluded.response_json,
                    updated_at=excluded.updated_at, complete=excluded.complete
            """, (target, match[0] if match else None,
                  json.dumps({"timestamp": item["timestamp"], "values": item["values"],
                              "issues": item.get("issues", {}),
                              "aggregate_issue": item.get("aggregate_issue")}, allow_nan=False),
                  json.dumps(response, allow_nan=False), utc_now(), int(complete)))
        if diagnostics["causes"]:
            level = logging.WARNING if any(p["reason"] not in {"no_matching_forecast", "total_power_zero"}
                                           for p in diagnostics["causes"]) else logging.INFO
            LOG.log(level, "Observation diagnostics: %s", json.dumps(
                diagnostics, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return response

    def counts(self) -> dict:
        with self.connect() as connection:
            return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("batches", "observations")}
