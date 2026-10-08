"""GET-only forecast synchronization and single-frame observation HTTP API."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .contract import EVENT_KEY
from .storage import Store, utc_now

LOG = logging.getLogger("observations")
MAX_BODY = 256 * 1024
ROUTE = "/api/v1/fluxcast/observations"


def reject_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ForecastSync:
    def __init__(self, store: Store, url: str, interval: float = 5, timeout: float = 3):
        self.store, self.url, self.interval, self.timeout = store, url, interval, timeout
        self.status = "starting"
        self.last_success = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="forecast-archive", daemon=True)
        # Local predictor access must not go through a host's HTTP proxy.
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def once(self) -> None:
        try:
            with self.opener.open(Request(self.url, method="GET"), timeout=self.timeout) as response:
                raw = response.read(MAX_BODY + 1)
            seen = utc_now()
            if len(raw) > MAX_BODY:
                raise ValueError("Forecast response is too large")
            self.store.archive(json.loads(raw, parse_constant=reject_constant,
                                          object_pairs_hook=unique_object), first_seen=seen)
            self.last_success = seen
            status = "ready"
        except HTTPError as exc:
            status = "no_forecast" if exc.code == 404 else "unavailable"
            exc.close()
        except (OSError, ValueError) as exc:
            status = "unavailable"
            if self.status != status:
                LOG.warning("Forecast sync unavailable: %s", exc)
        if status != self.status:
            LOG.info("Forecast sync: %s", status)
        self.status = status

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.once()
            except Exception:
                self.status = "storage_error"
                LOG.exception("Forecast archive failed")
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.timeout + 15)


def make_server(host: str, port: int, store: Store, sync: ForecastSync) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def reply(self, code: int, payload: dict):
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def error(self, code: int, reason: str, message: str):
            self.reply(code, {"result_point": [], "extra_info": [], "event_key": EVENT_KEY,
                              "reason": reason, "message": message})

        def do_GET(self):
            if self.path != "/health":
                self.error(404, "not_found", "Unknown route")
                return
            try:
                counts = store.counts()
                alive = sync.thread.is_alive()
                self.reply(200 if alive else 503, {
                    "status": "ok" if alive else "sync_stopped",
                    "forecast_sync": sync.status, "last_sync_utc_epoch": sync.last_success,
                    **counts,
                })
            except Exception:
                LOG.exception("Health check failed")
                self.error(503, "storage_unavailable", "Observation storage unavailable")

        def do_POST(self):
            if self.path != ROUTE:
                self.error(404, "not_found", "Unknown route")
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Send JSON with Content-Length, not Transfer-Encoding")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                lengths = self.headers.get_all("Content-Length") or []
                if len(lengths) != 1:
                    raise ValueError("Exactly one Content-Length is required")
                size = int(lengths[0])
                if size > MAX_BODY:
                    self.error(413, "request_too_large", "Request exceeds 256 KiB")
                    return
                if size <= 0:
                    raise ValueError("Request body is empty")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete request body")
                payload = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
                response = store.receive(payload)
            except (ValueError, UnicodeError) as exc:
                self.error(400, "invalid_request", str(exc))
                return
            except TimeoutError:
                self.error(408, "request_timeout", "Request body timed out")
                return
            except Exception:
                LOG.exception("Observation request failed")
                self.error(500, "internal_error", "Observation storage or processing failed")
                return
            self.reply(200, response)

        def log_message(self, fmt, *args):
            LOG.info("%s %s", self.client_address[0], fmt % args)

    return ThreadingHTTPServer((host, port), Handler)


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--database", default=os.environ.get(
        "OBSERVATIONS_DATABASE", "runtime_observations/observations.sqlite3"))
    parser.add_argument("--forecast-url", default="http://127.0.0.1:8000/api/v1/fluxcast/compute/latest")
    parser.add_argument("--sync-interval", type=positive, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    store = Store(args.database)
    sync = ForecastSync(store, args.forecast_url, interval=args.sync_interval)
    server = make_server(args.host, args.port, store, sync)
    sync.start()
    LOG.info("Observations listening on %s:%d; independent DB %s", args.host, server.server_port, store.path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sync.close()


if __name__ == "__main__":
    main()
