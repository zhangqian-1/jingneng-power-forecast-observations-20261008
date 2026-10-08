"""HTTP API for real-data power forecasting."""
from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from input_adapter import InputValidationError, POWER_POLICY_ID
from predict import DEFAULT_HISTORY_CACHE, DEFAULT_MODEL_PATH, PowerPredictor
from platform_adapter import PLATFORM_PATH, PlatformForecastService, empty_result
from time_policy import MODEL_TIMEZONE, TIME_POLICY_ID, TIMEZONE_BASIS


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LATEST_JSON = PACKAGE_ROOT / "runtime" / "latest_forecast_utc_to_asia_shanghai_v1.json"
MAX_REQUEST_BYTES = 50 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


class ForecastHandler(BaseHTTPRequestHandler):
    predictor: PowerPredictor
    latest_json: Path = DEFAULT_LATEST_JSON
    platform_service: PlatformForecastService | None = None
    inference_lock = threading.Lock()

    def send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != PLATFORM_PATH:
            self.send_json(404, empty_result("not_found", "接口不存在"))
            return

        try:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise InputValidationError("Content-Length 必须为整数") from exc
            if content_length <= 0:
                raise InputValidationError("请求体不能为空")
            if content_length > MAX_REQUEST_BYTES:
                self.send_json(413, empty_result("request_too_large", "请求体超过50 MiB"))
                return

            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InputValidationError("请求体不是有效的 UTF-8 JSON") from exc

            with self.inference_lock:
                if self.platform_service is None:
                    raise RuntimeError("Platform service is not initialized")
                result = self.platform_service.compute(payload)
                if result["result_point"]:
                    self._save_latest(result)
            self.send_json(200, result)
        except InputValidationError as exc:
            LOGGER.warning("invalid_platform_request: %s", exc)
            self.send_json(400, empty_result("invalid_request", str(exc)))
        except Exception:
            LOGGER.exception("platform_prediction_failed")
            self.send_json(500, empty_result("prediction_failed", "服务内部错误，请查看服务日志"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path != PLATFORM_PATH + "/latest":
            self.send_json(404, empty_result("not_found", "接口不存在"))
            return
        if not self.latest_json.exists():
            self.send_json(404, empty_result("no_forecast", "当前平台接口尚无成功预测"))
            return

        try:
            payload = json.loads(self.latest_json.read_text(encoding="utf-8"))
            if (payload.get("model") != self.predictor.model_name
                    or payload.get("training_timezone") != MODEL_TIMEZONE
                    or payload.get("time_policy") != TIME_POLICY_ID
                    or payload.get("power_policy") != POWER_POLICY_ID
                    or "response" not in payload):
                self.send_json(404, empty_result("no_forecast", "当前模型尚无平台预测结果"))
                return
            self.send_json(200, payload["response"])
        except Exception:
            LOGGER.exception("platform_latest_result_unavailable")
            self.send_json(500, empty_result("latest_result_unavailable", "最近结果读取失败"))

    def _save_latest(self, payload: dict) -> None:
        path = self.latest_json
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(
            {"model": self.predictor.model_name, "training_timezone": MODEL_TIMEZONE,
             "time_policy": TIME_POLICY_ID, "timezone_basis": TIMEZONE_BASIS,
             "power_policy": POWER_POLICY_ID, "response": payload},
            ensure_ascii=False, indent=2, allow_nan=False,
        ), encoding="utf-8")
        temporary.replace(path)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Jingneng power forecast API")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--latest-json", type=Path, default=DEFAULT_LATEST_JSON)
    parser.add_argument("--history-cache", type=Path, default=DEFAULT_HISTORY_CACHE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    paths = [args.history_cache, args.latest_json]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("历史缓存和最近结果必须使用不同文件")

    ForecastHandler.predictor = PowerPredictor(
        args.model_path,
        device=args.device,
        history_cache_path=args.history_cache,
    )
    ForecastHandler.latest_json = args.latest_json
    ForecastHandler.platform_service = PlatformForecastService(ForecastHandler.predictor)
    server = ThreadingHTTPServer((args.host, args.port), ForecastHandler)
    print(f"Power forecast API: http://{args.host}:{args.port}")
    print(f"POST {PLATFORM_PATH}")
    print(f"GET  {PLATFORM_PATH}/latest")
    LOGGER.info("API timezone=UTC; model timezone=%s; policy=%s; basis=%s",
                MODEL_TIMEZONE, TIME_POLICY_ID, TIMEZONE_BASIS)
    server.serve_forever()


if __name__ == "__main__":
    main()
