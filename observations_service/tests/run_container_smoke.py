"""Container smoke: real missing data plus a clearly separate illustrative complete frame."""
import argparse
import json
import math
from pathlib import Path
import time
from urllib.request import ProxyHandler, Request, build_opener

from observations_service.contract import POINTS, measurement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--forecast-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--save-response", type=Path)
    parser.add_argument("--verify-restored", type=Path)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}))

    def call(base, route, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        req = Request(base.rstrip("/") + route, data=body, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=10) as response:
            assert response.status == 200
            return json.load(response)

    root = Path(__file__).resolve().parents[2]
    original = json.loads((root / "examples/platform_input_example.json").read_text(encoding="utf-8"))
    frame = original["frames"][0]
    real_payload = {"point_table": list(POINTS), "frames": [
        {"timestamp": frame["timestamp"], **{point: frame.get(point) for point in POINTS}}]}
    real_item = measurement(real_payload)
    # Artificial numbers are confined to this explicit interface test, not model inputs.
    payload = json.loads((root / "observations_service/examples/input_complete.json").read_text(encoding="utf-8"))
    before = call(args.forecast_base_url, "/api/v1/fluxcast/compute/latest")
    payload["frames"][0]["timestamp"] = before["result_point"][0]["timestamp"]
    item = measurement(payload)
    assert item["slot"] < time.time(), "Replay smoke test must use a past measurement"
    deadline = time.monotonic() + 20
    while True:
        health = call(args.base_url, "/health")
        if health["status"] == "ok" and health["batches"] >= 1:
            break
        assert time.monotonic() < deadline, "Latest forecast was not archived"
        time.sleep(0.2)
    if args.verify_restored:
        assert health["observations"] >= 2, "Measured records did not survive container recreation"
    real_response = call(args.base_url, "/api/v1/fluxcast/observations", real_payload)
    real_info = {p["varname"]: p["value"] for p in real_response["extra_info"]}
    assert real_info["dataStatus"] == real_item["status"]
    if real_item["status"] != "complete":
        assert not any(p["varname"] == "totalPowerActual" for p in real_response["result_point"]), "Incomplete total must be omitted"
        assert real_info["missingPoints"] == ",".join(real_item["missing"])
    else:
        assert real_response["result_point"][0]["value"] == real_item["total"]
    response = call(args.base_url, "/api/v1/fluxcast/observations", payload)
    assert response["event_key"] == "JNH.Fluxcast.Compute"
    assert not any(p["varname"] == "totalPowerDeviation" for p in response["result_point"]), "Past replay must not use a newly observed forecast"
    actual = response["result_point"][0]
    assert actual["varname"] == "totalPowerActual"
    assert actual["timestamp"] == item["timestamp"]
    assert type(actual["value"]) is float
    assert math.isclose(actual["value"], item["total"], abs_tol=1e-9)
    info = {p["varname"]: p["value"] for p in response["extra_info"]}
    assert info["dataStatus"] == "complete" and info["reason"] == "no_matching_forecast"
    assert call(args.forecast_base_url, "/api/v1/fluxcast/compute/latest") == before
    if args.verify_restored:
        assert response == json.loads(args.verify_restored.read_text(encoding="utf-8"))
    if args.save_response:
        args.save_response.parent.mkdir(parents=True, exist_ok=True)
        args.save_response.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        real_path = args.save_response.with_name(args.save_response.stem + "_real_input.json")
        real_path.write_text(json.dumps(real_response, indent=2) + "\n", encoding="utf-8")
    print(f"Real frame status={real_item['status']}; separate illustrative sum=1795.0; "
          "UTC timestamp, archive, and prediction read-only checks passed")


if __name__ == "__main__":
    main()
