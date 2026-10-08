"""Container readiness: both HTTP processes, including the empty-history state."""
import json
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener


def main():
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:8000/api/v1/fluxcast/compute/latest", timeout=3) as response:
            payload = json.load(response)
        forecasts = [p for p in payload.get("result_point", []) if p.get("varname") == "totalPowerForecast"]
        forecast_ok = payload.get("event_key") == "JNH.Fluxcast.Compute" and len(forecasts) == 96
    except HTTPError as error:
        with error:
            payload = json.load(error)
        forecast_ok = (error.code == 404 and payload.get("event_key") == "JNH.Fluxcast.Compute"
                       and payload.get("reason") == "no_forecast")
    with opener.open("http://127.0.0.1:8002/health", timeout=3) as response:
        observations_ok = json.load(response).get("status") == "ok"
    raise SystemExit(0 if forecast_ok and observations_ok else 1)


if __name__ == "__main__":
    main()
