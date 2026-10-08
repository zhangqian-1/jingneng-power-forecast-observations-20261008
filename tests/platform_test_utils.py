"""Independent client-side checks for the platform contract."""
from datetime import datetime, timedelta
import math
import re


PLATFORM_PATH = "/api/v1/fluxcast/compute"


def forecast_rows(result: dict) -> list:
    return [row for row in result.get("result_point", []) if row.get("varname") == "totalPowerForecast"]


def validate_platform_prediction(result: dict, payload: dict) -> None:
    if result.get("event_key") != "JNH.Fluxcast.Compute":
        raise AssertionError("Incorrect platform event_key")
    rows = forecast_rows(result)
    if not isinstance(rows, list) or len(rows) != 96:
        raise AssertionError("Platform result must contain 96 predictions")
    cutoff = max(datetime.fromisoformat(row["timestamp"]) for row in payload["frames"])
    sample = payload["frames"][0]["timestamp"]
    for index, row in enumerate(rows, start=1):
        target = (cutoff + timedelta(minutes=15 * index)).strftime("%Y-%m-%d %H:%M:%S")
        target = target[:10] + sample[10] + target[11:] + sample[19:]
        if row.get("varname") != "totalPowerForecast" or row.get("timestamp") != target:
            raise AssertionError("Incorrect platform variable or target timestamp")
        if type(row.get("value")) is not float or not math.isfinite(row["value"]):
            raise AssertionError("Platform value must be a finite float")
    numbers = {row["varname"]: row for row in result["result_point"] if row["varname"] != "totalPowerForecast"}
    info = {row["varname"]: row for row in result.get("extra_info", [])}
    if len(result["result_point"]) != 98 or len(result.get("extra_info", [])) != 4:
        raise AssertionError("Expected 96 predictions, two statistics and four batch metadata entries")
    if set(numbers) != {"forecastPointCount", "forecastIntervalMinutes"} or set(info) != {
            "forecastBatchId", "forecastGeneratedAt", "forecastStartAt", "forecastEndAt"}:
        raise AssertionError("Incorrect result vocabulary")
    if numbers["forecastPointCount"]["value"] != 96.0 or numbers["forecastIntervalMinutes"]["value"] != 15.0:
        raise AssertionError("Incorrect forecast statistics")
    if any(type(row["value"]) is not float for row in numbers.values()):
        raise AssertionError("Statistics must use floats")
    if any(not isinstance(row["value"], str) for row in info.values()):
        raise AssertionError("Metadata must use strings")
    if re.fullmatch(r"dayahead-[0-9a-f]{32}", info["forecastBatchId"]["value"]) is None:
        raise AssertionError("Incorrect batch identifier")
    if info["forecastStartAt"]["value"] != rows[0]["timestamp"] or info["forecastEndAt"]["value"] != rows[-1]["timestamp"]:
        raise AssertionError("Incorrect forecast extent")
    generated = info["forecastGeneratedAt"]["value"]
    datetime.fromisoformat(generated)
    if generated[10] != sample[10] or generated[19:] != sample[19:]:
        raise AssertionError("Generation timestamp format differs from request")
    if any(row["timestamp"] != generated for row in [*numbers.values(), *info.values()]):
        raise AssertionError("Batch metadata timestamp must be generation time")


def validate_not_ready(result: dict) -> None:
    if (result.get("event_key") != "JNH.Fluxcast.Compute" or result.get("result_point") != []
            or result.get("reason") not in {"history_not_ready", "weather_history_not_ready"}
            or not isinstance(result.get("message"), str) or not result["message"]
            or result.get("required_points") != 672
            or not 0 <= result.get("continuous_points", -1) < 672):
        raise AssertionError("Incorrect platform not-ready response")
