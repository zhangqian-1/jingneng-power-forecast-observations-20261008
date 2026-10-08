"""Real-model HTTP check using isolated histories; no training or production writes."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from run_platform_verification import run, server


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@contextmanager
def observations(directory, forecast_url):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with (directory / "observations.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([
            sys.executable, "-u", "-m", "observations_service.api", "--port", str(port),
            "--database", str(directory / "observations.sqlite3"),
            "--forecast-url", forecast_url + "/api/v1/fluxcast/compute/latest",
            "--sync-interval", "0.2",
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Measured service exited; see observations.log")
                try:
                    with opener.open(base + "/health", timeout=2) as response:
                        if json.load(response)["status"] == "ok":
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Measured service did not start")
                time.sleep(0.2)
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "tests/results/observations_service"):
        raise ValueError("Use a new directory under tests/results/observations_service")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    protected = [*sorted((ROOT / "app").rglob("*.py")),
                 *sorted(p for p in (ROOT / "models").rglob("*") if p.is_file()),
                 ROOT / "requirements.txt"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    fixtures, smoke = output / "fixtures", output / "prediction"
    run("tests/build_real_test_payloads.py", "--raw-dir", "tests/real_data_raw",
        "--output-dir", fixtures, "--end-time", "2025-10-12 23:45:00")
    with server(smoke) as forecast_url:
        with observations(output, forecast_url) as measured_url:
            run("tests/run_api_test.py", "--base-url", forecast_url, "--fixture-dir", fixtures,
                "--save-response", output / "forecast_response.json")
            paths = (smoke / "platform.csv", smoke / "platform_latest.json")
            history_before = {p.name: digest(p) for p in paths}
            run("-m", "observations_service.tests.run_container_smoke", "--base-url", measured_url,
                "--forecast-base-url", forecast_url, "--save-response", output / "measured_response.json")
            assert history_before == {p.name: digest(p) for p in paths}
        with observations(output, forecast_url) as measured_url:
            run("-m", "observations_service.tests.run_container_smoke", "--base-url", measured_url,
                "--forecast-base-url", forecast_url, "--verify-restored", output / "measured_response.json")
            assert history_before == {p.name: digest(p) for p in paths}
    with server(smoke) as forecast_url:
        run("tests/run_api_test.py", "--base-url", forecast_url, "--fixture-dir", fixtures,
            "--verify-restored", output / "forecast_response.json")
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before == after, "Protected prediction files changed"
    record = {
        "real_model_96_point_http": "passed", "measured_http_real_input": "passed_missing_data_policy",
        "measured_http_complete_fixture": "passed_illustrative_19_point_sum",
        "measured_restart": "passed", "prediction_restart": "passed",
        "prediction_runtime_read_only": "passed", "protected_prediction_files": len(protected),
        "protected_sha256_unchanged": before == after,
        "past_replay_forecast_leakage_rejected": "passed",
        "deviation_matching": "covered_by_isolated_unit_and_HTTP_tests",
        "docker_container_test": "not_run",
        "note": "Local CPU processes, not Docker; this check does not measure forecast accuracy.",
    }
    (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
