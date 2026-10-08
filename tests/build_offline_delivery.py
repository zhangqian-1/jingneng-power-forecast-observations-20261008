"""Assemble a downloadable image bundle after the CI import test succeeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from platform_test_utils import forecast_rows


IMAGE_FILE = "image.tar.gz"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_delivery(root: Path, ci: Path, output: Path) -> dict:
    root, ci, output = root.resolve(), ci.resolve(), output.resolve()
    if output == root or output in root.parents:
        raise ValueError("Delivery output cannot be the source root or its parent")
    archive = output / IMAGE_FILE
    if not archive.is_file() or archive.stat().st_size == 0:
        raise ValueError("The exported image archive is missing or empty")
    if {p.name for p in output.iterdir()} != {IMAGE_FILE}:
        raise ValueError("Delivery directory must contain only the new image archive")

    release = json.loads((ci / "release.json").read_text(encoding="utf-8"))
    active = json.loads((root / "models/active_model.json").read_text(encoding="utf-8"))
    if release["model"] != active["version"]:
        raise ValueError("Tested release does not match the active model")
    if release["platform"] not in ("linux/amd64", "linux/arm64"):
        raise ValueError("Unsupported image platform")
    for check in ("compose_smoke_test", "container_recreation_test",
                  "observations_smoke_test", "observations_recreation_test"):
        if release.get(check) != "passed":
            raise ValueError(f"Required check did not pass: {check}")
    initial = json.loads((ci / "smoke_response.json").read_text(encoding="utf-8"))
    imported = json.loads((ci / "import_smoke_response.json").read_text(encoding="utf-8"))
    if (initial.get("event_key") != "JNH.Fluxcast.Compute"
            or imported.get("event_key") != initial["event_key"]
            or len(forecast_rows(initial)) != 96
            or forecast_rows(imported) != forecast_rows(initial)):
        raise ValueError("Imported predictions do not match the tested image")
    observed = json.loads((ci / "observations_smoke_response.json").read_text(encoding="utf-8"))
    imported_observed = json.loads((ci / "import_observations_smoke_response.json").read_text(encoding="utf-8"))
    if (observed != imported_observed or observed.get("event_key") != "JNH.Fluxcast.Compute"
            or sum(p.get("varname") == "totalPowerActual" for p in observed.get("result_point", [])) != 1
            or any(p.get("varname") == "totalPowerDeviation" for p in observed.get("result_point", []))):
        raise ValueError("Imported observations do not match the tested image")
    expected_id = (ci / "image_id_before_export.txt").read_text().strip()
    imported_id = (ci / "image_id_after_import.txt").read_text().strip()
    if not expected_id.startswith("sha256:") or imported_id != expected_id:
        raise ValueError("Imported image ID does not match the exported image")
    if (ci / "delivery/compose.yaml").read_bytes() != (root / "compose.yaml").read_bytes():
        raise ValueError("Delivery Compose does not match the tested configuration")

    for name in ("README.md", "compose.yaml", ".env.example"):
        shutil.copy2(root / name, output / name)
    for directory, pattern in (("docs", "*.md"), ("examples", "*.json")):
        (output / directory).mkdir()
        for source in sorted((root / directory).glob(pattern)):
            shutil.copy2(source, output / directory / source.name)
    observations = output / "observations_service"
    observations.mkdir()
    for source in sorted((root / "observations_service").glob("*.md")):
        shutil.copy2(source, observations / source.name)
    shutil.copytree(root / "observations_service/examples", observations / "examples")
    # These are only deployment settings, never CI credentials or runtime state.
    (output / ".env").write_text(
        f"POWER_FORECAST_IMAGE={release['image']}\nPOWER_FORECAST_BIND=127.0.0.1\n"
        "POWER_FORECAST_PORT=8000\nPOWER_FORECAST_RUNTIME_DIR=./runtime\n"
        "POWER_OBSERVATIONS_BIND=127.0.0.1\nPOWER_OBSERVATIONS_PORT=8002\n"
        "POWER_OBSERVATIONS_RUNTIME_DIR=./runtime_observations\n",
        encoding="utf-8",
    )
    release["image_id"] = expected_id
    release["offline_image"] = {
        "file": IMAGE_FILE,
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "export_import_test": "passed",
        "import_prediction_test": "passed",
        "import_observations_test": "passed",
    }
    (output / "release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(output).as_posix()}\n" for p in files),
        encoding="utf-8",
    )
    return release


def archive_delivery(output: Path, archive: Path) -> Path:
    """Keep the ZIP deployable at its root, including the hidden .env file."""
    output, archive = output.resolve(), archive.resolve()
    if output in archive.parents:
        raise ValueError("ZIP must be outside the delivery directory")
    checksums = output / "SHA256SUMS"
    expected = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = (output / name).resolve()
        if output not in path.parents or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Delivery checksum mismatch: {name}")
        expected[name] = path
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != set(expected) | {"SHA256SUMS"}:
        raise ValueError("Delivery contains unchecked files")
    required = {IMAGE_FILE, ".env", "compose.yaml", "release.json", "README.md"}
    if not required <= set(expected):
        raise ValueError("Delivery is missing required deployment files")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", allowZip64=True) as bundle:
        for name in sorted(actual):
            compression = zipfile.ZIP_STORED if name == IMAGE_FILE else zipfile.ZIP_DEFLATED
            bundle.write(output / name, name, compress_type=compression)
    checksum_path = archive.with_name(archive.name + ".sha256")
    checksum_path.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return checksum_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci-dir", type=Path, default=Path("tests/results/ci"))
    parser.add_argument("--output-dir", type=Path, default=Path("tests/results/offline_delivery"))
    parser.add_argument("--archive", type=Path, help="Optional deployable ZIP outside the output directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    release = build_delivery(root, args.ci_dir, args.output_dir)
    if args.archive:
        archive_delivery(args.output_dir, args.archive)
    print(json.dumps({"platform": release["platform"], "offline_image": release["offline_image"]}, indent=2))


if __name__ == "__main__":
    main()
