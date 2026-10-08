"""Unit checks for bundle assembly; no model execution or training."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from build_offline_delivery import archive_delivery, build_delivery, sha256


class OfflineDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(__file__).resolve().parents[1]
        self.ci = Path(self.temp.name) / "ci"
        self.output = Path(self.temp.name) / "delivery"
        (self.ci / "delivery").mkdir(parents=True)
        self.output.mkdir()
        # This small byte fixture tests packaging checks, not a Docker image.
        (self.output / "image.tar.gz").write_bytes(b"archive-test-fixture")
        self.release = {
            "image": "example/forecast:test", "platform": "linux/arm64",
            "model": "trend_detail_7station_2025_v1",
            "compose_smoke_test": "passed", "container_recreation_test": "passed",
            "observations_smoke_test": "passed", "observations_recreation_test": "passed",
        }
        (self.ci / "release.json").write_text(json.dumps(self.release))
        response = (self.root / "examples/platform_output_example.json").read_bytes()
        for name in ("smoke_response.json", "import_smoke_response.json"):
            (self.ci / name).write_bytes(response)
        observed = (self.root / "observations_service/examples/output_no_forecast.json").read_bytes()
        for name in ("observations_smoke_response.json", "import_observations_smoke_response.json"):
            (self.ci / name).write_bytes(observed)
        for name in ("image_id_before_export.txt", "image_id_after_import.txt"):
            (self.ci / name).write_text("sha256:" + "a" * 64)
        (self.ci / "delivery/compose.yaml").write_bytes((self.root / "compose.yaml").read_bytes())

    def build(self):
        return build_delivery(self.root, self.ci, self.output)

    def test_bundle_has_image_settings_and_docs_but_no_test_data(self):
        release = self.build()
        self.assertEqual(release["offline_image"]["sha256"], sha256(self.output / "image.tar.gz"))
        self.assertIn("POWER_FORECAST_IMAGE=example/forecast:test", (self.output / ".env").read_text())
        self.assertTrue((self.output / "examples/platform_input_example.json").is_file())
        self.assertTrue((self.output / "README.md").is_file())
        self.assertFalse((self.output / "tests").exists())
        self.assertFalse((self.output / "runtime").exists())
        self.assertTrue((self.output / "observations_service/README.md").is_file())
        self.assertTrue((self.output / "observations_service/examples/input_complete.json").is_file())
        self.assertFalse((self.output / "observations_service/tests").exists())
        self.assertFalse((self.output / "runtime_observations").exists())
        self.assertIn("POWER_OBSERVATIONS_PORT=8002", (self.output / ".env").read_text())
        for line in (self.output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(digest, sha256(self.output / name))

    def test_missing_image_rejected(self):
        (self.output / "image.tar.gz").unlink()
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            self.build()

    def test_new_computation_batch_allowed_but_curve_must_match(self):
        path = self.ci / "import_smoke_response.json"
        response = json.loads(path.read_text(encoding="utf-8"))
        for entry in response["extra_info"]:
            if entry["varname"] == "forecastBatchId":
                entry["value"] = "dayahead-" + "a" * 32
        path.write_text(json.dumps(response), encoding="utf-8")
        self.build()

    def test_release_zip_keeps_configuration_and_verifiable_image(self):
        self.build()
        archive = self.output.parent / "offline-image.zip"
        checksum = archive_delivery(self.output, archive)
        self.assertEqual(checksum.read_text().split()[0], sha256(archive))
        with zipfile.ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertIn(".env", bundle.namelist())
            self.assertIn("docs/接口交接说明.md", bundle.namelist())
            self.assertEqual(bundle.read("image.tar.gz"), (self.output / "image.tar.gz").read_bytes())
            self.assertFalse(any(name.startswith(("tests/", "runtime/")) for name in bundle.namelist()))

    def test_release_zip_rejects_changed_or_unchecked_files(self):
        self.build()
        archive = self.output.parent / "offline-image.zip"
        (self.output / ".env").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            archive_delivery(self.output, archive)
        self.assertFalse(archive.exists())

    def test_release_zip_cannot_overwrite_or_include_itself(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "outside"):
            archive_delivery(self.output, self.output / "bundle.zip")
        archive = self.output.parent / "offline-image.zip"
        archive_delivery(self.output, archive)
        with self.assertRaises(FileExistsError):
            archive_delivery(self.output, archive)

    def test_old_delivery_rejected(self):
        (self.output / "old.txt").write_text("old")
        with self.assertRaisesRegex(ValueError, "only the new image"):
            self.build()

    def test_changed_predictions_rejected(self):
        path = self.ci / "import_smoke_response.json"
        response = json.loads(path.read_text(encoding="utf-8"))
        response["result_point"][0]["value"] += 1
        path.write_text(json.dumps(response))
        with self.assertRaisesRegex(ValueError, "predictions"):
            self.build()

    def test_wrong_image_rejected(self):
        (self.ci / "image_id_after_import.txt").write_text("sha256:" + "b" * 64)
        with self.assertRaisesRegex(ValueError, "image ID"):
            self.build()

    def test_failed_acceptance_rejected(self):
        self.release["container_recreation_test"] = "failed"
        (self.ci / "release.json").write_text(json.dumps(self.release))
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.build()

    def test_changed_compose_rejected(self):
        (self.ci / "delivery/compose.yaml").write_text("services: {}")
        with self.assertRaisesRegex(ValueError, "Compose"):
            self.build()

    def test_changed_observations_rejected(self):
        path = self.ci / "import_observations_smoke_response.json"
        response = json.loads(path.read_text())
        response["result_point"][0]["value"] += 1
        path.write_text(json.dumps(response))
        with self.assertRaisesRegex(ValueError, "observations"):
            self.build()

    def test_failed_observations_acceptance_rejected(self):
        self.release["observations_recreation_test"] = "failed"
        (self.ci / "release.json").write_text(json.dumps(self.release))
        with self.assertRaisesRegex(ValueError, "observations_recreation_test"):
            self.build()


if __name__ == "__main__":
    unittest.main()
