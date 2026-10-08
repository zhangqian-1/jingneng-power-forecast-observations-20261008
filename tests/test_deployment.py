"""Negative checks for the container acceptance assertions; not accuracy data."""
import copy
import json
from pathlib import Path
import unittest

from platform_test_utils import validate_platform_prediction


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.payload = json.loads((root / "examples/platform_input_example.json").read_text(encoding="utf-8"))
        self.response = json.loads((root / "examples/platform_output_example.json").read_text(encoding="utf-8"))

    def test_valid_contract(self):
        validate_platform_prediction(self.response, self.payload)

    def test_wrong_event(self):
        self.response["event_key"] = "wrong.event"
        with self.assertRaises(AssertionError):
            validate_platform_prediction(self.response, self.payload)

    def test_wrong_time_or_sequence(self):
        for key, value in (("timestamp", "2025-01-01 00:00:00"), ("varname", "wrongVariable")):
            response = copy.deepcopy(self.response)
            response["result_point"][0][key] = value
            with self.assertRaises(AssertionError):
                validate_platform_prediction(response, self.payload)

    def test_missing_or_extra_predictions(self):
        for rows in (self.response["result_point"][:-1], self.response["result_point"] + self.response["result_point"][:1]):
            response = dict(self.response, result_point=rows)
            with self.assertRaises(AssertionError):
                validate_platform_prediction(response, self.payload)

    def test_invalid_power(self):
        for value in (None, True, "123.4", float("nan"), float("inf")):
            self.response["result_point"][0]["value"] = value
            with self.subTest(value=value), self.assertRaises(AssertionError):
                validate_platform_prediction(self.response, self.payload)

    def test_empty_result_is_not_a_prediction(self):
        self.response["result_point"] = []
        with self.assertRaises(AssertionError):
            validate_platform_prediction(self.response, self.payload)


if __name__ == "__main__":
    unittest.main()
