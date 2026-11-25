import json
import math
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import average_precision, binary_auroc, compute_metrics, linear_quantile, load_predictions, probability_stratified_score, validate_predictions

class MetricsTests(unittest.TestCase):
    def test_auroc_ties(self):
        self.assertEqual(binary_auroc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(binary_auroc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]), 1.0)
        self.assertIsNone(binary_auroc([1, 1], [0.2, 0.3]))

    def test_average_precision_ties(self):
        self.assertEqual(average_precision([1, 0], [0.5, 0.5]), 0.5)
        self.assertEqual(average_precision([0, 0], [0.8, 0.2]), 0.0)
        self.assertEqual(average_precision([1, 1], [0.8, 0.2]), 1.0)
        self.assertAlmostEqual(average_precision([1, 0, 1], [0.9, 0.8, 0.7]), 5 / 6)

    def test_linear_quantile(self):
        self.assertEqual(linear_quantile([0, 1, 2, 3], 0.5), 1.5)
        self.assertEqual(linear_quantile([2], 0.5), 2)

    def test_stratified_weighting(self):
        y = [0, 1, 0, 1, 1, 0, 1, 1]
        probability = [0.1, 0.1, 0.2, 0.2, 0.6, 0.6, 0.7, 0.7]
        uncertainty = [0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.3]
        result = probability_stratified_score(y, probability, uncertainty, bins=2, min_errors=0)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["valid_records"], 8)
        self.assertEqual(result["score"], 1.0)

    def test_repeated_quantile_edges(self):
        result = probability_stratified_score([0, 1, 0, 1], [0.1, 0.1, 0.9, 0.9], [0.1, 0.9, 0.9, 0.1], bins=10, min_errors=0)
        self.assertLess(result["strata"], 10)
        self.assertEqual(result["score"], 1.0)

    def test_constant_probability_has_no_stratum(self):
        result = probability_stratified_score([0, 1], [0.5, 0.5], [0.8, 0.2], min_errors=0)
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["score"])

    def test_minimum_error_eligibility(self):
        result = probability_stratified_score([0, 1], [0.1, 0.2], [0.1, 0.9], bins=1, min_errors=10)
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["score"])

    def test_binary_labels_and_range_validation(self):
        with self.assertRaises(ValueError):
            validate_predictions([2], [0.5], [0.5])
        with self.assertRaises(ValueError):
            validate_predictions([0], [math.nan], [0.5])
        with self.assertRaises(ValueError):
            validate_predictions([0], [0.5], [1.2])
        with self.assertRaises(ValueError):
            validate_predictions([], [], [])

    def test_quarter_rounds_ties_to_even(self):
        result = compute_metrics([0] * 10, [0.1] * 10, [0.1] * 10)
        self.assertEqual(result["most_certain_quarter_n"], 2)
        self.assertIsNone(result["missed_interaction_flag_rate"])
        self.assertIsNone(result["error_detection_auroc"])

    def test_csv_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "predictions.csv"
            source.write_text("y,p1,u\n0,0.1,0.2\n1,0.9,0.3\n", encoding="utf-8")
            result = compute_metrics(*load_predictions(source))
            self.assertEqual(result["prediction_auroc"], 1.0)
            self.assertEqual(result["accuracy"], 1.0)
            json.dumps(result, allow_nan=False)

if __name__ == "__main__":
    unittest.main()
