from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memora_ai.domain.image_quality import ImageQualityAnalyzer, ImageQualityResult


class ImageQualityAnalyzerTests(unittest.TestCase):
    def test_analyze_image_returns_scores_in_expected_range(self) -> None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (220, 220), (255, 255, 255), -1)
        cv2.circle(image, (128, 128), 70, (200, 200, 200), -1)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            cv2.imwrite(str(image_path), image)

            analyzer = ImageQualityAnalyzer()
            result = analyzer.analyze_file(image_path)

        self.assertIsInstance(result, ImageQualityResult)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 100.0)
        self.assertTrue(0.0 <= result.blur_score <= 100.0)
        self.assertTrue(0.0 <= result.brightness_score <= 100.0)
        self.assertTrue(0.0 <= result.contrast_score <= 100.0)
        self.assertTrue(0.0 <= result.sharpness_score <= 100.0)
        self.assertTrue(0.0 <= result.noise_score <= 100.0)
        self.assertTrue(0.0 <= result.resolution_score <= 100.0)


if __name__ == "__main__":
    unittest.main()
