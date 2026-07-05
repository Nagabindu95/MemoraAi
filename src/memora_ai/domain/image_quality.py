from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageQualityResult:
    """Represents image quality metrics and an overall score."""

    blur_score: float
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    noise_score: float
    resolution_score: float
    overall_score: float
    laplacian_variance: float
    brightness_mean: float
    contrast_std: float
    noise_estimate: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


class ImageQualityAnalyzer:
    """Analyze image quality using OpenCV-based heuristics."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        self.logger = logger_instance or logger

    def analyze_file(self, image_path: Union[str, Path]) -> ImageQualityResult:
        """Analyze an image file from disk and return quality metrics."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError(f"Unable to read image: {path}")

        return self.analyze_image(image, source_name=str(path))

    def analyze_image(self, image: np.ndarray, source_name: Optional[str] = None) -> ImageQualityResult:
        """Analyze an in-memory image array and return quality metrics."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim == 2:
            gray = image.astype(np.float32)
        elif image.shape[2] == 1:
            gray = image[:, :, 0].astype(np.float32)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
        else:
            raise ValueError(f"Unsupported image channel count: {image.shape[2]}")

        height, width = gray.shape
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions are invalid")

        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

        brightness_mean = float(np.mean(gray_u8))
        contrast_std = float(np.std(gray_u8))

        laplacian = cv2.Laplacian(gray_u8, cv2.CV_32F)
        laplacian_variance = float(np.var(laplacian))

        sharpness_score = self._score_from_laplacian(laplacian_variance)
        blur_score = max(0.0, 100.0 - sharpness_score)

        brightness_score = self._score_brightness(brightness_mean)
        contrast_score = self._score_contrast(contrast_std)
        noise_estimate = self._estimate_noise(gray_u8)
        noise_score = self._score_noise(noise_estimate)
        resolution_score = self._score_resolution(width, height)

        overall_score = float(
            0.25 * sharpness_score
            + 0.20 * blur_score
            + 0.15 * brightness_score
            + 0.15 * contrast_score
            + 0.15 * noise_score
            + 0.10 * resolution_score
        )

        result = ImageQualityResult(
            blur_score=round(blur_score, 2),
            brightness_score=round(brightness_score, 2),
            contrast_score=round(contrast_score, 2),
            sharpness_score=round(sharpness_score, 2),
            noise_score=round(noise_score, 2),
            resolution_score=round(resolution_score, 2),
            overall_score=round(overall_score, 2),
            laplacian_variance=round(laplacian_variance, 4),
            brightness_mean=round(brightness_mean, 2),
            contrast_std=round(contrast_std, 2),
            noise_estimate=round(noise_estimate, 4),
            width=width,
            height=height,
        )

        self.logger.debug(
            "Image quality analysis complete for %s: overall=%s sharpness=%s blur=%s brightness=%s contrast=%s noise=%s resolution=%s",
            source_name or "in-memory image",
            result.overall_score,
            result.sharpness_score,
            result.blur_score,
            result.brightness_score,
            result.contrast_score,
            result.noise_score,
            result.resolution_score,
        )
        return result

    @staticmethod
    def _score_from_laplacian(variance: float) -> float:
        """Map Laplacian variance into a 0-100 sharpness score."""
        return float(np.clip(100.0 * np.log1p(variance) / np.log1p(2000.0), 0.0, 100.0))

    @staticmethod
    def _score_brightness(mean_intensity: float) -> float:
        """Score brightness based on proximity to the mid-gray target."""
        target = 128.0
        return float(np.clip(100.0 * np.exp(-abs(mean_intensity - target) / target), 0.0, 100.0))

    @staticmethod
    def _score_contrast(std_dev: float) -> float:
        """Score contrast using the image standard deviation."""
        return float(np.clip((std_dev / 127.5) * 100.0, 0.0, 100.0))

    @staticmethod
    def _estimate_noise(gray_image: np.ndarray) -> float:
        """Estimate noise using a simple high-frequency difference metric."""
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        return float(np.mean(np.abs(gray_image.astype(np.float32) - blurred.astype(np.float32))))

    @staticmethod
    def _score_noise(noise_estimate: float) -> float:
        """Map the estimated noise to a 0-100 score where higher is better."""
        return float(np.clip(100.0 * np.exp(-noise_estimate / 20.0), 0.0, 100.0))

    @staticmethod
    def _score_resolution(width: int, height: int) -> float:
        """Score resolution against a 1080p reference."""
        target_pixels = 1920 * 1080
        actual_pixels = width * height
        if actual_pixels <= 0:
            return 0.0
        scaled = np.log1p(actual_pixels) / np.log1p(target_pixels)
        return float(np.clip(100.0 * scaled, 0.0, 100.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an image and print quality metrics")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    analyzer = ImageQualityAnalyzer()
    result = analyzer.analyze_file(args.image_path)
    print(result.to_dict())


if __name__ == "__main__":
    main()
