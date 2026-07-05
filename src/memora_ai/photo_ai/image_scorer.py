from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class ImageScore:
    """Structured quality metrics for a single image."""

    filename: str
    quality_score: float
    sharpness: float
    brightness: float
    blur: float
    face_count: int
    orientation: str
    width: int
    height: int
    duplicate_hash: str


class ImageScorer:
    """Analyze image quality and basic visual attributes with OpenCV and NumPy."""

    def __init__(self) -> None:
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    def score_image(self, image_path: str | Path) -> ImageScore:
        """Score an image file and return structured metrics."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        sharpness = self._compute_sharpness(gray)
        brightness = self._compute_brightness(gray)
        blur = self._compute_blur(gray)
        face_count = self._detect_faces(gray)
        smile_count = self._detect_smiles(gray)
        orientation = self._classify_orientation(width, height)
        duplicate_hash = self._compute_duplicate_hash(image)

        quality_score = self._compute_overall_score(
            sharpness=sharpness,
            brightness=brightness,
            blur=blur,
            face_count=face_count,
            smile_count=smile_count,
            width=width,
            height=height,
        )

        return ImageScore(
            filename=image_path.name,
            quality_score=round(float(quality_score), 2),
            sharpness=round(float(sharpness), 2),
            brightness=round(float(brightness), 2),
            blur=round(float(blur), 2),
            face_count=int(face_count),
            orientation=orientation,
            width=int(width),
            height=int(height),
            duplicate_hash=duplicate_hash,
        )

    def score_images(self, image_paths: list[str | Path]) -> list[ImageScore]:
        """Score multiple images and return a list of results."""
        return [self.score_image(path) for path in image_paths]

    def _compute_sharpness(self, gray: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(laplacian))
        return float(np.clip(variance / 1000.0, 0.0, 100.0))

    def _compute_brightness(self, gray: np.ndarray) -> float:
        mean_brightness = float(np.mean(gray))
        return float(np.clip(mean_brightness / 255.0 * 100.0, 0.0, 100.0))

    def _compute_blur(self, gray: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(laplacian))
        return float(np.clip(100.0 - (variance / 1000.0), 0.0, 100.0))

    def _detect_faces(self, gray: np.ndarray) -> int:
        if self._face_cascade.empty():
            return 0

        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        return int(len(faces))

    def _detect_smiles(self, gray: np.ndarray) -> int:
        if self._smile_cascade.empty():
            return 0

        smiles = self._smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=10,
            minSize=(20, 20),
        )
        return int(len(smiles))

    def _compute_duplicate_hash(self, image: np.ndarray) -> str:
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        avg = float(np.mean(gray))
        bits = "".join("1" if pixel >= avg else "0" for pixel in gray.flatten())
        return bits

    def _classify_orientation(self, width: int, height: int) -> str:
        if width > height:
            return "landscape"
        if height > width:
            return "portrait"
        return "square"

    def _compute_overall_score(
        self,
        *,
        sharpness: float,
        brightness: float,
        blur: float,
        face_count: int,
        smile_count: int,
        width: int,
        height: int,
    ) -> float:
        size_bonus = 0.0
        if width >= 1280 and height >= 720:
            size_bonus = 8.0

        face_bonus = 10.0 * min(face_count, 2)
        smile_bonus = 4.0 * min(smile_count, 2)
        brightness_factor = 1.0 - abs(brightness - 60.0) / 60.0
        sharpness_factor = np.clip(sharpness / 100.0, 0.0, 1.0)
        blur_factor = np.clip((100.0 - blur) / 100.0, 0.0, 1.0)

        score = 45.0 * sharpness_factor + 20.0 * brightness_factor + 20.0 * blur_factor + face_bonus + smile_bonus + size_bonus
        return float(np.clip(score, 0.0, 100.0))
