from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageAsset:
    """Represents a loaded image file and its metadata."""

    path: Path
    image: np.ndarray
    width: int
    height: int
    format_name: str


class ImageLoader:
    """Loads and validates images from a directory."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

    def __init__(self, input_dir: Path, logger_instance: Optional[logging.Logger] = None) -> None:
        self.input_dir = Path(input_dir)
        self.logger = logger_instance or logger

    def load_images(self) -> List[ImageAsset]:
        """Load all supported images from the input directory."""
        if not self.input_dir.exists():
            self.logger.error("Input directory does not exist: %s", self.input_dir)
            return []

        if not self.input_dir.is_dir():
            self.logger.error("Input path is not a directory: %s", self.input_dir)
            return []

        image_assets: List[ImageAsset] = []

        for image_path in sorted(self.input_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None or image.size == 0:
                    self.logger.warning("Skipping corrupted or unreadable image: %s", image_path)
                    continue

                height, width = image.shape[:2]
                if width <= 0 or height <= 0:
                    self.logger.warning("Skipping invalid image dimensions: %s", image_path)
                    continue

                image_assets.append(
                    ImageAsset(
                        path=image_path,
                        image=image,
                        width=width,
                        height=height,
                        format_name=image_path.suffix.lower().lstrip("."),
                    )
                )
                self.logger.info("Loaded image: %s", image_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.exception("Failed to load image %s: %s", image_path, exc)

        return image_assets
