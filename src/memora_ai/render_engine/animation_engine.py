from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from moviepy.editor import ImageClip, CompositeVideoClip

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnimationInstruction:
    """Describes how a photo should animate within a scene."""

    kind: str
    duration: float
    intensity: float = 1.0


class AnimationEngine:
    """Applies per-scene animation effects to images using OpenCV and MoviePy."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        self.logger = logger_instance or logger

    def build_clip(self, image_path: str, duration: float, animation: AnimationInstruction, size: tuple[int, int]) -> ImageClip:
        """Create a MoviePy clip for a photo with the requested animation."""
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = self._resize_for_frame(image, size)
        clip = ImageClip(image_path, duration=duration)

        if animation.kind == "Zoom In":
            clip = clip.resize(lambda t: 1.0 + 0.15 * t / max(duration, 1e-6))
        elif animation.kind == "Zoom Out":
            clip = clip.resize(lambda t: 1.0 + 0.15 * (duration - t) / max(duration, 1e-6))
        elif animation.kind == "Pan Left":
            clip = clip.set_position(lambda t: (-int(80 * t / max(duration, 1e-6)), 0))
        elif animation.kind == "Pan Right":
            clip = clip.set_position(lambda t: (int(80 * t / max(duration, 1e-6)), 0))
        elif animation.kind == "Pan Up":
            clip = clip.set_position(lambda t: (0, -int(60 * t / max(duration, 1e-6))))
        elif animation.kind == "Pan Down":
            clip = clip.set_position(lambda t: (0, int(60 * t / max(duration, 1e-6))))
        elif animation.kind == "Scale Pop":
            clip = clip.resize(lambda t: 0.9 + 0.15 * (t / max(duration, 1e-6)))
        elif animation.kind == "Fade":
            clip = clip.set_opacity(lambda t: max(0.2, min(1.0, t / max(duration, 1e-6))))

        return clip

    def _resize_for_frame(self, image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        height, width = size
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
