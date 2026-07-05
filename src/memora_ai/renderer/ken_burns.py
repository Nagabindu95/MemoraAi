from __future__ import annotations

from typing import List

import cv2
import numpy as np


class KenBurnsAnimator:
    """Generate high-quality Ken Burns-style animation frame sequences with OpenCV and NumPy."""

    def zoom_in(self, image: np.ndarray, duration: float, fps: int) -> list[np.ndarray]:
        """Return a zoom-in animation sequence that preserves the original aspect ratio."""
        return self._animate(image, duration, fps, zoom_start=1.0, zoom_end=1.08, shift_x=0.0, shift_y=0.0)

    def zoom_out(self, image: np.ndarray, duration: float, fps: int) -> list[np.ndarray]:
        """Return a zoom-out animation sequence that preserves the original aspect ratio."""
        return self._animate(image, duration, fps, zoom_start=1.08, zoom_end=1.0, shift_x=0.0, shift_y=0.0)

    def pan_left(self, image: np.ndarray, duration: float, fps: int) -> list[np.ndarray]:
        """Return a pan-left animation sequence that preserves the original aspect ratio."""
        return self._animate(image, duration, fps, zoom_start=1.0, zoom_end=1.0, shift_x=-0.08, shift_y=0.0)

    def pan_right(self, image: np.ndarray, duration: float, fps: int) -> list[np.ndarray]:
        """Return a pan-right animation sequence that preserves the original aspect ratio."""
        return self._animate(image, duration, fps, zoom_start=1.0, zoom_end=1.0, shift_x=0.08, shift_y=0.0)

    def _animate(
        self,
        image: np.ndarray,
        duration: float,
        fps: int,
        *,
        zoom_start: float,
        zoom_end: float,
        shift_x: float,
        shift_y: float,
    ) -> list[np.ndarray]:
        if image is None or image.size == 0:
            raise ValueError("Input image cannot be empty")
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if fps <= 0:
            raise ValueError("FPS must be positive")

        image = self._ensure_rgb(image)
        height, width = image.shape[:2]
        frame_count = max(2, int(round(duration * fps)))

        frames: list[np.ndarray] = []
        for step in range(frame_count):
            t = step / max(1, frame_count - 1)
            eased_t = self._ease_in_out(t)
            zoom = zoom_start + (zoom_end - zoom_start) * eased_t
            offset_x = shift_x * eased_t * width
            offset_y = shift_y * eased_t * height

            frame = self._render_frame(image, zoom, offset_x, offset_y)
            frames.append(frame)

        return frames

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def _ease_in_out(t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return 0.5 * (1.0 - np.cos(np.pi * t))

    def _render_frame(self, image: np.ndarray, zoom: float, offset_x: float, offset_y: float) -> np.ndarray:
        height, width = image.shape[:2]
        new_width = int(round(width / zoom))
        new_height = int(round(height / zoom))

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        x_start = int(round((new_width - width) / 2.0 + offset_x))
        y_start = int(round((new_height - height) / 2.0 + offset_y))
        x_start = max(0, min(x_start, max(0, new_width - width)))
        y_start = max(0, min(y_start, max(0, new_height - height)))

        x_end = x_start + width
        y_end = y_start + height

        if x_end > new_width or y_end > new_height:
            x_end = min(x_end, new_width)
            y_end = min(y_end, new_height)
            if x_end <= x_start or y_end <= y_start:
                return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        cropped = resized[y_start:y_end, x_start:x_end]
        if cropped.shape[0] != height or cropped.shape[1] != width:
            padded = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
            pad_h = max(0, height - cropped.shape[0])
            pad_w = max(0, width - cropped.shape[1])
            top = pad_h // 2
            left = pad_w // 2
            padded[top:top + cropped.shape[0], left:left + cropped.shape[1]] = cropped
            cropped = padded

        return cropped
