from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.memora_ai.photo_ai.image_scorer import ImageScore
from src.memora_ai.reference_ai.reference_understanding_engine import ReferenceStyle
from src.memora_ai.renderer.ken_burns import KenBurnsAnimator


class ReelRenderer:
    """Render a reel that recreates the motion and structure of a reference style."""

    def __init__(
        self,
        reference_style: ReferenceStyle,
        image_scores: list[ImageScore],
        photos_folder: str | Path,
        output_folder: str | Path,
    ) -> None:
        self.reference_style = reference_style
        self.image_scores = image_scores
        self.photos_folder = Path(photos_folder)
        self.output_folder = Path(output_folder)
        self._animator = KenBurnsAnimator()
        self._frame_width = max(1, int(reference_style.width or 1080))
        self._frame_height = max(1, int(reference_style.height or 1920))
        self._fps = max(24, int(reference_style.fps or 24))

    def render(self, output_filename: str = "final_reel.mp4") -> Path:
        """Build a scene-driven reel and export it as a video."""
        if not self.photos_folder.exists():
            raise FileNotFoundError(f"Photos folder not found: {self.photos_folder}")

        self.output_folder.mkdir(parents=True, exist_ok=True)
        photo_paths = self._load_photo_paths()
        if not photo_paths:
            raise ValueError("No photos were found to render")

        photo_cache = self._load_photo_cache(photo_paths)
        output_path = self.output_folder / output_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, (self._frame_width, self._frame_height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open video writer for {output_path}")

        try:
            used_photo_names: set[str] = set()
            for scene_index, scene in enumerate(self.reference_style.scenes):
                frame_count = max(2, int(round(float(scene.duration) * self._fps)))
                selected_photos = self._select_photos_for_scene(scene, photo_cache, used_photo_names, scene_index)
                selected_names = [entry["name"] for entry in selected_photos]
                used_photo_names.update(selected_names)
                self._write_scene(writer, scene, selected_photos, frame_count)
        finally:
            writer.release()

        return output_path

    def _load_photo_paths(self) -> list[Path]:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(
            path for path in self.photos_folder.iterdir() if path.is_file() and path.suffix.lower() in image_extensions
        )

    def _load_photo_cache(self, photo_paths: list[Path]) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        for path in photo_paths:
            try:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            except Exception:
                continue
            if image is not None:
                cache[path.name] = image
        return cache

    def _select_photos_for_scene(
        self,
        scene: Any,
        photo_cache: dict[str, np.ndarray],
        used_photo_names: set[str],
        scene_index: int,
    ) -> list[dict[str, Any]]:
        ranked = self._rank_photos(scene, photo_cache, used_photo_names, scene_index)
        if getattr(scene, "collage_layout", "single") != "single":
            count = 2 if len(ranked) >= 2 else 1
            selected = ranked[:count]
        else:
            selected = ranked[:1]

        return selected

    def _rank_photos(
        self,
        scene: Any,
        photo_cache: dict[str, np.ndarray],
        used_photo_names: set[str],
        scene_index: int,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for name, image in photo_cache.items():
            if name in used_photo_names and len(photo_cache) > len(self.reference_style.scenes):
                continue

            score = self._score_photo(name, scene, scene_index)
            scored.append({"name": name, "image": image, "score": score})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def _score_photo(self, photo_name: str, scene: Any, scene_index: int) -> float:
        image_score = next((item for item in self.image_scores if item.filename == photo_name), None)
        quality = image_score.quality_score if image_score is not None else 50.0
        if getattr(scene, "collage_layout", "single") != "single":
            quality += 8.0
        if getattr(scene, "zoom", 1.0) > 1.03:
            quality += 2.0
        return float(quality + (scene_index % 7) * 0.1)

    def _write_scene(
        self,
        writer: cv2.VideoWriter,
        scene: Any,
        selected_photos: list[dict[str, Any]],
        frame_count: int,
    ) -> None:
        if not selected_photos:
            return

        if getattr(scene, "collage_layout", "single") != "single" and len(selected_photos) > 1:
            self._write_collage_scene(writer, scene, selected_photos, frame_count)
            return

        self._write_single_scene(writer, scene, selected_photos[0]["image"], frame_count)

    def _write_single_scene(self, writer: cv2.VideoWriter, scene: Any, photo: np.ndarray, frame_count: int) -> None:
        motion_frames = self._build_motion_frames(photo, scene, frame_count)
        fade_frames = max(2, frame_count // 6)

        for frame_index in range(frame_count):
            frame = motion_frames[min(frame_index, len(motion_frames) - 1)]
            frame = self._fit_to_canvas(frame, self._frame_width, self._frame_height)
            alpha = 1.0
            if frame_index < fade_frames:
                alpha = frame_index / max(1, fade_frames)
            elif frame_index >= frame_count - fade_frames:
                alpha = (frame_count - frame_index) / max(1, fade_frames)
            writer.write(self._apply_fade(frame, alpha))

    def _write_collage_scene(
        self,
        writer: cv2.VideoWriter,
        scene: Any,
        selected_photos: list[dict[str, Any]],
        frame_count: int,
    ) -> None:
        fade_frames = max(2, frame_count // 6)
        for frame_index in range(frame_count):
            frame = self._build_collage_frame([item["image"] for item in selected_photos], scene)
            frame = self._fit_to_canvas(frame, self._frame_width, self._frame_height)
            alpha = 1.0
            if frame_index < fade_frames:
                alpha = frame_index / max(1, fade_frames)
            elif frame_index >= frame_count - fade_frames:
                alpha = (frame_count - frame_index) / max(1, fade_frames)
            writer.write(self._apply_fade(frame, alpha))

    def _build_motion_frames(self, photo: np.ndarray, scene: Any, frame_count: int) -> list[np.ndarray]:
        duration = max(0.25, float(getattr(scene, "duration", 1.0)))
        if getattr(scene, "zoom", 1.0) > 1.03:
            return self._animator.zoom_in(photo, duration, self._fps)
        if getattr(scene, "zoom", 1.0) < 0.97:
            return self._animator.zoom_out(photo, duration, self._fps)
        if getattr(scene, "pan_x", 0.0) < -0.01:
            return self._animator.pan_left(photo, duration, self._fps)
        if getattr(scene, "pan_x", 0.0) > 0.01:
            return self._animator.pan_right(photo, duration, self._fps)
        return self._animator.zoom_in(photo, duration, self._fps)

    def _build_collage_frame(self, photos: list[np.ndarray], scene: Any) -> np.ndarray:
        if not photos:
            return np.zeros((self._frame_height, self._frame_width, 3), dtype=np.uint8)
        if len(photos) == 1:
            return self._fit_to_canvas(photos[0], self._frame_width, self._frame_height)

        tile_count = min(len(photos), 2)
        tile_width = self._frame_width // tile_count
        canvas = np.zeros((self._frame_height, self._frame_width, 3), dtype=np.uint8)
        for index, photo in enumerate(photos[:tile_count]):
            tile = self._fit_to_canvas(photo, tile_width, self._frame_height)
            x_offset = index * tile_width
            canvas[:, x_offset:x_offset + tile_width] = tile[:, :tile_width]
        return canvas

    def _fit_to_canvas(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        image_rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

        src_height, src_width = image_rgb.shape[:2]
        target_ratio = width / height
        src_ratio = src_width / src_height

        if src_ratio > target_ratio:
            new_width = width
            new_height = max(1, int(round(width / src_ratio)))
        else:
            new_height = height
            new_width = max(1, int(round(height * src_ratio)))

        resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y_offset = max(0, (height - new_height) // 2)
        x_offset = max(0, (width - new_width) // 2)
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        return canvas

    def _apply_fade(self, frame: np.ndarray, alpha: float) -> np.ndarray:
        if alpha >= 1.0:
            return frame
        alpha = float(np.clip(alpha, 0.0, 1.0))
        base = np.zeros_like(frame)
        return cv2.addWeighted(base, 1.0 - alpha, frame, alpha, 0)
