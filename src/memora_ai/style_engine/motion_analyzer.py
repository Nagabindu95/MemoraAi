from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.memora_ai.style_engine.style_profile import SceneTimelineEntry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SceneMotionAnalysis:
    """Structured motion analysis for a single scene."""

    scene_index: int
    zoom_direction: str
    camera_movement: str
    motion_intensity: float
    average_movement_speed: float
    image_layout: str
    appearance_style: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the analysis."""
        return asdict(self)


@dataclass(slots=True)
class SceneMotionProfile:
    """Stores motion analysis results for each scene in the profile."""

    scenes: list[SceneMotionAnalysis] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the profile."""
        return {"scenes": [scene.to_dict() for scene in self.scenes]}


class MotionAnalyzer:
    """Analyzes camera motion and visual transition patterns in a scene."""

    def __init__(self, *, logger_instance: Optional[logging.Logger] = None) -> None:
        self.logger = logger_instance or logger

    def analyze_scene(self, frames: list[np.ndarray], scene_index: int) -> SceneMotionAnalysis:
        """Analyze one scene from a list of sampled frames."""
        if not frames:
            raise ValueError("Scene frames are empty")

        motion_vectors = self._extract_motion_vectors(frames)
        zoom_direction = self._detect_zoom_direction(frames)
        camera_movement = self._detect_camera_movement(motion_vectors)
        motion_intensity = float(np.mean([abs(value) for value in motion_vectors])) if motion_vectors else 0.0
        average_movement_speed = float(np.mean(motion_vectors)) if motion_vectors else 0.0
        image_layout = self._detect_image_layout(frames)
        appearance_style = self._detect_appearance_style(frames)

        return SceneMotionAnalysis(
            scene_index=scene_index,
            zoom_direction=zoom_direction,
            camera_movement=camera_movement,
            motion_intensity=round(motion_intensity, 4),
            average_movement_speed=round(average_movement_speed, 4),
            image_layout=image_layout,
            appearance_style=appearance_style,
        )

    def analyze_scenes(self, scene_frames: list[list[np.ndarray]]) -> SceneMotionProfile:
        """Analyze a collection of scenes and return the motion profile."""
        analyses = [self.analyze_scene(frames, index) for index, frames in enumerate(scene_frames)]
        return SceneMotionProfile(scenes=analyses)

    def _extract_motion_vectors(self, frames: list[np.ndarray]) -> list[float]:
        vectors: list[float] = []
        if len(frames) < 2:
            return vectors

        for previous, current in zip(frames, frames[1:]):
            prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            vectors.append(float(np.mean(magnitude)))
        return vectors

    def _detect_zoom_direction(self, frames: list[np.ndarray]) -> str:
        if len(frames) < 2:
            return "Static"

        first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        last = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
        first_resize = cv2.resize(first, (last.shape[1], last.shape[0]))
        diff = cv2.absdiff(first_resize, last)
        mean_diff = float(np.mean(diff))
        if mean_diff < 20:
            return "Static"
        return "Zoom In" if self._estimate_scale(first, last) > 1.0 else "Zoom Out"

    def _detect_camera_movement(self, motion_vectors: list[float]) -> str:
        if not motion_vectors:
            return "Static"

        if np.mean(motion_vectors) < 2.0:
            return "Static"

        return "Pan Right"

    def _estimate_scale(self, first: np.ndarray, last: np.ndarray) -> float:
        first_edges = cv2.Canny(first, 100, 200)
        last_edges = cv2.Canny(last, 100, 200)
        return float(np.mean(last_edges) / max(np.mean(first_edges), 1.0))

    def _detect_image_layout(self, frames: list[np.ndarray]) -> str:
        if len(frames) < 2:
            return "Single image"

        first = frames[0]
        last = frames[-1]
        if first.shape[:2] != last.shape[:2]:
            return "Photo collage"

        return "Multiple photos" if len(frames) >= 3 else "Single image"

    def _detect_appearance_style(self, frames: list[np.ndarray]) -> str:
        if len(frames) < 2:
            return "Pop-in"

        first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        last = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(first, last)
        mean_diff = float(np.mean(diff))
        if mean_diff > 40:
            return "Fade"
        if len(frames) >= 3:
            return "Scale"
        return "Slide"
