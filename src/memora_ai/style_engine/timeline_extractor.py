from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.memora_ai.style_engine.style_profile import SceneTimelineEntry

logger = logging.getLogger(__name__)


class TimelineExtractor:
    """Extracts scene-level timeline information from a video sequence."""

    def __init__(self, *, scene_threshold: float = 25.0, min_scene_duration: float = 0.5) -> None:
        self.scene_threshold = scene_threshold
        self.min_scene_duration = min_scene_duration

    def extract(self, frames: list[np.ndarray], fps: float) -> list[SceneTimelineEntry]:
        """Build a timeline of scenes using histogram and SSIM-based frame changes."""
        if not frames:
            return []

        scene_entries: list[SceneTimelineEntry] = []
        start_frame = 0
        previous_frame: Optional[np.ndarray] = None
        transition_window: list[float] = []

        for index, frame in enumerate(frames):
            if frame is None:
                continue

            prepared_frame = self._prepare_frame(frame)
            if previous_frame is None:
                previous_frame = prepared_frame
                continue

            change_score = self._frame_change_score(previous_frame, prepared_frame)
            transition_window.append(change_score)
            if len(transition_window) > 6:
                transition_window.pop(0)

            if self._is_scene_transition(transition_window) and index - start_frame >= 1:
                self._close_scene(
                    scene_entries=scene_entries,
                    start_frame=start_frame,
                    end_frame=index - 1,
                    fps=fps,
                )
                start_frame = index
                transition_window = []

            previous_frame = prepared_frame

        self._close_scene(
            scene_entries=scene_entries,
            start_frame=start_frame,
            end_frame=len(frames) - 1,
            fps=fps,
        )

        effective_min_duration = min(self.min_scene_duration, 0.15)
        return [entry for entry in scene_entries if entry.duration >= effective_min_duration]

    def _close_scene(
        self,
        *,
        scene_entries: list[SceneTimelineEntry],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> None:
        if end_frame < start_frame:
            return

        frame_count = max(1, end_frame - start_frame + 1)
        start_time = start_frame / fps if fps > 0 else 0.0
        end_time = end_frame / fps if fps > 0 else 0.0
        duration = max(0.0, end_time - start_time)

        if duration < min(self.min_scene_duration, 0.15):
            return

        scene_entries.append(
            SceneTimelineEntry(
                start_time=round(start_time, 4),
                end_time=round(end_time, 4),
                duration=round(duration, 4),
                start_frame=start_frame,
                end_frame=end_frame,
                frame_count=frame_count,
            )
        )

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_frame, (160, 90), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(resized, (5, 5), 0)

    def _frame_change_score(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
        hist_change = self._histogram_difference(previous_frame, current_frame)
        structural_change = self._structural_difference(previous_frame, current_frame)
        edge_change = self._edge_difference(previous_frame, current_frame)
        score = 0.4 * hist_change + 0.35 * structural_change + 0.25 * edge_change
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _histogram_difference(previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
        hist_prev = cv2.calcHist([previous_frame], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([current_frame], [0], None, [64], [0, 256])
        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)
        return float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA))

    @staticmethod
    def _structural_difference(previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
        diff = cv2.absdiff(previous_frame, current_frame)
        mean_diff = float(np.mean(diff) / 255.0)
        return float(np.clip(mean_diff, 0.0, 1.0))

    @staticmethod
    def _edge_difference(previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
        prev_edges = cv2.Canny(previous_frame.astype(np.uint8), 50, 150)
        curr_edges = cv2.Canny(current_frame.astype(np.uint8), 50, 150)
        diff = cv2.absdiff(prev_edges, curr_edges)
        return float(np.mean(diff) / 255.0)

    def _is_scene_transition(self, transition_window: list[float]) -> bool:
        if not transition_window:
            return False

        strong_threshold = max(0.12, min(0.32, self.scene_threshold / 100.0))
        fade_threshold = max(0.07, strong_threshold * 0.58)

        if len(transition_window) >= 2:
            peak = max(transition_window)
            mean = float(np.mean(transition_window))
            recent = transition_window[-1]
            previous = transition_window[-2]

            rapid_cut = peak >= strong_threshold and (recent >= strong_threshold * 0.8 or mean >= strong_threshold * 0.8)
            flash_or_fade = mean >= fade_threshold and peak >= fade_threshold * 1.6
            collage_shift = (
                mean >= fade_threshold
                and peak >= strong_threshold * 0.8
                and (recent >= fade_threshold or previous >= fade_threshold)
            )

            return rapid_cut or flash_or_fade or collage_shift

        return transition_window[-1] >= strong_threshold
