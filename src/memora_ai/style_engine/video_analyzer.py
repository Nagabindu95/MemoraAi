from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.memora_ai.style_engine.motion_analyzer import MotionAnalyzer
from src.memora_ai.style_engine.style_profile import StyleProfile, StyleProfileBuilder
from src.memora_ai.style_engine.timeline_extractor import TimelineExtractor

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyzes a reference video and produces a style profile for later transfer steps."""

    def __init__(
        self,
        *,
        sample_interval: int = 6,
        max_frames: Optional[int] = 180,
        scene_threshold: float = 25.0,
        min_scene_duration: float = 0.5,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        self.sample_interval = sample_interval
        self.max_frames = max_frames
        self.scene_threshold = scene_threshold
        self.min_scene_duration = min_scene_duration
        self.logger = logger_instance or logger
        self.timeline_extractor = TimelineExtractor(
            scene_threshold=scene_threshold,
            min_scene_duration=min_scene_duration,
        )
        self.motion_analyzer = MotionAnalyzer(logger_instance=self.logger)

    def analyze_video(self, video_path: str | Path) -> StyleProfile:
        """Load a video, sample frames, detect scenes, and build a style profile."""
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_file}")

        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_file}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            duration_seconds = total_frames / fps if fps > 0 else 0.0

            sampled_frames: list[np.ndarray] = []
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % self.sample_interval == 0:
                    sampled_frames.append(frame)
                    if self.max_frames is not None and len(sampled_frames) >= self.max_frames:
                        break

                frame_index += 1

            scene_entries = self.timeline_extractor.extract(sampled_frames, fps)
            scene_groups = self._group_frames_by_scene(sampled_frames, scene_entries)
            motion_analysis = [
                self.motion_analyzer.analyze_scene(frames, index).to_dict()
                for index, frames in enumerate(scene_groups)
            ]

            profile = StyleProfileBuilder.build(
                source_video=str(video_file),
                fps=fps,
                duration_seconds=duration_seconds,
                width=width,
                height=height,
                total_frames=total_frames,
                scene_entries=scene_entries,
                sampled_frame_count=len(sampled_frames),
                motion_analysis=motion_analysis,
            )
            self.logger.info(
                "Analyzed video %s: %.2f fps, %d frames, %d scenes",
                video_file,
                fps,
                total_frames,
                profile.scene_count,
            )
            return profile
        finally:
            capture.release()

    def export_profile(self, video_path: str | Path, output_path: str | Path) -> Path:
        """Analyze a video and immediately export the style profile to JSON."""
        profile = self.analyze_video(video_path)
        return profile.export_json(output_path)

    def _group_frames_by_scene(self, frames: list[np.ndarray], scene_entries: list) -> list[list[np.ndarray]]:
        """Split sampled frames into scene groups based on the extracted scene windows."""
        if not scene_entries:
            return [frames]

        grouped: list[list[np.ndarray]] = []
        for scene in scene_entries:
            start_index = scene.start_frame
            end_index = scene.end_frame
            grouped.append(frames[start_index : end_index + 1])
        return grouped
