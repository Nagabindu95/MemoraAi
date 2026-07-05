from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SceneTimelineEntry:
    """Represents a single scene segment extracted from the reference video."""

    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    frame_count: int


@dataclass(slots=True)
class StyleProfile:
    """Stores the learned characteristics of a reference video for later style transfer."""

    source_video: str
    fps: float
    duration_seconds: float
    width: int
    height: int
    total_frames: int
    scene_entries: list[SceneTimelineEntry] = field(default_factory=list)
    motion_analysis: list[dict[str, Any]] = field(default_factory=list)
    sampled_frame_count: int = 0
    scene_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the profile."""
        return asdict(self)

    def export_json(self, output_path: str | Path) -> Path:
        """Export the style profile to a JSON file."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Exported style profile to %s", destination)
        return destination


class StyleProfileBuilder:
    """Helper for assembling a StyleProfile from analyzer results."""

    @staticmethod
    def build(
        *,
        source_video: str,
        fps: float,
        duration_seconds: float,
        width: int,
        height: int,
        total_frames: int,
        scene_entries: list[SceneTimelineEntry],
        sampled_frame_count: int,
        motion_analysis: list[dict[str, Any]] | None = None,
    ) -> StyleProfile:
        """Create a StyleProfile instance from raw analysis output."""
        return StyleProfile(
            source_video=source_video,
            fps=fps,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            total_frames=total_frames,
            scene_entries=scene_entries,
            motion_analysis=motion_analysis or [],
            sampled_frame_count=sampled_frame_count,
            scene_count=len(scene_entries),
        )
