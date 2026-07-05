from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.memora_ai.style_engine.style_profile import StyleProfile

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlacedPhoto:
    """Represents a photo assigned to a scene."""

    scene_index: int
    image_path: Path
    layout: str
    duration_seconds: float


@dataclass(slots=True)
class PlacementPlan:
    """Stores the photo placement plan for the full timeline."""

    placements: list[PlacedPhoto] = field(default_factory=list)


class PhotoPlacer:
    """Assigns user photos to style-engine scenes based on scene metadata."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        self.logger = logger_instance or logger

    def create_plan(self, style_profile: StyleProfile, photo_paths: list[Path]) -> PlacementPlan:
        """Create a placement plan using the style profile and the selected photos."""
        if not photo_paths:
            raise ValueError("No photos were provided for placement")

        placements: list[PlacedPhoto] = []
        photo_iter = iter(photo_paths)

        for scene_index, scene in enumerate(style_profile.scene_entries):
            try:
                photo_path = next(photo_iter)
            except StopIteration:
                photo_path = photo_paths[scene_index % len(photo_paths)]

            layout = self._infer_layout(scene_index, style_profile)
            placements.append(
                PlacedPhoto(
                    scene_index=scene_index,
                    image_path=photo_path,
                    layout=layout,
                    duration_seconds=scene.duration,
                )
            )

        return PlacementPlan(placements=placements)

    def _infer_layout(self, scene_index: int, style_profile: StyleProfile) -> str:
        """Infer whether a scene should use a single photo or a collage layout."""
        motion_info = style_profile.motion_analysis[scene_index] if scene_index < len(style_profile.motion_analysis) else {}
        layout = str(motion_info.get("image_layout", "Single image"))
        if "collage" in layout.lower():
            return "collage"
        if "multiple" in layout.lower():
            return "multiple"
        return "single"
