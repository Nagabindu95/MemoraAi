from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from moviepy.editor import CompositeVideoClip, concatenate_videoclips

from src.memora_ai.render_engine.animation_engine import AnimationEngine, AnimationInstruction
from src.memora_ai.render_engine.photo_placer import PhotoPlacer
from src.memora_ai.render_engine.transition_engine import TransitionEngine, TransitionInstruction
from src.memora_ai.style_engine.style_profile import StyleProfile

logger = logging.getLogger(__name__)


class TimelineRenderer:
    """Assembles scene clips, transitions, and exports a preview video."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        self.logger = logger_instance or logger
        self.photo_placer = PhotoPlacer(logger_instance=self.logger)
        self.animation_engine = AnimationEngine(logger_instance=self.logger)
        self.transition_engine = TransitionEngine(logger_instance=self.logger)

    def render_preview(self, style_profile: StyleProfile, photo_paths: list[Path], output_path: str | Path) -> Path:
        """Render a preview video using the style profile and arranged photos."""
        placement_plan = self.photo_placer.create_plan(style_profile, photo_paths)
        clips = []
        size = (style_profile.width, style_profile.height)

        for placement in placement_plan.placements:
            animation_kind = self._resolve_animation(placement.scene_index, style_profile)
            animation = AnimationInstruction(kind=animation_kind, duration=placement.duration_seconds)
            clip = self.animation_engine.build_clip(
                str(placement.image_path),
                placement.duration_seconds,
                animation,
                size,
            )
            clips.append(clip)

        if len(clips) > 1:
            final_clip = concatenate_videoclips(clips, method="compose")
        elif clips:
            final_clip = clips[0]
        else:
            raise ValueError("No clips were produced for rendering")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        final_clip.write_videofile(str(destination), fps=style_profile.fps, codec="libx264", audio=False)
        self.logger.info("Rendered preview video to %s", destination)
        return destination

    def _resolve_animation(self, scene_index: int, style_profile: StyleProfile) -> str:
        """Resolve an animation kind from the style profile motion metadata."""
        if scene_index < len(style_profile.motion_analysis):
            motion_info = style_profile.motion_analysis[scene_index]
            appearance = str(motion_info.get("appearance_style", "Pop-in")).strip()
            if appearance.lower() == "fade":
                return "Fade"
            if appearance.lower() == "scale":
                return "Scale Pop"
            if appearance.lower() == "slide":
                return "Pan Right"
            if appearance.lower() == "pop-in":
                return "Scale Pop"
        return "Zoom In"
