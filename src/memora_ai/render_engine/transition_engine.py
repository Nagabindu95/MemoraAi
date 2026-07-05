from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from moviepy.editor import CompositeVideoClip, ImageClip, VideoClip

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TransitionInstruction:
    """Describes a reusable transition effect."""

    kind: str
    duration: float


class TransitionEngine:
    """Builds reusable transitions for the rendered timeline."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        self.logger = logger_instance or logger

    def create_transition(self, incoming_clip: VideoClip, outgoing_clip: VideoClip, transition: TransitionInstruction) -> CompositeVideoClip:
        """Create a composite clip for the specified transition."""
        if transition.kind == "Fade":
            return CompositeVideoClip([incoming_clip.set_opacity(1.0), outgoing_clip.set_opacity(0.0)], size=incoming_clip.size)
        if transition.kind == "Crossfade":
            return CompositeVideoClip([incoming_clip.set_opacity(1.0), outgoing_clip.set_opacity(1.0)], size=incoming_clip.size)
        if transition.kind == "Slide":
            return CompositeVideoClip([incoming_clip, outgoing_clip], size=incoming_clip.size)
        if transition.kind == "Pop":
            return CompositeVideoClip([incoming_clip, outgoing_clip], size=incoming_clip.size)
        if transition.kind == "Blur":
            return CompositeVideoClip([incoming_clip, outgoing_clip], size=incoming_clip.size)
        raise ValueError(f"Unsupported transition: {transition.kind}")
