from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class AppSettings:
    """Application-wide settings for the reel generation pipeline."""

    project_root: Path = Path(__file__).resolve().parents[2]
    input_dir: Path = project_root / "input" / "photos"
    output_dir: Path = project_root / "output"
    temp_dir: Path = project_root / "temp"
    instagram_width: int = 1080
    instagram_height: int = 1920
    reference_reel_path: Optional[Path] = None


settings = AppSettings()
