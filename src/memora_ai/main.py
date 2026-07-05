from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.memora_ai.photo_ai.image_scorer import ImageScorer
from src.memora_ai.reference_ai.reference_understanding_engine import ReferenceUnderstandingEngine
from src.memora_ai.renderer.reel_renderer import ReelRenderer
from src.memora_ai.style_engine.video_analyzer import VideoAnalyzer

class MemoraAIApp:
    """Command-line application entry point for the MemoraAI style analysis pipeline."""

    def __init__(self) -> None:
        self.logger = self._configure_logging()

    @staticmethod
    def _configure_logging() -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger("memora_ai")

    def run(self) -> int:
        """Run the interactive analysis workflow from the terminal."""
        print("\n=========================")
        print("Welcome to MemoraAI")
        print("=========================")

        try:
            reference_video = self._prompt_for_path("Enter the reference video path")
            photos_folder = self._prompt_for_path("Enter the photos folder path")
            output_folder = self._prompt_for_path("Enter output folder path")

            self.logger.info("Loading video...")
            analyzer = VideoAnalyzer()
            profile = analyzer.analyze_video(reference_video)

            self.logger.info("Extracting scenes...")
            self.logger.info("Extracting motion...")

            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Saving style profile...")
            profile_path = output_dir / "style_profile.json"
            profile.export_json(profile_path)

            self.logger.info("Analyzing reference reel...")
            reference_engine = ReferenceUnderstandingEngine()
            reference_style = reference_engine.analyze(reference_video)

            self.logger.info("Scoring photos...")
            print("Scoring photos...")
            scorer = ImageScorer()
            photo_dir = Path(photos_folder)
            image_paths = sorted(
                path for path in photo_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            image_scores = scorer.score_images(image_paths) if image_paths else []

            self.logger.info("Generating reel...")
            print("Generating reel...")
            renderer = ReelRenderer(
                reference_style,
                image_scores,
                photo_dir,
                output_dir,
            )
            renderer.render("final_reel.mp4")

            self.logger.info("Rendering...")
            print("Rendering...")
            print("Done.")
            print(f"Final reel exported to: {output_dir / 'final_reel.mp4'}")
            return 0
        except FileNotFoundError as exc:
            self.logger.error("Input path error: %s", exc)
            return 1
        except ValueError as exc:
            self.logger.error("Invalid input: %s", exc)
            return 1
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Unexpected error: %s", exc)
            return 1

    @staticmethod
    def _prompt_for_path(prompt: str) -> str:
        while True:
            value = input(f"{prompt}: ").strip().strip('"')
            if value:
                return value
            print("Please enter a valid path.")


def main() -> int:
    """Entry point for running MemoraAI from the terminal."""
    app = MemoraAIApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
