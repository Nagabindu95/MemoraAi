from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memora_ai.domain.duplicate_remover import DuplicateImageRemover


class DuplicateImageRemoverTests(unittest.TestCase):
    def test_filter_images_returns_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_one = temp_path / "one.png"
            image_two = temp_path / "two.png"

            image_one.write_bytes(b"fake-image")
            image_two.write_bytes(b"fake-image")

            remover = DuplicateImageRemover(similarity_threshold=0.999, embedding_provider=lambda paths: np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
            result = remover.filter_images([image_one, image_two])

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], image_one)


if __name__ == "__main__":
    unittest.main()
