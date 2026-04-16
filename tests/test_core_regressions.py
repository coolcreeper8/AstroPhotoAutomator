import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.processing import FrameAnalyzer
from core.stacking import Stacker


class FrameAnalyzerTests(unittest.TestCase):
    def test_crop_centered_preserves_requested_size_near_edges(self):
        frame = np.zeros((100, 100), dtype=np.uint8)

        top_left_crop = FrameAnalyzer.crop_centered(frame, (10, 10), (51, 51))
        bottom_right_crop = FrameAnalyzer.crop_centered(frame, (90, 90), (51, 51))

        self.assertEqual(top_left_crop.shape, (51, 51))
        self.assertEqual(bottom_right_crop.shape, (51, 51))

    def test_crop_centered_pads_when_requested_size_exceeds_frame(self):
        frame = np.ones((20, 30), dtype=np.uint8)

        cropped = FrameAnalyzer.crop_centered(frame, (15, 10), (40, 50))

        self.assertEqual(cropped.shape, (50, 40))
        self.assertEqual(int(cropped.sum()), int(frame.sum()))


class StackerTests(unittest.TestCase):
    def test_align_frames_handles_grayscale_reference(self):
        frame = np.zeros((32, 32), dtype=np.uint8)
        frame[8:24, 8:24] = 255

        stacker = Stacker()
        aligned = stacker.align_frames([frame, frame], reference_frame=frame, mode="translate")

        self.assertEqual(len(aligned), 2)
        for aligned_frame in aligned:
            np.testing.assert_array_equal(aligned_frame, frame)


if __name__ == "__main__":
    unittest.main()
