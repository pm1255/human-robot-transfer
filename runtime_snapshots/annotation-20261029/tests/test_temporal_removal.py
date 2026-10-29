"""Behavioral checks on a known moving camera and occluded static surface."""
import unittest
import cv2
import numpy as np
from human2robot.retarget_cloth.temporal_removal import TemporalRemoval


class TemporalRemovalTest(unittest.TestCase):
    def test_static_surface_flicker_and_visible_cloth_preservation(self):
        cv2.setNumThreads(1)
        rng = np.random.default_rng(17)
        truth = cv2.GaussianBlur(rng.integers(30, 210, (480, 640, 3), dtype=np.uint8), (5, 5), 1)
        filter_ = TemporalRemoval()
        outputs, baselines = [], []
        for frame in range(16):
            background = np.roll(truth, frame, axis=1)
            raw = background.copy()
            human = np.zeros((480, 640), bool); human[290:380, 220+frame:320+frame] = True
            cloth = np.zeros_like(human); cloth[160:240, 200:400] = True
            raw[cloth] = [210, 205, 198]
            raw[human] = [65, 95, 170]
            clean = background.copy()
            clean[human] = np.clip(background[human].astype(float) + (-22 if frame % 2 else 22), 0, 255)
            result = raw.copy(); result[human] = clean[human]
            baseline = result.copy()
            output, edit, record, _ = filter_.apply(raw, clean, result, cloth, human, np.zeros_like(human), frame)
            self.assertTrue(record['observed_cloth_unchanged'])
            np.testing.assert_array_equal(output[~edit], raw[~edit])
            baselines.append(np.roll(baseline, -frame, axis=1)[310:360, 240:300].astype(float))
            outputs.append(np.roll(output, -frame, axis=1)[310:360, 240:300].astype(float))
        before = np.abs(np.diff(baselines[4:], axis=0)).mean()
        after = np.abs(np.diff(outputs[4:], axis=0)).mean()
        self.assertLess(after, before * .5)
        self.assertFalse(record['history_reset'])
        # Abrupt cut to an unrelated scene must not carry the old patch.
        new = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        _, _, record, _ = filter_.apply(new, new.copy(), new.copy(), cloth, human, np.zeros_like(human), 16)
        self.assertTrue(record['history_reset'])


if __name__ == '__main__':
    unittest.main()
