import unittest
import numpy as np
from human2robot.object_annotation.confidence import pair_context, pair_evidence, final_instance, depth_score


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(42).integers(0, 256, (96, 96), dtype=np.uint8)
        self.mask = np.zeros((96, 96), bool); self.mask[20:60, 20:60] = True

    def test_static_consistent_mask_scores_high_but_is_not_probability(self):
        e = pair_evidence(self.mask, self.mask, pair_context(self.rgb, self.rgb))
        self.assertGreater(e['score'], 95)
        row = final_instance(dict(instance_id='pot:1', class_id='pot'), [e], [])
        self.assertIsNone(row['probability'])
        self.assertFalse(row['pseudo_label_candidate'])

    def test_wrong_region_shift_is_penalized(self):
        shifted = np.roll(self.mask, 35, axis=1)
        e = pair_evidence(self.mask, shifted, pair_context(self.rgb, self.rgb))
        self.assertLess(e['score'], 15)

    def test_flat_image_abstains_instead_of_rewarding_zero_flow(self):
        flat = np.zeros_like(self.rgb)
        self.assertIsNone(pair_evidence(self.mask, self.mask, pair_context(flat, flat))['score'])

    def test_known_error_overrides_high_consistency(self):
        e = pair_evidence(self.mask, self.mask, pair_context(self.rgb, self.rgb))
        row = final_instance(dict(instance_id='gripper:1', class_id='gripper'), [e], ['effector_oversegmentation'])
        self.assertEqual(row['status'], 'known_issue')
        self.assertGreater(row['temporal_score'], 95)
        self.assertFalse(row['pseudo_label_candidate'])

    def test_missing_evidence_is_not_zero_confidence(self):
        row = final_instance(dict(instance_id='pot:1', class_id='pot'), [], [])
        self.assertIsNone(row['temporal_score'])
        self.assertEqual(row['status'], 'insufficient')

    def test_depth_abstains_without_pair_and_falls_on_change(self):
        row = dict(positive_finite=True, adjacent_median_relative_change=None, adjacent_focal_relative_change=None)
        self.assertIsNone(depth_score(row)['stability_score'])
        row.update(adjacent_median_relative_change=.01, adjacent_focal_relative_change=.01)
        stable=depth_score(row)['stability_score']
        row['adjacent_focal_relative_change']=.2
        self.assertLess(depth_score(row)['stability_score'], stable)


if __name__ == '__main__': unittest.main()
