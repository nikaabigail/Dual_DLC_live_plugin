"""Independent synthetic checks; no machine-local data or geometry imports."""
from dataclasses import FrozenInstanceError, replace
import math
import unittest

from current_frame_knee import CircleCalibration, FusionConfig, fuse_current_frame


class CurrentFrameKneeTests(unittest.TestCase):
    def setUp(self):
        self.cal = CircleCalibration(200, 200, 5., 5., -1, 'synthetic-3-4-5')
        self.hip = [50., 50., .9]
        self.ankle = [50., 56., .9]
        self.knee = [54.5, 53., .8]
        self.nose = [100., 30., .9]

    def fuse(self, **overrides):
        args = dict(hip=self.hip, knee=self.knee, ankle=self.ankle,
                    calibration=self.cal, nose=self.nose)
        args.update(overrides)
        return fuse_current_frame(**args)

    def assert_raw(self, result, status):
        self.assertFalse(result.applied)
        self.assertEqual(result.status, status)
        self.assertEqual(result.raw_xy, result.fused_xy)
        self.assertEqual(result.correction_px, 0.)

    def test_analytic_3_4_5_circle_and_soft_blend(self):
        r = self.fuse()
        self.assertTrue(r.applied)
        self.assertEqual(r.geometry_xy, (54., 53.))
        self.assertAlmostEqual(math.dist(r.geometry_xy, r.hip_xy), 5.)
        self.assertAlmostEqual(math.dist(r.geometry_xy, r.ankle_xy), 5.)
        self.assertAlmostEqual(r.fused_xy[0], 54.5 - .5 * .25 / 1.05)
        self.assertEqual(r.fused_xy[1], 53.)
        self.assertEqual(r.likelihood, .8)

    def test_both_calibrated_branches_and_headings(self):
        for canonical in (-1, 1):
            for heading in (-1, 1):
                expected = (50 - 4 * canonical * heading, 53.)
                with self.subTest(canonical=canonical, heading=heading):
                    r = self.fuse(calibration=replace(self.cal, canonical_branch_sign=canonical),
                                  nose=[50 + heading * 30, 30, .9], knee=[*expected, .8])
                    self.assertEqual(r.geometry_xy, expected)
                    self.assertTrue(r.applied)

    def test_equal_x_candidates_still_use_calibrated_branch(self):
        # Unlike the discarded max-x heuristic, the old branch disambiguates this.
        r = self.fuse(ankle=[56., 50., .9], knee=[53., 46., .8])
        self.assertEqual(r.geometry_xy, (53., 46.))
        self.assertTrue(r.applied)

    def test_tangent_uses_only_candidate_even_for_opposite_branch_sign(self):
        r = self.fuse(ankle=[50., 60., .9], knee=[50., 55., .8])
        self.assertEqual(r.geometry_xy, (50., 55.))
        self.assertTrue(r.applied)

    def test_low_knee_confidence_keeps_raw_and_never_inserts(self):
        self.assert_raw(self.fuse(knee=[54., 53., .59]), 'raw_low_knee_confidence')
        self.assertTrue(self.fuse(knee=[54., 53., .6]) .applied)

    def test_missing_or_nan_knee_is_not_inserted(self):
        for row in (None, [math.nan, 53., .9], [54., math.nan, .9], [54., 53.]):
            r = self.fuse(knee=row)
            self.assertFalse(r.applied)
            self.assertEqual(r.status, 'raw_invalid_knee')
            self.assertTrue(all(math.isnan(v) for v in r.fused_xy))

    def test_invalid_knee_probabilities_preserve_finite_raw(self):
        for p in (-.01, 1.01, math.nan, math.inf):
            self.assert_raw(self.fuse(knee=[54., 53., p]), 'raw_invalid_knee')

    def test_bounds_are_applied_to_all_anchors_and_knee(self):
        for name in ('hip', 'ankle', 'knee'):
            for xy in ((-1, 50), (200, 50), (50, -1), (50, 200)):
                expected = 'raw_invalid_knee' if name == 'knee' else 'raw_invalid_anchor'
                self.assert_raw(self.fuse(**{name: [*xy, .9]}), expected)

    def test_weak_or_invalid_anchor_refuses_correction(self):
        for name in ('hip', 'ankle'):
            row = list(getattr(self, name))
            row[2] = .59
            self.assert_raw(self.fuse(**{name: row}), 'raw_low_anchor_confidence')
            row[2] = math.nan
            self.assert_raw(self.fuse(**{name: row}), 'raw_invalid_anchor')

    def test_heading_uses_current_nose_then_selected_eye(self):
        self.assertTrue(self.fuse(nose=None, eye=[100, 20, .9]).applied)
        self.assertTrue(self.fuse(nose=[55, 20, .9], eye=[100, 20, .9]).applied)
        self.assertTrue(self.fuse(nose=[201, 20, .9], eye=[100, 20, .9]).applied)
        self.assert_raw(self.fuse(nose=None, eye=None), 'raw_missing_current_heading')
        self.assert_raw(self.fuse(nose=[55, 20, .9]), 'raw_missing_current_heading')
        self.assert_raw(self.fuse(nose=[100, 20, .59]), 'raw_missing_current_heading')

    def test_unreachable_coincident_and_contained_circles(self):
        self.assert_raw(self.fuse(ankle=self.hip), 'raw_geometry_coincident_anchors')
        self.assert_raw(self.fuse(ankle=[50, 61, .9]), 'raw_geometry_incompatible_lengths')
        self.assert_raw(self.fuse(calibration=replace(self.cal, femur_px=20.)),
                        'raw_geometry_incompatible_lengths')

    def test_selected_branch_outside_image_is_not_swapped(self):
        cal = replace(self.cal, canonical_branch_sign=1)
        self.assert_raw(self.fuse(hip=[2, 50, .9], ankle=[2, 56, .9], knee=[5, 53, .8],
                                  calibration=cal), 'raw_geometry_recommended_out_of_bounds')

    def test_all_candidates_outside_image(self):
        cal = replace(self.cal, height=4)
        self.assert_raw(self.fuse(hip=[50, 1, .9], ankle=[56, 1, .9], knee=[53, 2, .8],
                                  nose=[100, 1, .9], calibration=cal),
                        'raw_geometry_all_candidates_out_of_bounds')

    def test_large_disagreement_refuses_to_snap(self):
        self.assert_raw(self.fuse(knee=[100, 100, .9]), 'raw_large_disagreement')

    def test_small_explicit_cap_limits_shift(self):
        r = self.fuse(config=FusionConfig(geometry_shift_cap_fraction=.001))
        self.assertAlmostEqual(r.correction_px, .01)
        self.assertAlmostEqual(math.dist(r.raw_xy, r.fused_xy), .01)

    def test_missing_calibration_and_zero_weight_are_explicit_raw_modes(self):
        self.assert_raw(self.fuse(calibration=None), 'raw_missing_calibration')
        self.assert_raw(self.fuse(config=FusionConfig(fusion_geometry_weight=0)), 'raw_geometry_disabled')

    def test_frozen_radii_do_not_follow_predicted_knee(self):
        a = self.fuse()
        b = self.fuse(knee=[53., 52., .8])
        self.assertEqual(a.radii, b.radii)
        self.assertEqual(a.geometry_xy, b.geometry_xy)

    def test_stateless_nonmutation_and_immutable_results(self):
        before = (self.hip[:], self.knee[:], self.ankle[:], self.nose[:])
        first = self.fuse()
        self.fuse(knee=[100, 100, .9], nose=[10, 20, .9])
        second = self.fuse()
        self.assertEqual(first, second)
        self.assertEqual(before, (self.hip, self.knee, self.ankle, self.nose))
        with self.assertRaises(FrozenInstanceError):
            first.applied = False

    def test_calibration_validation(self):
        for field, values in {'width': (0, -1, 1.5, True), 'height': (0, math.inf),
                              'femur_px': (0, -1, math.nan, math.inf, True),
                              'tibia_px': (0, -1, math.nan),
                              'canonical_branch_sign': (0, 2, None, 1.0, True),
                              'calibration_id': (None, 3)}.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.cal, **{field: value})

    def test_config_validation(self):
        for field, value in [('pcutoff', -1), ('pcutoff', 1.1), ('pcutoff', math.nan),
                             ('fusion_geometry_weight', -1), ('fusion_geometry_weight', math.inf),
                             ('geometry_shift_cap_fraction', -1), ('geometry_shift_cap_fraction', 2),
                             ('maximum_disagreement_fraction', 0), ('heading_minimum_dx_px', 0),
                             ('heading_minimum_dx_px', True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                FusionConfig(**{field: value})


if __name__ == '__main__':
    unittest.main(verbosity=2)
