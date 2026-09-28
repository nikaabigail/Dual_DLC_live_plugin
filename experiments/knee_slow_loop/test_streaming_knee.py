"""Correctness tests for a bounded offline-to-streaming numerical prototype."""
from pathlib import Path
import hashlib
import json
import unittest
import numpy as np
from streaming_knee import StreamingKnee, StreamConfig, PoseSample, synthetic_sample


class StreamingKneeTests(unittest.TestCase):
    def feed(self, stream, ids, **kw):
        return [o.result for i in ids if (o := stream.push(synthetic_sample(i, **kw))).result is not None]

    def test_matches_existing_offline_fusion_fully_observed_interior(self):
        folder = Path(__file__).resolve().parent / 'fixtures'
        path = folder / 'synthetic_offline_fusion.npz'
        provenance = json.loads((folder / 'synthetic_offline_fusion.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), provenance['fixture_sha256'])
        with np.load(path, allow_pickle=False) as frozen:
            points = frozen['points']
            geometry = frozen['geometry']
            radii = frozen['radii']
            times = frozen['capture_times_s']
            expected = frozen['expected_interior_knee_xy']
            ids = frozen['frame_ids']
        n = len(points)
        samples = [PoseSample(frame_id=int(ids[i]), capture_time_s=float(times[i]),
                              hip_xy=points[i,0,:2], knee_xy=points[i,1,:2],
                              ankle_xy=points[i,2,:2], knee_likelihood=points[i,1,2],
                              geometry_xy=geometry[i], radii=radii[i]) for i in range(n)]
        stream = StreamingKnee()
        emitted = [o.result for s in samples if (o := stream.push(s)).result is not None]
        self.assertEqual([r.frame_id for r in emitted], list(range(3, n-3)))
        np.testing.assert_allclose([r.knee_xy for r in emitted], expected, atol=1e-11, rtol=0)
        np.testing.assert_array_equal([r.hip_xy for r in emitted], points[3:-3,0,:2])
        np.testing.assert_array_equal([r.ankle_xy for r in emitted], points[3:-3,2,:2])

    def test_100hz_waits_exactly_three_future_samples(self):
        s = StreamingKnee()
        self.assertEqual(len(self.feed(s, range(6))), 0)
        r = s.push(synthetic_sample(6)).result
        self.assertEqual(r.frame_id, 3)
        self.assertEqual(r.evidence_frame_ids, tuple(range(7)))
        self.assertAlmostEqual(r.future_wait_s, 0.03)
        self.assertAlmostEqual(r.capture_age_at_start_s, 0.03)

    def test_50hz_declared_decimation_wait_is_60ms(self):
        s = StreamingKnee(StreamConfig(sample_period_s=0.02, expected_frame_stride=2,
                                      max_window_span_s=0.15, max_capture_age_s=0.09))
        r = self.feed(s, range(12), source_period_s=0.02, frame_stride=2)[0]
        self.assertEqual(r.frame_id, 6)
        self.assertEqual(r.evidence_frame_ids, tuple(range(0,14,2)))
        self.assertAlmostEqual(r.future_wait_s, 0.06)

    def test_no_access_to_later_than_plus_three(self):
        a, b = StreamingKnee(), StreamingKnee()
        ra, rb = [], []
        for i in range(40):
            oa = a.push(synthetic_sample(i))
            ob = b.push(synthetic_sample(i, knee_xy=np.array([10000., -10000.])) if i >= 20 else synthetic_sample(i))
            if oa.result is not None:
                ra.append(oa.result)
                rb.append(ob.result)
        for x, y in zip(ra, rb):
            if x.frame_id <= 16:
                np.testing.assert_array_equal(x.knee_xy, y.knee_xy)
            self.assertLessEqual(max(x.evidence_frame_ids), x.frame_id + 3)

    def test_buffer_constant_long_stream(self):
        s = StreamingKnee()
        for i in range(12000):
            s.push(synthetic_sample(i))
            self.assertLessEqual(s.buffered_samples, 7)
        self.assertEqual(s.max_buffer_seen, 7)
        self.assertEqual(s.counters['emitted_results'], 11994)
        self.assertLess(len(s.counters), 12)
        self.assertFalse(hasattr(s, 'results'))

    def test_frame_gap_resets_and_never_bridges_skipped_capture(self):
        s = StreamingKnee()
        self.feed(s, range(10))
        outcomes = [s.push(synthetic_sample(i)) for i in range(12,19)]
        self.assertTrue(all(o.result is None for o in outcomes[:6]))
        r = outcomes[-1].result
        self.assertEqual(r.evidence_frame_ids, tuple(range(12,19)))
        self.assertEqual(s.counters['reset_frame_gap'], 1)

    def test_timestamp_gap_resets_even_with_consecutive_ids(self):
        s = StreamingKnee()
        self.feed(s, range(10))
        for i in range(10,17):
            o = s.push(synthetic_sample(i, capture_time_s=i*.01+.1))
            if i < 16:
                self.assertIsNone(o.result)
        self.assertEqual(o.result.evidence_frame_ids, tuple(range(10,17)))
        self.assertEqual(s.counters['reset_time_gap'], 1)

    def test_duplicate_out_of_order_rejected_without_buffer_change(self):
        s = StreamingKnee()
        self.feed(s, range(10))
        before = tuple(r['frame_id'] for r in s.buffer)
        self.assertEqual(s.push(synthetic_sample(9), now_s=.1).reason, 'duplicate_or_reordered')
        self.assertEqual(s.push(synthetic_sample(8), now_s=.11).reason, 'duplicate_or_reordered')
        self.assertEqual(tuple(r['frame_id'] for r in s.buffer), before)
        self.assertEqual(s.counters['rejected_duplicate_or_reordered'], 2)

    def test_side_route_stimulation_changes_reset_each_generation(self):
        for key, value in [('side','left'), ('route','camera_1'), ('stim_version',1)]:
            with self.subTest(key=key):
                s = StreamingKnee()
                self.feed(s, range(10))
                rr = self.feed(s, range(10,17), **{key:value})
                self.assertEqual(len(rr), 1)
                self.assertEqual(rr[0].evidence_frame_ids, tuple(range(10,17)))
                self.assertEqual(getattr(rr[0], key), value)
                self.assertEqual(s.counters['reset_generation_boundary'], 1)

    def test_one_to_three_missing_knees_need_available_real_boundaries(self):
        for length in [1,2,3]:
            with self.subTest(length=length):
                s = StreamingKnee()
                emitted = {}
                for i in range(20):
                    p = synthetic_sample(i)
                    if 8 <= i < 8 + length:
                        p = synthetic_sample(i, knee_xy=[np.nan,np.nan])
                    o = s.push(p)
                    if o.result:
                        emitted[o.result.frame_id] = o.result
                for i in range(8,8+length):
                    r = emitted[i]
                    self.assertEqual(r.status, 'interpolated')
                    self.assertTrue(np.isfinite(r.knee_xy).all())
                    self.assertTrue(np.isnan(r.likelihood))
                    self.assertIn(7, r.evidence_frame_ids)
                    self.assertIn(8+length, r.evidence_frame_ids)
                    self.assertLessEqual(8+length, i+3)

    def test_four_missing_knees_never_filled(self):
        s = StreamingKnee()
        rr = {}
        for i in range(20):
            p = synthetic_sample(i, knee_xy=[np.nan,np.nan]) if 8 <= i < 12 else synthetic_sample(i)
            o = s.push(p)
            if o.result:
                rr[o.result.frame_id] = o.result
        for i in range(8,12):
            self.assertFalse(np.isfinite(rr[i].knee_xy).any())

    def test_geometry_cannot_fill_missing_knee_by_itself(self):
        s = StreamingKnee()
        rr = self.feed(s, range(20), knee_xy=[np.nan,np.nan])
        self.assertTrue(rr)
        self.assertTrue(all(not np.isfinite(r.knee_xy).any() for r in rr))

    def test_low_nan_and_invalid_likelihood_are_missing(self):
        for confidence in [0.1, np.nan, np.inf, -1.0, 1.1]:
            with self.subTest(confidence=confidence):
                s = StreamingKnee()
                rr = self.feed(s, range(12), knee_likelihood=confidence)
                self.assertTrue(all(not np.isfinite(r.knee_xy).any() for r in rr))

    def test_invalid_anchor_resets(self):
        s = StreamingKnee()
        self.feed(s, range(10))
        o = s.push(synthetic_sample(10, hip_xy=[np.nan,90]))
        self.assertEqual(o.reason, 'invalid_anchor')
        self.assertEqual(s.buffered_samples,0)
        rr = self.feed(s, range(11,18))
        self.assertEqual(rr[0].evidence_frame_ids, tuple(range(11,18)))

    def test_invalid_geometry_skipped_without_poisoning_knee(self):
        a, b = StreamingKnee(), StreamingKnee()
        ra = self.feed(a, range(20), geometry_xy=None, radii=None)
        rb = self.feed(b, range(20), geometry_xy=[np.nan,1], radii=[-1,50])
        np.testing.assert_array_equal([r.knee_xy for r in ra], [r.knee_xy for r in rb])

    def test_stale_arrival_and_stale_output_are_rejected(self):
        s = StreamingKnee()
        self.assertEqual(s.push(synthetic_sample(0), now_s=.051).reason, 'stale_input')
        self.assertEqual(s.buffered_samples,0)
        s = StreamingKnee()
        for i in range(7):
            o = s.push(synthetic_sample(i), now_s=i*.01+.025)
        self.assertEqual(o.reason, 'stale_output')
        self.assertIsNone(o.result)

    def test_invalid_and_backward_clock(self):
        s = StreamingKnee()
        self.assertEqual(s.push(synthetic_sample(0), now_s=-1).reason,'invalid_time')
        self.assertEqual(s.push(synthetic_sample(0,capture_time_s=np.nan)).reason,'invalid_time')
        s.push(synthetic_sample(0),now_s=.02)
        self.assertEqual(s.push(synthetic_sample(1),now_s=.015).reason,'nonmonotonic_arrival')

    def test_span_guard_even_when_individual_intervals_acceptable(self):
        s = StreamingKnee(StreamConfig(max_window_span_s=.06))
        for i in range(7):
            o = s.push(synthetic_sample(i,capture_time_s=i*.011))
        self.assertEqual(o.reason,'window_span')
        self.assertEqual(s.buffered_samples,1)

    def test_strict_deadline_50hz_config_rejected(self):
        with self.assertRaises(ValueError):
            StreamConfig(sample_period_s=.02,expected_frame_stride=2,max_window_span_s=.15,max_capture_age_s=.03)

    def test_no_end_flush_or_partial_startup_extrapolation(self):
        s = StreamingKnee()
        rr = self.feed(s, range(10))
        self.assertEqual([r.frame_id for r in rr],[3,4,5,6])
        self.assertFalse(hasattr(s,'flush'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
