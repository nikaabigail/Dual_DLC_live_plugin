"""CPU-only regression tests for the opt-in knee extension; no GPU/model files."""
from __future__ import annotations

import logging
import struct
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config_dual_rt_dlc_live as config
import dual_rt_dlc_live as dual
import live_profiles
import rt_dlc_live as live
import single_rt_dlc_live_bridge as single
from pose_layout import KNEE_BRIDGE_POINT_NAMES, LEGACY_BRIDGE_POINT_NAMES

OLD_MODEL_NAMES = [
    'nose', 'eye_l', 'eye_r', 'fl_toes_l', 'fl_toes_r',
    'hl_toes_l', 'hl_ankle_l', 'hl_hip_l', 'hl_iliac_l',
    'hl_toes_r', 'hl_ankle_r', 'hl_hip_r', 'hl_iliac_r', 'spine', 'tail',
]
PADDING = {'position': 'top_left', 'border_mode': 'constant', 'border_value': 0,
           'pad_height_divisor': 32, 'pad_width_divisor': 32}


class FakeDLCLive:
    """Exercise the actual native-boundary wrapper without loading a model."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.dynamic_cropping = None
        self.runner = SimpleNamespace(init_inference=self.infer, get_pose=self.infer)

    def read_config(self):
        return {'data': {'inference': {'auto_padding': dict(PADDING)}}}

    def process_frame(self, frame):
        if self.cropping is not None:
            x1, x2, y1, y2 = self.cropping
            frame = frame[y1:y2, x1:x2]
        if self.resize != 1:
            frame = cv2.resize(frame, None, fx=self.resize, fy=self.resize)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.convert2rgb else frame

    def infer(self, frame):
        self.seen_input = frame
        return np.array([[2., 3., .99]], dtype=np.float32)


class KneeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

    def settings(self, **values):
        for key, value in values.items():
            self.stack.enter_context(patch.object(config, key, value, create=True))

    def fixture(self, knees=False, dictionary=False):
        names = list(KNEE_BRIDGE_POINT_NAMES if knees else LEGACY_BRIDGE_POINT_NAMES)
        self.settings(DUAL_OE_BRIDGE_INCLUDE_KNEES=knees, DUAL_USE_POINTS=names,
                      DUAL_OE_BRIDGE_WIRE_FORMAT='binary', DUAL_OE_BRIDGE_PACKET_MODE='pose',
                      DUAL_OE_BRIDGE_REQUEST_ACK=False)
        pose = np.arange(len(names) * 3, dtype=np.float32).reshape(-1, 3) + .25
        result = {'infer_ms': 2.5, 'raw_visible': len(names)}
        if dictionary:
            result['raw_points'] = {name: dict(x=float(p[0]), y=float(p[1]), likelihood=float(p[2]))
                                    for name, p in zip(names, pose)}
        else:
            result['raw_pose_array'] = pose
        frame = live.FramePacket(7, np.zeros((2, 2, 3), np.uint8), 12.5, 9)
        pair = dual.PairInferenceResult(11, frame, frame, 1.25, 2.0, result, result)
        runtime = SimpleNamespace(source=SimpleNamespace(dropped_total=3))
        bridge = dual.OpenEphysBridge(logging.getLogger('test'))
        return bridge, pair, runtime, pose

    def test_legacy_packet_remains_byte_identical(self):
        for dictionary in (False, True):
            bridge, pair, runtime, pose = self.fixture(dictionary=dictionary)
            with patch.object(dual.time, 'time', return_value=100.5):
                payload = bridge._build_binary_pose_payload(pair, runtime, runtime)
            # Independent original DDLP/v1 six-point wire fixture.
            header = struct.pack('<4sHHqdffHH', b'DDLP', 1, 0, 11, 100.5, 1.25, 2.0, 6, 0)
            side = struct.pack('<qqdfIHH', 7, 9, 12.5, 2.5, 3, 6, 0)
            expected = header + (side + pose.astype('<f4').tobytes()) * 2
            self.assertEqual(payload, expected)
            self.assertEqual(len(payload), 252)

    def test_knee_packet_appends_two_points_in_fixed_order(self):
        bridge, pair, runtime, pose = self.fixture(knees=True)
        payload = bridge._build_binary_pose_payload(pair, runtime, runtime)
        self.assertEqual(dual.BINARY_HEADER_STRUCT.unpack_from(payload)[-2], 8)
        self.assertEqual(dual.BINARY_POSE_KNEE_POINT_NAMES[:6], dual.BINARY_POSE_POINT_NAMES)
        self.assertEqual(dual.BINARY_POSE_KNEE_POINT_NAMES[6:], ['hl_knee_l', 'hl_knee_r'])
        self.assertEqual(len(payload), 300)
        start = dual.BINARY_HEADER_STRUCT.size + dual.BINARY_SIDE_STRUCT.size
        np.testing.assert_array_equal(np.frombuffer(payload, '<f4', 24, start).reshape(8, 3), pose)
        second_start = start + 8 * 12 + dual.BINARY_SIDE_STRUCT.size
        np.testing.assert_array_equal(np.frombuffer(payload, '<f4', 24, second_start).reshape(8, 3), pose)

    def test_extended_packet_requires_explicit_opt_in_and_order(self):
        self.settings(DUAL_OE_BRIDGE_WIRE_FORMAT='binary', DUAL_USE_POINTS=list(KNEE_BRIDGE_POINT_NAMES),
                      DUAL_OE_BRIDGE_INCLUDE_KNEES=False)
        with self.assertRaises(ValueError):
            dual.validate_bridge_point_layout()
        config.DUAL_OE_BRIDGE_INCLUDE_KNEES = True
        dual.validate_bridge_point_layout()
        config.DUAL_USE_POINTS = list(reversed(KNEE_BRIDGE_POINT_NAMES))
        with self.assertRaises(ValueError):
            dual.validate_bridge_point_layout()

    def test_roi_anchors_match_old_model_and_exclude_added_landmarks(self):
        names = OLD_MODEL_NAMES + ['hl_knee_l', 'hl_knee_r', 'hl_extra']
        expected = [i for i, name in enumerate(OLD_MODEL_NAMES) if name.startswith('hl_')]
        self.assertEqual(single.leg_roi_indices(names), expected)
        self.settings(LEG_ROI_ENABLED=True, LEG_ROI_WIDTH=448)
        old = single.build_leg_roi_tracker(OLD_MODEL_NAMES, (220, 1920, 3))
        new = single.build_leg_roi_tracker(names, (220, 1920, 3))
        pose = np.zeros((len(names), 3), np.float32)
        pose[expected] = [1000., 100., .9]
        pose[15:] = [1800., 0., 1.]
        for _ in range(10):
            old.update(pose[:15])
            new.update(pose)
            self.assertEqual(old.window(), new.window())
        self.assertEqual(new.window()[1] - new.window()[0], 448)

    def test_knee_confidence_does_not_change_native_side_choice(self):
        self.settings(DUAL_USE_POINTS=list(KNEE_BRIDGE_POINT_NAMES), SINGLE_AUTO_PICK_SIDE=True,
                      SINGLE_EMIT_BOTH_LEGS=False)
        names = config.DUAL_USE_POINTS
        pose = np.zeros((8, 3), dtype=np.float32)
        pose[:, 2] = [.2, .9, .3, .8, .4, .7, 1., 0.]
        result = {'raw_pose_array': pose, 'infer_ms': 0., 'raw_visible': 8}
        packet = live.FramePacket(1, np.zeros((2, 2, 3), np.uint8), 0.)
        pair = single.make_pair_result(1, packet, result, 'left')
        self.assertIs(pair.right_result, result)
        self.assertTrue(np.isnan(pair.left_result['raw_pose_array']).all())
        self.assertAlmostEqual(single._triplet_min_confidence(result, 'left'), .2)
        self.assertAlmostEqual(single._triplet_min_confidence(result, 'right'), .7)
        self.assertEqual(config.DUAL_SIDE_POINT_SETS['right'], ('hl_hip_r', 'hl_ankle_r', 'hl_toes_r'))

    def test_padding_matches_training_and_preserves_origin(self):
        frame = np.full((220, 448, 3), 127, np.uint8)
        padding = live.constant_padding_config({'data': {'inference': {'auto_padding': PADDING}}})
        padded = live.pad_frame_for_model(frame, padding)
        self.assertEqual(padded.shape, (224, 448, 3))
        np.testing.assert_array_equal(padded[:220], frame)
        self.assertTrue((padded[220:] == 0).all())
        self.assertIs(live.pad_frame_for_model(padded, padding), padded)

    def test_padding_rejects_incompatible_model_contract(self):
        for changed in ({}, {**PADDING, 'border_mode': 'reflect'}, {**PADDING, 'border_value': 1},
                        {**PADDING, 'position': 'center'}, {**PADDING, 'pad_width_divisor': 0}):
            with self.assertRaises(ValueError):
                live.constant_padding_config({'data': {'inference': {'auto_padding': changed}}})

    def test_native_crop_color_and_coordinate_restore_with_padding(self):
        self.settings(MODEL_CONSTANT_ZERO_PADDING=True, CONVERT_TO_RGB=True, RESIZE=1.,
                      DUAL_TORCH_COMPILE_BACKEND='')
        with patch.dict(sys.modules, {'dlclive': SimpleNamespace(DLCLive=FakeDLCLive)}):
            dlc = live.build_dlc_live([3, 8, 1, 6])
        frame = np.zeros((10, 12, 3), np.uint8)
        frame[:] = [10, 20, 30]
        packet = live.FramePacket(1, frame, 0.)
        initialized, pose, _, _ = dual.run_raw_inference(dlc, False, packet)
        self.assertTrue(initialized)
        self.assertEqual(dlc.seen_input.shape, (32, 32, 3))
        np.testing.assert_array_equal(dlc.seen_input[:5, :5], np.tile([30, 20, 10], (5, 5, 1)))
        self.assertTrue((dlc.seen_input[5:] == 0).all())
        self.assertTrue((dlc.seen_input[:, 5:] == 0).all())
        np.testing.assert_allclose(pose, [[5., 4., .99]])
        _, pose2, _, _ = dual.run_raw_inference(dlc, True, packet)
        np.testing.assert_array_equal(pose2, pose)

    def test_legacy_build_keeps_native_class_and_unpadded_crop(self):
        self.settings(MODEL_CONSTANT_ZERO_PADDING=False, CONVERT_TO_RGB=False, RESIZE=1.)
        with patch.dict(sys.modules, {'dlclive': SimpleNamespace(DLCLive=FakeDLCLive)}):
            dlc = live.build_dlc_live([3, 8, 1, 6])
        self.assertIs(type(dlc), FakeDLCLive)
        self.assertEqual(dlc.process_frame(np.ones((10, 12, 3), np.uint8)).shape, (5, 5, 3))

    def test_padding_after_resize_does_not_change_coordinate_scale(self):
        self.settings(MODEL_CONSTANT_ZERO_PADDING=True, CONVERT_TO_RGB=False, RESIZE=.5,
                      DUAL_TORCH_COMPILE_BACKEND='')
        with patch.dict(sys.modules, {'dlclive': SimpleNamespace(DLCLive=FakeDLCLive)}):
            dlc = live.build_dlc_live([10, 26, 20, 36])
        packet = live.FramePacket(1, np.full((60, 60, 3), 127, np.uint8), 0.)
        _, pose, _, _ = dual.run_raw_inference(dlc, False, packet)
        self.assertEqual(dlc.seen_input.shape, (32, 32, 3))
        self.assertTrue((dlc.seen_input[:8, :8] == 127).all())
        self.assertTrue((dlc.seen_input[8:] == 0).all())
        self.assertTrue((dlc.seen_input[:, 8:] == 0).all())
        np.testing.assert_allclose(pose, [[14., 26., .99]])

    def test_knee_profile_requires_explicit_model_and_restores_legacy(self):
        cfg = SimpleNamespace(MODEL_PATH='original.pt')
        with patch.dict('os.environ', {'DLC_LIVE_KNEE_MODEL_PATH': ''}):
            with self.assertRaises(ValueError):
                live_profiles.apply_profile(cfg, 'single-knees-strict')
        self.assertEqual(vars(cfg), {'MODEL_PATH': 'original.pt'})
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model.pt'
            model.touch()
            with patch.dict('os.environ', {'DLC_LIVE_KNEE_MODEL_PATH': str(model)}):
                live_profiles.apply_profile(cfg, 'single-knees-strict')
            self.assertEqual(cfg.MODEL_PATH, str(model.resolve()))
            self.assertEqual(cfg.DUAL_USE_POINTS, list(KNEE_BRIDGE_POINT_NAMES))
            self.assertTrue(cfg.MODEL_CONSTANT_ZERO_PADDING)
            self.assertEqual(cfg.LEG_ROI_WIDTH, 448)
            self.assertFalse(cfg.DUAL_TORCH_ALLOW_TF32)
            live_profiles.apply_profile(cfg, 'single-strict')
            self.assertEqual(cfg.MODEL_PATH, 'original.pt')
            self.assertEqual(cfg.DUAL_USE_POINTS, list(LEGACY_BRIDGE_POINT_NAMES))
            self.assertFalse(cfg.MODEL_CONSTANT_ZERO_PADDING)
            self.assertFalse(cfg.DUAL_OE_BRIDGE_INCLUDE_KNEES)


if __name__ == '__main__':
    unittest.main()
