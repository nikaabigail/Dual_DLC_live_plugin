"""Synthetic checks of external calibration binding, side routing, and CLI I/O."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from run_current_frame import CONTEXT, run, sha256


class CurrentFrameIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.poses_path = self.root / "raw.npz"
        self.calibration_path = self.root / "calibration.json"
        self.names = np.asarray(["nose", "eye_l", "eye_r", "hl_hip_l", "hl_knee_l", "hl_ankle_l", "hl_hip_r", "hl_knee_r", "hl_ankle_r"])
        one = np.asarray([[200, 60, .95], [180, 65, .9], [180, 65, .9],
                          [100, 40, .9], [70, 73, .9], [100, 120, .9],
                          [100, 40, .9], [130, 73, .9], [100, 120, .9]], dtype=float)
        self.poses = np.repeat(one[None], 3, axis=0)
        self.poses[1, 7, 2] = .3
        self.frames = np.asarray([5, 9, 15], dtype=np.int64)
        self.data = dict(poses=self.poses, bodyparts=self.names, frame_index=self.frames,
            selected_side=np.asarray(["left", "right", "left"]), python_route=np.asarray(["right", "left", "right"]),
            width=320, height=160, coordinate_context=CONTEXT, fps=100.)
        self.calibration = json.loads((Path(__file__).parent / "synthetic_calibration.json").read_text(encoding="utf-8"))
        self.write_inputs()
        self.args = SimpleNamespace(poses=self.poses_path, calibration=self.calibration_path,
            output=self.root / "result", sides=None, native_replay=None, native_index_base=1,
            width=None, height=None, coordinate_context=None, overwrite=False)

    def write_inputs(self):
        np.savez_compressed(self.poses_path, **self.data)
        self.calibration["source_sha256"] = sha256(self.poses_path)
        self.write_calibration()

    def write_calibration(self):
        self.calibration_path.write_text(json.dumps(self.calibration), encoding="utf-8")

    def test_saved_result_retains_raw_source_and_low_confidence_coordinates(self):
        original_hash = sha256(self.poses_path)
        summary = run(self.args)
        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["applied_frames"], 2)
        self.assertEqual(sha256(self.poses_path), original_hash)
        with np.load(self.args.output / "correction.npz", allow_pickle=False) as result:
            np.testing.assert_array_equal(result["frame_index"], self.frames)
            np.testing.assert_array_equal(result["side"], self.data["selected_side"])
            np.testing.assert_array_equal(result["raw_xy"], [[70,73], [130,73], [70,73]])
            np.testing.assert_array_equal(result["fused_xy"][1], result["raw_xy"][1])
            self.assertEqual(result["status"][1], "raw_low_knee_confidence")
            self.assertTrue(np.isnan(result["geometry_xy"][1]).all())
        self.assertFalse(summary["native_coordinates_used"])
        self.assertEqual(summary["future_frames"], 0)
        self.assertFalse(summary["interpolation"])

    def test_native_route_is_not_used_as_anatomical_side(self):
        native = self.root / "replay.jsonl"
        rows = []
        for frame, route, side in zip(self.frames, self.data["python_route"], self.data["selected_side"]):
            index = int(frame) + 1
            row = {"pair_index": index, "left": {"frame_id": index, "picked_side": "left"}, "right": {"frame_id": index, "picked_side": "right"}}
            row[str(route)]["picked_side"] = str(side)
            # Clearly nonsensical native coordinates must never be consumed.
            row[str(route)]["points"] = {"hl_knee_l": {"x": -12345, "y": 99999, "valid": True}}
            rows.append(row)
        native.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.args.native_replay = native
        summary = run(self.args)
        self.assertEqual(summary["applied_frames"], 2)
        with np.load(self.args.output / "correction.npz") as result:
            np.testing.assert_array_equal(result["side"], ["left", "right", "left"])
            np.testing.assert_array_equal(result["raw_xy"], [[70,73], [130,73], [70,73]])
        rows[1]["left"]["frame_id"] = 123
        native.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.args.output = self.root / "bad_native_result"
        with self.assertRaisesRegex(ValueError, "frame_id mismatch"):
            run(self.args)

    def test_external_side_frame_indices_must_match(self):
        selected = self.root / "selected.npz"
        np.savez(selected, frame_index=self.frames + 1, side=self.data["selected_side"])
        self.args.sides = selected
        with self.assertRaisesRegex(ValueError, "frame_index must exactly match"):
            run(self.args)
        np.savez(selected, frame_index=self.frames, side=self.data["selected_side"])
        self.assertEqual(run(self.args)["frames"], 3)

    def test_calibration_hash_rejects_different_pose_file(self):
        self.calibration["source_sha256"] = "0" * 64
        self.write_calibration()
        with self.assertRaisesRegex(ValueError, "does not match"):
            run(self.args)
        self.assertFalse(self.args.output.exists())

    def test_coordinate_dimensions_and_explicit_metadata_must_agree(self):
        for field, value, pattern in (("width", 321, "width differ"), ("height", 161, "height differ"), ("coordinate_context", "roi_pixels", "coordinate_context")):
            with self.subTest(field=field):
                original = self.calibration[field]
                self.calibration[field] = value
                self.write_calibration()
                with self.assertRaisesRegex(ValueError, pattern):
                    run(self.args)
                self.calibration[field] = original
        self.write_calibration()
        self.args.width = 321
        with self.assertRaisesRegex(ValueError, "conflicts with RAW pose metadata"):
            run(self.args)

    def test_missing_input_context_requires_explicit_assertions(self):
        for key in ("width", "height", "coordinate_context"):
            self.data.pop(key)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Input coordinate context"):
            run(self.args)
        self.args.coordinate_context = CONTEXT
        with self.assertRaisesRegex(ValueError, "Input width is required"):
            run(self.args)
        self.args.width, self.args.height = 320, 160
        self.assertEqual(run(self.args)["frames"], 3)

    def test_cli_never_infers_selected_side_from_pose(self):
        self.data.pop("selected_side")
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "side is never inferred"):
            run(self.args)

    def test_frame_overrides_are_rejected_not_silently_ignored(self):
        self.calibration["frame_overrides"] = {"5": {"left": {"femur_px": 1}}}
        self.write_calibration()
        with self.assertRaisesRegex(ValueError, "Frame overrides are not supported"):
            run(self.args)

    def test_existing_result_requires_explicit_overwrite(self):
        run(self.args)
        before = sha256(self.args.output / "correction.npz")
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            run(self.args)
        self.assertEqual(sha256(self.args.output / "correction.npz"), before)

    def test_overwrite_cannot_replace_raw_input(self):
        renamed = self.root / "correction.npz"
        self.poses_path.rename(renamed)
        self.args.poses = renamed
        self.args.output = self.root
        self.args.overwrite = True
        before = sha256(renamed)
        with self.assertRaisesRegex(ValueError, "must not overwrite any input"):
            run(self.args)
        self.assertEqual(sha256(renamed), before)

    def test_help_does_not_import_numpy_model_or_gpu_packages(self):
        poison = self.root / "poison"
        poison.mkdir()
        for name in ("numpy", "torch", "tensorflow", "dlclive"):
            (poison / f"{name}.py").write_text("raise RuntimeError('Dependency imported while requesting --help')", encoding="utf-8")
        environment = dict(os.environ, PYTHONPATH=str(poison), PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).parent / "run_current_frame.py"), "--help"],
            env=environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--native-replay", completed.stdout)


if __name__ == "__main__":
    unittest.main()
