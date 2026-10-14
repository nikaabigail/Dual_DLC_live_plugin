"""Apply the optional current-frame knee experiment to stored RAW pose arrays.

No model, camera, native filter, GPU, or live plugin is loaded. The caller must
supply already-selected sides. Calibration is bound to the SHA256 of the exact
input pose NPZ and to an explicit pixel coordinate context. This CLI deliberately
does not support frame-dependent calibration overrides or historical label data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

CONTEXT = "full_frame_pixels_top_left"
XY_FIELDS = ("raw_xy", "fused_xy", "geometry_xy", "hip_xy", "ankle_xy", "radii")
RESULT_FIELDS = XY_FIELDS + ("status", "applied", "likelihood", "correction_px")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_metadata(archive, key):
    if key not in archive:
        return None
    value = archive[key]
    if value.shape != ():
        raise ValueError(f"Pose metadata {key!r} must be scalar")
    return value.item()


def positive_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def validate_context(calibration, metadata, args, input_hash):
    if calibration.get("format_version") != 1:
        raise ValueError("Calibration format_version must be 1")
    if "frame_overrides" in calibration:
        raise ValueError("Frame overrides are not supported; use the core API with explicitly bound per-frame calibrations")
    if calibration.get("source_hash_kind") != "raw_poses_npz_sha256":
        raise ValueError("Calibration source_hash_kind must be raw_poses_npz_sha256")
    supplied_hash = calibration.get("source_sha256", "")
    if not isinstance(supplied_hash, str) or supplied_hash.lower() != input_hash:
        raise ValueError("Calibration source_sha256 does not match the exact input RAW pose NPZ")
    if calibration.get("coordinate_context") != CONTEXT:
        raise ValueError(f"Calibration coordinate_context must be {CONTEXT!r}")
    context = args.coordinate_context if args.coordinate_context is not None else metadata.get("coordinate_context")
    if context != CONTEXT:
        raise ValueError(f"Input coordinate context must be provided as NPZ metadata or --coordinate-context {CONTEXT}")
    if args.coordinate_context is not None and metadata.get("coordinate_context") is not None and args.coordinate_context != metadata["coordinate_context"]:
        raise ValueError("CLI coordinate context conflicts with RAW pose metadata")
    dimensions = []
    for field in ("width", "height"):
        expected = positive_integer(calibration.get(field), f"Calibration {field}")
        explicit, embedded = getattr(args, field), metadata.get(field)
        if explicit is not None and embedded is not None and explicit != embedded:
            raise ValueError(f"CLI {field} conflicts with RAW pose metadata")
        actual = explicit if explicit is not None else embedded
        if actual is None:
            raise ValueError(f"Input {field} is required in pose NPZ metadata or --{field}; do not infer it from keypoint bounds")
        if positive_integer(actual, f"Input {field}") != expected:
            raise ValueError(f"Calibration and RAW pose {field} differ")
        dimensions.append(expected)
    return tuple(dimensions)


def read_sides(args, archive, frame_index):
    import numpy as np
    count = len(frame_index)
    if args.native_replay is not None:
        if "python_route" not in archive:
            raise ValueError("--native-replay requires python_route in the RAW pose NPZ")
        routes = np.asarray(archive["python_route"])
        if routes.shape != (count,):
            raise ValueError("python_route length differs from RAW poses")
        sides = []
        with args.native_replay.open(encoding="utf-8-sig") as stream:
            rows = (json.loads(line) for line in stream if line.strip())
            for i, route in enumerate(routes.astype(str)):
                row = next(rows, None)
                expected = int(frame_index[i]) + args.native_index_base
                pair_index = row.get("pair_index") if isinstance(row, dict) else None
                if row is None or route not in ("left", "right") or type(pair_index) is not int or pair_index != expected:
                    raise ValueError(f"Native replay pair/route mismatch at input position {i}")
                selected = row.get(route, {})
                frame_id = selected.get("frame_id")
                if type(frame_id) is not int or frame_id != expected:
                    raise ValueError(f"Native routed frame_id mismatch at input position {i}")
                sides.append(selected.get("picked_side", ""))
            if next(rows, None) is not None:
                raise ValueError("Native replay contains more frames than the RAW input")
        source = "native_replay[python_route].picked_side (side metadata only)"
    elif args.sides is not None:
        with np.load(args.sides, allow_pickle=False) as selected:
            if "frame_index" not in selected or not np.array_equal(selected["frame_index"], frame_index):
                raise ValueError("External selected-side NPZ frame_index must exactly match the RAW input")
            key = "selected_side" if "selected_side" in selected else "side"
            if key not in selected:
                raise ValueError("Selected-side NPZ must contain selected_side or side")
            sides = selected[key].copy()
        source = "external selected-side NPZ"
    else:
        key = "selected_side" if "selected_side" in archive else "side"
        if key not in archive:
            raise ValueError("Provide selected_side/side in the RAW NPZ, --sides, or --native-replay; side is never inferred here")
        if "selected_side" in archive and "side" in archive and not np.array_equal(archive["selected_side"], archive["side"]):
            raise ValueError("RAW selected_side and side fields conflict")
        sides = archive[key].copy()
        source = f"RAW pose NPZ {key}"
    sides = np.asarray(sides)
    if sides.shape != (count,) or sides.dtype.kind not in ("U", "S"):
        raise ValueError("Selected sides must be a string array of shape (frames,)")
    sides = sides.astype(str)
    if not set(sides).issubset({"left", "right"}):
        raise ValueError("Every selected side must be left or right; unknown sides require upstream handling")
    return sides, source


def run(args):
    import numpy as np
    from current_frame_knee import CircleCalibration, FusionConfig, fuse_current_frame

    started = time.perf_counter()
    out = args.output.resolve()
    inputs = [path.resolve() for path in (args.poses, args.calibration, args.sides, args.native_replay) if path is not None]
    targets = [out / name for name in ("correction.npz", "compute_summary.json", "correction.npz.tmp", "compute_summary.json.tmp")]
    if any(target == source for target in targets for source in inputs):
        raise ValueError("Output paths must not overwrite any input file, even with --overwrite")
    if not args.overwrite and any((out / name).exists() for name in ("correction.npz", "compute_summary.json")):
        raise ValueError("Output already exists; choose another directory or explicitly pass --overwrite")
    poses_hash = sha256(args.poses)
    calibration_json = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
    with np.load(args.poses, allow_pickle=False) as archive:
        if "poses" not in archive or "bodyparts" not in archive:
            raise ValueError("RAW NPZ must contain poses and bodyparts")
        poses = archive["poses"].copy()
        names_array = archive["bodyparts"]
        if names_array.ndim != 1 or names_array.dtype.kind not in ("U", "S"):
            raise ValueError("bodyparts must be a one-dimensional string array")
        names = names_array.astype(str).tolist()
        if poses.ndim != 3 or poses.shape[1:] != (len(names), 3) or not len(poses) or len(set(names)) != len(names) or poses.dtype.kind not in ("f", "i", "u"):
            raise ValueError("Expected nonempty numeric poses (frames, unique_bodyparts, 3)")
        count = len(poses)
        frame_index = archive["frame_index"].copy() if "frame_index" in archive else np.arange(count, dtype=np.int64)
        if frame_index.shape != (count,) or frame_index.dtype.kind not in ("i", "u") or np.any(frame_index < 0) or (count > 1 and np.any(frame_index[1:] <= frame_index[:-1])):
            raise ValueError("frame_index must contain strictly increasing, nonnegative integer source indices")
        metadata = {key: scalar_metadata(archive, key) for key in ("width", "height", "coordinate_context", "fps")}
        width, height = validate_context(calibration_json, metadata, args, poses_hash)
        sides, side_source = read_sides(args, archive, frame_index)
    indices = {name: j for j, name in enumerate(names)}
    required = {f"hl_{joint}_{'l' if side == 'left' else 'r'}" for side in set(sides) for joint in ("hip", "knee", "ankle")}
    if required - indices.keys():
        raise ValueError(f"Missing selected-side bodyparts: {sorted(required - indices.keys())}")
    calibration_id = calibration_json.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError("Calibration must have a nonempty calibration_id")
    side_calibrations = calibration_json.get("sides")
    if not isinstance(side_calibrations, dict) or set(side_calibrations) - {"left", "right"}:
        raise ValueError("Calibration sides must be an object with left/right keys")
    calibrations = {}
    for side, spec in side_calibrations.items():
        if spec is not None:
            calibrations[side] = CircleCalibration(width=width, height=height, femur_px=spec["femur_px"],
                tibia_px=spec["tibia_px"], canonical_branch_sign=spec["canonical_branch_sign"],
                calibration_id=spec.get("calibration_id", calibration_id + ":" + side))
    config = FusionConfig(**calibration_json.get("fusion_config", {}))
    results = []
    for pose, side in zip(poses, sides):
        suffix = "l" if side == "left" else "r"
        def observation(name):
            return tuple(float(v) for v in pose[indices[name]]) if name in indices else None
        result = fuse_current_frame(observation(f"hl_hip_{suffix}"), observation(f"hl_knee_{suffix}"),
            observation(f"hl_ankle_{suffix}"), calibration=calibrations.get(str(side)),
            nose=observation("nose"), eye=observation(f"eye_{suffix}"), config=config)
        results.append(result)
    arrays = {"frame_index": frame_index, "side": sides}
    arrays.update({field: np.asarray([getattr(result, field) for result in results]) for field in RESULT_FIELDS})
    applied = arrays["applied"].astype(bool)
    if not np.array_equal(arrays["fused_xy"][~applied], arrays["raw_xy"][~applied], equal_nan=True):
        raise AssertionError("Rejected corrections changed RAW knee coordinates")
    if not np.all(arrays["likelihood"][applied] >= config.pcutoff):
        raise AssertionError("A low-confidence RAW knee was corrected")
    if not np.all(arrays["correction_px"][applied] <= config.geometry_shift_cap_fraction * arrays["radii"][applied].sum(axis=1) + 1e-9):
        raise AssertionError("A correction exceeded its displacement cap")
    shifts = arrays["correction_px"][applied]
    summary = {"format_version": 1, "frames": count, "applied_frames": int(applied.sum()),
        "status_counts": dict(Counter(arrays["status"])), "side_counts": dict(Counter(sides)),
        "calibration_id": calibration_id, "coordinate_context": CONTEXT, "width": width, "height": height,
        "config": asdict(config), "future_frames": 0, "interpolation": False, "smoothing": False,
        "selected_side_source": side_source, "native_coordinates_used": False,
        "source_sha256": poses_hash, "source_hash_kind": "raw_poses_npz_sha256",
        "calibration_sha256": sha256(args.calibration),
        "side_source_sha256": sha256(args.sides or args.native_replay) if args.sides or args.native_replay else poses_hash,
        "correction_applied_only": {"n": len(shifts), "mean_px": float(shifts.mean()), "max_px": float(shifts.max())} if len(shifts) else {"n": 0},
        "elapsed_seconds": time.perf_counter() - started,
        "limits": ["No anatomical ground truth or accuracy gain is established by this correction.",
                   "The supplied radii are final projected pixel lengths; any anatomy prior must already be applied.",
                   "This optional offline experiment neither changes the live plugin nor sends stimulation output."]}
    if metadata["fps"] is not None:
        fps = float(metadata["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Stored fps must be positive and finite")
        summary["fps"] = fps
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "correction.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, out / "correction.npz")
    summary["correction_sha256"] = sha256(out / "correction.npz")
    temporary = out / "compute_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, out / "compute_summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True, help="RAW NPZ: poses[N,B,3], bodyparts, optional frame_index and metadata")
    parser.add_argument("--calibration", type=Path, required=True, help="External JSON with exact input NPZ SHA256, pixel context, and frozen per-side radii")
    parser.add_argument("--output", type=Path, required=True, help="Directory for correction.npz and compute_summary.json")
    side_group = parser.add_mutually_exclusive_group()
    side_group.add_argument("--sides", type=Path, help="External NPZ with frame_index and selected_side/side; default uses RAW NPZ side metadata")
    side_group.add_argument("--native-replay", type=Path, help="Native JSONL: read only pair/frame IDs and routed picked_side; RAW NPZ must contain python_route")
    parser.add_argument("--native-index-base", type=int, choices=(0, 1), default=1, help="Native replay IDs = source frame_index + this offset (default 1)")
    parser.add_argument("--width", type=int, help="Assert source width when absent from RAW NPZ metadata")
    parser.add_argument("--height", type=int, help="Assert source height when absent from RAW NPZ metadata")
    parser.add_argument("--coordinate-context", choices=(CONTEXT,), help="Assert coordinate space when absent from RAW NPZ metadata")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing correction outputs")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({key: summary[key] for key in ("frames", "applied_frames", "status_counts")}, indent=2))


if __name__ == "__main__":
    main()
