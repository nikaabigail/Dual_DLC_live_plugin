"""
Single-camera DLCLive runtime that feeds the DualDLCLiveBridge Open Ephys plugin.

The plugin packet format remains the dual DDLP/v1 binary packet. A single side
camera sees only one hind leg at a time, so by default (SINGLE_AUTO_PICK_SIDE)
the pose is routed to whichever side's triplet is actually present: the plugin
reports the visible leg only — L when the left flank shows, R when the rat turns
and the right flank shows. Set SINGLE_AUTO_PICK_SIDE=False to pin
SINGLE_PLUGIN_SIDE, or SINGLE_EMIT_BOTH_LEGS=True only for a camera that
genuinely sees both legs at once (top/rear view).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config_dual_rt_dlc_live as config
import dual_rt_dlc_live as dual
import live_profiles
import live_recorder
import rt_dlc_live as live


live.config = config

_OVERLAY_EMA_ALPHA = 0.12


def _ema(prev: Optional[float], value: float, alpha: float = _OVERLAY_EMA_ALPHA) -> float:
    return value if prev is None else (alpha * value + (1.0 - alpha) * prev)


class NullSource:
    dropped_total = 0


class LegRoiTracker:
    """Fixed-width sliding ROI that follows the hind legs (single camera).

    Drives DLCLive's ``cropping`` so inference runs on a small, constant-size
    crop instead of the full stripe. The window centre is the mean X of whatever
    hind-leg points are currently visible (>= ``detect_thresh``), so it is robust
    to a 1-2 point dropout, and EMA-smoothed so it does not jitter. On loss the
    window HOLDS at the last position for ``hold_frames`` frames (~1 s) -- riding
    out turns/occlusions where legs vanish then reappear near the same X -- and
    only then SWEEPS the fixed-size window across the frame to re-acquire. The
    width is always constant (never the full frame) so cudagraphs/torch.compile
    stay valid.
    Coordinates returned by DLCLive are already restored to the full
    frame (``postprocess_pose_without_processor`` adds the crop offset back), so
    everything downstream (pose_result, side auto-pick, plugin) is unchanged.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        leg_indices: list[int],
        width: int,
        detect_thresh: float,
        hold_frames: int,
        center_ema: float,
    ) -> None:
        self.fw = int(frame_w)
        self.fh = int(frame_h)
        self.leg_idx = list(leg_indices)
        self.width = max(16, min(int(width), self.fw))
        self.thresh = float(detect_thresh)
        self.hold_frames = int(hold_frames)
        self.ema = float(center_ema)
        # Start (and re-acquire) CENTRED, never full-frame. The crop is always a
        # FIXED width, so the model input shape never changes and cudagraphs /
        # torch.compile stay valid. A full-frame fallback would flip the shape
        # 256<->1920 and force a revert to slow eager inference (~2x slower).
        self.cx: Optional[float] = self.fw / 2.0
        self.misses = 0

    def window(self) -> Optional[list[int]]:
        """ROI [x1, x2, y1, y2] for the next frame (fixed width, always engaged)."""
        if self.cx is None:
            return None
        half = self.width // 2
        x1 = int(round(self.cx)) - half
        x1 = max(0, min(x1, self.fw - self.width))
        return [x1, x1 + self.width, 0, self.fh]

    def update(self, pose: np.ndarray) -> None:
        """Update the window centre from a full-frame pose."""
        if not self.leg_idx:
            return
        pts = np.asarray(pose)[self.leg_idx]
        visible = pts[pts[:, 2] >= self.thresh]
        if len(visible) >= 1:
            cx_new = float(np.mean(visible[:, 0]))
            # Snap straight onto the legs when (re)acquiring (cx unset, or we were
            # mid-sweep / just lost); EMA-smooth only during continuous tracking.
            # Otherwise coming out of a sweep the window would lag behind the
            # animal and immediately lose it again.
            if self.cx is None or self.misses > 0:
                self.cx = cx_new
            else:
                self.cx = self.ema * cx_new + (1.0 - self.ema) * self.cx
            self.misses = 0
        else:
            self.misses += 1
            if self.misses <= self.hold_frames:
                # HOLD at the last leg position (~1 s). A turn or side-switch makes
                # the legs vanish for a moment then reappear near the SAME X, so
                # staying put re-locks instantly instead of thrashing. cx is left
                # untouched -> the window does not move during the hold.
                pass
            else:
                # Only after a sustained loss -> SWEEP the fixed-size window across
                # the frame to re-acquire (the animal may have moved elsewhere).
                # One window per frame (wrap at the right edge) covers the stripe in
                # ~fw/width frames. Width stays constant -> cudagraphs stays valid
                # (never a full-frame scan); misses is NOT reset so it keeps sweeping.
                base = self.cx if self.cx is not None else self.width / 2.0  # 0.0 is a valid centre
                self.cx = base + self.width
                if self.cx > self.fw:
                    self.cx = self.width / 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = live_profiles.build_profile_parser("single", "single-best")
    parser.description = "Run one Daheng camera through DLCLive and send pose to Open Ephys."
    parser.add_argument("--camera", choices=["left", "right"], help="Camera config name from DUAL_CAMERAS.")
    parser.add_argument(
        "--plugin-side",
        choices=["left", "right"],
        help="Which plugin side receives this camera pose. Defaults to the profile setting.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means run forever.")
    return parser


def apply_args(args: argparse.Namespace) -> live_profiles.LiveProfile:
    if args.list_profiles:
        live_profiles.print_profile_table("single")
        raise SystemExit(0)
    profile = live_profiles.apply_profile(config, args.profile)
    display_value = live_profiles.display_value_for_profile(profile, args.display, args.no_display)
    live_profiles.apply_display_value(config, profile.target, display_value)
    if args.camera:
        config.SINGLE_CAMERA_NAME = args.camera
    if args.plugin_side:
        config.SINGLE_PLUGIN_SIDE = args.plugin_side
    print(live_profiles.profile_banner(profile, display_override=display_value), flush=True)
    print(
        f"Single camera: camera={config.SINGLE_CAMERA_NAME} -> plugin_side={config.SINGLE_PLUGIN_SIDE} "
        f"display={bool(getattr(config, 'SINGLE_DISPLAY_WINDOW', False))}",
        flush=True,
    )
    return profile


def find_camera_config(camera_name: str) -> dict[str, object]:
    for camera in config.DUAL_CAMERAS:
        if str(camera.get("name", "")).strip().lower() == camera_name.lower():
            return camera
    names = ", ".join(str(camera.get("name", "")) for camera in config.DUAL_CAMERAS)
    raise ValueError(f"Unknown camera {camera_name!r}. Available cameras: {names}")


def make_empty_result() -> dict[str, object]:
    return {
        "infer_ms": 0.0,
        "preprocess_ms": 0.0,
        "model_infer_ms": 0.0,
        "raw_pose_array": dual.empty_pose_array(),
        "raw_visible": 0,
        "filtered_visible": 0,
        "has_triplet": False,
        "hind_angle": None,
        "picked_side": None,
    }


def make_dummy_packet(reference: live.FramePacket) -> live.FramePacket:
    return live.FramePacket(
        frame_id=int(reference.frame_id),
        frame=np.empty((0, 0, 3), dtype=np.uint8),
        capture_ts=float(reference.capture_ts),
        source_frame_id=None,
        source_timestamp=None,
    )


def make_dummy_runtime(name: str) -> dual.CameraRuntime:
    return dual.CameraRuntime(
        name=name,
        sn="inactive",
        config_path=Path("inactive"),
        source=NullSource(),  # type: ignore[arg-type]
        native_config={},
    )


def _triplet_min_confidence(pose_result: dict[str, object], side: str) -> float:
    """Lowest likelihood among a side's hind-leg triplet (missing point -> 0).

    The plugin only reports a side when all three of its points are present, so
    the side with the higher minimum is the one whose triplet is most complete,
    i.e. the leg this single camera is actually looking at.
    """
    arr = pose_result.get("raw_pose_array")
    names = list(config.DUAL_USE_POINTS)
    points = [p for p in config.DUAL_SIDE_POINT_SETS.get(side, ()) if p in names]
    if not isinstance(arr, np.ndarray) or arr.shape[0] != len(names) or not points:
        return -1.0
    values = []
    for name in points:
        value = float(arr[names.index(name), 2])
        values.append(value if np.isfinite(value) else 0.0)
    return min(values) if values else -1.0


def make_pair_result(
    frame_index: int,
    packet: live.FramePacket,
    pose_result: dict[str, object],
    plugin_side: str,
) -> dual.PairInferenceResult:
    # A single side camera sees only ONE hind leg at a time (the rat's near
    # flank). The model still emits both hl_*_l and hl_*_r, but the occluded leg
    # is a low-confidence guess, so we must NOT report both. By default we auto-
    # pick the side whose triplet is actually present (the visible leg) and route
    # the pose only there: the plugin reports exactly one leg — L when the left
    # flank shows, R when the rat turns and the right flank shows. The opposite
    # side stays empty so its TTL line stays low.
    if bool(getattr(config, "SINGLE_EMIT_BOTH_LEGS", False)):
        # Opt-in: a camera that truly sees both legs (top/rear view). Reports
        # L and R at once from the same pose.
        return dual.PairInferenceResult(
            pair_index=frame_index,
            left_packet=packet,
            right_packet=packet,
            host_dt_ms=0.0,
            camera_dt_ms=None,
            left_result=pose_result,
            right_result=pose_result,
        )

    target_side = plugin_side
    if bool(getattr(config, "SINGLE_AUTO_PICK_SIDE", True)):
        target_side = (
            "right"
            if _triplet_min_confidence(pose_result, "right")
            > _triplet_min_confidence(pose_result, "left")
            else "left"
        )

    dummy_packet = make_dummy_packet(packet)
    empty_result = make_empty_result()
    if target_side == "left":
        return dual.PairInferenceResult(
            pair_index=frame_index,
            left_packet=packet,
            right_packet=dummy_packet,
            host_dt_ms=0.0,
            camera_dt_ms=None,
            left_result=pose_result,
            right_result=empty_result,
        )
    return dual.PairInferenceResult(
        pair_index=frame_index,
        left_packet=dummy_packet,
        right_packet=packet,
        host_dt_ms=0.0,
        camera_dt_ms=None,
        left_result=empty_result,
        right_result=pose_result,
    )


def log_cuda_status(dlc_live, logger: logging.Logger) -> None:
    try:
        import torch

        logger.info(
            "CUDA_CHECK torch=%s cuda=%s gpu=%s",
            torch.__version__,
            torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        )
        model = getattr(dlc_live, "model", None)
        if model is not None:
            first_param = next(model.parameters(), None)
            logger.info("DLC_MODEL_DEVICE=%s", first_param.device if first_param is not None else "no_params")
        else:
            logger.info("DLC_MODEL_DEVICE: dlc_live.model is None before init")
    except Exception as exc:
        logger.warning("CUDA/DLC device check failed: %s", exc)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    apply_args(args)

    logger = dual.setup_dual_logger()
    dual.configure_runtime_backends(logger)
    bridge = dual.OpenEphysBridge(logger)

    camera_cfg = find_camera_config(str(getattr(config, "SINGLE_CAMERA_NAME", "left")))
    runtime = dual.build_camera_runtime(camera_cfg)
    inactive_name = "right" if str(getattr(config, "SINGLE_PLUGIN_SIDE", "left")) == "left" else "left"
    inactive_runtime = make_dummy_runtime(inactive_name)
    left_runtime = runtime if config.SINGLE_PLUGIN_SIDE == "left" else inactive_runtime
    right_runtime = runtime if config.SINGLE_PLUGIN_SIDE == "right" else inactive_runtime

    source = runtime.source
    dlc_live = None
    recorder = None
    profiler = dual.StageProfiler(
        enabled=bool(getattr(config, "DUAL_ENABLE_STAGE_PROFILER", True)),
        alpha=float(getattr(config, "DUAL_PROFILE_EMA_ALPHA", 0.10)),
    )

    stats: dict[str, float] = {
        "frames": 0.0,
        "raw_visible_sum": 0.0,
        "infer_ms_sum": 0.0,
    }
    prev_capture_ts: Optional[float] = None
    prev_infer_end: Optional[float] = None
    total_start = time.perf_counter()
    profile_start = total_start
    profile_last_frame = 0
    visual_mode = bool(getattr(config, "SINGLE_DISPLAY_WINDOW", False))
    initialized = False

    try:
        bridge.open()
        source.open()
        logger.info(
            "Opened single camera name=%s sn=%s config=%s native=%s fps=%.1f plugin_side=%s",
            runtime.name,
            runtime.sn,
            runtime.config_path,
            runtime.native_config,
            source.nominal_fps(),
            getattr(config, "SINGLE_PLUGIN_SIDE", "left"),
        )

        read_start = time.perf_counter()
        ok, packet = source.read()
        profiler.observe("camera/read", (time.perf_counter() - read_start) * 1000.0)
        if not ok or packet is None:
            raise RuntimeError("Could not read the first frame from the selected camera.")

        effective_cropping = live.normalize_cropping_for_frame(
            getattr(config, "CROPPING", None),
            packet.frame,
            logger,
        )
        dlc_live = live.build_dlc_live(effective_cropping)
        log_cuda_status(dlc_live, logger)
        model_cfg = dlc_live.read_config()
        body_parts = live.extract_bodyparts(model_cfg)
        missing_points = [name for name in config.DUAL_USE_POINTS if name not in body_parts]
        if missing_points:
            raise ValueError(f"DUAL_USE_POINTS are missing from the exported model: {missing_points}")
        bodypart_to_idx = {name: i for i, name in enumerate(body_parts)}
        logger.info(
            "Single bridge started. profile=%s camera=%s plugin_side=%s crop=%s precision=%s convert2rgb=%s",
            args.profile,
            runtime.name,
            getattr(config, "SINGLE_PLUGIN_SIDE", "left"),
            effective_cropping,
            getattr(config, "PRECISION", "FP32"),
            getattr(config, "CONVERT_TO_RGB", True),
        )
        logger.info("Model bodyparts loaded: %d; bridge points=%s", len(body_parts), config.DUAL_USE_POINTS)

        base_cropping = effective_cropping
        roi_tracker: Optional[LegRoiTracker] = None
        if bool(getattr(config, "LEG_ROI_ENABLED", False)):
            if base_cropping is not None:
                logger.warning("LEG_ROI_ENABLED ignores a non-None base CROPPING; window uses camera-frame coords.")
            leg_indices = [i for i, name in enumerate(body_parts) if str(name).startswith("hl_")]
            frame_h, frame_w = packet.frame.shape[0], packet.frame.shape[1]
            roi_tracker = LegRoiTracker(
                frame_w=frame_w,
                frame_h=frame_h,
                leg_indices=leg_indices,
                width=int(getattr(config, "LEG_ROI_WIDTH", 256)),
                detect_thresh=float(getattr(config, "LEG_ROI_DETECT_THRESH", 0.30)),
                hold_frames=int(getattr(config, "LEG_ROI_HOLD_FRAMES", 100)),
                center_ema=float(getattr(config, "LEG_ROI_CENTER_EMA", 0.35)),
            )
            logger.info(
                "Leg ROI enabled: width=%d/%d thresh=%.2f hold_frames=%d ema=%.2f hind_points=%d",
                roi_tracker.width, frame_w, roi_tracker.thresh, roi_tracker.hold_frames, roi_tracker.ema, len(leg_indices),
            )

        recorder: Optional[live_recorder.ParallelRecorder] = None
        if bool(getattr(config, "SINGLE_RECORD_ENABLED", False)):
            recorder = live_recorder.ParallelRecorder(
                out_dir=getattr(config, "SINGLE_RECORD_DIR", Path("recordings")),
                stem="single_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                bodyparts=body_parts,
                fps=source.nominal_fps(),
                frame_is_rgb=str(getattr(config, "GALAXY_OUTPUT_COLOR", "bgr")).strip().lower() == "rgb",
                record_video=bool(getattr(config, "SINGLE_RECORD_VIDEO", True)),
                video_codec=str(getattr(config, "SINGLE_RECORD_VIDEO_CODEC", "mp4v")),
                record_keypoints=bool(getattr(config, "SINGLE_RECORD_KEYPOINTS", True)),
                kp_format=str(getattr(config, "SINGLE_KP_FORMAT", "binary")),
                queue_size=int(getattr(config, "SINGLE_RECORD_QUEUE", 128)),
                logger=logger,
            )
            recorder.start()

        while packet is not None:
            frame_index = int(stats["frames"]) + 1
            stats["frames"] = float(frame_index)
            if prev_capture_ts is not None:
                dt_cam = float(packet.capture_ts) - prev_capture_ts
                if dt_cam > 0:
                    stats["cam_fps"] = 1.0 / dt_cam
            prev_capture_ts = float(packet.capture_ts)

            if roi_tracker is not None:
                window = roi_tracker.window()
                dlc_live.cropping = window if window is not None else base_cropping
            infer_start = time.perf_counter()
            initialized, pose, preprocess_ms, model_infer_ms = dual.run_raw_inference(dlc_live, initialized, packet)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            if frame_index > 1:  # skip the one-time init/compile spike in the overlay EMA
                stats["infer_ms_ema"] = _ema(stats.get("infer_ms_ema"), infer_ms)
            if roi_tracker is not None:
                roi_tracker.update(pose)
            infer_end = time.perf_counter()
            profiler.observe("preprocess", preprocess_ms)
            profiler.observe("inference", model_infer_ms)
            if prev_infer_end is not None:
                dt_inf = infer_end - prev_infer_end
                if dt_inf > 0:
                    stats["infer_fps"] = 1.0 / dt_inf
                    stats["result_hz_ema"] = _ema(stats.get("result_hz_ema"), stats["infer_fps"])
            prev_infer_end = infer_end
            stats["infer_ms_sum"] += infer_ms

            pose_result = dual.raw_pose_result(
                runtime,
                np.asarray(pose),
                bodypart_to_idx,
                infer_ms,
                preprocess_ms=preprocess_ms,
                model_infer_ms=model_infer_ms,
            )
            stats["raw_visible_sum"] += float(pose_result["raw_visible"])
            pair_result = make_pair_result(
                frame_index,
                packet,
                pose_result,
                str(getattr(config, "SINGLE_PLUGIN_SIDE", "left")),
            )

            pack_start = time.perf_counter()
            bridge.send(pair_result, left_runtime, right_runtime)
            profiler.observe("pack/send", (time.perf_counter() - pack_start) * 1000.0)

            if recorder is not None:
                # raw frame + full-frame pose (+ the ROI window the model saw) ->
                # background thread; hot loop only copies + enqueues.
                recorder.submit(
                    packet.frame, frame_index, float(packet.capture_ts),
                    pose, window if roi_tracker is not None else None,
                )

            display_start = time.perf_counter()
            if visual_mode:
                display = dual.frame_for_opencv_display(packet.frame).copy()
                metrics = {
                    # throughput (results/s) and latency (ms), both EMA-smoothed
                    "result_hz": stats.get("result_hz_ema", 0.0),
                    "infer_ms_ema": stats.get("infer_ms_ema", infer_ms),
                    "infer_ms": infer_ms,  # raw fallback
                    "raw_visible": int(pose_result["raw_visible"]),
                    "tracked_points": len(config.DUAL_USE_POINTS),
                    "source_drops": getattr(source, "dropped_total", 0),
                }
                display = live.draw_overlay(display, dual.draw_points_for_result(pose_result), metrics)
                if roi_tracker is not None:
                    if window is not None:
                        x1, x2, y1, y2 = window
                        cv2.rectangle(display, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (0, 255, 255), 2)
                        cv2.putText(
                            display, f"ROI {x2 - x1}px", (min(x1 + 5, max(0, display.shape[1] - 95)), y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            display, "ROI: full frame (re-acquiring)", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA,
                        )
                cv2.imshow(str(getattr(config, "SINGLE_WINDOW_NAME", "DLC Live single bridge")), display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("Exit requested from display window.")
                    break
            profiler.observe("display", (time.perf_counter() - display_start) * 1000.0)

            log_every = max(1, int(getattr(config, "SINGLE_LOG_EVERY_N_FRAMES", 120)))
            if frame_index % log_every == 0:
                logger.info(
                    "frame=%d camera=%s plugin_side=%s raw_visible=%d/%d p=%s "
                    "cam_fps=%.1f live_fps=%.1f infer=%.1fms drops=%d",
                    frame_index,
                    runtime.name,
                    getattr(config, "SINGLE_PLUGIN_SIDE", "left"),
                    int(pose_result["raw_visible"]),
                    len(config.DUAL_USE_POINTS),
                    dual.pose_likelihood_summary(pose_result),
                    stats.get("cam_fps", 0.0),
                    stats.get("infer_fps", 0.0),
                    infer_ms,
                    getattr(source, "dropped_total", 0),
                )

            profile_every = max(1, int(getattr(config, "SINGLE_PROFILE_LOG_EVERY_N_FRAMES", 120)))
            if frame_index % profile_every == 0:
                now = time.perf_counter()
                elapsed = max(1e-9, now - profile_start)
                total_elapsed = max(1e-9, now - total_start)
                result_delta = frame_index - profile_last_frame
                logger.info(
                    "stage_profile frame=%d result_hz=%.2f total_hz=%.2f %s",
                    frame_index,
                    result_delta / elapsed,
                    frame_index / total_elapsed,
                    profiler.format_snapshot(),
                )
                profile_start = now
                profile_last_frame = frame_index

            if int(args.max_frames) > 0 and frame_index >= int(args.max_frames):
                logger.info("Reached --max-frames=%d.", int(args.max_frames))
                break

            read_start = time.perf_counter()
            ok, packet = source.read()
            profiler.observe("camera/read", (time.perf_counter() - read_start) * 1000.0)
            if not ok:
                packet = None

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        try:
            source.release()
        finally:
            bridge.close()
        if recorder is not None:
            try:
                recorder.close()  # drain the queue + flush/close the files
            except Exception:
                pass
        if dlc_live is not None:
            try:
                dlc_live.close()
            except Exception:
                pass
        if visual_mode:
            cv2.destroyAllWindows()

        total = max(1.0, stats["frames"])
        logger.info(
            "Finished single bridge. frames=%d avg_infer=%.1fms avg_raw=%.2f drops=%d",
            int(stats["frames"]),
            stats["infer_ms_sum"] / total,
            stats["raw_visible_sum"] / total,
            int(getattr(source, "dropped_total", 0)),
        )


if __name__ == "__main__":
    main()
