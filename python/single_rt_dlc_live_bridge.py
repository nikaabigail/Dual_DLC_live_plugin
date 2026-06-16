"""
Single-camera DLCLive runtime that feeds the DualDLCLiveBridge Open Ephys plugin.

The plugin packet format remains the dual DDLP/v1 binary packet. The active
camera side is filled with real pose points; the inactive side is filled with
NaN points so the plugin keeps the corresponding TTL lines low.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config_dual_rt_dlc_live as config
import dual_rt_dlc_live as dual
import live_profiles
import rt_dlc_live as live


live.config = config


class NullSource:
    dropped_total = 0


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


def make_pair_result(
    frame_index: int,
    packet: live.FramePacket,
    pose_result: dict[str, object],
    plugin_side: str,
) -> dual.PairInferenceResult:
    dummy_packet = make_dummy_packet(packet)
    empty_result = make_empty_result()
    if plugin_side == "left":
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

        while packet is not None:
            frame_index = int(stats["frames"]) + 1
            stats["frames"] = float(frame_index)
            if prev_capture_ts is not None:
                dt_cam = float(packet.capture_ts) - prev_capture_ts
                if dt_cam > 0:
                    stats["cam_fps"] = 1.0 / dt_cam
            prev_capture_ts = float(packet.capture_ts)

            infer_start = time.perf_counter()
            initialized, pose, preprocess_ms, model_infer_ms = dual.run_raw_inference(dlc_live, initialized, packet)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            infer_end = time.perf_counter()
            profiler.observe("preprocess", preprocess_ms)
            profiler.observe("inference", model_infer_ms)
            if prev_infer_end is not None:
                dt_inf = infer_end - prev_infer_end
                if dt_inf > 0:
                    stats["infer_fps"] = 1.0 / dt_inf
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

            display_start = time.perf_counter()
            if visual_mode:
                display = dual.frame_for_opencv_display(packet.frame).copy()
                metrics = {
                    "cam_fps": stats.get("cam_fps", 0.0),
                    "infer_fps": stats.get("infer_fps", 0.0),
                    "infer_ms": infer_ms,
                    "raw_visible": int(pose_result["raw_visible"]),
                    "filtered_visible": int(pose_result["raw_visible"]),
                    "tracked_points": len(config.DUAL_USE_POINTS),
                    "triplet": False,
                    "hind_angle": None,
                    "source_drops": getattr(source, "dropped_total", 0),
                }
                display = live.draw_overlay(display, dual.draw_points_for_result(pose_result), metrics)
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
