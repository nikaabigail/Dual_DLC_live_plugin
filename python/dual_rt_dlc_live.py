"""
Dual DLCLive runtime for two Daheng Galaxy USB3 Vision cameras.

The two GalaxyView config files are imported by Galaxy SDK, so camera-side ROI
and acquisition settings are native camera parameters rather than OpenCV crops.
"""
from __future__ import annotations

import csv
import json
import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np

import config_dual_rt_dlc_live as config
import rt_dlc_live as live


live.config = config


BINARY_POSE_MAGIC = b"DDLP"
BINARY_POSE_VERSION = 1
BINARY_FLAG_ACK = 1 << 0
BINARY_POSE_POINT_NAMES = [
    "hl_ankle_l",
    "hl_ankle_r",
    "hl_hip_l",
    "hl_hip_r",
    "hl_toes_l",
    "hl_toes_r",
]
BINARY_HEADER_STRUCT = struct.Struct("<4sHHqdffHH")
BINARY_SIDE_STRUCT = struct.Struct("<qqdfIHH")
BINARY_POINT_STRUCT = struct.Struct("<fff")


@dataclass
class CameraRuntime:
    name: str
    sn: str
    config_path: Path
    source: live.GalaxyCameraSource
    native_config: dict[str, float | int | str]
    processor: Optional[live.OnlinePoseProcessor] = None
    lock: Lock = field(default_factory=Lock)
    latest_packet: Optional[live.FramePacket] = None
    latest_seq: int = 0
    prev_capture_ts: Optional[float] = None
    fps_cam: float = 0.0
    fps_dlc: float = 0.0
    prev_infer_end_perf: Optional[float] = None
    last_error: Optional[BaseException] = None
    warned_shape: bool = False


@dataclass
class PairInferenceResult:
    pair_index: int
    left_packet: live.FramePacket
    right_packet: live.FramePacket
    host_dt_ms: float
    camera_dt_ms: Optional[float]
    left_result: dict[str, object]
    right_result: dict[str, object]


@dataclass
class InferenceState:
    lock: Lock = field(default_factory=Lock)
    latest_result: Optional[PairInferenceResult] = None
    latest_seq: int = 0
    processed_pairs: int = 0
    skipped_pairs: int = 0
    last_error: Optional[BaseException] = None


class OpenEphysBridge:
    """Sends dual-DLCLive packets to an Open Ephys bridge plugin."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.enabled = bool(getattr(config, "DUAL_OE_BRIDGE_ENABLED", False))
        self.host = str(getattr(config, "DUAL_OE_BRIDGE_HOST", "127.0.0.1"))
        self.port = int(getattr(config, "DUAL_OE_BRIDGE_PORT", 47000))
        self.send_every = max(1, int(getattr(config, "DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS", 1)))
        self.packet_mode = str(getattr(config, "DUAL_OE_BRIDGE_PACKET_MODE", "pose")).strip().lower()
        self.wire_format = str(getattr(config, "DUAL_OE_BRIDGE_WIRE_FORMAT", "json")).strip().lower()
        self.request_ack = bool(getattr(config, "DUAL_OE_BRIDGE_REQUEST_ACK", False))
        threshold = getattr(config, "DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG", None)
        self.angle_threshold_deg = None if threshold is None else float(threshold)
        self.sock: Optional[socket.socket] = None

    def open(self) -> None:
        if not self.enabled:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.logger.info(
            "Open Ephys bridge enabled: UDP %s:%d mode=%s wire=%s",
            self.host,
            self.port,
            self.packet_mode,
            self.wire_format if self.packet_mode == "pose" else "json",
        )

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def send(self, result: PairInferenceResult, left: CameraRuntime, right: CameraRuntime) -> None:
        if self.sock is None or result.pair_index % self.send_every != 0:
            return

        try:
            if self.packet_mode == "pose" and self.wire_format == "binary":
                data = self._build_binary_pose_payload(result, left, right)
            else:
                payload = self._build_payload(result, left, right)
                data = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            self.sock.sendto(data, (self.host, self.port))
        except Exception as exc:
            self.logger.warning("Open Ephys bridge send failed: %s", exc)

    def _side_payload(
        self,
        runtime: CameraRuntime,
        packet: live.FramePacket,
        result: dict[str, object],
    ) -> dict[str, object]:
        return {
            "name": runtime.name,
            "frame_id": int(packet.frame_id),
            "source_frame_id": packet.source_frame_id,
            "capture_ts": float(packet.capture_ts),
            "infer_ms": float(result["infer_ms"]),
            "drops": int(runtime.source.dropped_total),
            "raw_visible": int(result["raw_visible"]),
            "raw_points": result["raw_points"],
        }

    def _angle_trigger(self, result: dict[str, object]) -> bool:
        if self.angle_threshold_deg is None or not bool(result["has_triplet"]):
            return False
        angle = result["hind_angle"]
        return angle is not None and float(angle) <= self.angle_threshold_deg

    def _build_payload(
        self,
        result: PairInferenceResult,
        left: CameraRuntime,
        right: CameraRuntime,
    ) -> dict[str, object]:
        left_state = self._side_payload(left, result.left_packet, result.left_result)
        right_state = self._side_payload(right, result.right_packet, result.right_result)

        if self.packet_mode == "ttl":
            return self._build_ttl_payload(result, left_state, right_state)

        payload: dict[str, object] = {
            "schema": "dual_dlc_live.pose.v1",
            "pair_index": int(result.pair_index),
            "host_time": time.time(),
            "host_dt_ms": float(result.host_dt_ms),
            "camera_dt_ms": result.camera_dt_ms,
            "tracked_points": list(config.DUAL_USE_POINTS),
            "side_point_sets": {
                str(side): list(points)
                for side, points in getattr(config, "DUAL_SIDE_POINT_SETS", {}).items()
            },
            "left": left_state,
            "right": right_state,
        }
        if self.request_ack:
            payload["ack"] = True
        return payload

    def _build_binary_pose_payload(
        self,
        result: PairInferenceResult,
        left: CameraRuntime,
        right: CameraRuntime,
    ) -> bytes:
        flags = BINARY_FLAG_ACK if self.request_ack else 0
        camera_dt_ms = float("nan") if result.camera_dt_ms is None else float(result.camera_dt_ms)
        packet = bytearray()
        packet.extend(
            BINARY_HEADER_STRUCT.pack(
                BINARY_POSE_MAGIC,
                BINARY_POSE_VERSION,
                flags,
                int(result.pair_index),
                time.time(),
                float(result.host_dt_ms),
                camera_dt_ms,
                len(config.DUAL_USE_POINTS),
                0,
            )
        )
        self._append_binary_side(packet, left, result.left_packet, result.left_result)
        self._append_binary_side(packet, right, result.right_packet, result.right_result)
        return bytes(packet)

    def _append_binary_side(
        self,
        packet: bytearray,
        runtime: CameraRuntime,
        frame_packet: live.FramePacket,
        inference_result: dict[str, object],
    ) -> None:
        source_frame_id = -1 if frame_packet.source_frame_id is None else int(frame_packet.source_frame_id)
        drops = max(0, min(int(runtime.source.dropped_total), 0xFFFFFFFF))
        raw_visible = max(0, min(int(inference_result["raw_visible"]), 0xFFFF))
        packet.extend(
            BINARY_SIDE_STRUCT.pack(
                int(frame_packet.frame_id),
                source_frame_id,
                float(frame_packet.capture_ts),
                float(inference_result["infer_ms"]),
                drops,
                raw_visible,
                0,
            )
        )

        raw_points = inference_result.get("raw_points", {})
        if not isinstance(raw_points, dict):
            raw_points = {}
        for name in config.DUAL_USE_POINTS:
            point = raw_points.get(name, {})
            if not isinstance(point, dict):
                point = {}
            packet.extend(
                BINARY_POINT_STRUCT.pack(
                    self._finite_or_nan(point.get("x")),
                    self._finite_or_nan(point.get("y")),
                    self._finite_or_nan(point.get("likelihood")),
                )
            )

    @staticmethod
    def _finite_or_nan(value: object) -> float:
        if value is None:
            return float("nan")
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return result if np.isfinite(result) else float("nan")

    def _build_ttl_payload(
        self,
        result: PairInferenceResult,
        left_state: dict[str, object],
        right_state: dict[str, object],
    ) -> dict[str, object]:
        ttl_lines = [False] * 8
        ttl_lines[0] = bool(result.left_result["has_triplet"])
        ttl_lines[1] = bool(result.right_result["has_triplet"])
        ttl_lines[2] = self._angle_trigger(result.left_result)
        ttl_lines[3] = self._angle_trigger(result.right_result)

        payload: dict[str, object] = {
            "schema": "dual_dlc_live.v1",
            "pair_index": int(result.pair_index),
            "host_time": time.time(),
            "host_dt_ms": float(result.host_dt_ms),
            "camera_dt_ms": result.camera_dt_ms,
            "left": left_state,
            "right": right_state,
            "ttl_lines": ttl_lines,
        }
        if self.request_ack:
            payload["ack"] = True
        return payload


def setup_dual_logger() -> logging.Logger:
    logger = logging.getLogger("dual_rt_dlc_live")
    if logger.handlers:
        return logger

    level_name = str(getattr(config, "LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(config.LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_galaxy_config(path: Path) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {}
    keys = {
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
        "ExposureTime",
        "AcquisitionFrameRate",
        "TriggerMode",
        "TriggerSource",
        "PixelFormat",
    }
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("<"):
                continue
            parts = line.split()
            if len(parts) < 2 or parts[0] not in keys:
                continue
            key, value = parts[0], parts[1]
            if key in {"PixelFormat", "TriggerMode", "TriggerSource"}:
                result[key] = value
                continue
            try:
                number = float(value)
            except ValueError:
                result[key] = value
                continue
            result[key] = int(number) if number.is_integer() else number
    return result


def validate_dual_config() -> None:
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    if not config.DUAL_USE_POINTS:
        raise ValueError("DUAL_USE_POINTS must not be empty.")
    if float(getattr(config, "RESIZE", 1.0)) <= 0:
        raise ValueError("RESIZE must be positive.")
    if not (0.0 <= float(config.CONF_THRESH_USE) <= 1.0):
        raise ValueError("CONF_THRESH_USE must be in [0, 1].")
    if not (0.0 <= float(config.CONF_THRESH_DRAW) <= 1.0):
        raise ValueError("CONF_THRESH_DRAW must be in [0, 1].")
    if float(config.DESPIKE_THRESHOLD_PX) <= 0:
        raise ValueError("DESPIKE_THRESHOLD_PX must be positive.")
    if int(config.MAX_HOLD_FRAMES) < 0:
        raise ValueError("MAX_HOLD_FRAMES must be >= 0.")
    if int(config.MEDIAN_WINDOW) <= 0:
        raise ValueError("MEDIAN_WINDOW must be positive.")

    cameras = getattr(config, "DUAL_CAMERAS", [])
    if len(cameras) != 2:
        raise ValueError("DUAL_CAMERAS must contain exactly two camera definitions.")

    seen_names: set[str] = set()
    seen_sns: set[str] = set()
    for camera in cameras:
        name = str(camera.get("name", "")).strip()
        sn = str(camera.get("sn", "")).strip()
        path = Path(camera.get("config_path", ""))
        if not name:
            raise ValueError("Each DUAL_CAMERAS entry needs a non-empty name.")
        if not sn:
            raise ValueError(f"DUAL_CAMERAS[{name}] needs a non-empty sn.")
        if name in seen_names:
            raise ValueError(f"Duplicate camera name: {name}")
        if sn in seen_sns:
            raise ValueError(f"Duplicate camera serial number: {sn}")
        if not path.exists():
            raise FileNotFoundError(f"Galaxy config does not exist for {name}: {path}")
        seen_names.add(name)
        seen_sns.add(sn)

    if int(getattr(config, "DUAL_PROCESS_EVERY_N_PAIRS", 1)) <= 0:
        raise ValueError("DUAL_PROCESS_EVERY_N_PAIRS must be positive.")
    if int(getattr(config, "DUAL_PAIR_WAIT_TIMEOUT_MS", 2000)) <= 0:
        raise ValueError("DUAL_PAIR_WAIT_TIMEOUT_MS must be positive.")
    if int(getattr(config, "DUAL_ACQUISITION_BUFFER_COUNT", 2)) < 0:
        raise ValueError("DUAL_ACQUISITION_BUFFER_COUNT must be >= 0.")
    if int(getattr(config, "DUAL_MAX_DRAIN_FRAMES", 20)) < 0:
        raise ValueError("DUAL_MAX_DRAIN_FRAMES must be >= 0.")
    if int(getattr(config, "DUAL_OE_BRIDGE_PORT", 47000)) <= 0:
        raise ValueError("DUAL_OE_BRIDGE_PORT must be positive.")
    if int(getattr(config, "DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS", 1)) <= 0:
        raise ValueError("DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS must be positive.")
    bridge_mode = str(getattr(config, "DUAL_OE_BRIDGE_PACKET_MODE", "pose")).strip().lower()
    if bridge_mode not in {"pose", "ttl"}:
        raise ValueError('DUAL_OE_BRIDGE_PACKET_MODE must be "pose" or "ttl".')
    bridge_wire = str(getattr(config, "DUAL_OE_BRIDGE_WIRE_FORMAT", "json")).strip().lower()
    if bridge_wire not in {"binary", "json"}:
        raise ValueError('DUAL_OE_BRIDGE_WIRE_FORMAT must be "binary" or "json".')
    if bridge_wire == "binary" and list(config.DUAL_USE_POINTS) != BINARY_POSE_POINT_NAMES:
        raise ValueError(
            "Binary Open Ephys bridge packets use a fixed point order. "
            "Set DUAL_OE_BRIDGE_WIRE_FORMAT='json' for custom DUAL_USE_POINTS."
        )


def build_camera_runtime(camera_cfg: dict[str, object]) -> CameraRuntime:
    name = str(camera_cfg["name"])
    sn = str(camera_cfg["sn"])
    config_path = Path(camera_cfg["config_path"])
    source = live.GalaxyCameraSource(
        sdk_root=Path(config.GALAXY_SDK_ROOT),
        serial_number=sn,
        device_index=1,
        config_path=config_path,
        import_config=bool(getattr(config, "DUAL_IMPORT_CONFIG", True)),
        config_verify=bool(getattr(config, "DUAL_CONFIG_VERIFY", False)),
        fallback_apply_config=bool(getattr(config, "DUAL_FALLBACK_APPLY_CONFIG", True)),
        frame_timeout_ms=int(getattr(config, "DUAL_FRAME_TIMEOUT_MS", 1000)),
        force_trigger_off=bool(getattr(config, "DUAL_FORCE_TRIGGER_OFF", False)),
        low_latency=bool(getattr(config, "DUAL_LOW_LATENCY", True)),
        stream_buffer_handling_mode=str(getattr(config, "DUAL_STREAM_BUFFER_HANDLING_MODE", "NEWEST_ONLY")),
        acquisition_buffer_count=int(getattr(config, "DUAL_ACQUISITION_BUFFER_COUNT", 2)),
        drain_queued_frames=bool(getattr(config, "DUAL_DRAIN_QUEUED_FRAMES", True)),
        max_drain_frames=int(getattr(config, "DUAL_MAX_DRAIN_FRAMES", 20)),
    )
    return CameraRuntime(
        name=name,
        sn=sn,
        config_path=config_path,
        source=source,
        native_config=parse_galaxy_config(config_path),
    )


def reader_loop(runtime: CameraRuntime, stop_event: Event) -> None:
    sleep_s = max(0.0, float(getattr(config, "DUAL_READER_SLEEP_MS", 1.0)) / 1000.0)
    try:
        while not stop_event.is_set():
            ok, packet = runtime.source.read()
            if not ok or packet is None:
                time.sleep(sleep_s)
                continue

            if runtime.prev_capture_ts is not None:
                dt = packet.capture_ts - runtime.prev_capture_ts
                if dt > 0:
                    runtime.fps_cam = 1.0 / dt
            runtime.prev_capture_ts = packet.capture_ts

            with runtime.lock:
                runtime.latest_packet = packet
                runtime.latest_seq += 1
    except BaseException as exc:
        runtime.last_error = exc
        stop_event.set()


def get_latest(runtime: CameraRuntime) -> tuple[int, Optional[live.FramePacket]]:
    with runtime.lock:
        return runtime.latest_seq, runtime.latest_packet


def wait_for_initial_pair(
    left: CameraRuntime,
    right: CameraRuntime,
    stop_event: Event,
) -> tuple[live.FramePacket, live.FramePacket]:
    deadline = time.perf_counter() + int(getattr(config, "DUAL_PAIR_WAIT_TIMEOUT_MS", 2000)) / 1000.0
    while not stop_event.is_set() and time.perf_counter() < deadline:
        _, left_packet = get_latest(left)
        _, right_packet = get_latest(right)
        if left_packet is not None and right_packet is not None:
            return left_packet, right_packet
        time.sleep(0.002)
    raise RuntimeError("Timed out waiting for the first dual-camera frame pair.")


def score_triplet(
    points: dict[str, dict[str, float | None]],
    triplet: tuple[str, str, str],
) -> tuple[int, float]:
    count = 0
    likelihood_sum = 0.0
    for name in triplet:
        point = points.get(name, {})
        x, y, likelihood = point.get("x"), point.get("y"), point.get("likelihood")
        if x is not None and y is not None and likelihood is not None and likelihood >= config.CONF_THRESH_DRAW:
            count += 1
            likelihood_sum += float(likelihood)
    return count, likelihood_sum


def pick_triplet(
    camera_name: str,
    points: dict[str, dict[str, float | None]],
) -> tuple[str, tuple[str, str, str]]:
    side_sets = getattr(config, "DUAL_SIDE_POINT_SETS", {})
    if bool(getattr(config, "DUAL_AUTO_PICK_SIDE", True)):
        best_name = ""
        best_triplet: tuple[str, str, str] | None = None
        best_score = (-1, -1.0)
        for side_name, names in side_sets.items():
            if len(names) != 3:
                continue
            triplet = tuple(names)
            score = score_triplet(points, triplet)
            if score > best_score:
                best_name = side_name
                best_triplet = triplet
                best_score = score
        if best_triplet is not None:
            return best_name, best_triplet

    default_side = getattr(config, "DUAL_DEFAULT_CAMERA_SIDE", {}).get(camera_name, camera_name)
    triplet = tuple(side_sets.get(default_side, config.USE_POINTS[:3]))
    if len(triplet) != 3:
        raise ValueError(f"No valid triplet for camera {camera_name!r}.")
    return str(default_side), triplet


def normalize_triplet(
    points: dict[str, dict[str, float | None]],
    triplet: tuple[str, str, str],
) -> dict[str, dict[str, float | None]]:
    hip, ankle, toes = triplet
    empty = {"x": None, "y": None, "likelihood": None}
    return {
        "hip": dict(points.get(hip, empty)),
        "ankle": dict(points.get(ankle, empty)),
        "toes": dict(points.get(toes, empty)),
    }


def angle_from_canonical(points: dict[str, dict[str, float | None]]) -> Optional[float]:
    hip, ankle, toes = points["hip"], points["ankle"], points["toes"]
    coords = [hip.get("x"), hip.get("y"), ankle.get("x"), ankle.get("y"), toes.get("x"), toes.get("y")]
    if any(value is None for value in coords):
        return None
    return live.safe_angle_deg(
        (float(hip["x"]), float(hip["y"])),
        (float(ankle["x"]), float(ankle["y"])),
        (float(toes["x"]), float(toes["y"])),
    )


def packet_delta_ms(left: live.FramePacket, right: live.FramePacket) -> float:
    return (left.capture_ts - right.capture_ts) * 1000.0


def sdk_delta_ms(
    left_runtime: CameraRuntime,
    right_runtime: CameraRuntime,
    left_packet: live.FramePacket,
    right_packet: live.FramePacket,
) -> Optional[float]:
    left_ts = left_packet.source_timestamp
    right_ts = right_packet.source_timestamp
    left_freq = float(getattr(left_runtime.source, "timestamp_tick_frequency", 0.0) or 0.0)
    right_freq = float(getattr(right_runtime.source, "timestamp_tick_frequency", 0.0) or 0.0)
    if left_ts is None or right_ts is None or left_freq <= 0 or right_freq <= 0:
        return None
    return ((float(left_ts) / left_freq) - (float(right_ts) / right_freq)) * 1000.0


def validate_native_shape(runtime: CameraRuntime, packet: live.FramePacket, logger: logging.Logger) -> None:
    if runtime.warned_shape:
        return
    expected_w = runtime.native_config.get("Width")
    expected_h = runtime.native_config.get("Height")
    if not isinstance(expected_w, int) or not isinstance(expected_h, int):
        return
    h, w = packet.frame.shape[:2]
    if w != expected_w or h != expected_h:
        logger.warning(
            "camera=%s frame shape %dx%d does not match native config Width/Height %dx%d",
            runtime.name,
            w,
            h,
            expected_w,
            expected_h,
        )
    runtime.warned_shape = True


def bridge_packet_mode() -> str:
    return str(getattr(config, "DUAL_OE_BRIDGE_PACKET_MODE", "pose")).strip().lower()


def fast_pose_only_enabled() -> bool:
    return (
        bool(getattr(config, "DUAL_FAST_POSE_ONLY", True))
        and bool(getattr(config, "DUAL_OE_BRIDGE_ENABLED", False))
        and bridge_packet_mode() == "pose"
    )


def python_postprocess_enabled() -> bool:
    return not fast_pose_only_enabled()


def update_inference_fps(runtime: CameraRuntime, end_perf: float) -> None:
    if runtime.prev_infer_end_perf is not None:
        dt = end_perf - runtime.prev_infer_end_perf
        if dt > 0:
            runtime.fps_dlc = 1.0 / dt
    runtime.prev_infer_end_perf = end_perf


def raw_pose_result(
    runtime: CameraRuntime,
    pose: np.ndarray,
    bodypart_to_idx: dict[str, int],
    infer_ms: float,
) -> dict[str, object]:
    raw_points = live.pose_to_points(np.asarray(pose), bodypart_to_idx, config.DUAL_USE_POINTS)
    raw_visible = live.count_visible_points(raw_points)
    return {
        "infer_ms": infer_ms,
        "raw_points": raw_points,
        "raw_visible": raw_visible,
        "filtered_visible": raw_visible,
        "draw_points": raw_points,
        "has_triplet": False,
        "hind_angle": None,
        "picked_side": runtime.name,
        "reason": {},
        "python_postprocess": False,
    }


def processed_pose_result(
    runtime: CameraRuntime,
    pose: np.ndarray,
    bodypart_to_idx: dict[str, int],
    infer_ms: float,
) -> dict[str, object]:
    processor = runtime.processor
    raw_pose = processor.last_raw_pose if processor is not None and processor.last_raw_pose is not None else np.asarray(pose)
    filtered_pose = (
        processor.last_filtered_pose
        if processor is not None and processor.last_filtered_pose is not None
        else np.asarray(pose)
    )
    raw_points = live.pose_to_points(raw_pose, bodypart_to_idx, config.DUAL_USE_POINTS)
    filtered_points = live.pose_to_points(filtered_pose, bodypart_to_idx, config.DUAL_USE_POINTS)
    picked_side, picked_triplet = pick_triplet(runtime.name, filtered_points)
    canonical = normalize_triplet(filtered_points, picked_triplet)
    draw_points, has_triplet, reason = live.evaluate_triplet(canonical, ["hip", "ankle", "toes"])
    angle = angle_from_canonical(draw_points) if has_triplet else None

    return {
        "infer_ms": infer_ms,
        "raw_points": raw_points,
        "raw_visible": live.count_visible_points(raw_points),
        "filtered_visible": live.count_visible_points(filtered_points),
        "draw_points": draw_points,
        "has_triplet": has_triplet,
        "hind_angle": angle,
        "picked_side": picked_side,
        "reason": reason,
        "python_postprocess": True,
    }


def run_inference(
    dlc_live,
    initialized: bool,
    runtime: CameraRuntime,
    packet: live.FramePacket,
    bodypart_to_idx: dict[str, int],
) -> tuple[bool, dict[str, object]]:
    dlc_live.processor = runtime.processor
    start = time.perf_counter()
    if not initialized:
        pose = dlc_live.init_inference(
            packet.frame,
            frame_id=int(packet.frame_id),
            ure_ts=float(packet.capture_ts),
            stream_name=runtime.name,
        )

        try:
            model = getattr(dlc_live, "model", None)
            if model is not None:
                first_param = next(model.parameters(), None)
                print(
                    "DLC_MODEL_DEVICE_AFTER_INIT =",
                    first_param.device if first_param is not None else "no_params",
                )
            else:
                print("DLC_MODEL_DEVICE_AFTER_INIT = model_is_none")
        except Exception as exc:
            print("DLC_MODEL_DEVICE_CHECK_FAILED =", exc)

        initialized = True
    else:
        pose = dlc_live.get_pose(
            packet.frame,
            frame_id=int(packet.frame_id),
            capture_ts=float(packet.capture_ts),
            stream_name=runtime.name,
        )
    end = time.perf_counter()

    infer_ms = (end - start) * 1000.0
    update_inference_fps(runtime, end)

    if runtime.processor is None:
        return initialized, raw_pose_result(runtime, np.asarray(pose), bodypart_to_idx, infer_ms)
    return initialized, processed_pose_result(runtime, np.asarray(pose), bodypart_to_idx, infer_ms)


def batch_inference_supported(dlc_live) -> bool:
    if not fast_pose_only_enabled():
        return False
    if getattr(dlc_live, "display", None) is not None:
        return False
    if bool(getattr(dlc_live, "dynamic", (False, 0.5, 10))[0]):
        return False

    runner = getattr(dlc_live, "runner", None)
    if runner is None:
        return False
    if getattr(runner, "model", None) is None or getattr(runner, "pose_transform", None) is None:
        return False
    if getattr(runner, "detector", None) is not None or getattr(runner, "dynamic", None) is not None:
        return False
    if not bool(getattr(runner, "single_animal", True)):
        return False
    return True


def postprocess_pose_without_processor(dlc_live, pose: np.ndarray) -> np.ndarray:
    pose_arr = np.array(pose, copy=True)
    resize = getattr(dlc_live, "resize", None)
    if resize is not None:
        pose_arr[..., :2] *= 1.0 / float(resize)

    cropping = getattr(dlc_live, "cropping", None)
    if cropping is not None:
        pose_arr[..., 0] += cropping[0]
        pose_arr[..., 1] += cropping[2]

    dynamic_cropping = getattr(dlc_live, "dynamic_cropping", None)
    if dynamic_cropping is not None:
        pose_arr[..., 0] += dynamic_cropping[0]
        pose_arr[..., 1] += dynamic_cropping[2]
    return pose_arr


def run_batch_inference(
    dlc_live,
    left: CameraRuntime,
    right: CameraRuntime,
    left_packet: live.FramePacket,
    right_packet: live.FramePacket,
    bodypart_to_idx: dict[str, int],
) -> tuple[dict[str, object], dict[str, object]]:
    if not batch_inference_supported(dlc_live):
        raise RuntimeError("DLCLive runner does not support the fast dual-frame batch path.")

    import torch

    runner = dlc_live.runner
    dlc_live.processor = None
    if left_packet.frame.ndim >= 2 or right_packet.frame.ndim >= 2:
        dlc_live.convert2rgb = True

    start = time.perf_counter()
    processed_frames = [dlc_live.process_frame(left_packet.frame), dlc_live.process_frame(right_packet.frame)]
    if processed_frames[0].shape != processed_frames[1].shape:
        raise RuntimeError(f"Batch frames must have identical shapes, got {processed_frames[0].shape} and {processed_frames[1].shape}.")

    tensors = []
    for processed_frame in processed_frames:
        tensor = torch.from_numpy(processed_frame).permute(2, 0, 1)
        transformed = runner.pose_transform(tensor)
        if transformed.dim() != 3:
            raise RuntimeError(f"Expected 3D transformed pose tensor, got {tuple(transformed.shape)}.")
        tensors.append(transformed)

    with torch.inference_mode():
        model_input = torch.stack(tensors, dim=0).to(runner.device)
        if str(getattr(runner, "precision", "FP32")).upper() == "FP16":
            model_input = model_input.half()
        outputs = runner.model(model_input)
        batch_pose = runner.model.get_predictions(outputs)["bodypart"]["poses"]

    poses: list[np.ndarray] = []
    for index in range(2):
        pose = batch_pose[index]
        if len(pose) == 0:
            bodyparts, coords = pose.shape[-2:]
            pose = torch.zeros((bodyparts, coords), dtype=pose.dtype, device=pose.device)
        else:
            pose = pose[0]
        poses.append(postprocess_pose_without_processor(dlc_live, pose.detach().cpu().numpy()))

    if poses:
        dlc_live.pose = poses[-1]
    end = time.perf_counter()
    infer_ms_each = (end - start) * 500.0
    update_inference_fps(left, end)
    update_inference_fps(right, end)
    return (
        raw_pose_result(left, poses[0], bodypart_to_idx, infer_ms_each),
        raw_pose_result(right, poses[1], bodypart_to_idx, infer_ms_each),
    )


def inference_loop(
    stop_event: Event,
    dlc_live,
    left: CameraRuntime,
    right: CameraRuntime,
    bodypart_to_idx: dict[str, int],
    state: InferenceState,
    logger: Optional[logging.Logger] = None,
) -> None:
    initialized = False
    sdk_baseline_ms: Optional[float] = None
    last_pair_seq = (-1, -1)
    pair_index = 0
    process_every = max(1, int(getattr(config, "DUAL_PROCESS_EVERY_N_PAIRS", 1)))
    batch_disabled = False

    try:
        while not stop_event.is_set():
            left_seq, left_packet = get_latest(left)
            right_seq, right_packet = get_latest(right)
            if left_packet is None or right_packet is None:
                time.sleep(0.001)
                continue

            current_pair_seq = (left_seq, right_seq)
            if current_pair_seq == last_pair_seq:
                time.sleep(0.001)
                continue
            last_pair_seq = current_pair_seq
            pair_index += 1

            if pair_index % process_every != 0:
                state.skipped_pairs += 1
                continue

            host_dt_ms = packet_delta_ms(left_packet, right_packet)
            camera_dt_raw_ms = sdk_delta_ms(left, right, left_packet, right_packet)
            if camera_dt_raw_ms is not None and sdk_baseline_ms is None:
                sdk_baseline_ms = camera_dt_raw_ms
            camera_dt_ms = (
                None
                if camera_dt_raw_ms is None or sdk_baseline_ms is None
                else camera_dt_raw_ms - sdk_baseline_ms
            )

            use_batch = (
                initialized
                and not batch_disabled
                and bool(getattr(config, "DUAL_ENABLE_BATCH_INFERENCE", True))
                and batch_inference_supported(dlc_live)
            )
            if use_batch:
                try:
                    left_result, right_result = run_batch_inference(
                        dlc_live,
                        left,
                        right,
                        left_packet,
                        right_packet,
                        bodypart_to_idx,
                    )
                except Exception as exc:
                    if not bool(getattr(config, "DUAL_BATCH_FALLBACK_TO_SEQUENTIAL", True)):
                        raise
                    batch_disabled = True
                    if logger is not None:
                        logger.warning("Batch inference disabled; falling back to sequential DLCLive calls: %s", exc)
                    initialized, left_result = run_inference(
                        dlc_live,
                        initialized,
                        left,
                        left_packet,
                        bodypart_to_idx,
                    )
                    initialized, right_result = run_inference(
                        dlc_live,
                        initialized,
                        right,
                        right_packet,
                        bodypart_to_idx,
                    )
            else:
                initialized, left_result = run_inference(
                    dlc_live,
                    initialized,
                    left,
                    left_packet,
                    bodypart_to_idx,
                )
                initialized, right_result = run_inference(
                    dlc_live,
                    initialized,
                    right,
                    right_packet,
                    bodypart_to_idx,
                )

            result = PairInferenceResult(
                pair_index=pair_index,
                left_packet=left_packet,
                right_packet=right_packet,
                host_dt_ms=host_dt_ms,
                camera_dt_ms=camera_dt_ms,
                left_result=left_result,
                right_result=right_result,
            )
            with state.lock:
                state.latest_result = result
                state.latest_seq += 1
                state.processed_pairs += 1

            for runtime in (left, right):
                if runtime.last_error is not None:
                    raise RuntimeError(f"{runtime.name} reader failed") from runtime.last_error

    except BaseException as exc:
        state.last_error = exc
        stop_event.set()


def get_latest_inference(state: InferenceState) -> tuple[int, Optional[PairInferenceResult]]:
    with state.lock:
        return state.latest_seq, state.latest_result


def add_dual_text(
    frame,
    runtime: CameraRuntime,
    packet: live.FramePacket,
    pair_index: int,
    host_dt_ms: float,
    camera_dt_ms: Optional[float],
    picked_side: str,
) -> None:
    cfg = runtime.native_config
    roi_text = "roi {}x{} off {},{}".format(
        cfg.get("Width", "?"),
        cfg.get("Height", "?"),
        cfg.get("OffsetX", "?"),
        cfg.get("OffsetY", "?"),
    )
    sdk_dt_text = "sdk_rel_dt n/a" if camera_dt_ms is None else f"sdk_rel_dt {camera_dt_ms:+.2f}ms"
    lines = [
        f"{runtime.name} sn {runtime.sn} side {picked_side}",
        f"pair {pair_index} host_dt {host_dt_ms:+.2f}ms {sdk_dt_text}",
        f"sdk_frame {packet.source_frame_id} local_frame {packet.frame_id} {roi_text}",
    ]
    y = 90
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tuple(getattr(config, "TEXT_COLOR", (60, 255, 60))),
            2,
            cv2.LINE_AA,
        )
        y += 22


def write_csv_row(csv_path: Path, header_written: bool, row: list[object]) -> bool:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not header_written:
            writer.writerow(
                [
                    "pair_index",
                    "timestamp",
                    "left_local_frame",
                    "right_local_frame",
                    "left_sdk_frame",
                    "right_sdk_frame",
                    "host_dt_ms",
                    "sdk_rel_dt_ms",
                    "left_infer_ms",
                    "right_infer_ms",
                    "left_angle",
                    "right_angle",
                    "left_triplet",
                    "right_triplet",
                    "left_drops",
                    "right_drops",
                ]
            )
        writer.writerow(row)
    return True


def make_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*str(getattr(config, "DUAL_OUTPUT_VIDEO_CODEC", "mp4v")))
    writer = cv2.VideoWriter(str(path), fourcc, max(1.0, fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {path}")
    return writer


def main() -> None:
    validate_dual_config()
    logger = setup_dual_logger()
    bridge = OpenEphysBridge(logger)

    runtimes = [build_camera_runtime(camera_cfg) for camera_cfg in config.DUAL_CAMERAS]
    runtime_by_name = {runtime.name: runtime for runtime in runtimes}
    left, right = runtime_by_name["left"], runtime_by_name["right"]

    stop_event = Event()
    threads: list[Thread] = []
    dlc_live = None
    left_writer = None
    right_writer = None
    csv_header_written = False
    inference_state = InferenceState()

    try:
        bridge.open()

        for runtime in runtimes:
            runtime.source.open()
            logger.info(
                "Opened %s sn=%s config=%s native=%s fps=%.1f ts_freq=%.1f",
                runtime.name,
                runtime.sn,
                runtime.config_path,
                runtime.native_config,
                runtime.source.nominal_fps(),
                float(getattr(runtime.source, "timestamp_tick_frequency", 0.0) or 0.0),
            )

        for runtime in runtimes:
            thread = Thread(target=reader_loop, args=(runtime, stop_event), daemon=True)
            thread.start()
            threads.append(thread)

        first_left, first_right = wait_for_initial_pair(left, right, stop_event)
        logger.info(
            "First pair received. host_dt=%.2fms left_shape=%s right_shape=%s",
            packet_delta_ms(first_left, first_right),
            first_left.frame.shape,
            first_right.frame.shape,
        )

        dlc_live = live.build_dlc_live(None)
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
                logger.info(
                    "DLC_MODEL_DEVICE=%s",
                    first_param.device if first_param is not None else "no_params",
                )
            else:
                logger.info("DLC_MODEL_DEVICE: dlc_live.model is None before init")
        except Exception as exc:
            logger.warning("CUDA/DLC device check failed: %s", exc)
        model_cfg = dlc_live.read_config()
        body_parts = live.extract_bodyparts(model_cfg)
        missing_points = [name for name in config.DUAL_USE_POINTS if name not in body_parts]
        if missing_points:
            raise ValueError(f"DUAL_USE_POINTS are missing from the exported model: {missing_points}")

        bodypart_to_idx = {name: i for i, name in enumerate(body_parts)}
        if python_postprocess_enabled():
            for runtime in runtimes:
                runtime.processor = live.OnlinePoseProcessor(body_parts=body_parts, use_points=config.DUAL_USE_POINTS)
            logger.info("Python pose postprocess enabled for diagnostics/legacy TTL.")
        else:
            for runtime in runtimes:
                runtime.processor = None
            logger.info("Fast pose-only mode enabled: Python sends raw DLCLive points; plugin computes filters, angles and TTL.")

        logger.info("Model bodyparts loaded: %d; dual points=%s", len(body_parts), config.DUAL_USE_POINTS)

        inference_thread = Thread(
            target=inference_loop,
            args=(stop_event, dlc_live, left, right, bodypart_to_idx, inference_state, logger),
            daemon=True,
        )
        inference_thread.start()
        threads.append(inference_thread)

        visual_mode = bool(getattr(config, "DUAL_DISPLAY_WINDOW", True))
        last_result_seq = -1

        while not stop_event.is_set():
            result_seq, result = get_latest_inference(inference_state)
            if result is None or result_seq == last_result_seq:
                if inference_state.last_error is not None:
                    raise RuntimeError("Inference worker failed") from inference_state.last_error
                for runtime in runtimes:
                    if runtime.last_error is not None:
                        raise RuntimeError(f"{runtime.name} reader failed") from runtime.last_error
                time.sleep(0.001)
                continue

            last_result_seq = result_seq
            pair_index = result.pair_index
            left_packet = result.left_packet
            right_packet = result.right_packet
            host_dt_ms = result.host_dt_ms
            camera_dt_ms = result.camera_dt_ms
            left_result = result.left_result
            right_result = result.right_result
            bridge.send(result, left, right)
            validate_native_shape(left, left_packet, logger)
            validate_native_shape(right, right_packet, logger)

            left_metrics = {
                "cam_fps": left.fps_cam,
                "infer_fps": left.fps_dlc,
                "infer_ms": left_result["infer_ms"],
                "raw_visible": left_result["raw_visible"],
                "filtered_visible": left_result["filtered_visible"],
                "tracked_points": len(config.DUAL_USE_POINTS),
                "triplet": left_result["has_triplet"],
                "hind_angle": left_result["hind_angle"],
                "source_drops": left.source.dropped_total,
            }
            right_metrics = {
                "cam_fps": right.fps_cam,
                "infer_fps": right.fps_dlc,
                "infer_ms": right_result["infer_ms"],
                "raw_visible": right_result["raw_visible"],
                "filtered_visible": right_result["filtered_visible"],
                "tracked_points": len(config.DUAL_USE_POINTS),
                "triplet": right_result["has_triplet"],
                "hind_angle": right_result["hind_angle"],
                "source_drops": right.source.dropped_total,
            }

            left_display = live.draw_overlay(left_packet.frame.copy(), left_result["draw_points"], left_metrics)
            right_display = live.draw_overlay(right_packet.frame.copy(), right_result["draw_points"], right_metrics)
            add_dual_text(
                left_display,
                left,
                left_packet,
                pair_index,
                host_dt_ms,
                camera_dt_ms,
                str(left_result["picked_side"]),
            )
            add_dual_text(
                right_display,
                right,
                right_packet,
                pair_index,
                host_dt_ms,
                camera_dt_ms,
                str(right_result["picked_side"]),
            )

            if bool(getattr(config, "DUAL_SAVE_OUTPUT_VIDEO", False)):
                if left_writer is None:
                    left_writer = make_writer(Path(config.DUAL_OUTPUT_LEFT_PATH), left_display, left.source.nominal_fps())
                if right_writer is None:
                    right_writer = make_writer(Path(config.DUAL_OUTPUT_RIGHT_PATH), right_display, right.source.nominal_fps())
                left_writer.write(left_display)
                right_writer.write(right_display)

            if visual_mode:
                scale = float(getattr(config, "DUAL_SHOW_SCALE", 1.0))
                if scale != 1.0:
                    left_show = cv2.resize(left_display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    right_show = cv2.resize(right_display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    left_show = left_display
                    right_show = right_display
                cv2.imshow(f"{config.DUAL_WINDOW_NAME} | left", left_show)
                cv2.imshow(f"{config.DUAL_WINDOW_NAME} | right", right_show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("Exit requested.")
                    break

            if pair_index % max(1, int(getattr(config, "DUAL_LOG_EVERY_N_PAIRS", 30))) == 0:
                logger.info(
                    "pair=%d host_dt=%.2fms sdk_rel_dt=%s left_frame=%s right_frame=%s "
                    "left_triplet=%s right_triplet=%s left_angle=%s right_angle=%s drops=%d/%d",
                    pair_index,
                    host_dt_ms,
                    f"{camera_dt_ms:.2f}ms" if camera_dt_ms is not None else "n/a",
                    left_packet.source_frame_id,
                    right_packet.source_frame_id,
                    left_result["has_triplet"],
                    right_result["has_triplet"],
                    f"{float(left_result['hind_angle']):.1f}" if left_result["hind_angle"] is not None else "None",
                    f"{float(right_result['hind_angle']):.1f}" if right_result["hind_angle"] is not None else "None",
                    left.source.dropped_total,
                    right.source.dropped_total,
                )

                if bool(getattr(config, "ENABLE_BENCHMARK_CSV", True)):
                    csv_header_written = write_csv_row(
                        Path(config.BENCHMARK_CSV_PATH),
                        csv_header_written,
                        [
                            pair_index,
                            f"{time.time():.3f}",
                            left_packet.frame_id,
                            right_packet.frame_id,
                            left_packet.source_frame_id or "",
                            right_packet.source_frame_id or "",
                            f"{host_dt_ms:.3f}",
                            f"{camera_dt_ms:.3f}" if camera_dt_ms is not None else "",
                            f"{float(left_result['infer_ms']):.3f}",
                            f"{float(right_result['infer_ms']):.3f}",
                            f"{float(left_result['hind_angle']):.3f}" if left_result["hind_angle"] is not None else "",
                            f"{float(right_result['hind_angle']):.3f}" if right_result["hind_angle"] is not None else "",
                            int(bool(left_result["has_triplet"])),
                            int(bool(right_result["has_triplet"])),
                            left.source.dropped_total,
                            right.source.dropped_total,
                        ],
                    )

            for runtime in runtimes:
                if runtime.last_error is not None:
                    raise RuntimeError(f"{runtime.name} reader failed") from runtime.last_error
            if inference_state.last_error is not None:
                raise RuntimeError("Inference worker failed") from inference_state.last_error

    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
        for runtime in runtimes:
            runtime.source.release()
        if left_writer is not None:
            left_writer.release()
        if right_writer is not None:
            right_writer.release()
        if dlc_live is not None:
            try:
                dlc_live.close()
            except Exception:
                pass
        bridge.close()
        if bool(getattr(config, "DUAL_DISPLAY_WINDOW", True)):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
