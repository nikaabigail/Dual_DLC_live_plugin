"""
DLCLive-based realtime runtime.

Architecture:
  - DLCLive performs model loading, cropping, resizing, and coordinate restoration.
  - This script manages frame capture, overlay, logging, and output video writing.
  - Online filtering is implemented as a DLCLive-compatible processor.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config_rt_dlc_live as config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rt_dlc_live")
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


@dataclass
class FramePacket:
    frame_id: int
    frame: np.ndarray
    capture_ts: float
    source_frame_id: Optional[int] = None
    source_timestamp: Optional[int] = None


class FrameSource(ABC):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> tuple[bool, Optional[FramePacket]]:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def nominal_fps(self) -> float:
        raise NotImplementedError


class CameraSource(FrameSource):
    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self.cap: cv2.VideoCapture | None = None
        self.frame_id = 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_VIDEO_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

    def read(self) -> tuple[bool, Optional[FramePacket]]:
        if self.cap is None:
            raise RuntimeError("CameraSource is not opened.")
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        self.frame_id += 1
        return True, FramePacket(frame_id=self.frame_id, frame=frame, capture_ts=time.time())

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def nominal_fps(self) -> float:
        return float(getattr(config, "TARGET_VIDEO_FPS", 0.0))


class GalaxyCameraSource(FrameSource):
    """Daheng Galaxy SDK source for USB3 Vision cameras used by GalaxyView."""

    def __init__(
        self,
        sdk_root: Path,
        serial_number: str,
        device_index: int,
        config_path: Path | None,
        import_config: bool,
        config_verify: bool,
        fallback_apply_config: bool,
        frame_timeout_ms: int,
        force_trigger_off: bool,
        low_latency: bool,
        stream_buffer_handling_mode: str,
        acquisition_buffer_count: int,
        drain_queued_frames: bool,
        max_drain_frames: int,
    ) -> None:
        self.sdk_root = sdk_root
        self.serial_number = serial_number.strip()
        self.device_index = device_index
        self.config_path = config_path
        self.import_config = import_config
        self.config_verify = config_verify
        self.fallback_apply_config = fallback_apply_config
        self.frame_timeout_ms = frame_timeout_ms
        self.force_trigger_off = force_trigger_off
        self.low_latency = low_latency
        self.stream_buffer_handling_mode = stream_buffer_handling_mode.upper()
        self.acquisition_buffer_count = acquisition_buffer_count
        self.drain_queued_frames = drain_queued_frames
        self.max_drain_frames = max_drain_frames
        self.device_manager = None
        self.cam = None
        self.frame_id = 0
        self.source_fps = 0.0
        self.timestamp_tick_frequency = 0.0
        self.is_color = True
        self.dropped_total = 0
        self._dll_dir_handles: list[object] = []
        self.output_color = str(getattr(config, "GALAXY_OUTPUT_COLOR", "bgr")).strip().lower()
        if self.output_color not in {"bgr", "rgb"}:
            raise ValueError("GALAXY_OUTPUT_COLOR must be 'bgr' or 'rgb'.")

    def open(self) -> None:
        self._prepare_sdk_environment()
        try:
            import gxipy as gx
        except Exception as exc:
            raise RuntimeError(
                "Cannot import Daheng `gxipy`. Check GALAXY_SDK_ROOT and Galaxy SDK installation."
            ) from exc

        try:
            self.device_manager = gx.DeviceManager()
            dev_num, dev_info_list = self.device_manager.update_device_list()
        except Exception as exc:
            raise RuntimeError(
                "Galaxy SDK could not enumerate cameras. Ensure GENICAM_GENTL64_PATH points "
                "to the GalaxySDK GenTL Win64 directory."
            ) from exc

        if dev_num <= 0:
            raise RuntimeError("Galaxy SDK did not find any Daheng cameras.")

        try:
            if self.serial_number:
                self.cam = self.device_manager.open_device_by_sn(self.serial_number)
            else:
                self.cam = self.device_manager.open_device_by_index(self.device_index)
        except Exception as exc:
            devices = ", ".join(str(info.get("display_name", info)) for info in dev_info_list)
            raise RuntimeError(
                "Cannot open Galaxy camera. Close GalaxyView or stop its acquisition first. "
                f"Requested sn={self.serial_number!r} index={self.device_index}; found: {devices}"
            ) from exc

        if self.import_config and self.config_path is not None:
            self._import_or_apply_config()

        if self.force_trigger_off:
            self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        self.is_color = bool(self.cam.PixelColorFilter.is_implemented())
        self.source_fps = self._read_source_fps()
        self.timestamp_tick_frequency = self._read_timestamp_tick_frequency()
        self._configure_low_latency_stream(gx)
        self.cam.stream_on()

    def read(self) -> tuple[bool, Optional[FramePacket]]:
        if self.cam is None:
            raise RuntimeError("GalaxyCameraSource is not opened.")

        raw_image = self.cam.data_stream[0].get_image(int(self.frame_timeout_ms))
        if raw_image is None:
            return False, None

        if self.drain_queued_frames:
            raw_image = self._drain_to_latest(raw_image)

        source_frame_id = self._read_raw_image_int(raw_image, "get_frame_id")
        source_timestamp = self._read_raw_image_int(raw_image, "get_timestamp")
        frame = self._raw_image_to_bgr(raw_image)
        if frame is None:
            return False, None

        self.frame_id += 1
        return True, FramePacket(
            frame_id=self.frame_id,
            frame=frame,
            capture_ts=time.time(),
            source_frame_id=source_frame_id,
            source_timestamp=source_timestamp,
        )

    def release(self) -> None:
        if self.cam is not None:
            try:
                self.cam.stream_off()
            except Exception:
                pass
            try:
                self.cam.close_device()
            except Exception:
                pass
            self.cam = None
        self.device_manager = None

    def nominal_fps(self) -> float:
        if self.source_fps > 0:
            return self.source_fps
        return float(getattr(config, "TARGET_VIDEO_FPS", 0.0))

    def _prepare_sdk_environment(self) -> None:
        sdk_root = self.sdk_root
        python_sdk_path = Path(
            getattr(config, "GALAXY_PYTHON_SDK_PATH", sdk_root / "Samples" / "Python SDK")
        )
        dll_dirs = [
            sdk_root / "APIDll" / "Win64",
            sdk_root / "GenTL" / "Win64",
            sdk_root / "GenICam" / "bin" / "Win64_x64",
        ]

        for dll_dir in dll_dirs:
            if not dll_dir.exists():
                continue
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                self._dll_dir_handles.append(os.add_dll_directory(str(dll_dir)))

        gentl64_path = Path(getattr(config, "GALAXY_GENTL64_PATH", sdk_root / "GenTL" / "Win64"))
        if gentl64_path.exists():
            os.environ["GENICAM_GENTL64_PATH"] = str(gentl64_path)

        gentl32_path = Path(getattr(config, "GALAXY_GENTL32_PATH", sdk_root / "GenTL" / "Win32"))
        if gentl32_path.exists():
            os.environ["GENICAM_GENTL32_PATH"] = str(gentl32_path)

        if python_sdk_path.exists():
            python_sdk = str(python_sdk_path)
            if python_sdk not in sys.path:
                sys.path.insert(0, python_sdk)

    def _import_or_apply_config(self) -> None:
        if self.cam is None or self.config_path is None:
            return
        try:
            self.cam.import_config_file(str(self.config_path), verify=self.config_verify)
            return
        except Exception as exc:
            if not self.fallback_apply_config:
                raise
            logging.getLogger("rt_dlc_live").warning(
                "Galaxy import_config_file failed for %s (%s). Applying core camera settings manually.",
                self.config_path,
                exc,
            )
            self._apply_core_config_file(self.config_path)

    def _apply_core_config_file(self, path: Path) -> None:
        values = self._read_core_config_values(path)
        if not values:
            raise RuntimeError(f"Could not read any core Galaxy settings from: {path}")

        self._set_enum_by_name("TriggerSelector", values.get("TriggerSelector"))
        self._set_enum_by_name("TriggerMode", "Off")

        self._set_enum_by_name("PixelFormat", values.get("PixelFormat"))
        self._set_enum_by_name("RegionSendMode", values.get("RegionSendMode"))
        self._set_enum_by_name("RegionSelector", values.get("RegionSelector"))
        self._set_enum_by_name("RegionMode", values.get("RegionMode"))

        # ROI changes are safest from origin first, then size, then final offset.
        self._set_int_feature("OffsetX", 0)
        self._set_int_feature("OffsetY", 0)
        self._set_int_feature("Width", values.get("Width"))
        self._set_int_feature("Height", values.get("Height"))
        self._set_int_feature("OffsetX", values.get("OffsetX"))
        self._set_int_feature("OffsetY", values.get("OffsetY"))

        self._set_enum_by_name("AcquisitionMode", values.get("AcquisitionMode"))
        self._set_enum_by_name("AcquisitionFrameRateMode", values.get("AcquisitionFrameRateMode"))
        self._set_float_feature("AcquisitionFrameRate", values.get("AcquisitionFrameRate"))
        self._set_enum_by_name("ExposureMode", values.get("ExposureMode"))
        self._set_enum_by_name("ExposureAuto", values.get("ExposureAuto"))
        self._set_float_feature("ExposureTime", values.get("ExposureTime"))

        self._set_enum_by_name("TriggerSource", values.get("TriggerSource"))
        self._set_enum_by_name("TriggerActivation", values.get("TriggerActivation"))
        self._set_float_feature("TriggerDelay", values.get("TriggerDelay"))
        self._set_float_feature("TriggerFilterRaisingEdge", values.get("TriggerFilterRaisingEdge"))
        self._set_float_feature("TriggerFilterFallingEdge", values.get("TriggerFilterFallingEdge"))
        self._set_enum_by_name("TriggerMode", values.get("TriggerMode"))

    def _read_core_config_values(self, path: Path) -> dict[str, str]:
        keys = {
            "PixelFormat",
            "RegionSendMode",
            "RegionSelector",
            "RegionMode",
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
            "AcquisitionMode",
            "AcquisitionFrameRateMode",
            "AcquisitionFrameRate",
            "ExposureMode",
            "ExposureAuto",
            "ExposureTime",
            "TriggerSelector",
            "TriggerMode",
            "TriggerSource",
            "TriggerActivation",
            "TriggerDelay",
            "TriggerFilterRaisingEdge",
            "TriggerFilterFallingEdge",
        }
        values: dict[str, str] = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("<"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[0] in keys:
                    values[parts[0]] = parts[1]
        return values

    def _set_enum_by_name(self, feature_name: str, value: object) -> None:
        if value is None or self.cam is None:
            return
        feature = getattr(self.cam, feature_name, None)
        if feature is None:
            return
        try:
            if not feature.is_implemented() or not feature.is_writable():
                return
            range_dict = feature.get_range()
            if value in range_dict:
                feature.set(int(range_dict[value]))
        except Exception as exc:
            logging.getLogger("rt_dlc_live").debug("Could not set %s=%s: %s", feature_name, value, exc)

    def _set_int_feature(self, feature_name: str, value: object) -> None:
        if value is None or self.cam is None:
            return
        feature = getattr(self.cam, feature_name, None)
        if feature is None:
            return
        try:
            if not feature.is_implemented() or not feature.is_writable():
                return
            feature.set(int(float(value)))
        except Exception as exc:
            logging.getLogger("rt_dlc_live").debug("Could not set %s=%s: %s", feature_name, value, exc)

    def _set_float_feature(self, feature_name: str, value: object) -> None:
        if value is None or self.cam is None:
            return
        feature = getattr(self.cam, feature_name, None)
        if feature is None:
            return
        try:
            if not feature.is_implemented() or not feature.is_writable():
                return
            feature.set(float(value))
        except Exception as exc:
            logging.getLogger("rt_dlc_live").debug("Could not set %s=%s: %s", feature_name, value, exc)

    def _raw_image_to_bgr(self, raw_image) -> np.ndarray | None:
        if self.is_color:
            rgb_image = raw_image.convert("RGB")
            if rgb_image is None:
                return None
            arr = rgb_image.get_numpy_array()
            if arr is None:
                return None
            if arr.ndim == 2:
                code = cv2.COLOR_GRAY2RGB if self.output_color == "rgb" else cv2.COLOR_GRAY2BGR
                return cv2.cvtColor(arr, code)
            if self.output_color == "rgb":
                return arr
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        arr = raw_image.get_numpy_array()
        if arr is None:
            return None
        if arr.ndim == 2:
            code = cv2.COLOR_GRAY2RGB if self.output_color == "rgb" else cv2.COLOR_GRAY2BGR
            return cv2.cvtColor(arr, code)
        return arr

    def _read_source_fps(self) -> float:
        try:
            if self.cam is not None and self.cam.AcquisitionFrameRate.is_readable():
                return float(self.cam.AcquisitionFrameRate.get())
        except Exception:
            pass
        return 0.0

    def _read_timestamp_tick_frequency(self) -> float:
        try:
            if self.cam is not None and self.cam.TimestampTickFrequency.is_readable():
                return float(self.cam.TimestampTickFrequency.get())
        except Exception:
            pass
        return 0.0

    def _read_raw_image_int(self, raw_image, method_name: str) -> int | None:
        try:
            value = getattr(raw_image, method_name)()
        except Exception:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _configure_low_latency_stream(self, gx) -> None:
        if self.cam is None or not self.low_latency:
            return

        stream = self.cam.data_stream[0]
        if self.acquisition_buffer_count > 0:
            try:
                stream.set_acquisition_buffer_number(int(self.acquisition_buffer_count))
            except Exception:
                pass

        mode_map = {
            "OLDEST_FIRST": gx.GxDSStreamBufferHandlingModeEntry.OLDEST_FIRST,
            "OLDEST_FIRST_OVERWRITE": gx.GxDSStreamBufferHandlingModeEntry.OLDEST_FIRST_OVERWRITE,
            "NEWEST_ONLY": gx.GxDSStreamBufferHandlingModeEntry.NEWEST_ONLY,
        }
        mode_value = mode_map.get(self.stream_buffer_handling_mode)
        if mode_value is None:
            return

        try:
            if stream.StreamBufferHandlingMode.is_writable():
                stream.StreamBufferHandlingMode.set(mode_value)
        except Exception:
            pass

    def _drain_to_latest(self, raw_image):
        if self.cam is None:
            return raw_image

        latest = raw_image
        for _ in range(max(0, int(self.max_drain_frames))):
            next_image = self.cam.data_stream[0].get_image(0)
            if next_image is None:
                break
            latest = next_image
            self.dropped_total += 1
        return latest


class VideoFileSource(FrameSource):
    def __init__(self, video_path: Path, target_fps: float | None, skip_if_behind: bool) -> None:
        self.video_path = video_path
        self.target_fps = target_fps
        self.skip_if_behind = skip_if_behind
        self.cap: cv2.VideoCapture | None = None
        self.frame_id = 0
        self.start_perf: float | None = None
        self.frame_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else None
        self.source_fps = 0.0
        self.dropped_total = 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        self.start_perf = time.perf_counter()
        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self) -> tuple[bool, Optional[FramePacket]]:
        if self.cap is None:
            raise RuntimeError("VideoFileSource is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        self.frame_id += 1
        if self.frame_interval is not None and self.start_perf is not None:
            expected_elapsed = self.frame_id * self.frame_interval
            actual_elapsed = time.perf_counter() - self.start_perf
            lag = actual_elapsed - expected_elapsed

            if lag < 0:
                time.sleep(-lag)
            elif lag > self.frame_interval and self.skip_if_behind:
                drops_needed = max(1, min(int(lag / self.frame_interval), 10))
                dropped_now = 0
                for _ in range(drops_needed):
                    drop_ok, _ = self.cap.read()
                    if not drop_ok:
                        break
                    self.frame_id += 1
                    dropped_now += 1
                self.dropped_total += dropped_now

        return True, FramePacket(frame_id=self.frame_id, frame=frame, capture_ts=time.time())

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def nominal_fps(self) -> float:
        if self.source_fps > 0:
            return self.source_fps
        return float(self.target_fps or 0.0)


def build_frame_source() -> FrameSource:
    if config.USE_VIDEO_FILE:
        return VideoFileSource(
            video_path=Path(config.VIDEO_FILE_PATH),
            target_fps=float(getattr(config, "VIDEO_TARGET_FPS", 0.0)),
            skip_if_behind=bool(getattr(config, "VIDEO_SKIP_IF_BEHIND", False)),
        )
    camera_backend = str(getattr(config, "CAMERA_BACKEND", "opencv")).lower()
    if camera_backend == "galaxy":
        return GalaxyCameraSource(
            sdk_root=Path(getattr(config, "GALAXY_SDK_ROOT")),
            serial_number=str(getattr(config, "GALAXY_SN", "")),
            device_index=int(getattr(config, "GALAXY_INDEX", 1)),
            config_path=Path(getattr(config, "GALAXY_CONFIG_PATH")),
            import_config=bool(getattr(config, "GALAXY_IMPORT_CONFIG", True)),
            config_verify=bool(getattr(config, "GALAXY_CONFIG_VERIFY", False)),
            fallback_apply_config=bool(getattr(config, "GALAXY_FALLBACK_APPLY_CONFIG", True)),
            frame_timeout_ms=int(getattr(config, "GALAXY_FRAME_TIMEOUT_MS", 1000)),
            force_trigger_off=bool(getattr(config, "GALAXY_FORCE_TRIGGER_OFF", False)),
            low_latency=bool(getattr(config, "GALAXY_LOW_LATENCY", True)),
            stream_buffer_handling_mode=str(getattr(config, "GALAXY_STREAM_BUFFER_HANDLING_MODE", "NEWEST_ONLY")),
            acquisition_buffer_count=int(getattr(config, "GALAXY_ACQUISITION_BUFFER_COUNT", 2)),
            drain_queued_frames=bool(getattr(config, "GALAXY_DRAIN_QUEUED_FRAMES", True)),
            max_drain_frames=int(getattr(config, "GALAXY_MAX_DRAIN_FRAMES", 20)),
        )
    if camera_backend == "opencv":
        return CameraSource(int(config.CAM_INDEX))
    raise ValueError(f"Unsupported CAMERA_BACKEND={camera_backend!r}; use 'galaxy' or 'opencv'.")


def validate_config() -> None:
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    if config.USE_VIDEO_FILE and not Path(config.VIDEO_FILE_PATH).exists():
        raise FileNotFoundError(f"VIDEO_FILE_PATH does not exist: {config.VIDEO_FILE_PATH}")
    if not config.USE_VIDEO_FILE:
        camera_backend = str(getattr(config, "CAMERA_BACKEND", "opencv")).lower()
        if camera_backend not in {"galaxy", "opencv"}:
            raise ValueError("CAMERA_BACKEND must be 'galaxy' or 'opencv'.")
        if camera_backend == "galaxy":
            sdk_root = Path(getattr(config, "GALAXY_SDK_ROOT"))
            if not sdk_root.exists():
                raise FileNotFoundError(f"GALAXY_SDK_ROOT does not exist: {sdk_root}")
            if not str(getattr(config, "GALAXY_SN", "")).strip() and int(config.GALAXY_INDEX) < 1:
                raise ValueError("GALAXY_INDEX must be >= 1 when GALAXY_SN is empty.")
            if int(getattr(config, "GALAXY_FRAME_TIMEOUT_MS", 1000)) <= 0:
                raise ValueError("GALAXY_FRAME_TIMEOUT_MS must be positive.")
            if int(getattr(config, "GALAXY_ACQUISITION_BUFFER_COUNT", 2)) < 0:
                raise ValueError("GALAXY_ACQUISITION_BUFFER_COUNT must be >= 0.")
            if int(getattr(config, "GALAXY_MAX_DRAIN_FRAMES", 20)) < 0:
                raise ValueError("GALAXY_MAX_DRAIN_FRAMES must be >= 0.")
            stream_mode = str(getattr(config, "GALAXY_STREAM_BUFFER_HANDLING_MODE", "NEWEST_ONLY")).upper()
            if stream_mode not in {"OLDEST_FIRST", "OLDEST_FIRST_OVERWRITE", "NEWEST_ONLY"}:
                raise ValueError(
                    "GALAXY_STREAM_BUFFER_HANDLING_MODE must be OLDEST_FIRST, "
                    "OLDEST_FIRST_OVERWRITE, or NEWEST_ONLY."
                )
            if bool(getattr(config, "GALAXY_IMPORT_CONFIG", True)):
                galaxy_config_path = Path(getattr(config, "GALAXY_CONFIG_PATH"))
                if not galaxy_config_path.exists():
                    raise FileNotFoundError(f"GALAXY_CONFIG_PATH does not exist: {galaxy_config_path}")
    if not config.USE_POINTS:
        raise ValueError("USE_POINTS must not be empty.")
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

    cropping = getattr(config, "CROPPING", None)
    if cropping is not None:
        if len(cropping) != 4:
            raise ValueError("CROPPING must be [x1, x2, y1, y2] or None.")
        x1, x2, y1, y2 = [int(v) for v in cropping]
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("CROPPING must satisfy 0 <= x1 < x2 and 0 <= y1 < y2.")


def normalize_cropping_for_frame(
    cropping: list[int] | tuple[int, int, int, int] | None,
    frame: np.ndarray,
    logger: logging.Logger,
) -> list[int] | None:
    if cropping is None:
        return None

    x1, x2, y1, y2 = [int(v) for v in cropping]
    h, w = frame.shape[:2]
    if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
        return [x1, x2, y1, y2]

    crop_w = max(0, x2 - x1)
    crop_h = max(0, y2 - y1)
    if w == crop_w and h == crop_h:
        logger.warning(
            "Configured CROPPING matches the full input frame size (%dx%d). "
            "Internal DLCLive cropping will be disabled.",
            w,
            h,
        )
        return None

    logger.warning(
        "Configured CROPPING=%s does not fit the first frame size (%dx%d). "
        "Internal DLCLive cropping will be disabled.",
        cropping,
        w,
        h,
    )
    return None


def build_dlc_live(cropping: list[int] | None):
    try:
        from dlclive import DLCLive
    except ModuleNotFoundError as exc:
        if exc.name == "colorcet":
            raise RuntimeError(
                "DLCLive import failed because the optional dependency `colorcet` "
                "is missing in the active environment. Install it there and rerun."
            ) from exc
        raise

    return DLCLive(
        model_path=config.MODEL_PATH,
        model_type=config.MODEL_TYPE,
        precision=getattr(config, "PRECISION", "FP32"),
        single_animal=bool(getattr(config, "SINGLE_ANIMAL", True)),
        device=getattr(config, "DEVICE", None),
        cropping=cropping,
        dynamic=getattr(config, "DYNAMIC_CROPPING", (False, 0.5, 10)),
        resize=float(getattr(config, "RESIZE", 1.0)),
        processor=None,
        convert2rgb=bool(getattr(config, "CONVERT_TO_RGB", True)),
        display=False,
    )


def extract_bodyparts(model_cfg: dict) -> list[str]:
    metadata = model_cfg.get("metadata", {})
    if "bodyparts" in metadata:
        return list(metadata["bodyparts"])
    if "all_joints_names" in model_cfg:
        return list(model_cfg["all_joints_names"])
    raise KeyError("Could not find bodypart names in the exported model config.")


def safe_angle_deg(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> Optional[float]:
    bax, bay = float(a[0] - b[0]), float(a[1] - b[1])
    bcx, bcy = float(c[0] - b[0]), float(c[1] - b[1])
    n1, n2 = math.hypot(bax, bay), math.hypot(bcx, bcy)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_val = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (n1 * n2)))
    return math.degrees(math.acos(cos_val))


def dist2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def point_from_pose_row(row: np.ndarray) -> dict[str, float | None]:
    if row.shape[0] < 3:
        return {"x": None, "y": None, "likelihood": None}
    x, y, l = float(row[0]), float(row[1]), float(row[2])
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(l)):
        return {"x": None, "y": None, "likelihood": None}
    return {"x": x, "y": y, "likelihood": l}


def pose_to_points(
    pose: np.ndarray | None,
    bodypart_to_idx: dict[str, int],
    names: list[str],
) -> dict[str, dict[str, float | None]]:
    points = {name: {"x": None, "y": None, "likelihood": None} for name in names}
    if pose is None:
        return points

    arr = np.asarray(pose)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 3:
        return points

    for name in names:
        idx = bodypart_to_idx.get(name)
        if idx is None or idx >= arr.shape[0]:
            continue
        points[name] = point_from_pose_row(arr[idx])
    return points


def evaluate_triplet(
    points: dict[str, dict[str, float | None]],
    required_points: list[str],
) -> tuple[dict[str, dict[str, float | None]], bool, dict[str, str]]:
    reason: dict[str, str] = {}
    for name in required_points:
        p = points.get(name)
        if p is None:
            reason[name] = "missing"
            continue
        x, y, l = p.get("x"), p.get("y"), p.get("likelihood")
        if x is None or y is None:
            reason[name] = "none_xy"
        elif l is None:
            reason[name] = "none_likelihood"
        elif l < config.CONF_THRESH_DRAW:
            reason[name] = "low_conf"
        else:
            reason[name] = "ok"

    has_triplet = all(reason.get(name) == "ok" for name in required_points)
    if has_triplet:
        return {name: dict(points[name]) for name in required_points}, True, reason

    empty = {name: {"x": None, "y": None, "likelihood": None} for name in required_points}
    return empty, False, reason


def count_visible_points(points: dict[str, dict[str, float | None]]) -> int:
    visible = 0
    for point in points.values():
        if (
            point.get("x") is not None
            and point.get("y") is not None
            and point.get("likelihood") is not None
            and float(point["likelihood"]) >= config.CONF_THRESH_DRAW
        ):
            visible += 1
    return visible


@dataclass
class PointState:
    last_good_xy: Optional[tuple[float, float]] = None
    last_good_frame_id: Optional[int] = None
    x_hist: deque[float] = field(default_factory=lambda: deque(maxlen=config.MEDIAN_WINDOW))
    y_hist: deque[float] = field(default_factory=lambda: deque(maxlen=config.MEDIAN_WINDOW))


class OnlinePoseProcessor:
    """DLCLive-compatible processor that filters selected bodyparts online."""

    def __init__(self, body_parts: list[str], use_points: list[str]) -> None:
        self.body_parts = list(body_parts)
        self.use_points = list(use_points)
        self.bodypart_to_idx = {name: i for i, name in enumerate(self.body_parts)}
        self.states: dict[str, PointState] = defaultdict(PointState)
        self.last_raw_pose: np.ndarray | None = None
        self.last_filtered_pose: np.ndarray | None = None
        self.last_point_status: dict[str, str] = {}
        self.last_raw_visible = 0
        self.last_filtered_visible = 0

    def process(self, pose: np.ndarray, **kwargs) -> np.ndarray:
        pose_arr = np.asarray(pose, dtype=np.float32)
        if pose_arr.ndim == 3 and pose_arr.shape[0] == 1:
            pose_arr = pose_arr[0]

        raw_pose = np.array(pose_arr, copy=True)
        filtered_pose = np.array(pose_arr, copy=True)
        frame_id = int(kwargs.get("frame_id", 0))

        if not bool(getattr(config, "ENABLE_PROCESSOR", True)):
            self.last_raw_pose = raw_pose
            self.last_filtered_pose = filtered_pose
            self.last_point_status = {}
            self.last_raw_visible = self._count_visible(raw_pose)
            self.last_filtered_visible = self._count_visible(filtered_pose)
            return filtered_pose

        status: dict[str, str] = {}
        for name in self.use_points:
            idx = self.bodypart_to_idx.get(name)
            if idx is None or idx >= filtered_pose.shape[0]:
                status[name] = "missing"
                continue

            raw_point = point_from_pose_row(raw_pose[idx])
            x = raw_point["x"]
            y = raw_point["y"]
            l = raw_point["likelihood"]
            fx, fy, fl, state_name = self._process_point(name, x, y, l, frame_id)
            status[name] = state_name
            if fx is None or fy is None or fl is None:
                filtered_pose[idx, :3] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            else:
                filtered_pose[idx, 0] = float(fx)
                filtered_pose[idx, 1] = float(fy)
                filtered_pose[idx, 2] = float(fl)

        self.last_raw_pose = raw_pose
        self.last_filtered_pose = filtered_pose
        self.last_point_status = status
        self.last_raw_visible = self._count_visible(raw_pose)
        self.last_filtered_visible = self._count_visible(filtered_pose)
        return filtered_pose

    def save(self, file: str = "") -> int:
        return 0

    def _count_visible(self, pose: np.ndarray) -> int:
        visible = 0
        if pose.ndim != 2 or pose.shape[1] < 3:
            return visible
        for name in self.use_points:
            idx = self.bodypart_to_idx.get(name)
            if idx is None or idx >= pose.shape[0]:
                continue
            point = point_from_pose_row(pose[idx])
            if (
                point["x"] is not None
                and point["y"] is not None
                and point["likelihood"] is not None
                and point["likelihood"] >= config.CONF_THRESH_DRAW
            ):
                visible += 1
        return visible

    def _process_point(
        self,
        name: str,
        x: Optional[float],
        y: Optional[float],
        likelihood: Optional[float],
        frame_id: int,
    ) -> tuple[Optional[float], Optional[float], Optional[float], str]:
        state = self.states[name]

        is_good = (
            x is not None
            and y is not None
            and likelihood is not None
            and ((not config.ENABLE_PCUTOFF) or likelihood >= config.CONF_THRESH_USE)
        )

        if is_good and config.ENABLE_DESPIKE:
            current_xy = (float(x), float(y))
            if state.last_good_xy is not None:
                jump = dist2d(current_xy, state.last_good_xy)
                last_idx = state.last_good_frame_id
                gap = (frame_id - last_idx) if last_idx is not None else 0
                allow_reacquire = gap > int(getattr(config, "DESPIKE_RESET_GAP_FRAMES", 0))
                if jump > config.DESPIKE_THRESHOLD_PX and not allow_reacquire:
                    is_good = False

        if is_good:
            current_xy = (float(x), float(y))
            state.last_good_xy = current_xy
            state.last_good_frame_id = frame_id
            state.x_hist.append(current_xy[0])
            state.y_hist.append(current_xy[1])
            x_med = float(np.median(np.array(state.x_hist, dtype=np.float32)))
            y_med = float(np.median(np.array(state.y_hist, dtype=np.float32)))
            return x_med, y_med, float(likelihood), "ok"

        if config.ENABLE_HOLD and state.last_good_xy is not None and state.last_good_frame_id is not None:
            gap = frame_id - state.last_good_frame_id
            if gap <= config.MAX_HOLD_FRAMES:
                hold_x, hold_y = state.last_good_xy
                hold_l = float(max(config.CONF_THRESH_DRAW, config.CONF_THRESH_USE) + 0.01)
                return hold_x, hold_y, hold_l, "hold"

        return None, None, None, "drop"


def compute_hind_angle(points: dict[str, dict[str, float | None]]) -> Optional[float]:
    if not getattr(config, "COMPUTE_HIND_ANGLE", False):
        return None

    p1, p2, p3 = config.HIND_ANGLE_POINTS
    a, b, c = points.get(p1), points.get(p2), points.get(p3)
    if not (a and b and c):
        return None

    coords = [a.get("x"), a.get("y"), b.get("x"), b.get("y"), c.get("x"), c.get("y")]
    if any(v is None for v in coords):
        return None

    return safe_angle_deg(
        (float(a["x"]), float(a["y"])),
        (float(b["x"]), float(b["y"])),
        (float(c["x"]), float(c["y"])),
    )


def draw_overlay(
    frame: np.ndarray,
    points: dict[str, dict[str, float | None]],
    metrics: dict[str, float | int | bool | None | str],
) -> np.ndarray:
    if config.DRAW_POINTS:
        for name, point in points.items():
            x = point.get("x")
            y = point.get("y")
            l = point.get("likelihood")
            if x is None or y is None or l is None:
                continue
            if l < config.CONF_THRESH_DRAW:
                continue

            px = int(round(float(x)))
            py = int(round(float(y)))
            cv2.circle(
                frame,
                (px, py),
                int(getattr(config, "POINT_RADIUS", 4)),
                tuple(getattr(config, "POINT_COLOR", (0, 255, 0))),
                -1,
                lineType=cv2.LINE_AA,
            )

            label_parts: list[str] = []
            if config.DRAW_NAMES:
                label_parts.append(name)
            if config.DRAW_CONF:
                label_parts.append(f"{float(l):.2f}")
            if label_parts:
                cv2.putText(
                    frame,
                    " ".join(label_parts),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    tuple(getattr(config, "TEXT_COLOR", (60, 255, 60))),
                    1,
                    cv2.LINE_AA,
                )

    lines: list[str] = []
    if config.DRAW_FPS:
        lines.append(
            "cam {cam_fps:.1f}  live {infer_fps:.1f}  inf {infer_ms:.1f}ms".format(
                cam_fps=float(metrics.get("cam_fps", 0.0) or 0.0),
                infer_fps=float(metrics.get("infer_fps", 0.0) or 0.0),
                infer_ms=float(metrics.get("infer_ms", 0.0) or 0.0),
            )
        )
    angle = metrics.get("hind_angle")
    if angle is not None:
        lines.append(f"Hind angle: {float(angle):.1f}")
    if config.DEBUG_OVERLAY:
        lines.append(
            "raw {raw_visible}/{tracked_points}  filt {filtered_visible}/{tracked_points}  "
            "triplet {triplet}  drops {source_drops}".format(
                raw_visible=int(metrics.get("raw_visible", 0) or 0),
                filtered_visible=int(metrics.get("filtered_visible", 0) or 0),
                tracked_points=int(metrics.get("tracked_points", 0) or 0),
                triplet="yes" if bool(metrics.get("triplet", False)) else "no",
                source_drops=int(metrics.get("source_drops", 0) or 0),
            )
        )

    y = 24
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
    return frame


def open_video_writer(frame: np.ndarray, source: FrameSource) -> cv2.VideoWriter:
    output_path = Path(config.OUTPUT_VIDEO_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_h, out_w = frame.shape[:2]

    fps = float(getattr(config, "OUTPUT_VIDEO_FPS", 0.0))
    if fps <= 0:
        fps = source.nominal_fps()
    if fps <= 0:
        fps = float(getattr(config, "VIDEO_TARGET_FPS", 30.0))

    fourcc = cv2.VideoWriter_fourcc(*str(getattr(config, "OUTPUT_VIDEO_CODEC", "mp4v")))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")
    return writer


def write_benchmark_row(csv_path: Path, header_written: bool, row: list[object]) -> bool:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not header_written:
            writer.writerow(
                [
                    "frame_id",
                    "timestamp",
                    "cam_fps",
                    "infer_fps",
                    "infer_ms",
                    "raw_visible",
                    "filtered_visible",
                    "triplet",
                    "hind_angle",
                    "source_drops",
                ]
            )
        writer.writerow(row)
    return True


def main() -> None:
    validate_config()
    logger = setup_logger()

    source = build_frame_source()
    source.open()
    video_writer: cv2.VideoWriter | None = None
    dlc_live = None

    stats = defaultdict(float)
    csv_header_written = False
    prev_capture_ts: Optional[float] = None
    prev_infer_end_perf: Optional[float] = None
    visual_mode = bool(getattr(config, "DISPLAY_WINDOW", True))

    try:
        ok, first_packet = source.read()
        if not ok or first_packet is None:
            raise RuntimeError("Could not read the first frame from the selected source.")

        effective_cropping = normalize_cropping_for_frame(
            getattr(config, "CROPPING", None),
            first_packet.frame,
            logger,
        )
        dlc_live = build_dlc_live(effective_cropping)
        model_cfg = dlc_live.read_config()
        body_parts = extract_bodyparts(model_cfg)
        missing_points = [name for name in config.USE_POINTS if name not in body_parts]
        if missing_points:
            raise ValueError(f"USE_POINTS are missing from the exported model: {missing_points}")
        missing_angle_points = [name for name in config.HIND_ANGLE_POINTS if name not in body_parts]
        if missing_angle_points:
            raise ValueError(f"HIND_ANGLE_POINTS are missing from the exported model: {missing_angle_points}")

        bodypart_to_idx = {name: i for i, name in enumerate(body_parts)}
        processor = OnlinePoseProcessor(body_parts=body_parts, use_points=config.USE_POINTS)
        if bool(getattr(config, "ENABLE_PROCESSOR", True)):
            dlc_live.processor = processor

        logger.info(
            "Pipeline started. source=%s crop=%s resize=%s dynamic=%s precision=%s device=%s",
            type(source).__name__,
            effective_cropping,
            getattr(config, "RESIZE", 1.0),
            getattr(config, "DYNAMIC_CROPPING", (False, 0.5, 10)),
            getattr(config, "PRECISION", "FP32"),
            getattr(config, "DEVICE", None),
        )
        logger.info("Model bodyparts loaded: %d", len(body_parts))

        packet: Optional[FramePacket] = first_packet
        initialized = False
        while packet is not None:
            frame_id = int(packet.frame_id)
            frame = packet.frame
            capture_ts = float(packet.capture_ts)
            stats["frames"] += 1

            if prev_capture_ts is not None:
                dt_cam = capture_ts - prev_capture_ts
                if dt_cam > 0:
                    stats["cam_fps"] = 1.0 / dt_cam
            prev_capture_ts = capture_ts

            infer_start = time.perf_counter()
            if not initialized:
                pose = dlc_live.init_inference(frame, frame_id=frame_id, capture_ts=capture_ts)
                initialized = True
            else:
                pose = dlc_live.get_pose(frame, frame_id=frame_id, capture_ts=capture_ts)
            infer_end = time.perf_counter()

            infer_ms = (infer_end - infer_start) * 1000.0
            stats["infer_ms_sum"] += infer_ms
            if prev_infer_end_perf is not None:
                dt_inf = infer_end - prev_infer_end_perf
                if dt_inf > 0:
                    stats["infer_fps"] = 1.0 / dt_inf
            prev_infer_end_perf = infer_end

            raw_pose = processor.last_raw_pose if processor.last_raw_pose is not None else np.asarray(pose)
            filtered_pose = processor.last_filtered_pose if processor.last_filtered_pose is not None else np.asarray(pose)
            raw_points = pose_to_points(raw_pose, bodypart_to_idx, config.USE_POINTS)
            filtered_points = pose_to_points(filtered_pose, bodypart_to_idx, config.USE_POINTS)
            draw_points, has_triplet, reason = evaluate_triplet(filtered_points, config.USE_POINTS)
            hind_angle = compute_hind_angle(draw_points) if has_triplet else None

            raw_visible = int(
                processor.last_raw_visible
                if processor.last_raw_pose is not None
                else count_visible_points(raw_points)
            )
            filtered_visible = int(
                processor.last_filtered_visible
                if processor.last_filtered_pose is not None
                else count_visible_points(filtered_points)
            )
            stats["raw_visible_sum"] += raw_visible
            stats["filtered_visible_sum"] += filtered_visible
            stats["triplet_sum"] += 1 if has_triplet else 0

            metrics = {
                "cam_fps": stats.get("cam_fps", 0.0),
                "infer_fps": stats.get("infer_fps", 0.0),
                "infer_ms": infer_ms,
                "raw_visible": raw_visible,
                "filtered_visible": filtered_visible,
                "tracked_points": len(config.USE_POINTS),
                "triplet": has_triplet,
                "hind_angle": hind_angle,
                "source_drops": getattr(source, "dropped_total", 0),
            }

            display = frame.copy()
            display = draw_overlay(display, draw_points, metrics)
            if float(getattr(config, "SHOW_SCALE", 1.0)) != 1.0:
                display = cv2.resize(
                    display,
                    None,
                    fx=float(config.SHOW_SCALE),
                    fy=float(config.SHOW_SCALE),
                    interpolation=cv2.INTER_AREA,
                )

            if bool(getattr(config, "SAVE_OUTPUT_VIDEO", False)):
                if video_writer is None:
                    video_writer = open_video_writer(display, source)
                video_writer.write(display)

            key = -1
            if visual_mode:
                cv2.imshow(config.WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF

            if frame_id % max(1, int(config.LOG_EVERY_N_FRAMES)) == 0:
                status_text = ",".join(
                    f"{name}:{processor.last_point_status.get(name, 'na')}" for name in config.USE_POINTS
                )
                logger.info(
                    "frame=%d raw=%d/%d filt=%d/%d triplet=%s angle=%s "
                    "cam_fps=%.1f live_fps=%.1f infer=%.1fms drops=%d status={%s} reason=%s",
                    frame_id,
                    raw_visible,
                    len(config.USE_POINTS),
                    filtered_visible,
                    len(config.USE_POINTS),
                    has_triplet,
                    f"{hind_angle:.1f}" if hind_angle is not None else "None",
                    stats.get("cam_fps", 0.0),
                    stats.get("infer_fps", 0.0),
                    infer_ms,
                    getattr(source, "dropped_total", 0),
                    status_text,
                    reason,
                )

                if bool(getattr(config, "ENABLE_BENCHMARK_CSV", True)):
                    csv_header_written = write_benchmark_row(
                        Path(config.BENCHMARK_CSV_PATH),
                        csv_header_written,
                        [
                            frame_id,
                            f"{time.time():.3f}",
                            f"{stats.get('cam_fps', 0.0):.2f}",
                            f"{stats.get('infer_fps', 0.0):.2f}",
                            f"{infer_ms:.2f}",
                            raw_visible,
                            filtered_visible,
                            int(has_triplet),
                            f"{hind_angle:.2f}" if hind_angle is not None else "",
                            getattr(source, "dropped_total", 0),
                        ],
                    )

            if key in (27, ord("q")):
                logger.info("Exit requested.")
                break

            ok, packet = source.read()
            if not ok:
                packet = None

    finally:
        source.release()
        if video_writer is not None:
            video_writer.release()
        if dlc_live is not None:
            try:
                dlc_live.close()
            except Exception:
                pass
        if visual_mode:
            cv2.destroyAllWindows()

        total_frames = max(1.0, stats["frames"])
        logger.info(
            "Finished. frames=%d triplet_rate=%.1f%% avg_infer=%.1fms avg_raw=%.2f avg_filt=%.2f drops=%d",
            int(stats["frames"]),
            100.0 * stats["triplet_sum"] / total_frames,
            stats["infer_ms_sum"] / total_frames,
            stats["raw_visible_sum"] / total_frames,
            stats["filtered_visible_sum"] / total_frames,
            int(getattr(source, "dropped_total", 0)),
        )


if __name__ == "__main__":
    main()
