from __future__ import annotations

from pathlib import Path

from config_rt_dlc_live import *  # noqa: F401,F403


# ============================================================================
# Dual Galaxy camera source. Configs from Galaxy SDK.
# ============================================================================
DUAL_CAMERAS = [
    {
        "name": "left",
        "sn": "FDE22070174",
        "config_path": Path(r"C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt"),
    },
    {
        "name": "right",
        "sn": "FDE22070175",
        "config_path": Path(r"C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt"),
    },
]

DUAL_IMPORT_CONFIG = True
DUAL_CONFIG_VERIFY = False
DUAL_FALLBACK_APPLY_CONFIG = True
DUAL_FORCE_TRIGGER_OFF = False
DUAL_FRAME_TIMEOUT_MS = 1000

# Keep these on for live work: old frames are intentionally dropped.
DUAL_LOW_LATENCY = True
DUAL_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
DUAL_ACQUISITION_BUFFER_COUNT = 2
DUAL_DRAIN_QUEUED_FRAMES = True
DUAL_MAX_DRAIN_FRAMES = 20

# The camera .txt configs already crop to 1920x220 via hardware ROI.
CROPPING = None
RESIZE = 1.0
DYNAMIC_CROPPING = (False, 0.5, 10)

# Galaxy SDK already converts Bayer frames to RGB. In dual fast-mode we keep the
# frame RGB and disable DLCLive's BGR->RGB conversion to avoid a redundant CPU
# color shuffle. Single-camera configs keep their original BGR/OpenCV behavior.
GALAXY_OUTPUT_COLOR = "rgb"
CONVERT_TO_RGB = False


# ============================================================================
# Tracked points
# ============================================================================
DUAL_SIDE_POINT_SETS = {
    "left": ("hl_hip_l", "hl_ankle_l", "hl_toes_l"),
    "right": ("hl_hip_r", "hl_ankle_r", "hl_toes_r"),
}
DUAL_AUTO_PICK_SIDE = True
DUAL_DEFAULT_CAMERA_SIDE = {
    "left": "left",
    "right": "right",
}
DUAL_USE_POINTS = sorted({point for points in DUAL_SIDE_POINT_SETS.values() for point in points})


# ============================================================================
# Runtime/display
# ============================================================================
DUAL_WINDOW_NAME = "DLC Live dual Galaxy"
# Working stimulation mode should not draw OpenCV windows: display/resize/overlay
# can add tens of milliseconds per pair and does not affect UDP pose output.
DUAL_DISPLAY_WINDOW = False
DUAL_SHOW_SCALE = 0.5
DUAL_PROCESS_EVERY_N_PAIRS = 1
DUAL_PAIR_WAIT_TIMEOUT_MS = 2000
DUAL_READER_SLEEP_MS = 1.0
DUAL_LOG_EVERY_N_PAIRS = 120
DUAL_ENABLE_STAGE_PROFILER = True
DUAL_PROFILE_LOG_EVERY_N_PAIRS = 120
DUAL_PROFILE_EMA_ALPHA = 0.10

DUAL_SAVE_OUTPUT_VIDEO = False
DUAL_OUTPUT_LEFT_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_left.mp4")
DUAL_OUTPUT_RIGHT_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_right.mp4")
DUAL_OUTPUT_VIDEO_CODEC = "mp4v"


# ============================================================================
# Logging
# ============================================================================
LOG_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_debug.log")
BENCHMARK_CSV_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_benchmark.csv")
ENABLE_BENCHMARK_CSV = False


# ============================================================================
# Open Ephys bridge
# ============================================================================
# UDP packets are consumed by the DualDLCLiveBridge Open Ephys plugin.
# Packet modes:
#   "pose" - send raw DLCLive pose points; plugin computes validity, angle and TTL.
#   "ttl"  - legacy mode; Python computes ttl_lines itself.
#
# Plugin TTL lines:
#   0 - left pose triplet is valid
#   1 - right pose triplet is valid
#   2 - left hind angle trigger, if enabled in plugin
#   3 - right hind angle trigger, if enabled in plugin
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS = 1
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"  # "binary" or "json"; binary is used only with packet_mode="pose".
DUAL_OE_BRIDGE_REQUEST_ACK = False
DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG = None


# ============================================================================
# Optimization switches
# ============================================================================
# Optional CPU throttling. The measured working default leaves OpenCV/PyTorch
# threadpools untouched for better result_hz. Set these to 1 only if CPU load is
# more important than maximum throughput.
DUAL_CV2_NUM_THREADS = 1
DUAL_TORCH_NUM_THREADS = 12
DUAL_TORCH_INTEROP_THREADS = 12
DUAL_TORCH_CUDNN_BENCHMARK = True
DUAL_TORCH_ALLOW_TF32 = True

# torch.compile backend applied once to runner.model after init (single/sequential
# inference path via run_raw_inference). Empty string = off (plain eager, default).
# "cudagraphs" replays the identical eager kernels => bit-identical poses
# (accuracy-neutral, safe for the closed-loop trigger) with lower per-frame overhead.
# Any other backend (e.g. "inductor") changes kernels and MUST pass a pose/TTL
# accuracy gate before stimulation use. On compile or compiled-inference failure the
# runtime logs a warning and falls back to eager automatically.
DUAL_TORCH_COMPILE_BACKEND = ""

# In the normal plugin-driven mode Python sends only raw pose points. Filtering,
# side selection, angle calculation, refractory logic and TTL generation are done
# in the Open Ephys plugin. Disable this only when you need Python-side angle
# diagnostics or the legacy ttl_lines packet mode.
DUAL_FAST_POSE_ONLY = True

# If DLCLive uses the supported PyTorch runner path, the two camera frames are
# inferred as one mini-batch after the first warm-up frame. Unsupported model
# layouts automatically fall back to the sequential path.
DUAL_ENABLE_BATCH_INFERENCE = True
DUAL_BATCH_FALLBACK_TO_SEQUENTIAL = True
