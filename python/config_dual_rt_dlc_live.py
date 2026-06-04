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
DUAL_DISPLAY_WINDOW = True
DUAL_SHOW_SCALE = 0.5
DUAL_PROCESS_EVERY_N_PAIRS = 1
DUAL_PAIR_WAIT_TIMEOUT_MS = 2000
DUAL_READER_SLEEP_MS = 1.0
DUAL_LOG_EVERY_N_PAIRS = 120

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
DUAL_OE_BRIDGE_REQUEST_ACK = False
DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG = None
