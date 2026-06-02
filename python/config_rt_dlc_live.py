"""
DLCLive-centric configuration for rt_dlc_live.py.

This config follows the official DLCLive architecture:
  - DLCLive handles model loading, cropping, resizing, and coordinate restoration.
  - The runtime script handles frame capture, overlay, logging, and video writing.
  - A custom processor performs online filtering after DLCLive inference.
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# Frame source
# ============================================================================
USE_VIDEO_FILE = False  # Set to True to use a video file instead of a camera feed.
VIDEO_FILE_PATH = os.getenv(
    "DLC_LIVE_VIDEO_PATH",
    r"C:\dlc\videos\1_MER2-230-168U3C(FDE22070174)_20240604_152156.avi",
)
VIDEO_TARGET_FPS = 0.0  # 0.0 means use the source video FPS.
VIDEO_SKIP_IF_BEHIND = False  # Keep False for file analysis to avoid timeline distortion.

# Camera backend for live capture: "galaxy" for Daheng Galaxy SDK, "opencv" for cv2.VideoCapture.
CAMERA_BACKEND = os.getenv("DLC_LIVE_CAMERA_BACKEND", "galaxy").lower()

CAM_INDEX = 1  # OpenCV fallback camera index. Used only when CAMERA_BACKEND = "opencv".
FRAME_W = 1920
FRAME_H = 1080
TARGET_VIDEO_FPS = 100.0

# Daheng Galaxy SDK camera source. GalaxyView should be closed or not acquiring while this runs.
GALAXY_SDK_ROOT = Path(
    os.getenv("DLC_LIVE_GALAXY_SDK_ROOT", r"C:\Program Files\Daheng Imaging\GalaxySDK")
)
GALAXY_SN = os.getenv("DLC_LIVE_GALAXY_SN", "FDE22070173")
GALAXY_INDEX = int(os.getenv("DLC_LIVE_GALAXY_INDEX", "1"))
GALAXY_CONFIG_PATH = Path(
    os.getenv(
        "DLC_LIVE_GALAXY_CONFIG_PATH",
        r"C:\config_daheng\Rat_TREDMILL_Top_1920px_340px_100Hz_(FDE22070173).txt",
    )
)
GALAXY_IMPORT_CONFIG = True
GALAXY_CONFIG_VERIFY = False
GALAXY_FALLBACK_APPLY_CONFIG = True
GALAXY_FRAME_TIMEOUT_MS = 1000

# Low-latency mode drops stale queued frames and keeps the newest camera frame.
GALAXY_LOW_LATENCY = True
GALAXY_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
GALAXY_ACQUISITION_BUFFER_COUNT = 2
GALAXY_DRAIN_QUEUED_FRAMES = True
GALAXY_MAX_DRAIN_FRAMES = 20

# Set True for testing without external Line2 trigger. False preserves the GalaxyView config.
GALAXY_FORCE_TRIGGER_OFF = False

# ============================================================================
# DLCLive model
# ============================================================================
MODEL_PATH = os.getenv(
    "DLC_LIVE_MODEL_PATH",
    r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch"
    r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5"
    r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt",
)
MODEL_TYPE = "pytorch"
PRECISION = "FP32"
DEVICE = "cuda"
SINGLE_ANIMAL = True
CONVERT_TO_RGB = True

# ============================================================================
# Official DLCLive preprocessing
# ============================================================================
# CROPPING uses the official DLCLive convention [x1, x2, y1, y2].
# Leave as None for already cropped stripe videos. For full OBS frames, for example:
# CROPPING = [0, 1920, 430, 649]
CROPPING = None

# Official DLCLive resize factor. 1.0 keeps the input size unchanged.
RESIZE = 1.0

# Official DLCLive dynamic cropping tuple: (enabled, detection_threshold, margin_px)
DYNAMIC_CROPPING = (False, 0.5, 10)

# ============================================================================
# Tracked points
# ============================================================================
USE_POINTS = [
    "hl_hip_l",
    "hl_ankle_l",
    "hl_toes_l",
]

# ============================================================================
# Post-inference processor settings
# ============================================================================
ENABLE_PROCESSOR = True
ENABLE_PCUTOFF = True
ENABLE_DESPIKE = True
ENABLE_HOLD = False

CONF_THRESH_USE = 0.20
CONF_THRESH_DRAW = 0.15
DESPIKE_THRESHOLD_PX = 150.0
DESPIKE_RESET_GAP_FRAMES = 15
MAX_HOLD_FRAMES = 20
MEDIAN_WINDOW = 3

# ============================================================================
# Overlay and feature computation
# ============================================================================
WINDOW_NAME = "DLC Live realtime"
DISPLAY_WINDOW = True
SHOW_SCALE = 0.8

DRAW_POINTS = True
DRAW_NAMES = True
DRAW_CONF = True
DRAW_FPS = True
DEBUG_OVERLAY = True
POINT_RADIUS = 4
POINT_COLOR = (0, 255, 0)
TEXT_COLOR = (60, 255, 60)

COMPUTE_HIND_ANGLE = True
HIND_ANGLE_POINTS = ("hl_hip_l", "hl_ankle_l", "hl_toes_l")

# ============================================================================
# Output video
# ============================================================================
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_live_output.mp4")
OUTPUT_VIDEO_FPS = 0.0  # 0.0 means use the source FPS when available.
OUTPUT_VIDEO_CODEC = "mp4v"

# ============================================================================
# Logging
# ============================================================================
LOG_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_live_debug.log")
LOG_LEVEL = "INFO"
LOG_EVERY_N_FRAMES = 30

BENCHMARK_CSV_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_live_benchmark.csv")
ENABLE_BENCHMARK_CSV = True
