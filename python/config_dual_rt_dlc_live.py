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
DUAL_FORCE_TRIGGER_OFF = False  # Real experiments respect the camera .txt Line2 trigger. Set True only for bench/free-run.
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

# Single-camera mode: one side camera sees only ONE hind leg at a time (the
# rat's near flank), even though the model emits both hl_*_l and hl_*_r (the
# occluded leg is a low-confidence guess). SINGLE_AUTO_PICK_SIDE (default) routes
# the pose to whichever side's triplet is actually present, so the plugin reports
# exactly the visible leg — L when the left flank shows, R when the rat turns and
# the right flank shows. Set False to force the fixed SINGLE_PLUGIN_SIDE.
SINGLE_AUTO_PICK_SIDE = True
# Opt-in only for a camera that genuinely sees BOTH legs at once (e.g. a top or
# rear view): feed the full pose to both packet sides so the plugin reports L and
# R simultaneously. WRONG for a single side camera (it would report both legs
# from one visible flank, as L==R).
SINGLE_EMIT_BOTH_LEGS = False

# --- Dynamic leg ROI (single-camera) -----------------------------------------
# A fixed-WIDTH sliding window that follows the hind legs, so DLCLive runs on a
# small crop (default 448 px wide, full stripe height) instead of the whole
# 1920 px frame -> faster inference and lower closed-loop latency. Fixed size
# keeps cudagraphs working (constant input shape); we crop (not resize) so the
# leg scale is unchanged and accuracy is preserved.
#   - Window centre = mean X of whatever hind-leg points are visible (>=THRESH).
#     Works with 3, 2, or even 1 point, EMA-smoothed -> a 1-point dropout does
#     NOT move/widen the window.
#   - The crop is ALWAYS a fixed width (never the full frame). On loss the window
#     first HOLDS at the last leg position for LEG_ROI_HOLD_FRAMES frames (~1 s):
#     a turn or side-switch makes the legs vanish briefly then reappear near the
#     same X, so holding re-locks instantly instead of thrashing into a scan.
#     Only after that hold does it SWEEP the window across the frame to re-acquire
#     (covers the stripe in ~fw/width frames). Keeping the input shape constant is
#     required so cudagraphs / torch.compile stay valid -- a full-frame fallback
#     would flip the shape (256<->1920) and force a revert to ~2x slower eager.
# Assumes base CROPPING is None (the camera already hardware-crops to the stripe).
LEG_ROI_ENABLED = True
LEG_ROI_WIDTH = 256            # px, fixed window width
LEG_ROI_DETECT_THRESH = 0.30   # min likelihood for a hind-leg point to anchor the centre
LEG_ROI_HOLD_FRAMES = 100      # ~1 s @100fps: HOLD at the last leg position this long (rides out turns/
                               # occlusions where legs vanish then reappear near the same X) before sweeping
LEG_ROI_CENTER_EMA = 0.35      # window-centre smoothing (0..1; higher = follows faster)


# --- Parallel recording (single-camera): raw video + keypoints ---------------
# Records the RAW camera video (no overlay) and/or the per-frame keypoints in a
# background thread, so it does NOT add latency to the camera->pose->UDP loop.
# Keypoints are FULL-FRAME coords (the sliding-ROI offset is already restored);
# the ROI window [x1,x2] the model saw is stored per frame too, so crop-local
# coords for fine-tuning are just (x - roi_x1). Files share a timestamped stem.
SINGLE_RECORD_ENABLED = True           # master switch
SINGLE_RECORD_DIR = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\recordings")
SINGLE_RECORD_VIDEO = True              # write the raw camera video (no points)
SINGLE_RECORD_VIDEO_CODEC = "FFV1"      # fourcc; FFV1=lossless .avi (best for DLC re-labeling). "MJPG"=lighter/lossy.
                                        # The recorder auto-selects the .avi extension for FFV1/MJPG/XVID.
SINGLE_RECORD_KEYPOINTS = True          # write per-frame keypoints (full-frame coords)
SINGLE_KP_FORMAT = "csv"             # "binary" (fast .dlckp + scripts/kp_to_csv.py) or "csv"
SINGLE_RECORD_QUEUE = 128               # frames buffered to the writer thread before dropping

# Dual-camera recording: writes BOTH cameras (two files each, dual_<ts>_left.* and
# dual_<ts>_right.*), each on its own background thread. Reuses the SINGLE_RECORD_*
# params above (dir, video on/off, codec, keypoints on/off, format, queue).
DUAL_RECORD_ENABLED = True


# ============================================================================
# Runtime/display
# ============================================================================
DUAL_WINDOW_NAME = "DLC Live dual Galaxy"
# Working stimulation mode should not draw OpenCV windows: display/resize/overlay
# can add tens of milliseconds per pair and does not affect UDP pose output.
DUAL_DISPLAY_WINDOW = False  # Stimulation runs draw no OpenCV windows. Set True only for setup/debug (two overlays).
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


# --- Phase trigger: стимул в заданный ПРОЦЕНТ шага -------------------------
# Соглашение: опора 0..100%, перенос 100..200%. Процент считается проекцией
# текущей точки носка на усреднённую за последние N шагов траекторию
# относительно iliac; событий в рантайме не требует.
#
# ВАЖНО: работает только при DUAL_OE_BRIDGE_PACKET_MODE = "ttl". Плагин
# пересобирает TTL-слово из каждого пакета, и в бинарном pose-режиме состояния
# линий вычисляет он сам, поэтому бит от Python будет погашен следующим кадром.
PHASE_TRIGGER_ENABLED = False
PHASE_TRIGGER_FPS = 100.0        # ОБЯЗАН совпадать с частотой камеры из
                                 # config_daheng: от неё считаются все
                                 # выдержки детектора и темп фазы
PHASE_TRIGGER_LEG = "r"          # "l" или "r"; автовыбор не поддержан - эталон
                                 # строится для конкретной ноги
PHASE_TRIGGER_TARGET_PCT = 145.0  # куда целимся. Раньше ~137% попасть нельзя:
                                 # это первые 39 мс переноса, они уходят на
                                 # инференс и детектор. Лучшая зона 130..160%
PHASE_TRIGGER_TTL_LINE = 4       # 0..3 заняты has_triplet и angle-триггером
PHASE_TRIGGER_LATENCY_MS = 28.0  # на столько экстраполируем фазу вперёд:
                                 # инференс ~18 мс + период кадра 10 мс
PHASE_TRIGGER_HOLD_FRAMES = 2    # сколько кадров держать линию поднятой
PHASE_TRIGGER_REF_CYCLES = 10    # эталон насыщается на 10 циклах (~4 с ходьбы)
PHASE_TRIGGER_MIN_GAP_PCT = 100.0  # рефрактерность: полцикла между импульсами
PHASE_TRIGGER_EVENT_LAG = 1      # на столько кадров детектор опаздывает с
                                 # событием; без компенсации смещение +9.5%
PHASE_TRIGGER_MED_WIN = 1        # НЕ сглаживать: каждый кадр медианы добавляет
                                 # свой лаг, p90 растёт 9.8 -> 17.6 -> 24.0%
PHASE_TRIGGER_BOUT_STD_PX = 8.0  # порог признака движения: поджиг закрыт, пока
                                 # разброс носка относительно тела ниже него
