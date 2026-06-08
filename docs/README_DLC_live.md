# Single-Camera DLCLive Runtime Reference

This document covers the older single-camera runtime:

```text
rt_dlc_live.py
config_rt_dlc_live.py
```

The production dual-camera Open Ephys path uses:

```text
dual_rt_dlc_live.py
config_dual_rt_dlc_live.py
Dual DLCLive Bridge
```

Use this document for single-camera debugging, model checks, offline video
testing or as a reference for shared DLCLive settings.

## Navigation

| Section | Use it for |
| --- | --- |
| Role | What the single-camera runtime does. |
| Configuration | Main settings in `config_rt_dlc_live.py`. |
| Camera and Video Sources | Galaxy, OpenCV and file input. |
| DLCLive Model | Model path, GPU and preprocessing. |
| Online Processor | Python-side filter and angle logic. |
| Running | How to start the runtime. |
| Logs and CSV | Output files and diagnostics. |
| Troubleshooting | Common single-camera issues. |

## Role

`rt_dlc_live.py` is a DLCLive-based realtime pipeline for one camera or one
video file.

It handles:

- frame source I/O;
- DLCLive model setup;
- optional Galaxy SDK camera config import;
- Python-side online filtering;
- hind-angle calculation;
- OpenCV overlay;
- benchmark CSV and output video.

Unlike the production dual Open Ephys path, this single-camera runtime computes
its own filtered points and angle in Python.

## Configuration File

Main config:

```text
python/config_rt_dlc_live.py
```

The dual config imports this file and overrides selected values:

```python
from config_rt_dlc_live import *
```

Changing shared settings in `config_rt_dlc_live.py` can affect both single and
dual runtimes unless the dual config overrides them.

## Camera and Video Sources

Choose source:

```python
USE_VIDEO_FILE = False
CAMERA_BACKEND = "galaxy"  # "galaxy" or "opencv"
```

Video-file settings:

```python
VIDEO_FILE_PATH = r"C:\dlc\videos\..."
VIDEO_TARGET_FPS = 0.0
VIDEO_SKIP_IF_BEHIND = False
```

OpenCV fallback:

```python
CAM_INDEX = 1
FRAME_W = 1920
FRAME_H = 1080
TARGET_VIDEO_FPS = 100.0
```

Galaxy source:

```python
GALAXY_SDK_ROOT = Path(r"C:\Program Files\Daheng Imaging\GalaxySDK")
GALAXY_SN = "FDE22070173"
GALAXY_INDEX = 1
GALAXY_CONFIG_PATH = Path(r"C:\config_daheng\Rat_TREDMILL_Top_1920px_340px_100Hz_(FDE22070173).txt")
GALAXY_IMPORT_CONFIG = True
GALAXY_CONFIG_VERIFY = False
GALAXY_FALLBACK_APPLY_CONFIG = True
GALAXY_FRAME_TIMEOUT_MS = 1000
```

Low-latency Galaxy behavior:

```python
GALAXY_LOW_LATENCY = True
GALAXY_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
GALAXY_ACQUISITION_BUFFER_COUNT = 2
GALAXY_DRAIN_QUEUED_FRAMES = True
GALAXY_MAX_DRAIN_FRAMES = 20
```

For live behavior, stale frames are intentionally dropped. For offline video
analysis, keep `VIDEO_SKIP_IF_BEHIND = False` to preserve the timeline.

## DLCLive Model

Model settings:

```python
MODEL_PATH = r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\..."
MODEL_TYPE = "pytorch"
PRECISION = "FP32"
DEVICE = "cuda"
SINGLE_ANIMAL = True
CONVERT_TO_RGB = True
```

DLCLive preprocessing:

```python
CROPPING = None
RESIZE = 1.0
DYNAMIC_CROPPING = (False, 0.5, 10)
```

`CROPPING` follows the DLCLive convention:

```text
[x1, x2, y1, y2]
```

Leave it as `None` when the camera or video is already cropped to the needed
stripe.

## Points and Online Processor

Single-camera points:

```python
USE_POINTS = [
    "hl_hip_l",
    "hl_ankle_l",
    "hl_toes_l",
]
```

Processor settings:

```python
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
```

The processor runs after DLCLive inference. It stores the last raw and filtered
pose arrays and is used by the overlay, angle calculation and CSV output.

## Overlay and Angle

Display settings:

```python
WINDOW_NAME = "DLC Live realtime"
DISPLAY_WINDOW = True
SHOW_SCALE = 0.8
```

Drawing settings:

```python
DRAW_POINTS = True
DRAW_NAMES = True
DRAW_CONF = True
DRAW_FPS = True
DEBUG_OVERLAY = True
```

Angle settings:

```python
COMPUTE_HIND_ANGLE = True
HIND_ANGLE_POINTS = ("hl_hip_l", "hl_ankle_l", "hl_toes_l")
```

## Output Files

Video output:

```python
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_live_output.mp4")
OUTPUT_VIDEO_FPS = 0.0
OUTPUT_VIDEO_CODEC = "mp4v"
```

Log and CSV:

```python
LOG_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_live_debug.log")
LOG_LEVEL = "INFO"
LOG_EVERY_N_FRAMES = 30

BENCHMARK_CSV_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\rt_dlc_live_benchmark.csv")
ENABLE_BENCHMARK_CSV = True
```

`OUTPUT_VIDEO_FPS = 0.0` means the source FPS is used when available.

## Environment Overrides

Supported environment variables:

```text
DLC_LIVE_VIDEO_PATH
DLC_LIVE_MODEL_PATH
DLC_LIVE_CAMERA_BACKEND
DLC_LIVE_GALAXY_SDK_ROOT
DLC_LIVE_GALAXY_SN
DLC_LIVE_GALAXY_INDEX
DLC_LIVE_GALAXY_CONFIG_PATH
```

These are useful for quick tests without editing the config file.

## Running

Activate environment:

```powershell
& C:\dlc_live_env\Scripts\Activate.ps1
```

Run from the project directory:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
python rt_dlc_live.py
```

Or run directly:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe rt_dlc_live.py
```

## Expected Runtime Signals

The log should show:

- selected frame source;
- camera or video open success;
- DLCLive/model initialization;
- bodypart extraction;
- frame/inference FPS;
- visible point count;
- hind angle when valid.

## Troubleshooting

### DLCLive import fails with `colorcet`

Install the missing dependency in the active environment:

```powershell
C:\dlc_live_env\Scripts\python.exe -m pip install colorcet
```

### Camera does not open

Check:

- GalaxyView is closed;
- the serial number or camera index is correct;
- Galaxy SDK path exists;
- config file exists;
- camera is not held by another process.

### GPU is not used

Check:

- `DEVICE = "cuda"`;
- PyTorch in `C:\dlc_live_env` has CUDA support;
- the exported model is PyTorch;
- startup log reports CUDA available.

### Overlay is slow

Disable drawing features:

```python
DRAW_NAMES = False
DRAW_CONF = False
DEBUG_OVERLAY = False
```

Or disable display:

```python
DISPLAY_WINDOW = False
```

### Offline video timeline changes

For offline/export workflows:

```python
VIDEO_SKIP_IF_BEHIND = False
```

Enable frame skipping only when low visual latency matters more than preserving
every video frame.

## Relation to the Dual Runtime

Use `rt_dlc_live.py` for:

- single-camera debugging;
- checking model loading;
- checking DLCLive preprocessing;
- testing overlay and filtering;
- offline/video-file experiments.

Use `dual_rt_dlc_live.py` for:

- two-camera live experiments;
- Open Ephys bridge integration;
- production stimulation path;
- binary UDP pose transport;
- plugin-side TTL generation.
