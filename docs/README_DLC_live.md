# DLC Live Runtime

`rt_dlc_live.py` is a DLCLive-based realtime pipeline for low-latency pose inference.

## Current architecture

- DLCLive handles model loading, preprocessing, and coordinate restoration.
- `rt_dlc_live.py` handles source I/O, overlay, logging, benchmark CSV, and video save.
- Online filtering is enabled through a DLCLive-compatible processor:
  - `pcutoff` gate
  - `despike`
  - optional `hold`
  - median smoothing

## Run

```bash
conda activate dlc_live_env
python rt_dlc_live.py
```

## Required packages

- `numpy<2`
- `opencv-python`
- `deeplabcut-live[pytorch]`
- `colorcet`

Example install:

```bash
pip install "numpy<2" opencv-python==4.11.0.86 colorcet
pip install deeplabcut-live[pytorch] --no-deps
```

## Config overview (`config_rt_dlc_live.py`)

Frame source:
- `USE_VIDEO_FILE`, `VIDEO_FILE_PATH`, `CAM_INDEX`
- `VIDEO_TARGET_FPS`, `VIDEO_SKIP_IF_BEHIND`
- `FRAME_W`, `FRAME_H`, `TARGET_VIDEO_FPS`

Model:
- `MODEL_PATH`, `MODEL_TYPE`, `PRECISION`, `DEVICE`
- `SINGLE_ANIMAL`, `CONVERT_TO_RGB`

DLCLive preprocessing:
- `CROPPING` (`[x1, x2, y1, y2]` or `None`)
- `RESIZE`
- `DYNAMIC_CROPPING`

Points and filtering:
- `USE_POINTS`
- `ENABLE_PROCESSOR`, `ENABLE_PCUTOFF`, `ENABLE_DESPIKE`, `ENABLE_HOLD`
- `CONF_THRESH_USE`, `CONF_THRESH_DRAW`
- `DESPIKE_THRESHOLD_PX`, `DESPIKE_RESET_GAP_FRAMES`
- `MAX_HOLD_FRAMES`, `MEDIAN_WINDOW`

Overlay and angle:
- `DRAW_POINTS`, `DRAW_NAMES`, `DRAW_CONF`, `DRAW_FPS`, `DEBUG_OVERLAY`
- `COMPUTE_HIND_ANGLE`, `HIND_ANGLE_POINTS`

Output:
- `SAVE_OUTPUT_VIDEO`, `OUTPUT_VIDEO_PATH`, `OUTPUT_VIDEO_FPS`, `OUTPUT_VIDEO_CODEC`

Logging:
- `LOG_PATH`, `LOG_LEVEL`, `LOG_EVERY_N_FRAMES`
- `ENABLE_BENCHMARK_CSV`, `BENCHMARK_CSV_PATH`

## Speed and timeline behavior

- `VIDEO_TARGET_FPS = 0.0` means "no artificial pacing for file input".
- `OUTPUT_VIDEO_FPS = 0.0` means "use source FPS when available".
- Keep `VIDEO_SKIP_IF_BEHIND = False` for offline analysis/export to avoid timeline distortion.
- Enable `VIDEO_SKIP_IF_BEHIND = True` only when you prefer lower visual latency over full frame preservation.

## Environment variable overrides

- `DLC_LIVE_VIDEO_PATH`
- `DLC_LIVE_MODEL_PATH`

