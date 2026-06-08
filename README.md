# Dual DLC Live Plugin

Production-ready bridge for running dual-camera DLCLive inference and sending
pose-derived TTL events into Open Ephys.

This repository contains:

- Python runtime for two Daheng/Galaxy USB3 cameras;
- Open Ephys C++ processor `Dual DLCLive Bridge`;
- synthetic UDP test sender for checking the bridge without cameras;
- Windows debug DLL artifact;
- camera config snapshots and reference documentation.

## Documentation Map

| Document | Purpose |
| --- | --- |
| `README.md` | Project overview, quick start, architecture and operational checklist. |
| `python/README_OPEN_EPHYS_BRIDGE.md` | Detailed Python runtime runbook: cameras, DLCLive, binary UDP, profiler, logs and troubleshooting. |
| `open_ephys_plugin/DualDLCLiveBridge/README.md` | Detailed Open Ephys plugin reference: UI parameters, TTL lines, packet formats, build and diagnostics. |
| `docs/README_DLC_live.md` | Single-camera `rt_dlc_live.py` reference kept for legacy/debug work. |
| `camera_configs/*.txt` | Daheng Galaxy configuration snapshots for the left/right treadmill cameras. |

Recommended reading order:

1. Read this file.
2. Use `python/README_OPEN_EPHYS_BRIDGE.md` when running cameras and DLCLive.
3. Use `open_ephys_plugin/DualDLCLiveBridge/README.md` when configuring Open Ephys or stimulation.
4. Use `docs/README_DLC_live.md` only for the older single-camera runtime.

## Current Production Path

```text
Two Daheng USB3 cameras
  -> dual_rt_dlc_live.py
  -> DLCLive/PyTorch inference
  -> compact binary UDP packet (DDLP/v1)
  -> Open Ephys processor "Dual DLCLive Bridge"
  -> Open Ephys TTL event channel "Dual DLCLive TTL"
  -> downstream stimulation/output processor
```

The Python process owns camera acquisition and neural-network inference. The
Open Ephys plugin owns filtering, side/triplet selection, angle computation,
refractory logic and TTL state generation.

## Current Defaults

The production defaults are in `python/config_dual_rt_dlc_live.py`:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
DUAL_FAST_POSE_ONLY = True
DUAL_ENABLE_BATCH_INFERENCE = True
DUAL_BATCH_FALLBACK_TO_SEQUENTIAL = True
DUAL_ENABLE_STAGE_PROFILER = True
```

In this mode Python sends raw pose points only. It does not compute filters,
triplets, hind angles or stimulation TTL decisions for the normal Open Ephys
path.

## Cameras

The current dual camera mapping is:

| Side | Serial number | Config |
| --- | --- | --- |
| `left` | `FDE22070174` | `C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt` |
| `right` | `FDE22070175` | `C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt` |

The Galaxy `.txt` configs contain the native camera ROI. In Python, software
cropping is disabled for the dual runtime:

```python
CROPPING = None
RESIZE = 1.0
DYNAMIC_CROPPING = (False, 0.5, 10)
```

Keep GalaxyView closed before starting Python. GalaxyView can hold the cameras
open and prevent Galaxy SDK from opening them in `dual_rt_dlc_live.py`.

## Pose Points

The bridge currently tracks six points:

```python
DUAL_SIDE_POINT_SETS = {
    "left": ("hl_hip_l", "hl_ankle_l", "hl_toes_l"),
    "right": ("hl_hip_r", "hl_ankle_r", "hl_toes_r"),
}
DUAL_USE_POINTS = [
    "hl_ankle_l",
    "hl_ankle_r",
    "hl_hip_l",
    "hl_hip_r",
    "hl_toes_l",
    "hl_toes_r",
]
```

The binary protocol uses this fixed point order. If you change
`DUAL_USE_POINTS`, switch to JSON transport:

```python
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

## Binary Fast Path

In binary fast mode, Python stores the selected DLC output as one compact
NumPy array:

```text
shape: (6, 3)
dtype: float32
columns: x, y, likelihood
rows: DUAL_USE_POINTS order
```

The UDP packet is packed directly from this array. Python does not build a
`raw_points` dictionary during binary send. A dictionary is created lazily only
when JSON fallback is used or when the local OpenCV overlay needs point labels.

This keeps the stimulation path short:

```text
DLCLive pose ndarray -> raw_pose_array -> DDLP binary UDP -> C++ plugin -> TTL
```

## TTL Lines

The plugin emits Open Ephys TTL state changes on channel:

```text
Dual DLCLive TTL
```

Line mapping:

| Line | Meaning | Typical use |
| --- | --- | --- |
| `0` | Left camera has a valid selected hip/ankle/toes triplet. | Quality/gate signal. |
| `1` | Right camera has a valid selected hip/ankle/toes triplet. | Quality/gate signal. |
| `2` | Left hind angle trigger, if enabled in plugin UI. | Left stimulation rising edge. |
| `3` | Right hind angle trigger, if enabled in plugin UI. | Right stimulation rising edge. |
| `4..7` | Reserved. | Future conditions. |

Angle triggers are controlled in the plugin UI:

```text
angle_trigger_enabled = true
angle_threshold_deg = 55.0
refractory_ms = 0..60000
```

`DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` in Python is for legacy `ttl` packet mode
only. In production `pose` mode, the C++ plugin threshold is the source of
truth.

## Stage Profiler

The Python runtime logs rolling stage timings:

```python
DUAL_ENABLE_STAGE_PROFILER = True
DUAL_PROFILE_LOG_EVERY_N_PAIRS = 120
DUAL_PROFILE_EMA_ALPHA = 0.10
```

Example log line:

```text
stage_profile pair=120 last_ms camera/read=0.42 preprocess=1.30 inference=8.70 pack/send=0.06 display=3.10 | avg_ms camera/read=...
```

Stages:

| Stage | What it measures |
| --- | --- |
| `camera/read` | Successful Galaxy SDK frame reads in reader threads. |
| `preprocess` | DLCLive frame preprocessing for the current left/right pair. |
| `inference` | Runner/model inference path for the current left/right pair. |
| `pack/send` | UDP packet construction plus socket send. |
| `display` | OpenCV overlay, text, resize, video writer and `imshow`. |

If both `DUAL_DISPLAY_WINDOW = False` and `DUAL_SAVE_OUTPUT_VIDEO = False`,
overlay work is skipped and `display` should stay near zero.

## Quick Start

### 1. Start Open Ephys

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug
.\open-ephys.exe
```

Add processor:

```text
Dual DLCLive Bridge
```

Set:

```text
enabled = true
udp_port = 47000
```

For stimulation angle triggers, set in the plugin UI:

```text
angle_trigger_enabled = true
angle_threshold_deg = 55.0
```

### 2. Test UDP Without Cameras

Use this before touching the cameras:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
```

Expected ending:

```text
acked 5/5
```

Optional compatibility tests:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format json --count 5 --interval 0.025 --wait-ack
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode ttl --count 5 --interval 0.025 --wait-ack
```

### 3. Start Dual DLCLive

Close GalaxyView first.

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

Watch:

- Python console/log for camera and CUDA/model messages;
- Open Ephys bridge UI for increasing `pkts`, changing `pair`, fresh `age`;
- `dual_rt_dlc_live_debug.log` for `stage_profile` timing lines.

## Stimulation Setup

`Dual DLCLive Bridge` creates Open Ephys TTL events only. It does not directly
drive a physical stimulation device.

Downstream stimulation/output processor should listen to:

```text
Dual DLCLive TTL
```

Typical mapping:

| Stimulation side | Event line | Trigger mode |
| --- | --- | --- |
| Left | `2` | Rising edge |
| Right | `3` | Rising edge |

Use lines `0` and `1` as validity gates if the downstream processor supports
gating.

## Repository Layout

```text
camera_configs/
  Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt
  Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt

dist/windows-x64-debug/
  DualDLCLiveBridge.dll

docs/
  README_DLC_live.md

open_ephys_plugin/DualDLCLiveBridge/
  DualDLCLiveBridge.cpp
  DualDLCLiveBridge.h
  DualDLCLiveBridgeEditor.cpp
  CMakeLists.txt
  README.md

python/
  config_rt_dlc_live.py
  rt_dlc_live.py
  config_dual_rt_dlc_live.py
  dual_rt_dlc_live.py
  send_dual_dlc_bridge_test.py
  README_OPEN_EPHYS_BRIDGE.md
```

## Build Plugin

Close Open Ephys before rebuilding; otherwise Windows can keep
`DualDLCLiveBridge.dll` locked.

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
cmd.exe /s /c "`"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`" -arch=x64 && cmake --build out\build\x64-Debug --target DualDLCLiveBridge --config Debug"
```

Smoke-test DLL exports:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
python -B Plugins\DualDLCLiveBridge\check_plugin_load.py
```

Expected:

```text
PLUGIN_EXPORTS_OK
```

## Operational Checklist

Before an experiment:

1. Cameras are connected through the USB3 hub.
2. GalaxyView is closed.
3. `DUAL_CAMERAS` serial numbers match the physical cameras.
4. Galaxy `.txt` configs exist at the configured paths.
5. Open Ephys is running.
6. `Dual DLCLive Bridge` is in the signal chain and `enabled = true`.
7. `udp_port` matches `DUAL_OE_BRIDGE_PORT`.
8. UDP synthetic test returns `acked 5/5`.
9. Open Ephys acquisition/processing is running.
10. Downstream stimulation/output processor listens to `Dual DLCLive TTL`.
11. `dual_rt_dlc_live.py` is running.
12. Bridge UI shows increasing `pkts`, recent `age`, expected `ttl`.

## Troubleshooting

### `missing ack`

Check:

- Open Ephys is open.
- `Dual DLCLive Bridge` is in the signal chain.
- `enabled = true`.
- `udp_port = 47000`.
- Windows firewall or another process is not blocking UDP localhost.
- The DLL is the current build.

### `pkts` does not increase

Run:

```powershell
netstat -ano -p udp | Select-String ':47000'
```

Then run:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --count 5 --wait-ack
```

If the synthetic test works but live Python does not move `pkts`, inspect:

- `DUAL_OE_BRIDGE_ENABLED`;
- `DUAL_OE_BRIDGE_HOST`;
- `DUAL_OE_BRIDGE_PORT`;
- camera open errors;
- inference worker errors in `dual_rt_dlc_live_debug.log`.

### Cameras do not open

Check:

- GalaxyView is closed.
- Cameras are not held by another process.
- USB3 hub has enough bandwidth and power.
- Serial numbers match `FDE22070174` and `FDE22070175`.
- Config paths under `C:\config_daheng\...` exist.

### TTL exists but physical stimulation does not happen

The bridge emits Open Ephys TTL events. Physical output depends on the
downstream stimulation/output processor.

Check:

- event channel is `Dual DLCLive TTL`;
- trigger line is `2` or `3`;
- trigger mode is rising edge;
- acquisition/processing is running;
- physical output hardware is enabled and connected.

## Compatibility Modes

The current production mode is:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
```

Supported fallbacks:

| Mode | Use case |
| --- | --- |
| `pose` + `binary` | Production path, lowest allocation overhead. |
| `pose` + `json` | Debug/custom point names. |
| `ttl` + JSON | Legacy Python-computed TTL lines. |

Use legacy `ttl` only when specifically testing old behavior.
