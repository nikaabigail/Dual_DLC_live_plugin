# Python Dual DLCLive Runbook

This document describes the Python side of the dual-camera DLCLive system:

```text
dual_rt_dlc_live.py
  -> two Daheng/Galaxy cameras
  -> DLCLive/PyTorch pose inference
  -> binary or JSON UDP packets
  -> Open Ephys "Dual DLCLive Bridge"
```

For Open Ephys plugin parameters and TTL behavior, see:

```text
open_ephys_plugin/DualDLCLiveBridge/README.md
```

## Navigation

| Section | Use it for |
| --- | --- |
| Current Runtime Contract | What Python is responsible for now. |
| Launch Checklist | Minimal steps before a real run. |
| Camera Configuration | Left/right serials, Galaxy configs and ROI. |
| DLCLive Model | Model path, GPU mode and preprocessing. |
| Open Ephys Bridge | UDP modes, binary fast path and JSON fallback. |
| Stage Profiler | Where latency and CPU time are measured. |
| Running Tests Without Cameras | Synthetic packet checks. |
| Running With Cameras | Actual experiment startup order. |
| Logs and Diagnostics | What to inspect while debugging. |
| Troubleshooting | Common failures and fixes. |

## Current Runtime Contract

Default production configuration:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
DUAL_FAST_POSE_ONLY = True
DUAL_ENABLE_BATCH_INFERENCE = True
DUAL_BATCH_FALLBACK_TO_SEQUENTIAL = True
DUAL_ENABLE_STAGE_PROFILER = True
```

Python currently does:

- opens two Daheng/Galaxy USB3 cameras;
- imports the Galaxy `.txt` camera configs;
- keeps low-latency camera buffers by dropping stale frames;
- pairs latest left/right frames;
- runs DLCLive/PyTorch inference;
- sends raw pose points plus metadata to Open Ephys over UDP;
- optionally shows a local OpenCV diagnostic overlay;
- logs stage timings for performance diagnosis.

Python does not compute production TTL decisions in the default mode. Filtering,
side/triplet selection, hind angle, refractory logic and TTL output are computed
inside the Open Ephys C++ plugin.

## Launch Checklist

Before running `dual_rt_dlc_live.py`:

1. Connect both Daheng cameras through the USB3 hub.
2. Close GalaxyView.
3. Start Open Ephys.
4. Add `Dual DLCLive Bridge` to the signal chain.
5. Set bridge `enabled = true`.
6. Confirm `udp_port = 47000`.
7. Run synthetic UDP test and confirm `acked 5/5`.
8. Start Open Ephys acquisition/processing if TTL events must reach downstream processors.
9. Start `dual_rt_dlc_live.py`.

## Files

Runtime directory on the experiment machine:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation
  config_rt_dlc_live.py
  rt_dlc_live.py
  config_dual_rt_dlc_live.py
  dual_rt_dlc_live.py
  send_dual_dlc_bridge_test.py
  README_OPEN_EPHYS_BRIDGE.md
```

Repository source:

```text
C:\tmp\Dual_DLC_live_plugin\python
```

## Camera Configuration

Current dual camera mapping:

| Side | Serial | Galaxy config |
| --- | --- | --- |
| `left` | `FDE22070174` | `C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt` |
| `right` | `FDE22070175` | `C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt` |

Configured in `config_dual_rt_dlc_live.py`:

```python
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
```

Camera-side ROI comes from those Galaxy config files. Python does not add a
second software crop:

```python
CROPPING = None
RESIZE = 1.0
DYNAMIC_CROPPING = (False, 0.5, 10)
```

Low-latency camera settings:

```python
DUAL_LOW_LATENCY = True
DUAL_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
DUAL_ACQUISITION_BUFFER_COUNT = 2
DUAL_DRAIN_QUEUED_FRAMES = True
DUAL_MAX_DRAIN_FRAMES = 20
```

These settings intentionally prefer fresh frames over preserving every camera
frame. That is the correct behavior for live stimulation.

## DLCLive Model

`config_dual_rt_dlc_live.py` imports base model settings from
`config_rt_dlc_live.py`.

Important model settings:

```python
MODEL_TYPE = "pytorch"
PRECISION = "FP32"
DEVICE = "cuda"
SINGLE_ANIMAL = True
CONVERT_TO_RGB = True
```

The model path is configured in `config_rt_dlc_live.py`:

```python
MODEL_PATH = r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\..."
```

The script logs a CUDA/model device check at startup. If the model loads on CPU,
check the active environment, PyTorch CUDA installation and `DEVICE`.

## Tracked Points

Current point groups:

```python
DUAL_SIDE_POINT_SETS = {
    "left": ("hl_hip_l", "hl_ankle_l", "hl_toes_l"),
    "right": ("hl_hip_r", "hl_ankle_r", "hl_toes_r"),
}
```

`DUAL_USE_POINTS` is sorted:

```text
hl_ankle_l
hl_ankle_r
hl_hip_l
hl_hip_r
hl_toes_l
hl_toes_r
```

This fixed order is required for binary transport. If a future model uses
different point names, switch to JSON transport:

```python
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

## Open Ephys Bridge

Default bridge settings:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS = 1
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
DUAL_OE_BRIDGE_REQUEST_ACK = False
```

Supported modes:

| Packet mode | Wire format | Meaning |
| --- | --- | --- |
| `pose` | `binary` | Production default. Raw pose points are sent as `DDLP` binary packets. |
| `pose` | `json` | Debug/custom-point fallback. Raw pose points are sent as `dual_dlc_live.pose.v1`. |
| `ttl` | JSON | Legacy path. Python computes `ttl_lines` before sending. |

In production, keep:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
```

## Binary Fast Path

In binary fast mode, Python stores selected points as:

```text
raw_pose_array.shape == (6, 3)
raw_pose_array.dtype == float32
columns = x, y, likelihood
rows = DUAL_USE_POINTS order
```

Binary send packs directly from this array. It does not build:

```python
raw_points: dict[str, dict[str, float | None]]
```

`raw_points` is created only when:

- JSON fallback is used;
- local OpenCV overlay needs point dictionaries.

This reduces Python allocation overhead in the normal stimulation path.

## Binary Packet Summary

Binary pose packets use:

```text
magic: DDLP
version: 1
byte order: little-endian
```

High-level packet structure:

```text
header
left side block
left six [x, y, likelihood] points
right side block
right six [x, y, likelihood] points
```

The C++ plugin expects exactly the current six points in the configured order.
JSON fallback should be used for custom point sets because JSON carries point
names explicitly.

## Batch Inference

Batch inference is enabled by default:

```python
DUAL_ENABLE_BATCH_INFERENCE = True
DUAL_BATCH_FALLBACK_TO_SEQUENTIAL = True
```

Behavior:

- the first frame pair warms up DLCLive/model initialization;
- after warm-up, compatible PyTorch runner paths infer left and right as one
  mini-batch;
- unsupported paths automatically fall back to sequential `get_pose` calls.

Batch is skipped if the runner uses unsupported detector/dynamic-cropping paths.
The fallback is intentional and should not stop an experiment.

## Stage Profiler

Profiler settings:

```python
DUAL_ENABLE_STAGE_PROFILER = True
DUAL_PROFILE_LOG_EVERY_N_PAIRS = 120
DUAL_PROFILE_EMA_ALPHA = 0.10
```

Output goes to:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_debug.log
```

Example line:

```text
stage_profile pair=120 last_ms camera/read=0.42 preprocess=1.30 inference=8.70 pack/send=0.06 display=3.10 | avg_ms camera/read=...
```

Stage meanings:

| Stage | Meaning |
| --- | --- |
| `camera/read` | Successful camera frame read calls in reader threads. |
| `preprocess` | DLCLive `process_frame` for the current left/right pair. |
| `inference` | Model/runner inference for the current left/right pair. |
| `pack/send` | Binary or JSON packet construction and UDP socket send. |
| `display` | Local overlay, text, resize, video writer and `imshow`. |

Interpretation:

- High `camera/read`: camera SDK wait time, USB, trigger or frame timeout.
- High `preprocess`: RGB conversion, crop/resize or CPU transform cost.
- High `inference`: model/GPU path or GPU synchronization.
- High `pack/send`: serialization or socket issue.
- High `display`: OpenCV overlay/window/video writing is consuming CPU.

For lowest CPU usage during experiments:

```python
DUAL_DISPLAY_WINDOW = False
DUAL_SAVE_OUTPUT_VIDEO = False
```

Then `display` should remain close to zero.

## Running Tests Without Cameras

Open Ephys must be running and `Dual DLCLive Bridge` must be enabled.

Binary pose test:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
```

Expected ending:

```text
acked 5/5
```

JSON pose fallback:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format json --count 5 --interval 0.025 --wait-ack
```

Legacy TTL:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode ttl --count 5 --interval 0.025 --wait-ack
```

Use `--check-ack-ttl` when you want the sender to verify the returned TTL word.
Remember that angle trigger lines `2` and `3` are disabled unless enabled in
the plugin UI.

## Running With Cameras

1. Close GalaxyView.
2. Start Open Ephys.
3. Add and enable `Dual DLCLive Bridge`.
4. Start acquisition/processing in Open Ephys if downstream TTL processing is
   required.
5. Start Python:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

Optional environment activation:

```powershell
& C:\dlc_live_env\Scripts\Activate.ps1
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
python dual_rt_dlc_live.py
```

## Expected Runtime Signals

Python log should show:

- both cameras opened with the expected serial numbers;
- first frame pair received;
- CUDA availability and GPU name;
- bodypart count and `DUAL_USE_POINTS`;
- periodic pair logs;
- periodic `stage_profile` lines.

Open Ephys plugin UI should show:

```text
pkts increasing
mode bin
pair increasing
ttl 0x..
L/R angle values when valid
q usually near 0
age small when packets are fresh
```

## Output for Stimulation

The Python process does not directly stimulate. It sends pose packets to the
C++ plugin. The plugin creates Open Ephys TTL events:

```text
Dual DLCLive TTL
```

Typical downstream setup:

| Use | Line | Trigger |
| --- | --- | --- |
| Left stimulation | `2` | Rising edge |
| Right stimulation | `3` | Rising edge |
| Left validity gate | `0` | State/gate |
| Right validity gate | `1` | State/gate |

Set `angle_trigger_enabled` and `angle_threshold_deg` in the plugin UI, not in
Python, for production `pose` mode.

## Logs and CSV

Main log:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_debug.log
```

Optional benchmark CSV:

```python
ENABLE_BENCHMARK_CSV = True
BENCHMARK_CSV_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_benchmark.csv")
```

CSV is disabled by default in the dual runtime because file I/O can add CPU
overhead during live work.

## Display and Video

Local OpenCV windows:

```python
DUAL_DISPLAY_WINDOW = True
DUAL_SHOW_SCALE = 0.5
```

Output video:

```python
DUAL_SAVE_OUTPUT_VIDEO = False
DUAL_OUTPUT_LEFT_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_left.mp4")
DUAL_OUTPUT_RIGHT_PATH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live_right.mp4")
```

For lowest CPU usage, set both display and video save to `False`.

## Troubleshooting

### Camera import/config error

Check:

- GalaxyView is closed.
- Camera serials match `DUAL_CAMERAS`.
- `.txt` config files exist.
- USB3 hub is stable.
- No other program is using the cameras.

### Open Ephys does not receive packets

Check:

```powershell
netstat -ano -p udp | Select-String ':47000'
```

Then run the synthetic test:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --count 5 --wait-ack
```

If synthetic packets work, inspect live Python logs for camera/model errors and
confirm:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
```

### GPU is not loaded enough

Check stage profiler first. If `preprocess` or `display` is high, CPU work is
the bottleneck. If `inference` is high and GPU utilization is low, inspect:

- PyTorch CUDA availability;
- active environment `C:\dlc_live_env`;
- model runner compatibility with batch path;
- CPU thread contention from OpenCV/display.

### `display` is high

Use:

```python
DUAL_DISPLAY_WINDOW = False
DUAL_SAVE_OUTPUT_VIDEO = False
```

### Need custom bodyparts

For custom points, use JSON while updating the plugin:

```python
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

Binary mode validates the fixed six-point order and will reject incompatible
`DUAL_USE_POINTS`.

## Maintenance Notes

When changing the protocol:

1. Update `dual_rt_dlc_live.py`.
2. Update `send_dual_dlc_bridge_test.py`.
3. Update `open_ephys_plugin/DualDLCLiveBridge`.
4. Update this README and the plugin README.
5. Run `py_compile`.
6. Run synthetic binary and JSON tests.
7. Rebuild the plugin if C++ changed.
