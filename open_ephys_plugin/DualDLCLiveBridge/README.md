# Dual DLCLive Bridge for Open Ephys

`Dual DLCLive Bridge` is an Open Ephys processor that receives dual-camera
DLCLive pose packets over UDP and emits Open Ephys TTL state changes.

The plugin does not open cameras and does not run DLCLive. Cameras and neural
network inference are handled by Python:

```text
dual_rt_dlc_live.py -> UDP 127.0.0.1:47000 -> Dual DLCLive Bridge -> Dual DLCLive TTL
```

## Navigation

| Section | Use it for |
| --- | --- |
| Current Role | What the plugin owns in production. |
| UI Parameters | Every Open Ephys parameter and default. |
| TTL Lines | Output line mapping and stimulation use. |
| Packet Formats | Binary, JSON pose and legacy TTL input. |
| Filtering and Angle Logic | How raw points become valid triplets and triggers. |
| Status Line | How to read plugin UI status. |
| Build and Install | Rebuild and smoke-test the DLL. |
| Diagnostics | UDP test, ACK and common failures. |

## Current Role

In production `pose` mode, Python sends raw pose points and metadata. The plugin
computes:

- online point filtering;
- left/right triplet validity;
- selected side/triplet score;
- hind angle at ankle;
- angle threshold triggers;
- refractory gating;
- TTL state word;
- Open Ephys event-channel updates.

The plugin supports three input paths:

| Input | Status | Use case |
| --- | --- | --- |
| `DDLP` binary pose v1 | Production default | Lowest Python allocation overhead. |
| JSON `dual_dlc_live.pose.v1` | Supported fallback | Debugging and custom point names. |
| JSON `dual_dlc_live.v1` with `ttl_lines` | Legacy compatibility | Old Python-computed TTL mode. |

## Installation Layout

Open Ephys tree:

```text
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
  Plugins\DualDLCLiveBridge
  out\build\x64-Debug\plugins\DualDLCLiveBridge.dll
```

Repository source:

```text
C:\tmp\Dual_DLC_live_plugin\open_ephys_plugin\DualDLCLiveBridge
```

The plugin folder should be included from Open Ephys `Plugins/CMakeLists.txt`:

```cmake
add_subdirectory(DualDLCLiveBridge)
```

## UI Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Opens/closes the UDP listener. |
| `udp_port` | `47000` | Local UDP port for Python packets. |
| `angle_trigger_enabled` | `false` | Enables TTL lines `2` and `3`. |
| `angle_threshold_deg` | `55.0` | Hind angle threshold for angle triggers. |
| `conf_thresh_use` | `0.20` | Likelihood gate before filtering. |
| `conf_thresh_draw` | `0.15` | Likelihood gate for accepting a visible triplet. |
| `use_filter` | `true` | Enables the online point filter. |
| `enable_pcutoff` | `true` | Drops points below `conf_thresh_use`. |
| `enable_despike` | `true` | Rejects sudden implausible point jumps. |
| `despike_threshold_px` | `150.0` | Maximum accepted point jump before rejection. |
| `despike_reset_gap_frames` | `15` | Frame gap after which reacquisition is allowed. |
| `median_window` | `3` | Median smoothing window in frames. |
| `enable_hold` | `false` | Holds the last good point for short dropouts. |
| `max_hold_frames` | `20` | Maximum frames to hold the last good point. |
| `refractory_ms` | `0` | Minimum delay between angle-trigger rising edges. |

Production note: `angle_trigger_enabled` is disabled by default. Enable it in
the UI when lines `2` and `3` should drive stimulation.

## TTL Lines

The plugin emits one Open Ephys event channel:

```text
Dual DLCLive TTL
```

Line mapping:

| Line | Meaning | Typical downstream use |
| --- | --- | --- |
| `0` | Left valid selected hip/ankle/toes triplet. | Quality gate. |
| `1` | Right valid selected hip/ankle/toes triplet. | Quality gate. |
| `2` | Left angle trigger. | Left stimulation rising edge. |
| `3` | Right angle trigger. | Right stimulation rising edge. |
| `4..7` | Reserved. | Future conditions. |

Angle trigger condition:

```text
angle_trigger_enabled
and side has a valid triplet
and angle_deg <= angle_threshold_deg
and refractory_ms allows a new rising edge
```

The current TTL word is a bit mask:

```text
line0 -> 0x01
line1 -> 0x02
line2 -> 0x04
line3 -> 0x08
```

Examples:

| Active lines | TTL word |
| --- | --- |
| none | `0x00` |
| `0`, `1` | `0x03` |
| `0`, `1`, `2` | `0x07` |
| `1`, `3` | `0x0A` |

## Filtering and Angle Logic

For each side, the plugin receives six raw points:

```text
hl_ankle_l
hl_ankle_r
hl_hip_l
hl_hip_r
hl_toes_l
hl_toes_r
```

Default triplets:

```text
left:  hl_hip_l, hl_ankle_l, hl_toes_l
right: hl_hip_r, hl_ankle_r, hl_toes_r
```

Processing order:

1. Read raw points from binary or JSON input.
2. Apply likelihood cutoff if `enable_pcutoff = true`.
3. Apply despike rejection if enabled.
4. Apply median smoothing over `median_window`.
5. Optionally hold last good point when `enable_hold = true`.
6. Score left/right triplets by visible point count and likelihood sum.
7. Pick the better triplet.
8. Accept the triplet if hip, ankle and toes pass `conf_thresh_draw`.
9. Compute angle at ankle.
10. Emit TTL validity and optional angle trigger lines.

The angle is computed at the ankle from:

```text
hip -> ankle -> toes
```

## Packet Formats

### Binary Pose: `DDLP` v1

Production default:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
```

Properties:

```text
magic: DDLP
version: 1
endianness: little-endian
point order: fixed six-point DUAL_USE_POINTS order
```

High-level layout:

```text
packet header
left frame metadata
left 6 * [x, y, likelihood] float32
right frame metadata
right 6 * [x, y, likelihood] float32
```

Binary mode is compact and avoids Python `raw_points` dictionary creation. Use
JSON if point names or point count need to change.

### JSON Pose: `dual_dlc_live.pose.v1`

Supported fallback:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

JSON pose packets carry explicit point names:

```json
{
  "schema": "dual_dlc_live.pose.v1",
  "pair_index": 123,
  "tracked_points": ["hl_ankle_l", "..."],
  "side_point_sets": {
    "left": ["hl_hip_l", "hl_ankle_l", "hl_toes_l"],
    "right": ["hl_hip_r", "hl_ankle_r", "hl_toes_r"]
  },
  "left": {
    "frame_id": 123,
    "raw_points": {
      "hl_hip_l": {"x": 1.0, "y": 2.0, "likelihood": 0.9}
    }
  },
  "right": {}
}
```

Use JSON pose when debugging or when custom point names are needed.

### Legacy TTL: `dual_dlc_live.v1`

Legacy mode:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "ttl"
```

Python computes `ttl_lines` and the plugin forwards those states:

```json
{
  "schema": "dual_dlc_live.v1",
  "pair_index": 123,
  "ttl_lines": [true, true, false, false, false, false, false, false]
}
```

This mode remains for compatibility. It is not the production path.

## ACK Behavior

Normal live Python does not request ACK:

```python
DUAL_OE_BRIDGE_REQUEST_ACK = False
```

The synthetic sender can request ACK:

```powershell
python send_dual_dlc_bridge_test.py --wait-ack
```

ACK examples:

```text
dual_dlc_live.ack pair=5 mode=binary ttl=0x03 left_angle=135.00 right_angle=135.00
dual_dlc_live.ack pair=5 mode=pose ttl=0x03 left_angle=135.00 right_angle=135.00
dual_dlc_live.ack pair=5 mode=ttl ttl=0x03
```

## Status Line

The plugin UI status text:

```text
pkts 128 | mode bin | pair 128 | ttl 0x03 | L 135.0 | R 134.8 | q 0 | age 12ms
```

Fields:

| Field | Meaning |
| --- | --- |
| `pkts` | Number of accepted UDP packets. |
| `mode` | `bin`, `pose`, `ttl` or `-`. |
| `pair` | Last Python `pair_index`. |
| `ttl` | Current TTL bit mask. |
| `L` | Last left angle in degrees, or `-`. |
| `R` | Last right angle in degrees, or `-`. |
| `q` | Pending TTL word queue length. |
| `age` | Milliseconds since last accepted packet. |

Expected production mode is `bin`.

## Build and Install

Close Open Ephys before rebuilding. Windows can keep the DLL locked while the
GUI is open.

Build:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
cmd.exe /s /c "`"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`" -arch=x64 && cmake --build out\build\x64-Debug --target DualDLCLiveBridge --config Debug"
```

Smoke-test exports:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
python -B Plugins\DualDLCLiveBridge\check_plugin_load.py
```

Expected:

```text
PLUGIN_MAP_OK
EXPORT_OK getLibInfo
EXPORT_OK getPluginInfo
PLUGIN_EXPORTS_OK
```

Repository DLL artifact:

```text
dist/windows-x64-debug/DualDLCLiveBridge.dll
```

## Synthetic Tests

Run with Open Ephys open and the bridge enabled.

Binary production path:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
```

JSON pose fallback:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format json --count 5 --interval 0.025 --wait-ack
```

Legacy TTL:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode ttl --count 5 --interval 0.025 --wait-ack
```

Expected ending:

```text
acked 5/5
```

## Downstream Stimulation

This plugin emits Open Ephys events. Physical pulses are produced by a
downstream stimulation/output processor.

Recommended downstream mapping:

| Purpose | Event channel | Line | Trigger |
| --- | --- | --- | --- |
| Left stimulation | `Dual DLCLive TTL` | `2` | Rising edge |
| Right stimulation | `Dual DLCLive TTL` | `3` | Rising edge |
| Left validity gate | `Dual DLCLive TTL` | `0` | State/gate |
| Right validity gate | `Dual DLCLive TTL` | `1` | State/gate |

If you need repeated pulse trains while a condition remains true, implement the
pulse train in the downstream stimulation processor or add a dedicated pulse
mode to this bridge. The bridge currently emits TTL state changes.

## Troubleshooting

### Build fails with `LNK1168`

Open Ephys is probably still holding `DualDLCLiveBridge.dll`.

Fix:

1. Close Open Ephys.
2. Confirm no `open-ephys.exe` process is running.
3. Rebuild.

### `missing ack`

Check:

- Open Ephys is running.
- The signal chain contains `Dual DLCLive Bridge`.
- `enabled = true`.
- `udp_port = 47000`.
- Python sender uses the same port.
- DLL was rebuilt after source changes.

### `pkts` does not increase

Check UDP listener:

```powershell
netstat -ano -p udp | Select-String ':47000'
```

Run the synthetic test. If synthetic test works but live Python does not, the
problem is on the Python/camera/model side.

### `ttl` never reaches lines `2` or `3`

Check:

- `angle_trigger_enabled = true`;
- `angle_threshold_deg` is appropriate;
- valid points are reaching lines `0` and `1`;
- `conf_thresh_draw` is not too strict;
- angles shown in `L` and `R` cross the threshold;
- `refractory_ms` is not suppressing expected triggers.

### TTL events exist but stimulation does not fire

Check downstream processor:

- event channel is `Dual DLCLive TTL`;
- trigger line is `2` or `3`;
- trigger mode is rising edge;
- output hardware is enabled and connected;
- Open Ephys acquisition/processing is running.

## Maintenance Checklist

When changing the bridge protocol:

1. Update Python sender and runtime.
2. Update C++ parser.
3. Update synthetic tests.
4. Rebuild the plugin.
5. Run `check_plugin_load.py`.
6. Run binary, JSON and legacy synthetic UDP tests.
7. Update root README, Python README and this plugin README.
