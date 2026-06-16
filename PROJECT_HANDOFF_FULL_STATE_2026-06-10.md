# Project Handoff: DLC Live + DualDLCLiveBridge

Дата состояния: 2026-06-10, Europe/Moscow.

Этот файл предназначен для следующего агента. Он описывает текущее состояние проекта, реальные пути на машине, структуру папок, рабочие команды, архитектуру Python/Open Ephys и известные проблемы. Пользователь просил не делать commit/push до его явного подтверждения.

## Короткий Контекст

Проект делает realtime pose detection через DLCLive/PyTorch для Daheng/Galaxy USB3 камер и отправляет результат в Open Ephys plugin по UDP. Сейчас основная рабочая схема:

```text
Daheng Galaxy camera
  -> Python DLCLive runtime
  -> raw pose points [6, 3]
  -> UDP 127.0.0.1:47000, binary DDLP/v1
  -> Open Ephys plugin DualDLCLiveBridge
  -> C++ filtering / triplet / hind angle / TTL word
  -> Open Ephys event channel "Dual DLCLive TTL"
  -> downstream stimulation/output processor
```

На текущий момент реально используется single-camera режим по левой treadmill-камере `FDE22070174`. Правая камера `FDE22070175` физически подключена и имеет конфиг, но при проверках ее изображение не давало полезный treadmill/animal view: likelihood справа были около нуля. Поэтому для эксперимента сейчас безопаснее использовать `single-*` профили.

## Самые Важные Пути

| Назначение | Путь |
| --- | --- |
| Рабочая Python-папка | `C:\dlc\DLC_OBS_Spinal_cord_stimulation` |
| Python env | `C:\dlc_live_env\Scripts\python.exe` |
| User-home shim | `C:\Users\Владимир\run_live_profile.py` |
| GitHub/release repo copy | `C:\tmp\Dual_DLC_live_plugin` |
| Open Ephys source tree | `C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main` |
| Installed plugin source in Open Ephys tree | `C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\Plugins\DualDLCLiveBridge` |
| Built plugin DLL | `C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug\Plugins\DualDLCLiveBridge.dll` |
| Open Ephys exe | `C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug\open-ephys.exe` |
| Daheng configs | `C:\config_daheng` |
| DLCLive model | `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt` |

## Текущий Runtime Status

Проверено 2026-06-10 перед созданием этого файла:

- активный `single_rt_dlc_live_bridge.py` или `dual_rt_dlc_live.py` Python-процесс не обнаружен;
- активный `open-ephys.exe` процесс этим фильтром не обнаружен;
- `py_compile` основных Python-файлов прошел без ошибок;
- `run_live_profile.py --list` работает через shim из `C:\Users\Владимир`;
- live camera не запускалась заново после создания этого handoff-файла.

Проверка компиляции была такой:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe -m py_compile live_profiles.py run_live_profile.py single_rt_dlc_live_bridge.py dual_rt_dlc_live.py rt_dlc_live.py send_dual_dlc_bridge_test.py config_dual_rt_dlc_live.py config_rt_dlc_live.py
```

## Структура Рабочей Папки Python

Корень: `C:\dlc\DLC_OBS_Spinal_cord_stimulation`

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation
|-- .git\
|-- __pycache__\                         # generated Python cache
|-- debug_snapshots\                      # diagnostic snapshots from camera checks
|   |-- left_FDE22070174_snapshot.png
|   |-- right_FDE22070175_snapshot.png
|   |-- right_roi_scan.png
|   |-- right_offset_0.png
|   |-- right_offset_110.png
|   |-- right_offset_220.png
|   |-- right_offset_330.png
|   |-- right_offset_440.png
|   |-- right_offset_550.png
|   |-- right_offset_660.png
|   |-- right_offset_770.png
|   |-- right_offset_880.png
|   |-- right_offset_980.png
|-- .gitignore
|-- check_dlc_dataset.py
|-- check_dlc_shuffles.py
|-- check_online_buffering.py
|-- config_dual_rt_dlc_live.py            # dual/single bridge camera + UDP config
|-- config_rt_dlc.py                      # older non-DLCLive realtime config
|-- config_rt_dlc_live.py                 # base DLCLive config and model path
|-- dual_rt_dlc_live.py                   # two-camera runtime and shared bridge helpers
|-- dual_rt_dlc_live_benchmark.csv        # generated/diagnostic
|-- dual_rt_dlc_live_debug.log            # generated/diagnostic
|-- dual_rt_dlc_obs.py                    # older OBS/video dual script
|-- HISTORY_DLC_LIVE.md
|-- live_profiles.py                      # current launch profiles and overrides
|-- PROJECT_HANDOFF_FULL_STATE_2026-06-10.md
|-- PROJECT_REVIEW.md
|-- README.md
|-- README_DLC_live.md
|-- README_LIVE_PROFILES.md               # user-facing launch docs
|-- README_OPEN_EPHYS_BRIDGE.md
|-- requirements.txt
|-- rt_dlc_benchmark.csv                  # generated/diagnostic
|-- rt_dlc_config_gui.py
|-- rt_dlc_debug.log                      # generated/diagnostic
|-- rt_dlc_live.py                        # main reusable DLCLive/Galaxy utilities
|-- rt_dlc_live_benchmark.csv             # generated/diagnostic
|-- rt_dlc_live_debug.log                 # generated/diagnostic
|-- rt_dlc_live_output.mp4                # generated/diagnostic
|-- rt_dlc_obs.py                         # older OBS/video script
|-- rt_dlc_output.mp4                     # generated/diagnostic
|-- run_dlc.py
|-- run_live_profile.py                   # interactive/direct profile launcher
|-- run_one_eval.py
|-- send_dual_dlc_bridge_test.py          # synthetic UDP sender for plugin tests
|-- single_rt_dlc_live_bridge.py          # current one-camera runtime for plugin
|-- single_rt_dlc_live_bridge_debug.log   # generated/diagnostic
```

Generated files/logs/videos should not be treated as clean source unless the user explicitly asks to preserve them. Do not delete them without asking, because they contain useful experiment diagnostics.

## Структура Repo Copy

Корень: `C:\tmp\Dual_DLC_live_plugin`

```text
C:\tmp\Dual_DLC_live_plugin
|-- .git\
|-- .gitignore
|-- README.md
|-- camera_configs\
|   |-- Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt
|   |-- Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt
|-- dist\
|   |-- windows-x64-debug\
|       |-- DualDLCLiveBridge.dll
|-- docs\
|   |-- README_DLC_live.md
|-- open_ephys_plugin\
|   |-- DualDLCLiveBridge\
|       |-- CMakeLists.txt
|       |-- DualDLCLiveBridge.cpp
|       |-- DualDLCLiveBridge.h
|       |-- DualDLCLiveBridgeEditor.cpp
|       |-- DualDLCLiveBridgeEditor.h
|       |-- OpenEphysLib.cpp
|       |-- README.md
|       |-- check_plugin_load.py
|-- python\
    |-- config_dual_rt_dlc_live.py
    |-- config_rt_dlc_live.py
    |-- dual_rt_dlc_live.py
    |-- live_profiles.py
    |-- README_LIVE_PROFILES.md
    |-- README_OPEN_EPHYS_BRIDGE.md
    |-- rt_dlc_live.py
    |-- run_live_profile.py
    |-- send_dual_dlc_bridge_test.py
    |-- single_rt_dlc_live_bridge.py
```

На 2026-06-10 repo copy не закоммичена:

```text
## main...origin/main
 M python/dual_rt_dlc_live.py
?? python/README_LIVE_PROFILES.md
?? python/live_profiles.py
?? python/run_live_profile.py
?? python/single_rt_dlc_live_bridge.py
```

Пользователь явно сказал: пока тестируем, в git не пушить до тех пор, пока он не поймет, что результат хороший.

## Структура Open Ephys Plugin Source

Корень: `C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\Plugins\DualDLCLiveBridge`

```text
DualDLCLiveBridge
|-- CMakeLists.txt
|-- DualDLCLiveBridge.cpp
|-- DualDLCLiveBridge.h
|-- DualDLCLiveBridgeEditor.cpp
|-- DualDLCLiveBridgeEditor.h
|-- OpenEphysLib.cpp
|-- README.md
|-- check_plugin_load.py
```

Проверенная DLL:

```text
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug\Plugins\DualDLCLiveBridge.dll
size: 385024 bytes
LastWriteTime: 04.06.2026 18:55:16
```

## Камеры И Daheng/Galaxy Configs

Рабочие treadmill configs лежат в `C:\config_daheng`:

```text
Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt
Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt
```

Также рядом есть старые/альтернативные configs:

```text
Rat_TREDMILL_ANGLE15_Left_1920px_808px_100Hz_(FDE22070174).txt
Rat_TREDMILL_ANGLE15_Right_1920px_808px_100Hz_(FDE22070175).txt
Rat_TREDMILL_Top_1920px_340px_100Hz_(FDE22070173).txt
Rat_TREDMILL_WDS_Right_1200px_220px_100Hz_(FDE22070175).txt
Rat_TREDMILL_WDS_Top_1600px_340px_200Hz_(FDE22070173).txt
```

Current config in `config_dual_rt_dlc_live.py`:

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

Observed native values from logs:

| Camera | Serial | Width | Height | OffsetY | Exposure | FPS | Trigger |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| left | `FDE22070174` | `1920` | `220` | `510` | `8000` | `100` | `Off` |
| right | `FDE22070175` | `1920` | `220` | `530` | `4000` | `100` | `Off` |

Important Daheng/Galaxy behavior:

- Galaxy SDK cannot open a camera if GalaxyView or another Python process already owns it.
- Typical error when camera is busy: `DeviceManager.open_device_by_sn ... TL Error:The device has been open`.
- The launcher now checks for existing live Python processes and suggests `--replace` or `--stop`.
- Camera config import can throw U3V ack errors if the device is busy or the hub/camera state is unstable.

## Model И DLCLive Base Config

Base file: `config_rt_dlc_live.py`.

Model:

```python
MODEL_PATH = r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt"
MODEL_TYPE = "pytorch"
DEVICE = "cuda"
SINGLE_ANIMAL = True
PRECISION = "FP32"
```

Base preprocessing:

```python
CROPPING = None
RESIZE = 1.0
DYNAMIC_CROPPING = (False, 0.5, 10)
```

Base thresholds:

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
COMPUTE_HIND_ANGLE = True
HIND_ANGLE_POINTS = ("hl_hip_l", "hl_ankle_l", "hl_toes_l")
```

Note: in the current optimized plugin-driven mode, Python mostly sends raw pose; C++ plugin does filtering/TTL.

## Current Profile Launcher

Primary entry point:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py
```

The shim file `C:\Users\Владимир\run_live_profile.py` only redirects into the real project:

```python
WORK_DIR = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation")
sys.path.insert(0, str(WORK_DIR))
os.chdir(WORK_DIR)
runpy.run_path(str(WORK_DIR / "run_live_profile.py"), run_name="__main__")
```

Available profiles on 2026-06-10:

| # | Profile | Target | Purpose |
| ---: | --- | --- | --- |
| 1 | `single-best` | single | Recommended. One left camera, RGB no-convert, FP32+TF32. |
| 2 | `single-strict` | single | Strict FP32, TF32 off. Slower, more conservative numeric check. |
| 3 | `single-fp16` | single | FP16 experiment. Validate coordinates before stimulation. |
| 4 | `single-cpu` | single | Lower CPU profile, FP32+TF32 with smaller Torch thread pools. |
| 5 | `single-debug` | single | Visual debug, more frequent logging. |
| 6 | `single-rgb-on` | single | BGR input + DLCLive RGB conversion fallback. |
| 7 | `dual-best` | dual | Two cameras, RGB no-convert, FP32+TF32. Not recommended until right view is fixed. |
| 8 | `dual-cpu` | dual | Two cameras, CPU-balanced. |
| 9 | `dual-fp16` | dual | Two cameras, FP16 experiment. |

Preview state:

- `live_profiles.py` currently sets `SINGLE_DISPLAY_WINDOW=True` and `DUAL_DISPLAY_WINDOW=True` in profile settings so all profiles show video/points by default.
- This was requested so the user can inspect detections.
- For final stimulation/headless performance, run with `--no-display`.
- Base `config_dual_rt_dlc_live.py` still says `DUAL_DISPLAY_WINDOW=False`; profile settings override this at runtime.

Important commands:

```powershell
# list only; does not launch anything
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --list

# interactive menu; enter 1, 2, 3...
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py

# recommended current preview run
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --replace

# same mode but no OpenCV window for stimulation/performance
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --no-display --replace

# inspect whether a live Python owns the camera
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --status

# stop current live Python
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --stop

# dry run only
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --dry-run
```

Important UI note: if user runs `--list`, then types `1`, PowerShell only echoes `1`; it will not launch a profile. To select by number, run without `--list` or pass the number directly:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py 1
```

## Current Recommended Runtime

For the current physical setup:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --replace
```

This launches:

```text
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\single_rt_dlc_live_bridge.py --profile single-best
```

Runtime behavior:

- opens only left camera `FDE22070174`;
- sends it into plugin side `left`;
- right side of the binary packet is filled with `NaN` points;
- Open Ephys plugin therefore should keep right-side TTL lines inactive;
- current default opens OpenCV preview with overlay points;
- press `q` or `Esc` in preview window to stop.

For real stimulation:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --no-display --replace
```

## Python Runtime Files

### `rt_dlc_live.py`

Reusable single-camera/DLCLive utility layer:

- imports Galaxy SDK `gxipy`;
- manages Daheng camera open/import-config/read/release;
- supports low-latency frame handling (`NEWEST_ONLY`, small buffers, drain queued frames);
- builds DLCLive runner with model path, cropping, resize, precision;
- contains overlay drawing and old Python-side online filtering utilities;
- used by both `single_rt_dlc_live_bridge.py` and `dual_rt_dlc_live.py`.

### `dual_rt_dlc_live.py`

Two-camera runtime plus shared bridge helpers:

- defines binary `DDLP` packet format;
- defines `OpenEphysBridge`;
- defines `StageProfiler`;
- opens two camera reader threads;
- pairs latest left/right packets;
- runs DLCLive inference, with optional batch path;
- packs raw pose arrays into UDP;
- can show dual preview windows;
- can still do legacy JSON/TTL paths for debugging.

Important internals:

```python
BINARY_POSE_MAGIC = b"DDLP"
BINARY_POSE_VERSION = 1
BINARY_POSE_POINT_NAMES = [
    "hl_ankle_l",
    "hl_ankle_r",
    "hl_hip_l",
    "hl_hip_r",
    "hl_toes_l",
    "hl_toes_r",
]
```

Binary mode requires `DUAL_USE_POINTS` to match this exact order/list. If custom points are needed, switch to JSON pose mode.

### `single_rt_dlc_live_bridge.py`

Current most useful runtime:

- imports `config_dual_rt_dlc_live.py`;
- applies `live_profiles.py`;
- opens one selected camera (`left` by default);
- runs DLCLive;
- creates a fake inactive side with `NaN`;
- sends a normal dual-format `DDLP` packet to the plugin;
- shows preview if `SINGLE_DISPLAY_WINDOW=True`;
- supports `--camera`, `--plugin-side`, `--max-frames`, `--display`, `--no-display`.

### `live_profiles.py`

Profile table and runtime overrides:

- defines `single-best`, `single-strict`, `single-fp16`, `single-cpu`, `single-debug`, `single-rgb-on`, `dual-best`, `dual-cpu`, `dual-fp16`;
- default preview currently on for all profiles;
- defines `--display` and `--no-display`;
- prints banners with actual display state.

### `run_live_profile.py`

Launcher:

- interactive or direct profile selection;
- `--list`, `--status`, `--stop`, `--replace`, `--dry-run`;
- passes `--display` / `--no-display` to child script;
- prevents starting a second live process if camera is already owned.

## Open Ephys Plugin State

Plugin name: `Dual DLCLive Bridge`.

Role:

- receives UDP;
- parses `DDLP` binary pose, JSON pose, or legacy TTL JSON;
- filters raw points;
- chooses valid triplet;
- computes hind angle;
- applies angle threshold and refractory;
- creates TTL word;
- emits Open Ephys TTL state changes on event channel `Dual DLCLive TTL`.

Default UI parameters from plugin README/source:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | UDP listener on/off. |
| `udp_port` | `47000` | Local port Python sends to. |
| `angle_trigger_enabled` | `false` | Enables TTL lines 2/3 for angle trigger. |
| `angle_threshold_deg` | `55.0` | Hind angle threshold. |
| `conf_thresh_use` | `0.20` | Likelihood cutoff before filtering. |
| `conf_thresh_draw` | `0.15` | Likelihood cutoff for valid triplet. |
| `use_filter` | `true` | C++ online filter. |
| `enable_pcutoff` | `true` | Drop low-likelihood points. |
| `enable_despike` | `true` | Reject jumps. |
| `despike_threshold_px` | `150.0` | Max allowed jump. |
| `despike_reset_gap_frames` | `15` | Reacquire gap. |
| `median_window` | `3` | Median smoothing window. |
| `enable_hold` | `false` | Hold last good point. |
| `max_hold_frames` | `20` | Hold duration. |
| `refractory_ms` | `0` | Minimum interval between angle trigger rising edges. |

TTL mapping:

| Line | Bit | Meaning |
| ---: | ---: | --- |
| 0 | `0x01` | Left triplet valid. |
| 1 | `0x02` | Right triplet valid. |
| 2 | `0x04` | Left angle trigger, if enabled. |
| 3 | `0x08` | Right angle trigger, if enabled. |

Critical note: angle trigger lines 2/3 will not fire unless `angle_trigger_enabled=true` in the plugin UI.

## UDP Packet Modes

Current working mode:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
DUAL_FAST_POSE_ONLY = True
```

Modes:

| Mode | Status | Notes |
| --- | --- | --- |
| Binary `DDLP` pose v1 | Main working mode | Fastest, minimal Python allocations. |
| JSON `dual_dlc_live.pose.v1` | Debug/fallback | Slower, supports explicit point names/custom set. |
| JSON `dual_dlc_live.v1` + `ttl_lines` | Legacy | Python computes TTL itself; not the current target. |

Synthetic plugin test:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
```

Requires Open Ephys plugin running/listening on UDP `47000`.

## Current Performance Observations

These are prior experimental observations from logs, not a fresh 2026-06-10 camera run.

Single preview with `single-best`:

- log time around `2026-06-09 15:53`;
- GPU detected: `NVIDIA GeForce RTX 5070 Laptop GPU`;
- Torch: `2.10.0+cu128`;
- `precision=FP32`, `convert2rgb=False`;
- example stage profile after warm-up:
  - `result_hz` about `35-37`;
  - inference around `18-21 ms`, with spikes;
  - display cost around `4-6 ms`;
  - camera/read around `1-1.5 ms`;
  - pack/send around `0.12-0.15 ms`.

Single FP16 preview:

- can be similar or faster but likelihood/coordinates must be validated before stimulation;
- some logs show weaker likelihoods on some frames;
- use as experiment, not as final without comparison.

Dual headless:

- dual mode was tested, but right camera likelihood was effectively zero;
- result rate varied widely because right side/view and two-camera inference were not the current working target;
- do not use `dual-*` as primary until the right camera view is physically fixed.

General lesson:

- Task Manager GPU graph can be misleading; `nvidia-smi pmon` previously showed Python using GPU SM even when Windows task manager looked idle.
- The CPU load remains nonzero because camera SDK, frame conversion, DLCLive preprocessing, OpenCV display, UDP packing, and Python scheduling all involve CPU.
- Display preview is useful for point validation but should be disabled for final stimulation via `--no-display`.

## Current Known Issues / Risks

1. Right camera physical view is not ready.
   - It opens with serial `FDE22070175`.
   - It uses the right treadmill config.
   - But detections were not useful in previous checks.
   - Stick with `single-*` until right image truly contains the correct treadmill/animal view.

2. All profiles currently show preview by default.
   - This is intentional per user request.
   - It lowers FPS.
   - Use `--no-display` for stimulation.

3. Base config and profile config differ.
   - `config_dual_rt_dlc_live.py` has `DUAL_DISPLAY_WINDOW=False`.
   - `live_profiles.py` overrides `DUAL_DISPLAY_WINDOW=True` and `SINGLE_DISPLAY_WINDOW=True`.
   - Trust the profile banner printed at launch for actual runtime state.

4. Git state is dirty in both locations.
   - Do not revert user/previous changes.
   - Do not push without explicit user approval.

5. GalaxyView / camera ownership.
   - If GalaxyView or another live process owns the camera, Python cannot open it.
   - Use `run_live_profile.py --status`, `--stop`, or `--replace`.

6. `--list` confusion.
   - `--list` only prints options.
   - To select profile 1, use no `--list` and type `1`, or run `run_live_profile.py 1`.

7. Angle TTL is plugin-side.
   - Python no longer computes final TTL in current fast path.
   - If angle TTL lines are expected, check plugin UI `angle_trigger_enabled=true`.

## Git State Details

Working Python project `C:\dlc\DLC_OBS_Spinal_cord_stimulation` is a dirty git repo with many historical/untracked changes. It includes old scripts, logs, videos, and docs. Treat it as a live lab working directory.

Repo copy `C:\tmp\Dual_DLC_live_plugin` is the one associated with GitHub repo `nikaabigail/Dual_DLC_live_plugin`. Current status:

```text
## main...origin/main
 M python/dual_rt_dlc_live.py
?? python/README_LIVE_PROFILES.md
?? python/live_profiles.py
?? python/run_live_profile.py
?? python/single_rt_dlc_live_bridge.py
```

No commit/push should be done until the user explicitly says the result is good.

## How To Continue

Recommended next actions for another agent:

1. If user wants to inspect points, launch:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --replace
```

2. If user wants final stimulation/headless:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --no-display --replace
```

3. If camera is busy:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --status
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --stop
```

4. If testing alternative precision:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-strict --replace
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-fp16 --replace
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-rgb-on --replace
```

5. If Open Ephys is involved, verify plugin status:

- plugin added to signal chain;
- `enabled=true`;
- `udp_port=47000`;
- status label `pkts` increases;
- status label `mode` becomes `bin`;
- for angle-trigger stimulation, `angle_trigger_enabled=true`;
- downstream processor receives event channel `Dual DLCLive TTL`.

6. Before editing:

- read `live_profiles.py`, `single_rt_dlc_live_bridge.py`, `dual_rt_dlc_live.py`, `config_dual_rt_dlc_live.py`;
- do not revert dirty files;
- keep repo copy in `C:\tmp\Dual_DLC_live_plugin\python` synced manually if user wants packaging, but do not commit/push without permission.

## Files Most Likely To Edit Next

| File | Why |
| --- | --- |
| `C:\dlc\DLC_OBS_Spinal_cord_stimulation\live_profiles.py` | Add/change launch profiles, display defaults, precision/TF32/thread settings. |
| `C:\dlc\DLC_OBS_Spinal_cord_stimulation\single_rt_dlc_live_bridge.py` | Current one-camera experiment path. |
| `C:\dlc\DLC_OBS_Spinal_cord_stimulation\dual_rt_dlc_live.py` | UDP packing, profiler, dual camera pairing, shared bridge helpers. |
| `C:\dlc\DLC_OBS_Spinal_cord_stimulation\config_dual_rt_dlc_live.py` | Camera serials/configs, bridge mode, base thresholds. |
| `C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\Plugins\DualDLCLiveBridge\DualDLCLiveBridge.cpp` | C++ filtering/TTL logic. |
| `C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\Plugins\DualDLCLiveBridge\DualDLCLiveBridgeEditor.cpp` | Plugin UI/status. |
| `C:\dlc\DLC_OBS_Spinal_cord_stimulation\README_LIVE_PROFILES.md` | User-facing launch docs. |

## Current Source-Of-Truth Summary

- Current practical runtime: `single-best`.
- Camera: left `FDE22070174`.
- UDP: `127.0.0.1:47000`.
- Wire format: binary `DDLP`.
- Python sends raw pose points, not final TTL.
- Plugin computes filters/angles/TTL.
- Preview windows are currently enabled by default.
- Use `--no-display` for real stimulation/performance.
- Right camera exists but is not currently trusted for detections.
- Do not push to GitHub until user approval.
