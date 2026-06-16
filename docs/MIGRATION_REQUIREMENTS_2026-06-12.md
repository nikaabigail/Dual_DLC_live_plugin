# Migration requirements for DLC live + Open Ephys bridge

Date: 2026-06-12  
Target: move the current working realtime setup to a clean Windows desktop PC with NVIDIA RTX 5070.

This document describes what must exist on the new machine before we try to automate installation.
It is based on the current working machine and current project state, not on a generic DeepLabCut setup.

## 1. Scope

The system to reproduce is:

```text
Daheng/Galaxy USB3 camera(s)
  -> Galaxy SDK / gxipy
  -> Python live runtime
  -> DLCLive + PyTorch CUDA inference
  -> UDP 127.0.0.1:47000, binary DDLP/v1 pose packet
  -> Open Ephys / Plugin-GUI processor: DualDLCLiveBridge
  -> Open Ephys event channel "Dual DLCLive TTL"
  -> downstream stimulation/output processor
```

Current practical working mode:

```text
single camera: left camera FDE22070174
runtime profile: single-best
frame: 1920 x 220 px
camera FPS: 100 Hz
GPU inference: RTX 5070, CUDA through PyTorch wheel
UDP: 127.0.0.1:47000
wire format: binary pose DDLP/v1
plugin computes filtering, angle and TTL
```

The dual-camera path exists, but it should only be used after the physical right-camera view is corrected and validated.

## 2. Current reference machine snapshot

These are the versions observed on the working machine.

| Component | Current value |
| --- | --- |
| OS | Windows 11 Home, 64-bit, build 10.0.26200 |
| CPU | AMD Ryzen 9 8945HX, 16 cores / 32 logical processors |
| RAM | 32 GB class |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| NVIDIA driver | 592.01 |
| Driver-reported CUDA capability | CUDA 13.1 via `nvidia-smi` |
| CUDA Toolkit installed locally | 13.0.88, but not required for Python inference |
| Python | CPython 3.10.11 x64 |
| Python env | `C:\dlc_live_env` |
| pip | 26.0.1 |
| PyTorch | `torch==2.10.0+cu128` |
| Torchvision | `torchvision==0.25.0+cu128` |
| Torchaudio | `torchaudio==2.10.0+cu128` |
| PyTorch CUDA runtime | 12.8 |
| cuDNN seen by PyTorch | 91002 |
| Galaxy SDK | 1.18.2208.9301, release date 2022-08-30 |
| CMake | 3.31.6 |
| Visual Studio | VS 2022 Community 17.14.26 installed; VS Build Tools 2026 18.4 also present |
| GitHub repo | `https://github.com/nikaabigail/Dual_DLC_live_plugin.git` |

Important GPU note: PyTorch uses its own CUDA 12.8 runtime from the `+cu128` wheel. The new machine does not need CUDA Toolkit for inference. It does need an NVIDIA driver new enough for RTX 5070 and CUDA 12.8 runtime compatibility. For lowest risk, install the same or newer NVIDIA desktop driver than 592.01.

## 3. Required hardware

Minimum practical hardware:

| Requirement | Notes |
| --- | --- |
| Windows desktop PC | Windows 10/11 64-bit. Current validated system is Windows 11. |
| NVIDIA RTX 5070 desktop GPU | Use desktop NVIDIA driver, not laptop package. |
| At least 32 GB RAM | Current setup uses about 24 GB while working. 16 GB is risky with Open Ephys + Python + GPU stack. |
| USB3 controller/hub | Daheng USB3 Vision cameras are sensitive to bandwidth and ownership. A powered USB3 hub is preferable. |
| Daheng cameras | Expected serials: left `FDE22070174`, right `FDE22070175`. |
| Treadmill camera geometry | Current model and ROI expect the side treadmill view, 1920 x 220 stripe. |

Recommended hardware checks:

```powershell
nvidia-smi
Get-CimInstance Win32_Processor
Get-CimInstance Win32_OperatingSystem
```

Expected GPU check after driver install:

```text
NVIDIA GeForce RTX 5070
Driver Version: 592.01 or newer
```

## 4. Required system software

### 4.1 NVIDIA driver

Required for runtime.

Install a current NVIDIA desktop driver for RTX 5070. The tested source machine uses `592.01`.

Verify:

```powershell
nvidia-smi
```

The Python test later must print:

```text
torch.cuda.is_available True
torch.cuda.device_name NVIDIA GeForce RTX 5070 ...
```

### 4.2 Python

Required for runtime.

Install:

```text
Python 3.10.11 x64
```

Recommended install options:

- install for all users or a stable admin-controlled location;
- add Python launcher if desired;
- do not rely on Microsoft Store Python;
- create the project virtual environment at `C:\dlc_live_env`.

Do not move to Python 3.11/3.12 until validated. DLCLive, OpenCV, Torch and old camera SDK interaction were tested on Python 3.10.

### 4.3 Daheng Galaxy SDK

Required for camera runtime.

Install:

```text
Daheng GalaxySDK 1.18.2208.9301
Default path: C:\Program Files\Daheng Imaging\GalaxySDK
```

Required subpaths:

```text
C:\Program Files\Daheng Imaging\GalaxySDK\Samples\Python SDK\gxipy
C:\Program Files\Daheng Imaging\GalaxySDK\GenTL\Win64
C:\Program Files\Daheng Imaging\GalaxySDK\APIDll\Win64
```

`gxipy` is not installed with pip. The Python runtime adds the Galaxy SDK Python folder and DLL directories at runtime.

GalaxyView is useful for camera discovery/configuration, but it must not own/acquire the camera while Python is running. If GalaxyView has the camera open, Python can fail with:

```text
TL Error: The device has been open
```

### 4.4 Open Ephys / OpenSIS / Plugin-GUI

Required for stimulation path.

Assumption: when the target machine says it has "OpenSIS", it must be checked whether this is the same Open Ephys / Plugin-GUI environment that can load custom processor plugins. The current C++ plugin is an Open Ephys Plugin-GUI processor.

Current source-machine plugin-GUI layout:

```text
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
  Plugins\DualDLCLiveBridge
  out\build\x64-Debug\plugins\DualDLCLiveBridge.dll
  out\build\x64-Debug\open-ephys.exe
```

For the new machine there are two options:

1. If Open Ephys / OpenSIS can load external plugin DLLs compiled against the same ABI, copy `DualDLCLiveBridge.dll` into the correct plugin folder.
2. If not guaranteed, install/build the matching Plugin-GUI source tree and compile `DualDLCLiveBridge` there.

The second option is safer because Open Ephys plugins are ABI-sensitive.

### 4.5 Visual Studio / Build Tools

Required only if rebuilding the C++ Open Ephys plugin.

Install one of:

```text
Visual Studio 2022 Community 17.x with Desktop development with C++
Visual Studio Build Tools 2022/2026 with MSVC C++ and Windows SDK
```

The current machine has:

```text
Visual Studio Community 2022 17.14.26
Visual Studio Build Tools 2026 18.4.0
```

Required workloads/components:

- MSVC C++ x64 compiler;
- Windows 10/11 SDK;
- CMake integration or standalone CMake;
- Ninja optional, depending on generator.

Check with:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -all -products * -format json
```

`cl.exe` not being visible in a normal PowerShell is not enough to decide that Visual Studio is missing. It is usually available only inside a Developer PowerShell/Developer Command Prompt.

### 4.6 CMake

Required if rebuilding Open Ephys / Plugin-GUI.

Tested:

```text
CMake 3.31.6
```

Plugin minimum from `CMakeLists.txt` is `3.15`, but for a clean migration use a modern CMake matching the current source machine if possible.

### 4.7 Git

Required to clone/update repositories.

Current installed Git shown by Windows app metadata:

```text
Git 2.53.0.1
```

Any current Git for Windows should be sufficient. GitHub CLI is optional.

### 4.8 CUDA Toolkit

Not required for the Python realtime inference path.

Current source machine has:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
nvcc: release 13.0, V13.0.88
```

Do not make CUDA Toolkit a hard runtime dependency. PyTorch `+cu128` includes the CUDA runtime needed by the wheel. Install CUDA Toolkit only if later we add custom CUDA compilation or profiling workflows that need `nvcc`.

## 5. Required filesystem layout

The current scripts contain absolute paths. The lowest-risk migration is to reproduce these paths exactly first.

Required paths:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\tmp\Dual_DLC_live_plugin
C:\dlc_live_env
C:\config_daheng
C:\dlc\project\r_tm_side-og-2024-10-25
```

If these paths are changed, these files must be updated:

```text
python/live_profiles.py
python/config_rt_dlc_live.py
python/config_dual_rt_dlc_live.py
```

Current hard-coded important paths:

```python
WORK_DIR = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation")
PYTHON_EXE = Path(r"C:\dlc_live_env\Scripts\python.exe")
GALAXY_SDK_ROOT = r"C:\Program Files\Daheng Imaging\GalaxySDK"
GALAXY_CONFIG_PATH = r"C:\config_daheng\..."
MODEL_PATH = r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\..."
```

Environment variable overrides exist for some single-camera values:

```text
DLC_LIVE_VIDEO_PATH
DLC_LIVE_CAMERA_BACKEND
DLC_LIVE_GALAXY_SDK_ROOT
DLC_LIVE_GALAXY_SN
DLC_LIVE_GALAXY_INDEX
DLC_LIVE_GALAXY_CONFIG_PATH
DLC_LIVE_MODEL_PATH
```

Dual camera config paths are currently hard-coded in `config_dual_rt_dlc_live.py`.

## 6. GitHub repositories and local copies

Main GitHub repo for the new plugin/live bridge:

```text
https://github.com/nikaabigail/Dual_DLC_live_plugin.git
```

Current local copy:

```text
C:\tmp\Dual_DLC_live_plugin
```

Working Python folder used during experiments:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation
```

Important: the working folder and GitHub-copy folder may not be byte-identical. Before final migration, decide which is authoritative. For the current state, the working runtime is in `C:\dlc\DLC_OBS_Spinal_cord_stimulation`, while the GitHub repo is the intended transfer package.

Do not rely on the old `requirements.txt` in `C:\dlc\DLC_OBS_Spinal_cord_stimulation`; it does not match the current working live environment.

## 7. Python environment

Create environment:

```powershell
py -3.10 -m venv C:\dlc_live_env
C:\dlc_live_env\Scripts\python.exe -m pip install --upgrade pip
```

The current pip version is:

```text
pip==26.0.1
```

Install PyTorch first, from the CUDA 12.8 PyTorch wheel index:

```powershell
C:\dlc_live_env\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128
```

Then install the live runtime dependencies:

```powershell
C:\dlc_live_env\Scripts\python.exe -m pip install deeplabcut-live==1.1.0 dlclibrary==0.0.11 numpy==1.26.4 opencv-python==4.11.0.86 pandas==2.3.3 scipy==1.15.3 PyYAML==6.0.3 tqdm==4.67.3 timm==1.0.26 pillow==12.1.1 tables==3.10.1 requests==2.33.1 rich==14.3.3
```

Current key package versions:

| Package | Version |
| --- | --- |
| torch | `2.10.0+cu128` |
| torchvision | `0.25.0+cu128` |
| torchaudio | `2.10.0+cu128` |
| deeplabcut-live | `1.1.0` |
| dlclibrary | `0.0.11` |
| numpy | `1.26.4` |
| opencv-python | `4.11.0.86` |
| pandas | `2.3.3` |
| scipy | `1.15.3` |
| PyYAML | `6.0.3` |
| tqdm | `4.67.3` |
| timm | `1.0.26` |
| pillow | `12.1.1` |
| tables | `3.10.1` |

Current full freeze is stored separately in:

```text
python/requirements-live-lock-2026-06-12.txt
```

Important package notes:

- The live runtime uses `deeplabcut-live`, not full `deeplabcut` training package.
- `deeplabcut==3.0.0rc13` in older notes/requirements is not the current validated live runtime dependency.
- `opencv-python` is required for preview windows. A headless-only OpenCV install is insufficient for visual point checking.
- `sklearn` and `matplotlib` are not installed in the current live env. They are not required for the current live bridge.

Python validation command:

```powershell
C:\dlc_live_env\Scripts\python.exe -c "import torch, cv2, numpy, dlclive; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0)); print(cv2.__version__, numpy.__version__, dlclive.__version__)"
```

Expected:

```text
2.10.0+cu128 True 12.8 NVIDIA GeForce RTX 5070 ...
4.11.0 1.26.4 1.1.0
```

## 8. DLCLive model artifact

Required model path:

```text
C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt
```

Current model file:

```text
Size: 97,426,993 bytes
SHA256: BE67BD29F01529E4FAC2378659535536A447D9908290457D7896DDB9ED3F1CCE
```

For realtime inference, this `.pt` is the critical artifact. For training/re-export/offline analysis, also copy the full DeepLabCut project:

```text
C:\dlc\project\r_tm_side-og-2024-10-25
```

The model expects these bodyparts for the binary plugin path:

```text
hl_ankle_l
hl_ankle_r
hl_hip_l
hl_hip_r
hl_toes_l
hl_toes_r
```

Do not change the binary point order without changing both Python packing and C++ plugin parsing.

## 9. Camera configuration

Expected cameras:

| Side | Daheng model in GalaxyView | Serial |
| --- | --- | --- |
| left | `MER2-230-168U3C` | `FDE22070174` |
| right | `MER2-230-168U3C` | `FDE22070175` |

Required config directory:

```text
C:\config_daheng
```

Required config files:

```text
C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt
C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt
```

Current working config hashes:

```text
left  SHA256 C6D947D09E0F4309D79277B0D401EA3DADFD99AF0E6B1A81AD0F7787B6133A78
right SHA256 C9889CE3A5D1965AB2093F1E37D8377223162CB1D52184DF3ABB72E37F6EB86E
```

Current key settings:

| Setting | Left | Right |
| --- | --- | --- |
| PixelFormat | `BayerRG8` | `BayerRG8` |
| Width | `1920` | `1920` |
| Height | `220` | `220` |
| OffsetX | `0` | `0` |
| OffsetY | `510` | `530` |
| ExposureTime | `4000` | `4000` |
| AcquisitionFrameRateMode | `On` | `On` |
| AcquisitionFrameRate | `100` | `100` |
| TriggerMode | `Off` | `Off` |
| TriggerSource | `Software` | `Software` |
| GainAuto | `Off` | `Off` |
| Gain | `24` | `10` |

Important repo discrepancy:

```text
repo camera_configs left Gain is 10
current working C:\config_daheng left Gain is 24
```

For identical migration, copy the current `C:\config_daheng` files from the working machine, or update the repo camera configs before using them as the source of truth.

## 10. Open Ephys plugin

Plugin source:

```text
C:\tmp\Dual_DLC_live_plugin\open_ephys_plugin\DualDLCLiveBridge
```

Plugin files:

```text
DualDLCLiveBridge.cpp
DualDLCLiveBridge.h
DualDLCLiveBridgeEditor.cpp
DualDLCLiveBridgeEditor.h
CMakeLists.txt
OpenEphysLib.cpp
```

The plugin has no extra third-party C++ dependencies beyond Open Ephys Plugin-GUI/JUCE and Windows socket APIs.

Expected integration into Plugin-GUI source tree:

```text
<plugin-GUI-root>\Plugins\DualDLCLiveBridge
```

`<plugin-GUI-root>\Plugins\CMakeLists.txt` must include:

```cmake
add_subdirectory(DualDLCLiveBridge)
```

Current tested output on source machine:

```text
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug\plugins\DualDLCLiveBridge.dll
Size: 385,024 bytes
```

Plugin UI/runtime defaults:

| Parameter | Default / current meaning |
| --- | --- |
| `enabled` | UDP listener enabled |
| `udp_port` | `47000` |
| packet mode | pose/raw points from Python |
| binary magic | `DDLP` |
| version | `1` |
| `conf_thresh_use` | `0.20` |
| `conf_thresh_draw` | `0.15` |
| `despike_threshold_px` | `150.0` |
| `median_window` | `3` |
| `angle_threshold_deg` | about `55.0`, UI-controlled |
| `angle_trigger_enabled` | must be explicitly enabled for angle TTL lines |

TTL line meaning:

| TTL line | Meaning |
| --- | --- |
| 0 | left pose triplet valid |
| 1 | right pose triplet valid |
| 2 | left hind angle trigger, if enabled |
| 3 | right hind angle trigger, if enabled |

For the current single-camera mode, Python sends left pose and fills right side as NaN. Right-side TTL lines should remain inactive.

## 11. Runtime profiles and launch commands

Main launcher:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py
```

List profiles:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py --list
```

Recommended visual check:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py single-best --display --replace
```

Recommended stimulation run:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py single-best --no-display --replace
```

Stop live Python process:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py --stop
```

Status:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py --status
```

Profiles that exist:

| Profile | Use |
| --- | --- |
| `single-best` | recommended current mode: one left camera, FP32+TF32, RGB no-convert |
| `single-strict` | strict FP32, TF32 disabled |
| `single-fp16` | experimental, validate before stimulation |
| `single-cpu` | lower CPU thread pressure |
| `single-debug` | visual debug |
| `single-rgb-on` | color fallback, BGR input + DLCLive RGB conversion |
| `dual-best` | two-camera path after right view is corrected |
| `dual-cpu` | two-camera CPU-balanced variant |
| `dual-fp16` | experimental dual FP16 |

## 12. Open Ephys + Python startup order

Recommended order for experiment:

1. Boot Windows.
2. Confirm cameras are connected and visible in GalaxyView.
3. Close GalaxyView or stop acquisition so it does not own the camera.
4. Start Open Ephys / OpenSIS with `DualDLCLiveBridge` plugin loaded.
5. In plugin UI, set/confirm:
   - enabled;
   - UDP port `47000`;
   - binary/pose mode expected;
   - threshold parameters;
   - angle trigger enabled only when intended.
6. Start Python:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py single-best --no-display --replace
```

7. Confirm Open Ephys plugin packet counter increases.
8. Confirm TTL/event channel behavior before connecting real stimulation.

For visual point check, use `--display`; for stimulation, use `--no-display`.

## 13. Validation checklist on the new PC

### 13.1 GPU and PyTorch

```powershell
nvidia-smi
C:\dlc_live_env\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

Expected:

```text
2.10.0+cu128
True
12.8
NVIDIA GeForce RTX 5070 ...
```

### 13.2 Galaxy SDK

Check folders:

```powershell
Test-Path "C:\Program Files\Daheng Imaging\GalaxySDK\Samples\Python SDK\gxipy"
Test-Path "C:\Program Files\Daheng Imaging\GalaxySDK\GenTL\Win64"
```

Expected:

```text
True
True
```

### 13.3 Model file

```powershell
Get-FileHash -Algorithm SHA256 "C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt"
```

Expected hash:

```text
BE67BD29F01529E4FAC2378659535536A447D9908290457D7896DDB9ED3F1CCE
```

### 13.4 Camera config files

```powershell
Get-FileHash -Algorithm SHA256 "C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt"
Get-FileHash -Algorithm SHA256 "C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt"
```

Expected hashes:

```text
left  C6D947D09E0F4309D79277B0D401EA3DADFD99AF0E6B1A81AD0F7787B6133A78
right C9889CE3A5D1965AB2093F1E37D8377223162CB1D52184DF3ABB72E37F6EB86E
```

### 13.5 Open Ephys plugin without camera

Start Open Ephys with `DualDLCLiveBridge`, then send synthetic packets:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\send_dual_dlc_bridge_test.py --host 127.0.0.1 --port 47000 --mode pose --wire binary
```

Expected:

- plugin packet counter increases;
- plugin status line reports binary pose packets;
- TTL validity lines can change with synthetic points depending on test payload and thresholds.

### 13.6 Live camera smoke test

With GalaxyView closed:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py single-best --display --replace --max-frames 300
```

Expected:

- camera opens by serial `FDE22070174`;
- preview window shows left treadmill view;
- points are drawn when animal is visible;
- log reports stage profiler lines;
- no accumulating one-second video delay.

### 13.7 Stimulation/headless smoke test

```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\run_live_profile.py single-best --no-display --replace
```

Expected:

- Open Ephys packet counter increases;
- `pack/send` is around 0.1 ms class;
- no OpenCV preview cost;
- GPU used by Python process during inference.

## 14. Known failure modes

### Camera already open

Symptom:

```text
TL Error: The device has been open
```

Fix:

- close GalaxyView;
- stop acquisition;
- stop old Python live process with `run_live_profile.py --stop`;
- unplug/replug camera if SDK still holds state.

### Galaxy import config U3V error

Symptom:

```text
USB3 Vision Error: Wrong status code of U3V acknowledgment packet from device: 0x8002
```

Likely causes:

- camera is busy;
- hub/controller issue;
- camera state inconsistent after previous acquisition;
- config import attempted while another tool owns the device.

Fix:

- close GalaxyView;
- use one camera first;
- power-cycle cameras/hub;
- verify the exact serial-specific config file;
- if needed, apply config in GalaxyView and run Python with import disabled only after confirming settings.

### GPU looks idle in Task Manager

Do not use only Task Manager 3D graph as proof. Check CUDA directly:

```powershell
nvidia-smi
C:\dlc_live_env\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The current bottleneck is usually inference and Python/camera/display scheduling, not GPU memory copy alone.

### Low FPS with preview

Preview adds measurable cost. Current source-machine measurements:

```text
single-best preview: about 38-42 FPS after warm-up
average inference: about 18.6 ms in one 4700-frame run
display cost: about 4.5-6.9 ms
pack/send: about 0.11-0.15 ms
drops: about 0.28-0.32%
```

For stimulation, use:

```text
--no-display
```

### Plugin not visible in Open Ephys

Likely causes:

- DLL copied to wrong plugin folder;
- plugin built against different Open Ephys/Plugin-GUI ABI;
- missing Visual C++ runtime;
- Plugin-GUI `Plugins/CMakeLists.txt` did not include `add_subdirectory(DualDLCLiveBridge)`;
- build output not copied to the Open Ephys runtime plugin path.

Fix:

- rebuild against the exact Plugin-GUI/Open Ephys source used on the target machine;
- confirm `DualDLCLiveBridge.dll` exists in the runtime plugin folder;
- start Open Ephys from the matching build output if using source tree.

## 15. Required files to transfer

Minimum runtime transfer set:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation\*.py
C:\dlc\DLC_OBS_Spinal_cord_stimulation\README_OPEN_EPHYS_BRIDGE.md
C:\dlc\DLC_OBS_Spinal_cord_stimulation\README_LIVE_PROFILES.md
C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt
C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt
C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt
DualDLCLiveBridge source or compiled DLL
```

Recommended transfer set:

```text
C:\tmp\Dual_DLC_live_plugin
C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\config_daheng
C:\dlc\project\r_tm_side-og-2024-10-25
```

Do not assume GitHub contains the model file or the latest working camera config values. Verify hashes.

## 16. What the future installer should check

The later executable/installer should perform these checks before starting the system:

1. Windows version and 64-bit architecture.
2. NVIDIA driver present and RTX 5070 visible.
3. `nvidia-smi` works.
4. Python 3.10 exists or install it.
5. `C:\dlc_live_env\Scripts\python.exe` exists.
6. Python package versions match the lock file.
7. `torch.cuda.is_available()` is true.
8. Galaxy SDK exists at default path.
9. `gxipy` folder exists.
10. Camera configs exist and match expected hashes.
11. Model file exists and matches expected hash.
12. Project scripts exist at expected paths.
13. Open Ephys / OpenSIS process/plugin path exists.
14. `DualDLCLiveBridge.dll` exists or source tree can be built.
15. UDP port 47000 is not blocked by another local listener.
16. No old Python live process owns cameras.
17. GalaxyView is not acquiring cameras.
18. Cameras with serials `FDE22070174` and optionally `FDE22070175` are visible.
19. Test packet can be received by the plugin.
20. A short `single-best --display --max-frames 300` run succeeds.

## 17. Installation order for a clean PC

Recommended manual order before automation:

1. Install Windows updates and chipset/USB controller drivers.
2. Install NVIDIA desktop driver for RTX 5070, version 592.01 or newer.
3. Install Python 3.10.11 x64.
4. Install Daheng GalaxySDK 1.18.2208.9301.
5. Connect cameras and verify in GalaxyView.
6. Create `C:\config_daheng` and copy the current working camera config files.
7. Create `C:\dlc\project\...` and copy the model / DLC project.
8. Clone/copy `Dual_DLC_live_plugin`.
9. Copy or prepare `C:\dlc\DLC_OBS_Spinal_cord_stimulation`.
10. Create `C:\dlc_live_env`.
11. Install PyTorch `+cu128`.
12. Install remaining Python packages.
13. Validate Python imports and CUDA.
14. Install or locate Open Ephys / OpenSIS / Plugin-GUI.
15. Build or copy `DualDLCLiveBridge.dll`.
16. Start Open Ephys and test synthetic UDP.
17. Run `single-best --display` with camera.
18. Run `single-best --no-display` with Open Ephys.

## 18. Items that are intentionally not hard requirements

These exist on the source machine but should not be treated as mandatory for first runtime migration:

| Item | Reason |
| --- | --- |
| CUDA Toolkit 13.0 | PyTorch wheel supplies CUDA runtime for inference. |
| Full `deeplabcut` package | Current live runtime uses `deeplabcut-live`. |
| `matplotlib` / `scikit-learn` | Not installed in current live env and not needed for runtime. |
| Visual Studio IDE | Not needed if plugin DLL is already compatible and copied. Build tools are enough for rebuilding. |
| GitHub CLI | Useful for auth/push, not required for runtime. |

## 19. Open questions before final automation

These must be clarified on the target PC:

1. What exactly is installed there under "OpenSIS": Open Ephys GUI, Plugin-GUI source tree, or another distribution?
2. Where does that installation expect custom processor DLLs?
3. Is it ABI-compatible with the current `DualDLCLiveBridge.dll`, or must we rebuild?
4. Does the desktop RTX 5070 driver report CUDA support correctly through `nvidia-smi`?
5. Are both Daheng cameras visible through the same serial numbers?
6. Does the USB hub sustain one or two 1920 x 220 @ 100 Hz camera streams without U3V errors?
7. Should the repository be updated so `camera_configs` contains the actual current left `Gain=24` config?

## 20. Current recommended first migration target

For the first successful transfer, do not start with dual-camera stimulation. Start with:

```text
single-best
left camera FDE22070174
RGB no-convert
FP32 + TF32
binary pose UDP
Open Ephys plugin computes TTL
preview for validation, then headless for experiment
```

Only after this is stable should dual mode be tested.
