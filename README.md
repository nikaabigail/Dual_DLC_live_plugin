# Dual DLC Live Plugin

## Current protocol

Default runtime mode is now `DUAL_OE_BRIDGE_PACKET_MODE = "pose"` with
`DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"`.

In this mode Python keeps only camera acquisition, frame pairing and
DLCLive/PyTorch inference for the Open Ephys path. It sends raw pose points and
metadata as compact UDP binary packets (`DDLP`/v1). Set
`DUAL_OE_BRIDGE_WIRE_FORMAT = "json"` to return to JSON packets with schema
`dual_dlc_live.pose.v1`.

Optimized defaults:

```python
DUAL_FAST_POSE_ONLY = True
DUAL_ENABLE_BATCH_INFERENCE = True
DUAL_BATCH_FALLBACK_TO_SEQUENTIAL = True
```

With `DUAL_FAST_POSE_ONLY = True`, Python does not compute filters, triplets or
angles for the Open Ephys path. The plugin is the source of truth for angles
and TTL.

`Dual DLCLive Bridge` computes inside the Open Ephys plugin:

- point filtering: p-cutoff, despike, median window and optional hold;
- left/right triplet selection;
- valid triplet lines;
- hind angle;
- optional angle-trigger TTL lines;
- refractory gating for angle trigger rising edges.

TTL map:

```text
line 0 = left valid triplet
line 1 = right valid triplet
line 2 = left angle trigger, if angle_trigger_enabled is on in plugin UI
line 3 = right angle trigger, if angle_trigger_enabled is on in plugin UI
line 4..7 = reserved
```

Legacy `dual_dlc_live.v1` packets with `ttl_lines` are still accepted for
compatibility and can be sent by `send_dual_dlc_bridge_test.py --mode ttl`.
JSON pose compatibility can be tested with
`send_dual_dlc_bridge_test.py --mode pose --wire-format json --wait-ack`.

Интеграция dual DLCLive с Open Ephys:

```text
2 Daheng USB3 cameras -> dual_rt_dlc_live.py -> UDP 127.0.0.1:47000
-> Open Ephys processor "Dual DLCLive Bridge"
-> TTL event channel "Dual DLCLive TTL"
-> stimulation/output processor
```

## Что внутри

| Папка | Содержимое |
| --- | --- |
| `open_ephys_plugin/DualDLCLiveBridge` | C++ исходники Open Ephys plugin |
| `python` | dual DLCLive runtime, configs и UDP test sender |
| `camera_configs` | Galaxy/Daheng `.txt` configs для left/right камер |
| `dist/windows-x64-debug` | собранный `DualDLCLiveBridge.dll` |
| `docs` | дополнительные старые заметки по DLC live |

## Быстрый запуск

### 1. Open Ephys

Запустить GUI:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug
.\open-ephys.exe
```

В signal chain должен быть processor:

```text
Dual DLCLive Bridge
```

Параметры:

```text
enabled = true
udp_port = 47000
```

### 2. Проверка UDP без камер

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --count 5 --interval 0.025 --wait-ack
```

Ожидаемый результат:

```text
acked 5/5
```

### 3. Запуск dual DLCLive

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

## TTL mapping

| Line | Значение |
| --- | --- |
| `0` | left pose triplet валиден |
| `1` | right pose triplet валиден |
| `2` | left hind angle `<= DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` |
| `3` | right hind angle `<= DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` |
| `4..7` | резерв |

Для стимуляции обычно использовать:

```text
left stimulation trigger  -> Dual DLCLive TTL, line 2, rising edge
right stimulation trigger -> Dual DLCLive TTL, line 3, rising edge
```

Важно: `Dual DLCLive Bridge` не управляет физическим USB/TTL устройством сам.
Он создает Open Ephys TTL events. Физический импульс делает следующий
stimulation/output processor в Open Ephys.

## Документация

Подробные инструкции:

- `python/README_OPEN_EPHYS_BRIDGE.md` - рабочая инструкция со стороны Python/DLC.
- `open_ephys_plugin/DualDLCLiveBridge/README.md` - подробная документация C++ plugin.

## Сборка plugin

Скопировать папку:

```text
open_ephys_plugin\DualDLCLiveBridge
```

в:

```text
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\Plugins\DualDLCLiveBridge
```

В `Plugins/CMakeLists.txt` Open Ephys должна быть строка:

```cmake
add_subdirectory(DualDLCLiveBridge)
```

Сборка:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
cmd.exe /s /c "`"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`" -arch=x64 && cmake --build out\build\x64-Debug --target DualDLCLiveBridge --config Debug"
```

Smoke-test DLL:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
python -B Plugins\DualDLCLiveBridge\check_plugin_load.py
```

Ожидаемо:

```text
PLUGIN_EXPORTS_OK
```

## Current validated state

На машине разработки было проверено:

- `DualDLCLiveBridge.dll` экспортирует `getLibInfo` и `getPluginInfo`.
- Open Ephys слушает UDP `127.0.0.1:47000`.
- `send_dual_dlc_bridge_test.py --wait-ack` получает ACK от C++ plugin.
- `OpenEphysBridge.send()` из `dual_rt_dlc_live.py` формирует binary pose production packet by default;
  JSON `dual_dlc_live.pose.v1` с raw pose points и metadata remains available as fallback.
