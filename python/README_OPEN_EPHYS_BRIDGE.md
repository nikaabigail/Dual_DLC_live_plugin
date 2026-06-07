# Dual DLCLive -> Open Ephys Bridge

## Current protocol

Default mode in `config_dual_rt_dlc_live.py`:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
DUAL_FAST_POSE_ONLY = True
DUAL_ENABLE_BATCH_INFERENCE = True
DUAL_ENABLE_STAGE_PROFILER = True
```

Python sends compact binary pose packets with raw pose points and metadata.
The Open Ephys plugin computes filtering, valid triplets, hind angles and TTL
lines itself. Python no longer has to send `ttl_lines` or compute angles in the
default mode. If binary transport is not desired, set
`DUAL_OE_BRIDGE_WIRE_FORMAT = "json"`; the payload then uses
`dual_dlc_live.pose.v1`.

## Current Fast Path

In binary fast mode the Python result object stores selected pose points as a
small NumPy array:

```text
raw_pose_array.shape == (6, 3)
columns = x, y, likelihood
rows = DUAL_USE_POINTS order
```

The UDP binary packet is packed directly from this array. No `raw_points`
dictionary is built for binary send. `raw_points` is created lazily only when:

- `DUAL_OE_BRIDGE_WIRE_FORMAT = "json"` is used;
- the local OpenCV overlay needs point dictionaries for drawing.

This keeps the normal stimulation path as:

```text
camera frame -> DLCLive pose ndarray -> raw_pose_array -> binary UDP -> C++ plugin -> TTL
```

## Stage Profiler

The profiler is enabled by default:

```python
DUAL_ENABLE_STAGE_PROFILER = True
DUAL_PROFILE_LOG_EVERY_N_PAIRS = 120
DUAL_PROFILE_EMA_ALPHA = 0.10
```

It writes rolling stage timings to `dual_rt_dlc_live_debug.log`:

```text
stage_profile pair=120 last_ms camera/read=0.42 preprocess=1.30 inference=8.70 pack/send=0.06 display=3.10 | avg_ms camera/read=...
```

Stages:

| Stage | Meaning |
| --- | --- |
| `camera/read` | Successful Galaxy SDK frame reads in reader threads. |
| `preprocess` | DLCLive `process_frame` for both cameras in the pair. |
| `inference` | Runner/model path for both cameras in the pair. |
| `pack/send` | Binary or JSON packet build plus UDP socket send. |
| `display` | Local overlay, text, resize, video writer and `imshow`. |

Notes:

- The profiler is CPU-side wall time. GPU kernels may be asynchronous, so use it
  mainly to find Python/CPU overhead.
- If `DUAL_DISPLAY_WINDOW = False` and `DUAL_SAVE_OUTPUT_VIDEO = False`, overlay
  work is skipped and `display` should be near zero.
- `camera/read` is measured per successful camera read call, while `preprocess`
  and `inference` are measured per processed left/right pair.

Synthetic test without cameras:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --count 5 --interval 0.025 --wait-ack
```

JSON pose fallback test:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format json --count 5 --interval 0.025 --wait-ack
```

Legacy compatibility test:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode ttl --count 5 --interval 0.025 --wait-ack
```

Important current behavior:

- Default pose mode sends raw pose points, not `ttl_lines`.
- `DUAL_FAST_POSE_ONLY = True` disables Python-side filter/triplet/angle work
  for the Open Ephys path.
- The plugin UI is the source of truth for `angle_trigger_enabled`,
  `angle_threshold_deg`, filter settings and refractory timing.
- Python `DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` is used only by the legacy
  `DUAL_OE_BRIDGE_PACKET_MODE = "ttl"` path.

Этот файл - короткая рабочая инструкция со стороны DLC-проекта. Полная
документация плагина лежит здесь:

```text
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\Plugins\DualDLCLiveBridge\README.md
```

## Общая схема

```text
Daheng left/right cameras
  -> dual_rt_dlc_live.py
  -> UDP 127.0.0.1:47000
  -> Open Ephys processor "Dual DLCLive Bridge"
  -> TTL event channel "Dual DLCLive TTL"
  -> stimulation/output processor
```

Важное отличие от старого USB-пути: Open Ephys не забирает видеопоток с камер.
USB здесь используется для камер Daheng, а Open Ephys получает только готовые
цифровые события. Физическую стимуляцию делает следующий Open Ephys output node,
который слушает TTL events от `Dual DLCLive Bridge`.

## Текущие камеры

Настройки находятся в:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation\config_dual_rt_dlc_live.py
```

Сейчас заданы:

| Side | Serial | Config |
| --- | --- | --- |
| left | `FDE22070174` | `C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt` |
| right | `FDE22070175` | `C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt` |

ROI берется из этих Galaxy `.txt` конфигов. В Python дополнительный crop
отключен: `CROPPING = None`.

## TTL-линии

`dual_rt_dlc_live.py` отправляет в Open Ephys массив `ttl_lines[0..7]`.

| Line | Значение |
| --- | --- |
| `0` | left pose triplet валиден |
| `1` | right pose triplet валиден |
| `2` | left angle `<= DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` |
| `3` | right angle `<= DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` |
| `4..7` | резерв |

Для стимуляции обычно использовать:

```text
left stimulation trigger  -> Dual DLCLive TTL, line 2, rising edge
right stimulation trigger -> Dual DLCLive TTL, line 3, rising edge
```

Линии `0` и `1` лучше считать quality/gate сигналом: они показывают, что точки
найдены валидно, но сами по себе не являются фазовым trigger.

Чтобы включить линии `2` и `3`, задай порог:

```python
DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG = 55.0
```

Если стоит `None`, линии `2` и `3` не будут активироваться.

## Настройки bridge в Python

В `config_dual_rt_dlc_live.py`:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS = 1
DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG = None
```

Порт должен совпадать с `udp_port` в Open Ephys plugin UI.

Для низкой задержки держать:

```python
DUAL_PROCESS_EVERY_N_PAIRS = 1
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS = 1
DUAL_LOW_LATENCY = True
DUAL_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
```

## Порядок запуска

### 1. Закрыть GalaxyView

GalaxyView может держать камеры. Перед запуском Python live лучше закрыть его.

### 2. Запустить Open Ephys

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

Статус в plugin UI:

```text
pkts 0 | pair - | ttl 0x00 | q 0 | age -
```

Когда Python начнет отправлять данные, `pkts` должен расти.

### 3. Запустить acquisition/processing в Open Ephys

Для реальной стимуляции нужен запущенный Open Ephys processing clock. Если
тестируешь без железа, можно использовать `File Reader`; для эксперимента -
обычный acquisition source.

### 4. Проверить UDP без камер

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
python send_dual_dlc_bridge_test.py --count 5 --interval 0.025 --wait-ack
```

Ожидаемый конец вывода:

```text
acked 5/5
```

Это проверяет путь:

```text
Python test packet -> UDP -> C++ Open Ephys plugin -> ACK back to Python
```

Обычный `dual_rt_dlc_live.py` ACK не запрашивает; ACK нужен только для проверки.

### 5. Запустить live DLCLive

```powershell
& C:\dlc_live_env\Scripts\Activate.ps1
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
python dual_rt_dlc_live.py
```

Или без активации, напрямую через Python из окружения:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

## Как понять, что все работает

В Open Ephys plugin UI:

| Поле | Что должно происходить |
| --- | --- |
| `pkts` | растет при работе Python |
| `pair` | показывает последний индекс пары кадров |
| `ttl` | меняется при изменении валидности/угла |
| `q` | обычно около `0`; большие значения означают, что Open Ephys не успевает обработать очередь |
| `age` | маленькое число в ms, если данные свежие |

В PowerShell можно проверить, что Open Ephys слушает порт:

```powershell
netstat -ano -p udp | Select-String ':47000'
```

Пример:

```text
UDP    127.0.0.1:47000        *:*        16652
```

PID должен быть процессом `open-ephys.exe`.

## Как output пойдет в стимуляцию

`Dual DLCLive Bridge` создает внутренний Open Ephys event channel:

```text
Dual DLCLive TTL
```

Он вызывает Open Ephys TTL state changes по линиям `0..7`. Для дальнейшей
стимуляции нужно настроить output/stimulation processor:

1. Source event channel: `Dual DLCLive TTL`.
2. Trigger line: `2` для left или `3` для right.
3. Trigger mode: rising edge.
4. Pulse parameters: задаются уже в вашем output/stimulation processor.

Если нужно стимулировать не только на входе в условие, а пачкой импульсов пока
условие остается активным, эту pulse train логику надо задавать downstream или
добавить отдельный pulse-mode в bridge.

## Сборка плагина

Перед сборкой закрыть Open Ephys, иначе DLL будет занята.

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
cmd.exe /s /c "`"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`" -arch=x64 && cmake --build out\build\x64-Debug --target DualDLCLiveBridge --config Debug"
```

Проверка DLL:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
python -B Plugins\DualDLCLiveBridge\check_plugin_load.py
```

Ожидаемо:

```text
PLUGIN_EXPORTS_OK
```

## Быстрый troubleshooting

### `missing ack`

Проверить:

- Open Ephys запущен.
- `Dual DLCLive Bridge` добавлен в signal chain.
- `enabled = true`.
- `udp_port = 47000`.
- `netstat` показывает `127.0.0.1:47000`.

### `pkts` не растет

Сначала выполнить:

```powershell
python send_dual_dlc_bridge_test.py --count 5 --wait-ack
```

Если тест проходит, проблема не в Open Ephys bridge, а в live Python path или
`DUAL_OE_BRIDGE_ENABLED`.

### Камеры не открываются

Проверить:

- GalaxyView закрыт.
- Серийники в `DUAL_CAMERAS` совпадают.
- `.txt` configs доступны.
- USB3 hub не перегружен.

### TTL есть, но физической стимуляции нет

Проверить downstream processor:

- выбран `Dual DLCLive TTL`;
- выбрана line `2` или `3`;
- trigger mode стоит на rising edge;
- output hardware подключен;
- acquisition запущен.
