# Инструкция по Python-части dual DLCLive

Этот файл описывает Python-часть dual-camera системы:

```text
dual_rt_dlc_live.py
  -> две Daheng/Galaxy камеры
  -> DLCLive/PyTorch inference
  -> UDP-пакеты на 127.0.0.1:47000
  -> плагин Open Ephys "Dual DLCLive Bridge"
```

Плагин Open Ephys, его параметры и TTL-логика описаны отдельно:

```text
open_ephys_plugin/DualDLCLiveBridge/README.md
```

## Навигация

| Раздел | Для чего нужен |
| --- | --- |
| Роль Python | Что Python делает сейчас, а что перенесено в C++ plugin. |
| Файлы и конфиги | Где лежит runtime и какие файлы важны. |
| Камеры | Serial numbers, Galaxy configs, ROI, low-latency режим. |
| Модель DLCLive | Откуда берется модель, как проверяется GPU. |
| Порядок выполнения | Дотошный путь данных внутри `dual_rt_dlc_live.py`. |
| UDP bridge | Binary/JSON/legacy TTL режимы. |
| Binary fast-mode | Почему не строится `raw_points` dict. |
| Профилировщик стадий | Как понять, где тратятся миллисекунды. |
| Запуск | Synthetic test и live-run с камерами. |
| Логи | Какие строки искать при проверке. |
| Устранение проблем | Частые ошибки. |

## Роль Python

Текущий рабочий контракт:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
DUAL_FAST_POSE_ONLY = True
DUAL_ENABLE_BATCH_INFERENCE = True
DUAL_BATCH_FALLBACK_TO_SEQUENTIAL = True
DUAL_ENABLE_STAGE_PROFILER = True
DUAL_DISPLAY_WINDOW = False
GALAXY_OUTPUT_COLOR = "rgb"
CONVERT_TO_RGB = False
```

Для CPU throttling есть ручные переключатели:

```python
DUAL_CV2_NUM_THREADS = -1
DUAL_TORCH_NUM_THREADS = 0
DUAL_TORCH_INTEROP_THREADS = 0
```

`-1/0` означает “оставить стандартные OpenCV/PyTorch threadpool-настройки”.
Если нужно снизить CPU любой ценой, можно поставить все три значения в `1`, но
по live A/B это уменьшало `result_hz`.

Python делает:

- открывает две Daheng/Galaxy USB3 камеры;
- импортирует `.txt` конфиги Galaxy;
- держит low-latency поток, сбрасывая старые кадры;
- берет последнюю пару left/right кадров;
- запускает DLCLive/PyTorch inference;
- выбирает шесть нужных точек позы;
- упаковывает точки и metadata в UDP-пакет;
- отправляет пакет в плагин Open Ephys;
- опционально показывает OpenCV overlay;
- пишет profiler/log строки.

Python не делает в рабочем режиме:

- не фильтрует точки для решения о стимуляции;
- не выбирает рабочий triplet;
- не считает рабочий hind angle;
- не применяет refractory для стимуляции;
- не формирует рабочие TTL lines.

Эти решения делает C++ plugin `Dual DLCLive Bridge`.

## Файлы и конфиги

Live-папка на экспериментальном компьютере:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation
  config_rt_dlc_live.py
  rt_dlc_live.py
  config_dual_rt_dlc_live.py
  dual_rt_dlc_live.py
  send_dual_dlc_bridge_test.py
  README_OPEN_EPHYS_BRIDGE.md
```

Репозиторий:

```text
C:\tmp\Dual_DLC_live_plugin
```

Виртуальная среда:

```text
C:\dlc_live_env\Scripts\python.exe
```

Основной конфиг dual runtime:

```text
python/config_dual_rt_dlc_live.py
```

Он импортирует базовый single-camera config:

```python
from config_rt_dlc_live import *
```

Поэтому `MODEL_PATH`, `DEVICE`, часть DLCLive-настроек и некоторые shared
параметры берутся из `config_rt_dlc_live.py`, если dual config их не
переопределил. В рабочем dual-конфиге `CONVERT_TO_RGB` переопределен в `False`,
потому что Galaxy SDK уже отдает RGB-кадр после Bayer-конверсии.

## Камеры

Текущий список:

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

Что делает `runtime.source.open()`:

1. Находит камеру по serial number.
2. Открывает устройство через Galaxy SDK.
3. Импортирует `.txt` конфиг, если `DUAL_IMPORT_CONFIG = True`.
4. Если import не прошел, включает резервные настройки, если разрешено.
5. Запускает acquisition.
6. Читает native параметры и пишет их в лог.

Ожидаемые native параметры:

| Камера | Width | Height | OffsetY | FPS |
| --- | --- | --- | --- | --- |
| left `FDE22070174` | `1920` | `220` | `510` | `100` |
| right `FDE22070175` | `1920` | `220` | `530` | `100` |

Low-latency настройки:

```python
DUAL_LOW_LATENCY = True
DUAL_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
DUAL_ACQUISITION_BUFFER_COUNT = 2
DUAL_DRAIN_QUEUED_FRAMES = True
DUAL_MAX_DRAIN_FRAMES = 20
```

Смысл: если pipeline не успел обработать все кадры, старые кадры выбрасываются.
Для real-time стимуляции это правильнее, чем копить задержку.

## Модель DLCLive

Путь берется из `config_rt_dlc_live.py`:

```python
MODEL_PATH = r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\...\snapshot-best-380.pt"
MODEL_TYPE = "pytorch"
DEVICE = "cuda"
PRECISION = "FP32"
SINGLE_ANIMAL = True
GALAXY_OUTPUT_COLOR = "rgb"
CONVERT_TO_RGB = False
```

В single-camera конфиге `CONVERT_TO_RGB` может оставаться `True`; dual runtime
переопределяет его, чтобы убрать лишнюю CPU-конверсию RGB->BGR->RGB.

При запуске лог должен показать:

```text
CUDA_CHECK torch=... cuda=True gpu=NVIDIA GeForce RTX 5070 Laptop GPU
DLC_MODEL_DEVICE_AFTER_INIT = cuda:0
Model bodyparts loaded: 15; dual points=[...]
```

Важно: до `init_inference` у `dlc_live.model` может быть `None`. Это нормально.
Фактическая проверка GPU появляется после первого init.

## Порядок выполнения внутри `dual_rt_dlc_live.py`

### 1. Проверка конфига

`validate_config()` проверяет:

- есть две камеры;
- имена камер уникальны;
- serial numbers заданы;
- `DUAL_USE_POINTS` не пустой;
- UDP port положительный;
- `DUAL_OE_BRIDGE_PACKET_MODE` равен `pose` или `ttl`;
- `DUAL_OE_BRIDGE_WIRE_FORMAT` равен `binary` или `json`;
- binary mode используется только с фиксированным порядком точек.

Фиксированный binary порядок:

```text
hl_ankle_l
hl_ankle_r
hl_hip_l
hl_hip_r
hl_toes_l
hl_toes_r
```

### 2. Создание bridge object

`OpenEphysBridge(logger)` читает:

```python
DUAL_OE_BRIDGE_ENABLED
DUAL_OE_BRIDGE_HOST
DUAL_OE_BRIDGE_PORT
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS
DUAL_OE_BRIDGE_PACKET_MODE
DUAL_OE_BRIDGE_WIRE_FORMAT
DUAL_OE_BRIDGE_REQUEST_ACK
```

Если bridge включен, `bridge.open()` создает UDP socket:

```text
socket.AF_INET, socket.SOCK_DGRAM
non-blocking
target = 127.0.0.1:47000
```

### 3. Создание runtime для камер

Для каждой записи `DUAL_CAMERAS` создается `CameraRuntime`.

Внутри хранится:

- имя камеры: `left` или `right`;
- serial number;
- путь к Galaxy config;
- объект source;
- последний кадр;
- счетчики FPS/drops;
- last error.

### 4. Reader threads

Для каждой камеры запускается отдельный поток `reader_loop`.

Он делает:

```text
source.read()
  -> FramePacket
  -> runtime.latest_packet = packet
  -> runtime.latest_seq += 1
  -> profiler.observe("camera/read", read_ms)
```

Если источник вернул ошибку, поток сохраняет `runtime.last_error`, и основной
loop потом поднимет исключение.

### 5. Первая пара

`wait_for_initial_pair(left, right, stop_event)` ждет, пока у обеих камер
появится хотя бы один кадр.

Ожидаемая строка:

```text
First pair received. host_dt=... left_shape=(220, 1920, 3) right_shape=(220, 1920, 3)
```

### 6. Инициализация DLCLive

Вызов:

```python
dlc_live = live.build_dlc_live(None)
model_cfg = dlc_live.read_config()
body_parts = live.extract_bodyparts(model_cfg)
```

Затем код проверяет, что все точки из `DUAL_USE_POINTS` есть в exported model.
Если какой-то точки нет, запуск останавливается с понятной ошибкой.

### 7. Fast pose-only mode

Если:

```python
DUAL_FAST_POSE_ONLY = True
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
```

то Python-side processor отключается:

```text
runtime.processor = None
```

Значит Python не строит filtered pose, triplet и angle. Он только отправляет
сырые точки.

### 8. Inference thread

`inference_loop` берет последние кадры:

```text
left_packet = latest left frame
right_packet = latest right frame
```

Дальше:

```text
host_dt_ms = packet_delta_ms(left_packet, right_packet)
camera_dt_ms = sdk_delta_ms(...)
```

Если batch поддерживается:

```text
process_frame(left)
process_frame(right)
torch.stack(...)
runner.model(batch)
postprocess two poses
```

Если batch не поддерживается:

```text
run_inference(left)
run_inference(right)
```

На выходе создается `PairInferenceResult`:

```text
pair_index
left_packet
right_packet
host_dt_ms
camera_dt_ms
left_result
right_result
```

### 9. Компактный pose result

В fast mode `raw_pose_result(...)` создает:

```python
{
    "infer_ms": ...,
    "preprocess_ms": ...,
    "model_infer_ms": ...,
    "raw_pose_array": np.ndarray shape (6, 3), dtype float32,
    "raw_visible": ...,
    "filtered_visible": raw_visible,
    "has_triplet": False,
    "hind_angle": None,
    "picked_side": runtime.name,
    "python_postprocess": False,
}
```

Здесь `has_triplet=False` и `hind_angle=None` не означают ошибку. В fast mode
Python специально не считает triplet/angle, потому что это делает plugin.

### 10. Отправка UDP

Основной loop берет свежий `PairInferenceResult` и вызывает:

```python
bridge.send(result, left, right)
```

Если выбран рабочий binary mode:

```text
_build_binary_pose_payload(...)
  -> BINARY_HEADER_STRUCT
  -> BINARY_SIDE_STRUCT для left
  -> 6 * float32 points для left
  -> BINARY_SIDE_STRUCT для right
  -> 6 * float32 points для right
  -> sock.sendto(data, ("127.0.0.1", 47000))
```

Если выбран резервный JSON-режим:

```text
_build_payload(...)
  -> raw_points dict
  -> json.dumps(...)
  -> sock.sendto(...)
```

Если legacy TTL:

```text
_build_ttl_payload(...)
  -> ttl_lines[0..7]
  -> json.dumps(...)
  -> sock.sendto(...)
```

## UDP bridge режимы

| Режим пакета | Формат передачи | Что отправляет Python | Что делает plugin |
| --- | --- | --- | --- |
| `pose` | `binary` | `DDLP`-пакет с `[6,3]` float32 точками. | Сам считает valid/triplet/angle/TTL. |
| `pose` | `json` | JSON `dual_dlc_live.pose.v1` с `raw_points`. | Сам считает valid/triplet/angle/TTL. |
| `ttl` | JSON | JSON `dual_dlc_live.v1` с готовым `ttl_lines`. | Переносит готовые TTL states в Open Ephys. |

Рабочий режим:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
```

## Binary packet `DDLP`

Header:

```text
magic: b"DDLP"
version: 1
flags: bit 0 = request ACK
pair_index: int64
host_time: float64
host_dt_ms: float32
camera_dt_ms: float32
point_count: uint16
reserved: uint16
```

Side block для каждой камеры:

```text
frame_id: int64
source_frame_id: int64
capture_ts: float64
infer_ms: float32
drops: uint32
raw_visible: uint16
reserved: uint16
points: point_count * (x float32, y float32, likelihood float32)
```

Binary packet не несет имена точек, поэтому порядок точек должен совпадать с
`BINARY_POSE_POINT_NAMES`.

## ACK

`DUAL_OE_BRIDGE_REQUEST_ACK = False` по умолчанию.

Для тестов можно включить ACK. Тогда:

- Python ставит ACK-флаг в binary packet или поле `ack=true` в JSON;
- plugin отправляет назад текст:

```text
dual_dlc_live.ack pair=5 mode=binary ttl=0x03 left_angle=135.00 right_angle=135.00
```

Обычный live sender не читает ACK в рабочем loop, потому что это лишняя
нагрузка. Для проверки используется `send_dual_dlc_bridge_test.py --wait-ack`
или временный диагностический monkeypatch.

## Профилировщик стадий

Включение:

```python
DUAL_ENABLE_STAGE_PROFILER = True
DUAL_PROFILE_LOG_EVERY_N_PAIRS = 120
DUAL_PROFILE_EMA_ALPHA = 0.10
```

Пример:

```text
stage_profile pair=120 result_hz=23.4 total_hz=20.1 skipped=0 last_ms camera/read=10.23 preprocess=1.10 inference=43.93 pack/send=0.20 display=0.00 | avg_ms ...
```

| Stage | Что входит |
| --- | --- |
| `camera/read` | Время успешного `source.read()` в reader thread. |
| `preprocess` | DLCLive `process_frame` для пары. |
| `inference` | PyTorch runner/model inference. |
| `pack/send` | Упаковка UDP и `socket.sendto`. |
| `display` | OpenCV overlay, resize, video writer, `imshow`. |
| `result_hz` | Частота обработанных left/right пар за последний интервал профайлера. |
| `total_hz` | Средняя частота обработанных пар с начала live loop. |

Как читать:

- высокий `camera/read`: SDK/USB/camera timing;
- высокий `preprocess`: CPU preprocessing, RGB conversion, resize/crop;
- высокий `inference`: модель/GPU path;
- высокий `pack/send`: сериализация или UDP socket;
- высокий `display`: окно/overlay/video writer грузят CPU.

Если:

```python
DUAL_DISPLAY_WINDOW = False
DUAL_SAVE_OUTPUT_VIDEO = False
```

то `display` должен быть около `0.00`.

## Запуск без камер: synthetic UDP test

Сначала запусти Open Ephys и добавь `Dual DLCLive Bridge`.

Рабочий binary-test:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
```

Ожидаемый результат:

```text
sent pair=1 mode=pose wire=binary ...
ack pair=1 ... dual_dlc_live.ack pair=1 mode=binary ttl=...
acked 5/5
```

Резервный JSON pose:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format json --count 5 --interval 0.025 --wait-ack
```

Legacy TTL:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode ttl --count 5 --interval 0.025 --wait-ack
```

## Запуск с камерами

1. Подключить обе Daheng камеры через USB3 hub.
2. Закрыть GalaxyView.
3. Запустить Open Ephys.
4. Добавить `Dual DLCLive Bridge`.
5. Проверить `enabled = true`, `udp_port = 47000`.
6. Запустить synthetic test и убедиться, что `acked 5/5`.
7. Запустить Python:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

## Нормальные runtime-сигналы

В `dual_rt_dlc_live_debug.log` должны быть строки:

```text
Open Ephys bridge enabled: UDP 127.0.0.1:47000 mode=pose wire=binary
Opened left sn=FDE22070174 ...
Opened right sn=FDE22070175 ...
First pair received. host_dt=...
CUDA_CHECK torch=... cuda=True gpu=...
Fast pose-only mode enabled ...
Model bodyparts loaded: 15; dual points=[...]
stage_profile pair=...
```

Если мыши нет, ожидаемо:

```text
left_triplet=False
right_triplet=False
left_angle=None
right_angle=None
ttl=0x00
```

Это означает, что нет валидных точек, а не то, что UDP/plugin сломаны.

## Что идет дальше на стимуляцию

Python не стимулирует напрямую. Python только передает точки позы в plugin.

Стимуляционный путь:

```text
сырые точки позы
  -> plugin фильтрует точки
  -> plugin считает triplet validity
  -> plugin считает hind angle
  -> plugin формирует TTL word
  -> Open Ephys event channel "Dual DLCLive TTL"
  -> следующий stimulation/output processor
```

Настройки порога угла ставятся в UI плагина:

```text
angle_trigger_enabled
angle_threshold_deg
refractory_ms
```

`DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` в Python нужен только для legacy `ttl`
mode. В рабочем `pose` mode он не является источником истины.

## Логи и файлы

| Файл | Назначение |
| --- | --- |
| `dual_rt_dlc_live_debug.log` | Основной live log, камеры, CUDA, profiler. |
| `dual_rt_dlc_live_benchmark.csv` | CSV benchmark, если `ENABLE_BENCHMARK_CSV=True`. |
| `dual_rt_dlc_live_left.mp4` | Видео левой камеры, если включено сохранение. |
| `dual_rt_dlc_live_right.mp4` | Видео правой камеры, если включено сохранение. |

## Устранение проблем

### Камеры не открываются

Проверь:

- GalaxyView закрыт;
- serial numbers совпадают с `DUAL_CAMERAS`;
- hub подключен к USB3;
- `.txt` configs существуют;
- Galaxy SDK видит `device_count 2`;
- `access_status = 0`.

### Open Ephys не получает пакеты

Проверь:

- Open Ephys запущен;
- plugin добавлен в chain;
- `enabled = true`;
- `udp_port = 47000`;
- synthetic test возвращает `acked 5/5`.

### GPU почти не загружен

Сначала смотри profiler:

- если высокие `preprocess` или `display`, bottleneck на CPU;
- если высокий `inference`, но GPU низкий, возможно модель/runner синхронизируется или batch path неэффективен;
- если `camera/read` около 10 ms, это нормально для 100 Hz камеры.

### Нет TTL при пустой дорожке

Это нормально. Без мыши нет валидных точек:

```text
ttl=0x00
left_angle=nan
right_angle=nan
```

Проверять plugin в таком случае лучше synthetic test, где точки создаются
искусственно и ACK показывает ожидаемый TTL.

### Нужно поменять bodyparts

Для custom point set:

1. Убедись, что новые точки есть в exported model config.
2. Измени `DUAL_SIDE_POINT_SETS`.
3. Проверь `DUAL_USE_POINTS`.
4. Если binary order изменился, поставь:

```python
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

Binary path требует фиксированный six-point порядок.
