# Dual DLC Live Plugin

Основная задача системы: получить видеопоток с двух камер, посчитать
позу через DLCLive/PyTorch, передать сырые точки в плагин Open Ephys по UDP и
уже внутри плагина сформировать TTL-состояния для дальнейшей стимуляции.

> **Установка с нуля** (куда клонировать, как поставить окружение, как запустить): **[`docs/INSTALL_AND_RUN.md`](docs/INSTALL_AND_RUN.md)**.

## Карта документации

| Файл | Что в нем искать |
| --- | --- |
| `README.md` | Общая схема, порядок выполнения, быстрый запуск, связь UDP/TTL. |
| `docs/INSTALL_AND_RUN.md` | Установка с нуля: клонирование, драйвер, Python, venv, SDK, модель, плагин, запуск. |
| `python/README_OPEN_EPHYS_BRIDGE.md` | Подробно про Python runtime: камеры, DLCLive, binary fast-mode, profiler, логи. |
| `open_ephys_plugin/DualDLCLiveBridge/README.md` | Подробно про C++ плагин Open Ephys: параметры, входные пакеты, фильтрация, TTL. |
| `docs/README_DLC_live.md` | Старый single-camera `rt_dlc_live.py`, нужен для отладки одной камеры и модели. |
| `camera_configs/*.txt` | Снимки Daheng/Galaxy конфигов для левой и правой treadmill камер. |

Рекомендуемый порядок чтения:

1. Этот ReadMe - общая навигация по плагину.
2. `python/README_OPEN_EPHYS_BRIDGE.md`, когда запускаешь камеры и DLCLive.
3. `open_ephys_plugin/DualDLCLiveBridge/README.md`, когда настраиваешь Open Ephys и TTL.
4. `docs/README_DLC_live.md`, если нужен однокамерный режим (изначальный MVP).

## Коротко: кто за что отвечает

В рабочем режиме обязанности разделены так:

| Часть | Что делает | Чего не делает |
| --- | --- | --- |
| Daheng/Galaxy камеры | Отдают уже аппаратно обрезанные кадры `1920x220` за счет импорта соответсвующих config файлов. | Не считают точки, углы или TTL. |
| `dual_rt_dlc_live.py` | Открывает камеры, спаривает кадры, запускает DLCLive/PyTorch, отправляет сырые точки позы по UDP. | В обычном режиме не принимает решение о стимуляции. |
| Open Ephys `Dual DLCLive Bridge` | Принимает UDP, фильтрует точки, выбирает triplet, считает угол, формирует TTL-линии. | Не открывает камеры и не запускает нейросеть. |
| Downstream stimulation/output processor | Использует TTL-события из Open Ephys для физического выхода или стимуляции. | Не знает про камеры и DLCLive напрямую. |

## Главный рабочий путь

```text
Левая камера FDE22070174
Правая камера FDE22070175
  -> Daheng Galaxy SDK
  -> dual_rt_dlc_live.py
  -> FramePacket(left) + FramePacket(right)
  -> DLCLive/PyTorch
  -> raw_pose_array [6, 3]
  -> UDP 127.0.0.1:47000, binary DDLP/v1
  -> Open Ephys plugin "Dual DLCLive Bridge"
  -> фильтр точек + выбор triplet + расчет hind angle
  -> TTL word 8 bit
  -> Open Ephys event channel "Dual DLCLive TTL"
  -> следующий processor стимуляции/output
```

Важно: в нормальном режиме Python отправляет не готовые `ttl_lines`, а
сырые координаты точек и metadata. Решение “какая TTL-линия должна быть активна”
принимает C++ плагин внутри Open Ephys.

## Текущие рабочие настройки

Файл: `python/config_dual_rt_dlc_live.py`.

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS = 1
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

Пояснение:

- `pose`: Python отправляет точки позы, а не готовые TTL.
- `binary`: точки упакованы в компактный UDP-пакет `DDLP`, без JSON.
- `DUAL_FAST_POSE_ONLY = True`: Python не строит фильтрованные точки и углы для рабочего пути.
- `DUAL_ENABLE_BATCH_INFERENCE = True`: две камеры по возможности считаются одним mini-batch.
- `DUAL_ENABLE_STAGE_PROFILER = True`: логируются стадии `camera/read`, `preprocess`, `inference`, `pack/send`, `display`.
- `DUAL_DISPLAY_WINDOW = False`: рабочий режим не тратит время на OpenCV overlay/window.
- `GALAXY_OUTPUT_COLOR = "rgb"` и `CONVERT_TO_RGB = False`: Daheng/Galaxy уже отдает RGB после Bayer-конверсии, поэтому DLCLive не делает лишний BGR->RGB shuffle.

CPU throttling оставлен как ручной переключатель:

```python
DUAL_CV2_NUM_THREADS = -1
DUAL_TORCH_NUM_THREADS = 0
DUAL_TORCH_INTEROP_THREADS = 0
```

`-1/0` означает “не менять стандартные threadpool-настройки”. Если CPU важнее
максимальной частоты, можно поставить все три значения в `1`, но на текущем
тесте это снижало `result_hz`.

## Камеры и ROI

Текущая привязка камер:

| Сторона | Серийный номер | Конфиг Galaxy |
| --- | --- | --- |
| `left` | `FDE22070174` | `C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt` |
| `right` | `FDE22070175` | `C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt` |

В этих `.txt` файлах уже задан аппаратный ROI камеры. При проверке live-run
ожидаемые значения были такими:

| Камера | Width | Height | OffsetY | FPS |
| --- | --- | --- | --- | --- |
| left `FDE22070174` | `1920` | `220` | `510` | `100` |
| right `FDE22070175` | `1920` | `220` | `530` | `100` |

Поэтому в dual runtime программная обрезка выключена:

```python
CROPPING = None
RESIZE = 1.0
DYNAMIC_CROPPING = (False, 0.5, 10)
GALAXY_OUTPUT_COLOR = "rgb"
CONVERT_TO_RGB = False
```

Если GalaxyView открыт и держит камеры, Python может не открыть устройства.
Перед запуском `dual_rt_dlc_live.py` GalaxyView лучше закрывать.

## Точки позы

В рабочем binary-режиме используется фиксированный набор из шести точек:

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

Почему порядок именно такой: `DUAL_USE_POINTS` собирается через `sorted(...)`.
Binary packet не передает имена точек в каждом кадре, поэтому C++ плагин ожидает
строго этот порядок. Если порядок или состав точек нужно менять, надо временно
переключиться на JSON:

```python
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

## Структура и порядок выполнения

Унифицированнный путь одного live pair от камер до TTL.

### 1. Камера отдает кадр

Источник: Daheng/Galaxy SDK.

Что берется:

- серийный номер из `DUAL_CAMERAS`;
- путь к `.txt` конфигу Galaxy;
- аппаратные настройки из конфига: ROI, FPS, exposure, trigger mode;
- сырой кадр камеры.

Что получается в Python:

```text
FramePacket
  frame: np.ndarray, shape примерно (220, 1920, 3)
  frame_id: локальный номер кадра
  source_frame_id: номер кадра SDK, если доступен
  capture_ts: время получения кадра на host
  source_timestamp: timestamp камеры, если доступен
```

Кадры читаются в отдельных reader threads. Старые кадры намеренно сбрасываются,
потому что для real-time стимуляции важнее свежий кадр, а не обработка каждого
кадра из очереди.

### 2. Python спаривает левый и правый кадр

`dual_rt_dlc_live.py` берет последнюю доступную пару:

```text
left.latest_packet + right.latest_packet -> PairInferenceResult
```

В metadata пары записывается:

- `pair_index`: номер пары;
- `host_dt_ms`: разница времен получения left/right на host;
- `camera_dt_ms`: разница timestamp камер, если SDK ее отдал;
- счетчики dropped frames.

### 3. DLCLive/PyTorch считает позу

Python создает DLCLive из `MODEL_PATH` в `config_rt_dlc_live.py`.

Дальше для каждого кадра:

```text
camera frame
  -> dlc_live.process_frame(...)
  -> dlc_live.runner.init_inference(...) или get_pose(...)
  -> pose ndarray в порядке bodyparts модели
```

Если batch path поддерживается, левая и правая картинки после preprocessing
складываются в mini-batch и проходят модель одним вызовом. Если runner не
поддерживает быстрый batch path, код откатывается к последовательным вызовам.

### 4. Python сжимает pose в маленький массив

Из полного `pose ndarray` выбираются только шесть нужных точек.

Результат:

```text
raw_pose_array
  shape: (6, 3)
  dtype: float32
  columns: x, y, likelihood
  rows: DUAL_USE_POINTS order
```

В binary fast-mode Python не строит `raw_points` dict для отправки. Это важно
для производительности: меньше Python-объектов, меньше JSON, меньше CPU.

`raw_points` dict создается только если:

- выбран резервный JSON-режим;
- включен local OpenCV overlay и нужны имена точек для рисования;
- включен legacy/diagnostic Python-side postprocess.

### 5. Python пакует UDP

Рабочий UDP-пакет:

```text
UDP datagram
  magic: "DDLP"
  version: 1
  flags: ACK bit, если DUAL_OE_BRIDGE_REQUEST_ACK=True
  pair_index
  host_time
  host_dt_ms
  camera_dt_ms
  point_count = 6
  left side metadata + 6 * (x, y, likelihood)
  right side metadata + 6 * (x, y, likelihood)
```

Куда отправляется:

```text
127.0.0.1:47000
```

Это UDP, не TCP. UDP не создает соединение и сам по себе не гарантирует
доставку. Поэтому для тестов есть ACK-режим: Python или тестовый sender ставит
ACK-флаг, а плагин возвращает текстовый ответ `dual_dlc_live.ack ...`.

В обычном live config:

```python
DUAL_OE_BRIDGE_REQUEST_ACK = False
```

ACK выключен, чтобы не добавлять лишний обмен на каждом кадре.

### 6. Open Ephys plugin принимает UDP

Плагин `Dual DLCLive Bridge` слушает UDP port `47000`.

Если первые 4 байта равны `DDLP`, плагин идет по binary path:

```text
applyDatagram(...)
  -> applyBinaryPosePacket(...)
  -> parse left/right raw points
  -> evaluateSidePosePoints(...)
```

Если пакет JSON:

- `schema = "dual_dlc_live.pose.v1"`: плагин берет `raw_points` и сам считает TTL;
- `schema = "dual_dlc_live.v1"`: legacy mode, Python уже прислал `ttl_lines`.

### 7. Плагин фильтрует точки

Для каждой стороны плагин применяет параметры из UI:

| Параметр | Что делает |
| --- | --- |
| `conf_thresh_use` | Минимальный likelihood, чтобы точка участвовала в фильтрации. |
| `enable_pcutoff` | Включает отбрасывание точек ниже `conf_thresh_use`. |
| `enable_despike` | Отбрасывает резкие скачки точки. |
| `despike_threshold_px` | Максимальный разрешенный скачок в пикселях. |
| `despike_reset_gap_frames` | Через сколько пропущенных кадров разрешить заново поймать точку. |
| `median_window` | Размер медианного сглаживания. |
| `enable_hold` | Разрешает временно удерживать последнюю хорошую точку. |
| `max_hold_frames` | Сколько кадров можно удерживать последнюю хорошую точку. |
| `conf_thresh_draw` | Порог, чтобы triplet считался видимым/валидным. |

Если модели крысы нет и точек не видно, triplet будет `False`, угол будет
`nan/None`, а TTL будет `0x00`. Это нормальный результат пустого кадра.

### 8. Плагин выбирает triplet и считает угол

Для левой камеры используется:

```text
hl_hip_l -> hl_ankle_l -> hl_toes_l
```

Для правой камеры:

```text
hl_hip_r -> hl_ankle_r -> hl_toes_r
```

Hind angle считается в точке ankle:

```text
angle = angle(hip - ankle, toes - ankle)
```

Если включен `angle_trigger_enabled`, плагин сравнивает угол с
`angle_threshold_deg`.

### 9. Плагин собирает TTL word

TTL word - это 8 бит, где каждый бит соответствует линии TTL.

| TTL line | Бит | Что означает |
| --- | --- | --- |
| `0` | `0x01` | Левый triplet валиден. |
| `1` | `0x02` | Правый triplet валиден. |
| `2` | `0x04` | Левый angle trigger активен. |
| `3` | `0x08` | Правый angle trigger активен. |
| `4..7` | `0x10..0x80` | Зарезервировано. |

Примеры:

| TTL word | Значение |
| --- | --- |
| `0x00` | Ничего не активно, точки невалидны или триггеров нет. |
| `0x01` | Валиден только левый triplet. |
| `0x02` | Валиден только правый triplet. |
| `0x03` | Валидны левый и правый triplet. |
| `0x07` | Левый triplet, правый triplet и левый angle trigger. |

Важно: `angle_trigger_enabled` по умолчанию выключен. Пока он выключен, линии
`2` и `3` не будут активироваться даже при подходящем угле.

### 10. TTL попадает в Open Ephys

Плагин создает Open Ephys event channel:

```text
Dual DLCLive TTL
```

Внутри C++ логика такая:

```text
queueTtlWord(ttlWord)
  -> если word изменился, положить новое состояние в очередь
  -> process(...)
  -> emitPendingTtlState(...)
  -> setTTLState(sampleIndex, line, nextState)
```

То есть плагин сообщает Open Ephys, что TTL line изменила состояние. Дальше
уже downstream processor/Open Ephys output использует эти event states для
физического выхода или стимуляции.

## Быстрый запуск

### 1. Запустить Open Ephys

Нужен Open Ephys из debug build:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug
.\open-ephys.exe
```

В signal chain добавить `Dual DLCLive Bridge`.

Минимальные параметры:

```text
enabled = true
udp_port = 47000
angle_trigger_enabled = true, если нужны линии 2/3 для стимуляции
angle_threshold_deg = нужный порог
```

### 2. Проверить плагин без камер

Рабочий binary-путь:

```powershell
cd C:\dlc\Dual_DLC_live_plugin\python
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
```

Ожидаемый результат:

```text
acked 5/5
dual_dlc_live.ack ... mode=binary ...
```

Если `missing ack`, значит Open Ephys не запущен, плагин не добавлен в chain,
`enabled=false`, или порт не `47000`.

### 3. Запустить dual DLCLive

Закрыть GalaxyView, затем:

```powershell
cd C:\dlc\Dual_DLC_live_plugin\python
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

(Однокамерный боевой режим: `C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --no-display`.)

В логе должны появиться:

```text
Open Ephys bridge enabled: UDP 127.0.0.1:47000 mode=pose wire=binary
Opened left sn=FDE22070174 ...
Opened right sn=FDE22070175 ...
CUDA_CHECK ... cuda=True gpu=NVIDIA GeForce RTX 5070 Laptop GPU
Fast pose-only mode enabled ...
stage_profile pair=... result_hz=... total_hz=... skipped=... last_ms ... | avg_ms ...
```

## Как понять, что все работает

| Признак | Что значит |
| --- | --- |
| `device_count 2` в Galaxy SDK | Обе камеры видны. |
| `Opened left/right ... native={'Width': 1920, 'Height': 220}` | Камеры открылись с нужным ROI. |
| `CUDA_CHECK ... cuda=True` | PyTorch видит GPU. |
| `DLC_MODEL_DEVICE_AFTER_INIT = cuda:0` | Модель реально переехала на GPU после init. |
| `stage_profile ... pack/send=...` | Python отправляет UDP-пакеты. |
| `dual_dlc_live.ack ... mode=binary` | Плагин принял binary UDP и ответил. |
| В plugin UI растет `pkts` | Open Ephys plugin получает live-пакеты. |
| `ttl 0x..` меняется | Плагин меняет TTL state word. |

Если мыши нет, нормальная картина:

```text
left_triplet=False
right_triplet=False
left_angle=None
right_angle=None
ttl=0x00
```

Это значит, что pipeline работает, но модель не видит валидные hip/ankle/toes.

## Профилировщик стадий

Строка:

```text
stage_profile pair=120 result_hz=23.4 total_hz=20.1 skipped=0 last_ms camera/read=10.0 preprocess=1.5 inference=60.0 pack/send=0.4 display=0.0 | avg_ms ...
```

Смысл стадий:

| Stage | Что измеряет |
| --- | --- |
| `camera/read` | Чтение кадра Galaxy SDK в reader thread. |
| `preprocess` | DLCLive preprocessing: RGB, resize/crop transforms. |
| `inference` | PyTorch/DLCLive inference для текущей пары. |
| `pack/send` | Упаковка binary/JSON UDP и `sendto`. |
| `display` | OpenCV overlay, resize, video writer, `imshow`. |
| `result_hz` | Частота обработанных left/right пар за последний интервал профайлера. |
| `total_hz` | Средняя частота обработанных пар с начала live loop. |

Если `display=0.00`, значит окно и сохранение видео выключены или не тратят
значимое CPU. Если `inference` прыгает до сотен ms, узкое место сейчас модель/GPU
path, а не UDP и не TTL.

## Совместимые режимы

Рабочий режим:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
```

Резервный JSON pose:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

Legacy TTL mode:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "ttl"
```

В legacy TTL mode Python сам формирует `ttl_lines`, а плагин только переносит
эти состояния в Open Ephys TTL channel. Этот режим оставлен для совместимости и
отладки, но не является текущим рабочим путем.

## Типичные проблемы

### `missing ack`

Проверить:

- Open Ephys запущен;
- `Dual DLCLive Bridge` добавлен в signal chain;
- `enabled = true`;
- `udp_port = 47000`;
- firewall не блокирует localhost UDP;
- тест запускается с `--wait-ack`.

### Камеры не открываются

Проверить:

- GalaxyView закрыт;
- оба serial number видны в Galaxy SDK;
- hub подключен к USB3;
- `.txt` конфиги существуют;
- камеры не заняты другим процессом.

### TTL не появляется на стимуляции

Проверить:

- в plugin UI растет `pkts`;
- `ttl` меняется с `0x00` на ожидаемые значения;
- `angle_trigger_enabled = true`, если нужны линии `2` и `3`;
- следующий stimulation/output processor подключен к channel `Dual DLCLive TTL`;
- Open Ephys acquisition/processing запущен.

### Точки не детектятся

Если мыши нет, это нормально. Если мышь есть:

- проверь освещение и ROI;
- проверь, что модель обучена на такой вид камеры;
- смотри `raw_visible`, `left_triplet`, `right_triplet`;
- временно включи `DUAL_DISPLAY_WINDOW = True`, чтобы увидеть overlay.
