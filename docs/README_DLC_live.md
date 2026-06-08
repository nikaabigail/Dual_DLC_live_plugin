# Однокамерный DLCLive runtime

Этот файл описывает старый runtime для одной камеры:

```text
rt_dlc_live.py
config_rt_dlc_live.py
```

Рабочая система с двумя камерами и Open Ephys сейчас использует:

```text
dual_rt_dlc_live.py
config_dual_rt_dlc_live.py
Dual DLCLive Bridge
```

Однокамерный runtime полезен для проверки модели, Galaxy SDK, offline video и
отладки Python-side обработки без Open Ephys.

## Навигация

| Раздел | Для чего нужен |
| --- | --- |
| Роль однокамерного runtime | Что делает `rt_dlc_live.py`. |
| Конфиг | Главные настройки `config_rt_dlc_live.py`. |
| Источник кадров | Galaxy camera, резервный OpenCV-режим или video file. |
| Модель DLCLive | Путь к модели, GPU, preprocessing. |
| Python-процессор | Фильтр, точки и угол в однокамерном режиме. |
| Порядок выполнения | От кадра до overlay/log/video. |
| Запуск | Команды запуска. |
| Логи | Где смотреть результат. |
| Связь с dual runtime | Какие настройки общие. |

## Роль однокамерного runtime

`rt_dlc_live.py` - это realtime pipeline для одной камеры или одного видеофайла.

Он делает:

- открывает frame source;
- строит DLCLive object;
- запускает `init_inference`/`get_pose`;
- применяет Python-side online processor;
- считает hind angle в Python;
- рисует OpenCV overlay;
- пишет benchmark CSV;
- при необходимости сохраняет output video.

В отличие от рабочего dual path, здесь Python сам считает filtered points и
angle. В dual рабочем path это перенесено в C++ plugin Open Ephys.

## Конфиг

Основной файл:

```text
python/config_rt_dlc_live.py
```

Dual config импортирует его:

```python
from config_rt_dlc_live import *
```

Поэтому изменения в `config_rt_dlc_live.py` могут затронуть и single-camera, и
dual-camera runtime, если `config_dual_rt_dlc_live.py` не переопределяет
конкретный параметр.

## Источник кадров

Выбор источника:

```python
USE_VIDEO_FILE = False
CAMERA_BACKEND = "galaxy"  # "galaxy" или "opencv"
```

Видео файл:

```python
VIDEO_FILE_PATH = r"C:\dlc\videos\..."
VIDEO_TARGET_FPS = 0.0
VIDEO_SKIP_IF_BEHIND = False
```

Резервный OpenCV-режим:

```python
CAM_INDEX = 1
FRAME_W = 1920
FRAME_H = 1080
TARGET_VIDEO_FPS = 100.0
```

Galaxy camera:

```python
GALAXY_SN = "FDE22070173"
GALAXY_INDEX = 1
GALAXY_CONFIG_PATH = r"C:\config_daheng\Rat_TREDMILL_Top_1920px_340px_100Hz_(FDE22070173).txt"
GALAXY_IMPORT_CONFIG = True
GALAXY_CONFIG_VERIFY = False
```

Low-latency:

```python
GALAXY_LOW_LATENCY = True
GALAXY_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
GALAXY_ACQUISITION_BUFFER_COUNT = 2
GALAXY_DRAIN_QUEUED_FRAMES = True
GALAXY_MAX_DRAIN_FRAMES = 20
```

## Модель DLCLive

Параметры:

```python
MODEL_PATH = r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\...\snapshot-best-380.pt"
MODEL_TYPE = "pytorch"
PRECISION = "FP32"
DEVICE = "cuda"
SINGLE_ANIMAL = True
CONVERT_TO_RGB = True
```

Preprocessing DLCLive:

```python
CROPPING = None
RESIZE = 1.0
DYNAMIC_CROPPING = (False, 0.5, 10)
```

Если кадр уже аппаратно обрезан камерой, `CROPPING = None` правильнее, иначе
DLCLive может повторно обрезать уже узкую полосу.

## Python-процессор

В single-camera runtime processor включен:

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
```

Точки:

```python
USE_POINTS = [
    "hl_hip_l",
    "hl_ankle_l",
    "hl_toes_l",
]
```

Угол:

```python
COMPUTE_HIND_ANGLE = True
HIND_ANGLE_POINTS = ("hl_hip_l", "hl_ankle_l", "hl_toes_l")
```

Здесь angle считается в Python, потому что это автономный однокамерный режим.

## Порядок выполнения

### 1. Источник отдает кадр

Источник может быть:

- Galaxy camera;
- OpenCV camera;
- video file.

На выходе получается:

```text
frame
frame_id
capture_ts
source metadata
```

### 2. DLCLive preprocessing

Кадр проходит:

```text
convert to RGB, если CONVERT_TO_RGB=True
cropping, если CROPPING задан
resize, если RESIZE != 1.0
dynamic cropping, если включен
```

### 3. DLCLive inference

Первый кадр:

```text
dlc_live.init_inference(frame)
```

Следующие кадры:

```text
dlc_live.get_pose(frame)
```

Результат - pose array в порядке bodyparts модели.

### 4. Online processor

Processor берет pose:

```text
raw pose
  -> confidence cutoff
  -> despike
  -> median smoothing
  -> optional hold
  -> filtered pose
```

### 5. Triplet и angle

Из filtered pose выбираются:

```text
hl_hip_l
hl_ankle_l
hl_toes_l
```

Если все точки валидны, считается hind angle в ankle.

### 6. Overlay, log и video

Runtime рисует:

- точки;
- имена точек;
- confidence;
- FPS;
- angle;
- debug text.

Затем пишет log/CSV/video, если это включено.

## Запуск

Из live-папки:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe rt_dlc_live.py
```

Для проверки видеофайла поставить:

```python
USE_VIDEO_FILE = True
VIDEO_FILE_PATH = r"C:\path\to\video.avi"
```

## Ожидаемые признаки работы

В логе:

```text
Model bodyparts loaded: ...
frame=...
infer_ms=...
angle=...
```

В окне OpenCV:

- виден кадр;
- поверх кадра рисуются точки;
- FPS/angle обновляются.

Если мыши нет или модель не видит точки, angle будет `None`, а triplet будет
невалиден. Это нормально для пустого кадра.

## Логи и output-файлы

| Файл | Назначение |
| --- | --- |
| `rt_dlc_live_debug.log` | Основной log single-camera runtime. |
| `rt_dlc_live_benchmark.csv` | Benchmark CSV, если включен. |
| `rt_dlc_live_output.mp4` | Output video, если `SAVE_OUTPUT_VIDEO=True`. |

## Связь с dual runtime

Dual runtime импортирует `config_rt_dlc_live.py`, но переопределяет важные вещи:

| Параметр | Single-camera | Dual рабочий режим |
| --- | --- | --- |
| Камеры | Одна камера или video file. | Две Daheng камеры. |
| ROI | Может быть software cropping. | ROI уже в Galaxy `.txt`, `CROPPING=None`. |
| Processor | Python-side filter/angle. | C++ plugin filter/angle/TTL. |
| UDP | Нет рабочего UDP. | UDP `DDLP` на Open Ephys. |
| TTL | Нет Open Ephys TTL. | `Dual DLCLive TTL` в plugin. |

Если нужно проверить только модель или одну камеру, используй `rt_dlc_live.py`.
Если нужно проверить stimulation path, используй `dual_rt_dlc_live.py` и Open
Ephys plugin.

## Устранение проблем

### DLCLive import падает из-за `colorcet`

Установить dependency в активную среду `C:\dlc_live_env`.

### Камера не открывается

Проверить:

- GalaxyView закрыт;
- serial number правильный;
- Galaxy SDK видит камеру;
- `.txt` config существует;
- камера не занята другим процессом.

### GPU не используется

Проверить:

```python
DEVICE = "cuda"
MODEL_TYPE = "pytorch"
```

И в логах/диагностике:

```text
torch.cuda.is_available() == True
```

### Overlay тормозит

Для чистой проверки inference выключить:

```python
DISPLAY_WINDOW = False
SAVE_OUTPUT_VIDEO = False
```

### Offline video идет не в том темпе

Проверить:

```python
VIDEO_TARGET_FPS
VIDEO_SKIP_IF_BEHIND
```

Для анализа записи обычно лучше не пропускать кадры, чтобы не ломать timeline.
