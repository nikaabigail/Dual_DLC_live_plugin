# Профили запуска DLCLive

Сейчас рабочий режим такой: одна левая камера `FDE22070174` -> DLCLive/PyTorch -> UDP `DDLP` binary -> Open Ephys `DualDLCLiveBridge`.

Правая камера временно не используется. При проверке ее кадр смотрел не на treadmill/крысу, поэтому DLCLive давал почти нулевые likelihood (`right_p ~= 0.01`). Для single-режима Python отправляет реальные точки в левую сторону пакета, а правую сторону заполняет `NaN`, чтобы правые TTL-линии в плагине не срабатывали.

## Лучший вариант сейчас

Рекомендованный запуск:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-best
```

`single-best` включает:

```text
1 camera: left FDE22070174
GALAXY_OUTPUT_COLOR=rgb
CONVERT_TO_RGB=False
PRECISION=FP32
DUAL_TORCH_ALLOW_TF32=True
DUAL_CV2_NUM_THREADS=1
DUAL_TORCH_NUM_THREADS=12
DUAL_TORCH_INTEROP_THREADS=12
display=True
UDP=127.0.0.1:47000
```

Почему он лучший пока: это самый удачный компромисс из проверенных вариантов. GPU реально занят, лишний RGB shuffle выключен, CPU ниже, чем при полном auto, а скорость остается близкой к хорошему рабочему уровню. Сейчас OpenCV-окно включено у всех профилей, чтобы можно было смотреть, как точки ложатся на животное. Для финальной стимуляции запускай тот же профиль с `--no-display`, потому что рисование окна снижает FPS.

## Сейчас: просмотр точек

Для проверки детекции запускай любой профиль обычной командой, окно с видео и точками откроется автоматически:

```powershell
C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-best
C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-fp16
C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-strict
```

В окне можно нажать `q` или `Esc`, чтобы остановить live-процесс. Если нужно принудительно включить окно даже после будущих изменений профиля:

```powershell
C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-best --display
```

Для рабочего headless-режима без окна:

```powershell
C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-best --no-display
```

## Меню выбора

Можно запустить меню:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe run_live_profile.py
```

Из любой папки тоже можно, потому что добавлен shim:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --list
```

Если live-процесс уже запущен, камера занята. Это нормально: Galaxy SDK не дает открыть одну и ту же камеру вторым Python-процессом. Проверить статус:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --status
```

Остановить текущий live-процесс:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py --stop
```

Перезапустить текущий процесс с выбранным профилем:

```powershell
C:\dlc_live_env\Scripts\python.exe C:\Users\Владимир\run_live_profile.py single-best --replace
```

Посмотреть все профили без запуска:

```powershell
C:\dlc_live_env\Scripts\python.exe run_live_profile.py --list
```

`--list` только печатает список и завершает программу. Чтобы выбрать номер, запускай без `--list` или передавай номер сразу:

```powershell
C:\dlc_live_env\Scripts\python.exe run_live_profile.py
C:\dlc_live_env\Scripts\python.exe run_live_profile.py 1
```

Короткий тест без бесконечного запуска:

```powershell
C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-best --max-frames 300
```

## Профили

| Профиль | Когда выбирать | Команда |
| --- | --- | --- |
| `single-best` | Основной рабочий режим сейчас. Одна левая камера, Open Ephys plugin, FP32+TF32, RGB no-convert. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-best` |
| `single-strict` | Если нужна максимально консервативная численная проверка без TF32. Медленнее. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-strict` |
| `single-fp16` | Быстрый эксперимент. Перед стимуляцией обязательно сравнить точки/TTL с FP32. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-fp16` |
| `single-cpu` | Если CPU мешает Open Ephys или другим программам. Обычно ниже CPU, но возможна потеря FPS. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-cpu` |
| `single-debug` | Визуальная проверка точек поверх видео. Не для финальной стимуляции, окно снижает скорость. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-debug` |
| `single-rgb-on` | Fallback для проверки цвета: BGR из Galaxy + `CONVERT_TO_RGB=True`. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py single-rgb-on` |
| `dual-best` | Позже, когда обе камеры физически смотрят на treadmill. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py dual-best` |
| `dual-cpu` | Dual-вариант с меньшей CPU-нагрузкой. Сейчас не основной. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py dual-cpu` |
| `dual-fp16` | Dual FP16 эксперимент. Только для проверки. | `C:\dlc_live_env\Scripts\python.exe run_live_profile.py dual-fp16` |

## Прямой запуск без меню

Single runtime можно запускать напрямую:

```powershell
C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best
```

Можно явно выбрать камеру и сторону пакета:

```powershell
C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --camera left --plugin-side left
```

Dual runtime тоже теперь принимает профили:

```powershell
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py --profile dual-best
```

## Что получает Open Ephys

В single-режиме формат UDP не меняется: это тот же binary `DDLP/v1`, который уже понимает плагин.

```text
left side:
  frame metadata
  6 raw pose points [x, y, likelihood]

right side:
  frame metadata-заглушка
  6 points = NaN
```

В плагине это означает:

```text
TTL bit 0 / line 0 -> left triplet valid
TTL bit 2 / line 2 -> left angle trigger
TTL bit 1 / line 1 -> right triplet valid, сейчас должен молчать
TTL bit 3 / line 3 -> right angle trigger, сейчас должен молчать
```

## Логи

Основной single log:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation\single_rt_dlc_live_bridge_debug.log
```

Ищи строки:

```text
LIVE PROFILE: single-best
CUDA_CHECK ... cuda=True gpu=NVIDIA GeForce RTX 5070 Laptop GPU
DLC_MODEL_DEVICE_AFTER_INIT=cuda:0 ... precision=FP32 convert2rgb=False
frame=... raw_visible=... p=...
stage_profile frame=... result_hz=... camera/read=... inference=... pack/send=...
```

`raw_visible` и `p=` показывают, сколько из 6 точек реально видно и какие likelihood у каждой точки. Если `raw_visible` низкий, стимуляция не будет надежной независимо от скорости.

## Что не использовать сейчас

`dual-*` профили не являются рабочими для текущей физической расстановки, потому что правая камера сейчас не дает treadmill-видео. Их стоит включать только после того, как правая картинка в snapshot будет содержать крысу/дорожку.
