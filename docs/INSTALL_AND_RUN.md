# Установка и запуск с нуля

Полная инструкция: что скачать, **куда клонировать**, как поставить окружение и **как запустить** систему (камеры → DLCLive → UDP → Open Ephys плагин → TTL). Под чистый Windows-ПК с NVIDIA RTX 5070.

Архитектуру и поток данных см. в [`README.md`](../README.md). Этот файл — про инсталляцию.

---

## 0. Карта «что и куда» (создать эти папки)

| Что | Откуда | Куда (точный путь) |
|---|---|---|
| **Этот репозиторий** | `git clone` (см. §1) | `C:\dlc\Dual_DLC_live_plugin\` |
| Python 3.10.11 | python.org | установщик ставит сам |
| venv | создаётся вручную (см. §4) | `C:\dlc_live_env\` |
| Модель DLC `.pt` (93 МБ) | перенести с рабочей машины | `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\` |
| Конфиги камер `.txt` | из репозитория `camera_configs\` → скопировать | `C:\config_daheng\` |
| Galaxy SDK | daheng-imaging | `C:\Program Files\Daheng Imaging\GalaxySDK\` |
| Папка для логов | создать пустую | `C:\dlc\DLC_OBS_Spinal_cord_stimulation\` (см. §7) |
| Open Ephys плагин DLL | собрать/скопировать (см. §8) | `%LOCALAPPDATA%\Open Ephys\plugins-api10\` |

> ⚠️ Код использует **абсолютные пути** (раздел §7). Самый простой путь — воссоздать ту же раскладку `C:\`, тогда править ничего не нужно.

---

## 1. Куда клонировать репозиторий

```powershell
mkdir C:\dlc 2>$null
cd C:\dlc
git clone https://github.com/nikaabigail/Dual_DLC_live_plugin.git
# получится C:\dlc\Dual_DLC_live_plugin
```

Состав:
- `python\` — live-рантайм (запускается отсюда);
- `open_ephys_plugin\DualDLCLiveBridge\` — исходники C++ плагина Open Ephys;
- `camera_configs\` — `.txt` конфиги камер Daheng (Left/Right);
- `dist\windows-x64-release\DualDLCLiveBridge.dll` — собранный плагин (Release, API 10; для боевого OE). Debug-вариант — в `dist\windows-x64-debug\`;
- `docs\` — документация.

## 2. Драйвер NVIDIA

Скачать с **nvidia.com** последний драйвер для **RTX 5070, Windows 11** (Game Ready/Studio), «Custom → clean install», перезагрузка.
Проверка: `nvidia-smi` → видит `NVIDIA GeForce RTX 5070`, `CUDA Version ≥ 12.8`.
**CUDA Toolkit ставить НЕ нужно** — torch несёт CUDA 12.8 внутри колеса.

## 3. Python 3.10.11 (64-bit)

Скачать: https://www.python.org/downloads/release/python-31011/ → Windows installer (64-bit). Отметить «Add python.exe to PATH» (или использовать `py -3.10`).
Версия строго **3.10** — закреплённые колёса собраны под `cp310`.

## 4. Окружение (venv) + зависимости

```powershell
# 1) создать venv
py -3.10 -m venv C:\dlc_live_env
C:\dlc_live_env\Scripts\python.exe -m pip install --upgrade pip

# 2a) torch/torchvision/torchaudio — ТОЛЬКО с индекса CUDA 12.8 (на PyPI их нет):
C:\dlc_live_env\Scripts\python.exe -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2b) остальное — с обычного PyPI:
C:\dlc_live_env\Scripts\python.exe -m pip install -r C:\dlc\Dual_DLC_live_plugin\python\requirements.txt
```

Точный слепок версий — `python\requirements-live-lock-2026-06-16.txt` (57 пакетов).

Проверка GPU:
```powershell
C:\dlc_live_env\Scripts\python.exe -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# ожидается: True NVIDIA GeForce RTX 5070 (12, 0)
```

Нюансы:
- **torch строго `+cu128`** (CUDA внутри колеса; Toolkit не нужен). Не вешать `--index-url` на весь `requirements.txt` — иначе остальные пакеты не найдутся.
- `numpy` строго `1.26.4` (<2).
- `gxipy` ставится НЕ через pip, а из Galaxy SDK (§5).
- Предупреждение `pip check` про `opencv-python-headless` — безвредно (нужен полный `opencv-python` для окна).

## 5. Galaxy SDK + камеры

Установить **Galaxy SDK 1.18.2208.9301** (Windows x64) от Daheng в путь по умолчанию `C:\Program Files\Daheng Imaging\GalaxySDK`. Инсталлятор ставит USB3-драйвер, GenTL, GenICam, `gxipy` (в `Samples\Python SDK`) и системные env-переменные. Перезагрузка.
Камеры (Left `FDE22070174`, Right `FDE22070175`) — в порт **USB 3.0**.
Проверка: GalaxyView видит камеры → потом **закрыть** его (иначе Python не откроет устройства).

## 6. Модель и конфиги камер

**Модель** (93 МБ) положить в:
```
C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\
  DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\
    DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt
```
**Конфиги камер**: скопировать из репозитория `camera_configs\*.txt` (и/или с рабочей машины) в `C:\config_daheng\`.
⚠️ При правке `.txt` держать строки чистыми (`Feature<TAB>Value`, без inline `#`-комментариев и хвостовых пробелов) — иначе ломается `import_config_file`.

## 7. Зашитые пути (важно)

Часть путей берётся из **переменных окружения** (можно переопределить):
`DLC_LIVE_MODEL_PATH`, `DLC_LIVE_GALAXY_SDK_ROOT`, `DLC_LIVE_GALAXY_CONFIG_PATH`, `DLC_LIVE_GALAXY_SN`, `DLC_LIVE_VIDEO_PATH`, `DLC_LIVE_CAMERA_BACKEND`.

Часть путей **зашита жёстко** (без override) в `python\`:
- `WORK_DIR = C:\dlc\DLC_OBS_Spinal_cord_stimulation` (`live_profiles.py`) — туда пишутся логи/бенчмарки профилей;
- `LOG_PATH`/`BENCHMARK_CSV_PATH = C:\dlc\DLC_OBS_Spinal_cord_stimulation\...` (`config_dual_rt_dlc_live.py`, `config_rt_dlc_live.py`);
- `DUAL_CAMERAS[*].config_path = C:\config_daheng\...` (`config_dual_rt_dlc_live.py`).

**Поэтому проще всего:**
1. создать пустую папку `C:\dlc\DLC_OBS_Spinal_cord_stimulation` (туда лягут логи), и
2. держать модель в `C:\dlc\project\...`, конфиги в `C:\config_daheng\`, venv в `C:\dlc_live_env`.

Если такая раскладка не подходит — поправить `WORK_DIR` и две строки `LOG_PATH` в `python\live_profiles.py` / `python\config_*_rt_dlc_live.py` под свои пути (это единственное, что не покрыто env-переменными).

## 8. C++ плагин Open Ephys (DualDLCLiveBridge)

Open Ephys на целевом — **1.0.1 (plugin API 10)**, Release. Готовый `dist\windows-x64-debug\DualDLCLiveBridge.dll` — **Debug**, в Release-хост не загрузится. Поэтому:

1. На машине с Visual Studio собрать плагин в **Release** против дерева `plugin-GUI` 1.0.1:
   ```powershell
   # Developer PowerShell for VS, <root> = ваш plugin-GUI 1.0.1
   # положить open_ephys_plugin\DualDLCLiveBridge\ в <root>\Plugins\DualDLCLiveBridge\
   cmake -S <root> -B <root>\out\build\x64-Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DOE_DONT_CHECK_BUILD_PATH=TRUE
   cmake --build <root>\out\build\x64-Release --target DualDLCLiveBridge
   ```
2. Скопировать `DualDLCLiveBridge.dll` (Release) → `%LOCALAPPDATA%\Open Ephys\plugins-api10\`.
3. Если не грузится с ошибкой про `vcruntime140.dll` — поставить **VC++ Redistributable x64** (`vc_redist.x64.exe`, это не Visual Studio).
4. Запустить Open Ephys → в signal chain должен появиться узел `Dual DLCLive Bridge`.

> 💡 Сборку+деплой автоматизирует `scripts\build_plugin.ps1` (см. [`BUILD_PLUGIN.md`](BUILD_PLUGIN.md)): `.\scripts\build_plugin.ps1 -GuiRoot "<путь к plugin-GUI 1.0.1>" -DeployDir "<папка плагинов OE>"`. Портативная сборка OE грузит плагины из `<install>\plugins\`, официальный установщик — из `%LOCALAPPDATA%\Open Ephys\plugins-api10\`.

Параметры узла: `enabled=true`, `udp_port=47000`; для линий 2/3 — `angle_trigger_enabled=true`, `angle_threshold_deg=<порог>`. Детали — `open_ephys_plugin\DualDLCLiveBridge\README.md`.

## 9. Запуск

> Запускать из папки `python\` репозитория. Логи пишутся в `C:\dlc\DLC_OBS_Spinal_cord_stimulation` (см. §7).

**Проверка плагина без камер** (синтетический UDP):
```powershell
cd C:\dlc\Dual_DLC_live_plugin\python
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
# ожидается: acked 5/5
```

**Одна камера (боевой режим, рекомендуется):**
```powershell
cd C:\dlc\Dual_DLC_live_plugin\python
C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --no-display
```
Профили: `... single_rt_dlc_live_bridge.py --list-profiles`.

**Две камеры:**
```powershell
cd C:\dlc\Dual_DLC_live_plugin\python
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

В логе ждать: `TORCH_COMPILE applied backend=cudagraphs`, `Opened ... fps=100.0`, `CUDA_CHECK ... cuda=True`, `stage_profile ... result_hz=...`.

Запись видео+keypoints (обе камеры) включена в `config_dual_rt_dlc_live.py` (`DUAL_RECORD_ENABLED`/`SINGLE_RECORD_ENABLED`); видео пишется в lossless `.avi` (FFV1), файлы — в `C:\dlc\DLC_OBS_Spinal_cord_stimulation\recordings`.

> **Полная процедура эксперимента** (запуск OE + плагин, угловой триггер, правила валидности, запись, проверка TTL) — [`RUN_EXPERIMENT.md`](RUN_EXPERIMENT.md).

## 10. Проверка и диагностика

Признаки «всё работает», ожидаемая картина без животного, и типичные проблемы (`missing ack`, камеры не открываются, точки не детектятся) — в [`README.md`](../README.md), разделы «Как понять, что всё работает» и «Типичные проблемы».
Полный перенос/версии — [`MIGRATION_GUIDE_2026-06-16.md`](MIGRATION_GUIDE_2026-06-16.md); разбор кода и поведения — [`CODE_WALKTHROUGH_2026-06-15.md`](CODE_WALKTHROUGH_2026-06-15.md).
