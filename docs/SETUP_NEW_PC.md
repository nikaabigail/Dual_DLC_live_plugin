# Установка на новый ПК — пошагово (что качать, какую папку создать, куда распаковывать)

Линейный runbook для чистого десктопа с RTX 5070. Подробности и обоснования — в [MIGRATION_GUIDE_2026-06-16.md](MIGRATION_GUIDE_2026-06-16.md) и [MIGRATION_DETAIL_2026-06-16.md](MIGRATION_DETAIL_2026-06-16.md). Решения зафиксированы: **те же пути `C:\`, только live-режим, те же камеры** → правок в коде нет.

## Карта «что и куда» (создать эти папки)

| Что | Откуда взять | Куда положить (создать папку) |
|---|---|---|
| Код проекта | архив рабочего дерева **или** `git clone` (после мержа PR #1) | `C:\dlc\DLC_OBS_Spinal_cord_stimulation\` |
| Python 3.10.11 | python.org | (установщик, ставит сам) |
| venv | создаётся скриптом `setup\install_python_env.ps1` | `C:\dlc_live_env\` |
| Модель `.pt` (93 МБ) | перенести с рабочей машины | `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\` |
| Конфиги камер (25 шт) | перенести из `C:\config_daheng\` | `C:\config_daheng\` |
| Galaxy SDK | уже стоит на целевом ✅ | `C:\Program Files\Daheng Imaging\GalaxySDK\` |
| Плагин `DualDLCLiveBridge.dll` (Release) | собрать на рабочей машине | `%LOCALAPPDATA%\Open Ephys\plugins-api10\` |

---

## Шаг 1. Драйвер NVIDIA
Скачать с **nvidia.com** последний драйвер для **RTX 5070 (Desktop), Windows 11** (Game Ready или Studio), установка «Custom → clean install», перезагрузка.
Проверка: `nvidia-smi` → видит `NVIDIA GeForce RTX 5070`, `CUDA Version ≥ 12.8`.
**CUDA Toolkit НЕ ставить** (torch несёт CUDA внутри).

## Шаг 2. Python 3.10.11 (64-bit)
Скачать: https://www.python.org/downloads/release/python-31011/ → **Windows installer (64-bit)**. При установке отметить «Add python.exe to PATH» (или потом использовать `py -3.10`).

## Шаг 3. Код проекта → `C:\dlc\DLC_OBS_Spinal_cord_stimulation`
Создать папку `C:\dlc`. Затем:
- **Вариант с git** (после мержа PR #1 — репо обновлён):
  ```powershell
  cd C:\dlc
  git clone https://github.com/nikaabigail/DLC_Spinal_cord_stimulation.git DLC_OBS_Spinal_cord_stimulation
  ```
- **Вариант с архивом** (если PR ещё не смержен — git отстаёт): распаковать архив рабочего дерева **строго** в `C:\dlc\DLC_OBS_Spinal_cord_stimulation` (тот же путь — иначе `WORK_DIR`/пути не совпадут).

## Шаг 4. Окружение → `C:\dlc_live_env`
Из корня проекта в PowerShell:
```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
.\setup\install_python_env.ps1
```
Скрипт создаст venv `C:\dlc_live_env`, поставит `torch +cu128` с нужного индекса, затем остальное, и прогонит smoke-тест. (Подробности и ручной вариант — в README, раздел «Создание окружения».)

## Шаг 5. Модель → `C:\dlc\project\...`
Создать путь и положить экспортированный снапшот (93 МБ):
```
C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\
  DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\
    DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt
```
(Полный проект 2.3 ГБ и видео для live НЕ нужны.)

## Шаг 6. Конфиги камер → `C:\config_daheng`
Создать `C:\config_daheng` и скопировать туда все `.txt` из `C:\config_daheng\` рабочей машины (25 файлов). Боевой профиль — `Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt`.
⚠️ При правке `.txt` держать строки чистыми (`Feature<TAB>Value`, без inline `#`-комментариев и хвостовых пробелов) — иначе ломается `import_config_file`.

## Шаг 7. Galaxy SDK
На целевом ПК уже установлен ✅. (Если бы не был — поставить Galaxy SDK 1.18.2208.9301 в путь по умолчанию; он ставит USB3-драйвер, gxipy и env-переменные.) Камеру подключить в порт **USB 3.0**.

## Шаг 8. Плагин Open Ephys (Release DLL)
Open Ephys на целевом = **1.0.1 (API 10)**, Release. Текущий DLL — Debug, в Release-хост не загрузится. Поэтому:
1. **На рабочей машине** собрать плагин в **Release** (там есть Visual Studio):
   ```powershell
   # Developer PowerShell for VS, <root> = ...\plugin-GUI-main\plugin-GUI-main
   cmake -S <root> -B <root>\out\build\x64-Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DOE_DONT_CHECK_BUILD_PATH=TRUE
   cmake --build <root>\out\build\x64-Release --target DualDLCLiveBridge
   ```
2. Скопировать `DualDLCLiveBridge.dll` (Release) → `%LOCALAPPDATA%\Open Ephys\plugins-api10\` на целевом.
3. Если DLL не грузится с ошибкой про `vcruntime140.dll` — поставить **VC++ Redistributable x64** (`vc_redist.x64.exe`, это не Visual Studio).
4. Запустить Open Ephys → должен появиться узел `Dual DLCLive Bridge`. Тест без камеры: `python send_dual_dlc_bridge_test.py`.

**Visual Studio на целевом ПК НЕ нужна.**

## Шаг 9. Проверка готовности
```powershell
C:\dlc_live_env\Scripts\python.exe setup\check_environment.py
```
Должно быть `OK` по всем блокам (Python, пакеты, CUDA/GPU, драйвер, Galaxy SDK, модель, конфиги, скрипты). Что `MISSING/MISMATCH` — доустановить.

## Шаг 10. Запуск
```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --no-display
```
В логе ждать: `TORCH_COMPILE applied backend=cudagraphs`, `Opened single camera ... fps=100.0`, `raw_visible=N/6`, `result_hz` ~50+.
