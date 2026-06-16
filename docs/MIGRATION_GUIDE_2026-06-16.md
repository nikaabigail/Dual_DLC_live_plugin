# Перенос DLC realtime на новый ПК — мастер-гайд

Цель: поднять рабочую систему (камера Daheng → DLC-Live инференс на GPU → UDP в плагин Open Ephys → TTL-стимуляция) на чистом десктопном ПК с **RTX 5070 (Desktop)**, на котором уже стоит Open Ephys. Версии и зависимости — 1-в-1 с рабочей машиной.

**Документы и артефакты:**
- Этот файл — мастер-гайд (что, в каком порядке, какие решения принять).
- `MIGRATION_DETAIL_2026-06-16.md` — подробный разбор по 6 доменам (Python, NVIDIA, Galaxy SDK, код/пути, Open Ephys/плагин, скрипт проверки) с командами и подводными камнями.
- `setup/requirements.lock.txt` — точный слепок (57 пакетов).
- `setup/requirements.txt` — очищенный pip-устанавливаемый список (torch отдельно, см. ниже).
- `setup/install_python_env.ps1` — создаёт venv и ставит всё.
- `setup/check_environment.py` (+ `.ps1`) — **проверяет готовность целевого ПК** и печатает PASS/FAIL/MISSING. На рабочей машине даёт `46 OK / 0 MISSING / 0 MISMATCH`.

---

## 0. Самое важное — 4 предупреждения

1. **🔴 git устарел — НЕ переносите код через `git clone`.** Боевые файлы (`single_rt_dlc_live_bridge.py`, `dual_rt_dlc_live.py`, `config_dual_rt_dlc_live.py`, `live_profiles.py`, `run_live_profile.py`, `optimization/`) **не закоммичены**, а у tracked-файлов огромные несохранённые правки. Клон даст старую нерабочую версию **без файла, который вы запускаете**. Переносить надо **архив рабочего дерева**, потом закоммитить.
2. **🟢 CUDA Toolkit ставить НЕ нужно.** `torch==2.10.0+cu128` несёт CUDA 12.8 и cuDNN 9.10 **внутри колеса**. Нужен только свежий **драйвер NVIDIA** с поддержкой Blackwell (sm_120). Не поддавайтесь соблазну «доустановить CUDA, чтобы заработало».
3. **🟢 Visual Studio для Python-части НЕ нужна.** Бэкенд `cudagraphs` использует только CUDA-рантайм. VS нужна **только** для пересборки C++-плагина Open Ephys (если придётся, см. §6).
4. **🟢 Держите те же пути `C:\`.** Часть путей (`DUAL_CAMERAS[*].config_path`, `WORK_DIR`, `PYTHON_EXE`) **не имеют env-override**. Воссоздадите ту же раскладку `C:\dlc`, `C:\dlc_live_env`, `C:\config_daheng` — правок в коде НОЛЬ.

---

## 1. Решения (зафиксировано 2026-06-16)

✅ **ПОДТВЕРЖДЕНО пользователем:**
- **Те же пути `C:\`** (`C:\dlc`, `C:\dlc_live_env`, `C:\config_daheng`) → **zero-edit, правок в коде ноль.**
- **Только LIVE-запуск** → минимальный набор: 93 МБ модели + lean venv. Полный `deeplabcut` и проект 2.3 ГБ **не нужны**; видео 1.9 ГБ **не нужны**.
- **Те же физические камеры** (FDE22070174 для single «left») → конфиги `C:\config_daheng\*.txt` и `GALAXY_SN` подходят как есть.
- **Целевой репо для консолидации:** `https://github.com/nikaabigail/Dual_DLC_live_plugin` (туда заливаем и C++-плагин, и `python/` копию live-кода, и `docs/`).

✅ **Тоже подтверждено (2026-06-16):**
- **Galaxy SDK на целевом ПК уже стоит и работает** — этот домен закрыт.
- **Open Ephys на целевом = 1.0.1** (та же версия, что на источнике → API 10, плагин API-совместим).
- **Visual Studio на целевом НЕТ** → следовательно тамошний Open Ephys 1.0.1 — Release-сборка → нужен **Release**-плагин (собрать на источнике, см. §6). **VS на целевом не понадобится.**

> **ВАЖНО про репозитории — оба устарели.** И `nikaabigail/DLC_Spinal_cord_stimulation`, и `nikaabigail/Dual_DLC_live_plugin` имеют незакоммиченные/untracked правки. В репо плагина уже есть папка `python/` с копиями live-кода и прежний `docs/MIGRATION_REQUIREMENTS_2026-06-12.md` (этот гайд его расширяет/замещает) — но эти `python/`-копии тоже могут отставать. **Источник истины для live-кода — рабочее дерево `C:\dlc\DLC_OBS_Spinal_cord_stimulation`.** При заливке в репо плагина копировать `python/`-часть именно оттуда, а не из устаревшей `C:\tmp\Dual_DLC_live_plugin\python\`.

---

## 2. Эталонные версии (источник истины)

| Компонент | Версия |
|---|---|
| ОС | Windows 11 (10.0.26200) x64 |
| GPU | RTX 5070, compute capability (12,0) = sm_120 Blackwell |
| Драйвер NVIDIA | 592.01 (репортит CUDA 13.1); достаточно любого, дающего **CUDA ≥ 12.8** + Blackwell |
| Python | **3.10.11 x64** (именно ветка 3.10 — колёса cp310) |
| torch / torchvision / torchaudio | **2.10.0+cu128 / 0.25.0+cu128 / 2.10.0+cu128** (index `download.pytorch.org/whl/cu128`) |
| CUDA / cuDNN | 12.8 / 9.10.02 — **внутри колеса torch**, отдельно не ставить |
| deeplabcut-live (dlclive) | **1.1.0** |
| numpy / opencv-python / scipy / pandas | 1.26.4 / 4.11.0.86 / 1.15.3 / 2.3.3 |
| прочее | pillow 12.1.1, PyYAML 6.0.3, ruamel.yaml 0.19.1, tables 3.10.1, timm 1.0.26, huggingface_hub 1.9.0, networkx 3.4.2 (всего 57 пинов) |
| Daheng Galaxy SDK | **1.18.2208.9301** (gxipy — из SDK, не pip) |
| Open Ephys (источник) | GUI 1.0.1, **Plugin API 10**, плагин `DualDLCLiveBridge` |

---

## 3. Порядок установки (раннбук)

> Детали и точные команды по каждому шагу — в `MIGRATION_DETAIL_2026-06-16.md` (соответствующий домен).

**Шаг 1 — Драйвер NVIDIA.** Поставить свежий драйвер для RTX 5070 Desktop (Game Ready/Studio), «чистая установка», перезагрузка. Проверка: `nvidia-smi` → видит RTX 5070, `CUDA Version ≥ 12.8`. *(CUDA Toolkit НЕ ставить.)*

**Шаг 2 — Python 3.10.11 x64.** Установить (Add to PATH или `py -3.10`).

**Шаг 3 — venv + зависимости.** Запустить `setup\install_python_env.ps1` (создаёт `C:\dlc_live_env`, ставит torch с cu128-индекса, затем `requirements.txt` с PyPI). Либо вручную:
```powershell
py -3.10 -m venv C:\dlc_live_env
C:\dlc_live_env\Scripts\python.exe -m pip install --upgrade pip
C:\dlc_live_env\Scripts\python.exe -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
C:\dlc_live_env\Scripts\python.exe -m pip install -r C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\requirements.txt
```
Самотест GPU: `C:\dlc_live_env\Scripts\python.exe -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"` → `True NVIDIA GeForce RTX 5070 (12, 0)`.

**Шаг 4 — Daheng Galaxy SDK 1.18.2208.9301.** Установить от админа в путь по умолчанию `C:\Program Files\Daheng Imaging\GalaxySDK` (ставит USB3-драйвер, GenTL, GenICam, gxipy, env-переменные). Перезагрузка. Подключить камеру в **USB 3.0**. Проверка: GalaxyView видит `MER2-230-168U3C` (потом **закрыть** его).

**Шаг 5 — Код и ассеты.**
- Распаковать архив рабочего дерева в **`C:\dlc\DLC_OBS_Spinal_cord_stimulation`** (архив без `*.mp4/*.log/*_benchmark.csv/debug_snapshots/__pycache__`).
- Положить модель: `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\...snapshot-best-380.pt` (93 МБ).
- Скопировать `C:\config_daheng\*.txt` (25 файлов).

**Шаг 6 — Open Ephys + плагин.** На целевом ПК определить версию Open Ephys и plugin-API. Сначала попробовать **Путь A** (скопировать готовый `DualDLCLiveBridge.dll` в `%LOCALAPPDATA%\Open Ephys\plugins-api10`) — но учесть, что исходный DLL **Debug-сборки** и может не загрузиться в Release-хост; тогда **Путь A1** (перенести всё Debug-дерево с его `open-ephys.exe`) или **Путь B** (пересборка Release через Visual Studio). Плагин можно тестировать **без камеры** через `send_dual_dlc_bridge_test.py`. См. §6 и домен 4 детали.

**Шаг 7 — Проверка готовности.**
```powershell
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\check_environment.py
```
Должно дать `OK` по всем 8 блокам (Python, пакеты, CUDA/GPU, драйвер, Galaxy SDK+gxipy, модель, конфиги, скрипты). Что `MISSING/MISMATCH` — то и доустановить.

**Шаг 8 — Запуск.**
```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --no-display
```
Ожидать в логе: `TORCH_COMPILE applied backend=cudagraphs`, `Opened single camera ... fps=100.0`, `raw_visible=N/6`, `result_hz` ~50+. 

**Шаг 9 — Снять устаревание git.** На источнике (предпочтительно) или на целевом: `git add -A; git commit -m "..."; git push origin main` (`.gitignore` уже исключает `*.pt/*.mp4/logs/videos`).

---

## 4. Что НЕ нужно ставить (частые лишние действия)

- ❌ CUDA Toolkit (torch несёт CUDA внутри).
- ❌ `pip install gxipy` (берётся из SDK).
- ❌ Visual Studio для Python-части.
- ❌ `opencv-python-headless` рядом с `opencv-python` (конфликт `cv2`; предупреждение `pip check` про headless — безвредно).
- ❌ NumPy 2.x (нужен строго 1.26.4).
- ❌ `setuptools/wheel` вручную (lean-стек — чистые колёса).

## 5. Минимальный набор внешних ассетов для LIVE

| Ассет | Размер | Куда |
|---|---|---|
| Рабочее дерево (архив, без артефактов) | ~5 МБ | `C:\dlc\DLC_OBS_Spinal_cord_stimulation` |
| Экспортированная модель `.pt` | 93 МБ | `C:\dlc\project\...\snapshot-best-380.pt` |
| Конфиги камер | ~124 КБ | `C:\config_daheng\` |
| (опц.) видео | 1.9 ГБ | `C:\dlc\videos\` — только офлайн |
| (опц.) полный проект | 2.3 ГБ | только для дообучения |

## 6. Open Ephys плагин (целевой Open Ephys 1.0.1, VS на целевом НЕТ)

**Вывод: Visual Studio на целевом ПК НЕ нужна.** Раз Open Ephys 1.0.1 на целевом запускается без установленной VS — это **Release**-сборка (Debug требует отладочный CRT, который идёт только с VS). Версия GUI та же (1.0.1, API 10), поэтому плагин API-совместим; нужна лишь **Release**-версия `.dll`.

**Проблема:** текущий `DualDLCLiveBridge.dll` — **Debug-сборка** (VS18 Insiders, MSVC 14.51) → в Release-хост, скорее всего, не загрузится (несовпадение Debug/Release CRT).

**Рекомендуемый путь (VS только на ИСТОЧНИКЕ, где она уже есть):**
1. На **источнике** пересобрать плагин в **Release** против дерева `plugin-GUI` 1.0.1 (там уже стоят VS18 Insiders + Ninja + CMake):
   ```powershell
   # из Developer PowerShell for VS, <root> = ...\plugin-GUI-main\plugin-GUI-main
   cmake -S <root> -B <root>\out\build\x64-Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DOE_DONT_CHECK_BUILD_PATH=TRUE
   cmake --build <root>\out\build\x64-Release --target DualDLCLiveBridge
   # => <root>\out\build\x64-Release\Plugins\DualDLCLiveBridge.dll  (Release)
   ```
2. Скопировать **Release** `DualDLCLiveBridge.dll` на целевой в `%LOCALAPPDATA%\Open Ephys\plugins-api10\`.
3. На целевом убедиться, что стоит **VC++ Redistributable x64** (обычно есть; если DLL не грузится с ошибкой про `vcruntime140.dll` — поставить `vc_redist.x64.exe`, это крошечный redistributable, НЕ Visual Studio).
4. Запустить Open Ephys → проверить, что появился узел **`Dual DLCLive Bridge`** (Utility, UDP `127.0.0.1:47000`, DDLP/v1, TTL-канал). Тест без камеры — `send_dual_dlc_bridge_test.py`.

**Запасной вариант (если Release-DLL всё же не подходит по ABI):** собрать плагин против **официального** дерева plugin-GUI 1.0.1 (тем, что соответствует установленному на целевом релизу), либо в крайнем случае — поставить VS на целевой и собрать там (Путь B из детального документа). Но при совпадении версии 1.0.1 это, как правило, не требуется.

**Важно:** Release-DLL надо собирать против ровно той версии GUI/SDK (1.0.1), что бежит на целевом. Если целевой Open Ephys — официальный установщик 1.0.1, а исходное дерево `plugin-GUI-main` — кастомный коммит, при несовпадении ABI собирать против официального 1.0.1 SDK. Исходники плагина: `C:\tmp\Dual_DLC_live_plugin\open_ephys_plugin\DualDLCLiveBridge\` (и в репо `nikaabigail/Dual_DLC_live_plugin`).

---

## 7. Остаточные открытые вопросы

1. **Целевой Open Ephys 1.0.1 — официальный установщик или копия твоей сборки из исходников?** Это решает, против какого дерева plugin-GUI собирать Release-плагин (§6). Проверка на целевом: путь, откуда запускается `open-ephys.exe`, и `Test-Path "$env:LOCALAPPDATA\Open Ephys\plugins-api10"`.
2. **Есть ли на целевом VC++ Redistributable x64?** Нужен Release-плагину; обычно уже стоит — если нет, поставить `vc_redist.x64.exe` (это не Visual Studio).

✅ Закрыто: Galaxy SDK на целевом уже стоит; Open Ephys = 1.0.1 (API 10); Visual Studio на целевом не нужна (Release-плагин собираем на источнике).

> Следующие шаги: (а) на источнике собрать **Release** `DualDLCLiveBridge.dll` и перенести в `plugins-api10` целевого; (б) залить рабочее дерево live-кода (из `C:\dlc\DLC_OBS_Spinal_cord_stimulation`) в `nikaabigail/Dual_DLC_live_plugin` (`python/`) и в `nikaabigail/DLC_Spinal_cord_stimulation`, чтобы снять устаревание; (в) самораспаковывающийся бандл; (г) расширить `check_environment.py` до «проверил → выполнил установку» — заготовка авто-инсталлятора-exe.
