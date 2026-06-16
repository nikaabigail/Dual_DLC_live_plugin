

==========================================================================================
# DOMAIN 0: Python + виртуальное окружение (точное воссоздание)
==========================================================================================

# Python и виртуальное окружение (lean live-inference)

Этот раздел описывает точное воссоздание рабочего venv `C:\dlc_live_env` на чистом ПК. Окружение «тощее» (lean): в нём установлен ТОЛЬКО `deeplabcut-live` (== `dlclive` 1.1.0) для онлайн-инференса с камеры, а НЕ полный `deeplabcut` (обучение/анализ). Это проверено на исходной машине: `pip show deeplabcut` → пакет не найден; импорт `deeplabcut` срабатывает лишь потому, что в базовом Python лежит пустая папка-namespace `...Python310\deeplabcut` (без `__init__.py`, `origin: None`) — это мусорный артефакт, на чистой машине его не будет и воспроизводить его не нужно.

## 0. Что проверено на исходной машине

- `C:\dlc_live_env\Scripts\python.exe` → **Python 3.10.11** (MSC v.1929, 64-bit), pip **26.0.1**.
- `torch` → **2.10.0+cu128**, `torch.cuda.is_available()` = **True**, `torch.version.cuda` = **12.8**, capability = **(12, 0) = sm_120 (Blackwell)**.
- `pip freeze` даёт ровно **57 пакетов** и совпадает с `requirements.lock.txt` **бит-в-бит** (нулевой diff).
- Зависимости torch (`filelock, fsspec, jinja2, networkx, sympy, typing-extensions`) и torchvision (`numpy, pillow, torch`) все присутствуют и закреплены. Никаких `nvidia-*` pip-пакетов нет — колёса `+cu128` для Windows УЖЕ содержат рантайм CUDA 12.8 и cuDNN 9.10 внутри себя.
- `setuptools`/`wheel` в окружении вообще отсутствуют (pip 26 их не ставит) — это нормально: весь lean-стек ставится из готовых wheel-колёс, шага сборки нет.

## 1. Предусловия на целевом ПК

1. **NVIDIA-драйвер для Blackwell.** Десктопный RTX 5070 — та же архитектура sm_120, что и Laptop-версия на источнике. Нужен свежий драйвер, поддерживающий Blackwell (исходный был NVIDIA-SMI 592.01, CUDA 13.1-capable). Если после установки `torch.cuda.is_available()` вернёт `False` — почти всегда виноват старый драйвер, обновите его. **CUDA Toolkit ставить НЕ нужно** — torch несёт CUDA внутри колеса.
2. **Python 3.10.11 (64-bit).** Ставьте именно минорную ветку 3.10 (на источнике 3.10.11). НЕ берите 3.11/3.12/3.13: ряд закреплённых колёс — `cp310` (например `tables`, `blosc2`, и сами `torch +cu128`), на другой минорной версии pip их не найдёт. При установке отметьте «Add python.exe to PATH» (или используйте лаунчер `py -3.10`). Скачать: https://www.python.org/downloads/release/python-31011/

## 2. Установка двумя шагами (почему именно так)

Ключевой момент: `torch/torchvision/torchaudio` версии `+cu128` ЛЕЖАТ ТОЛЬКО на индексе колёс PyTorch CUDA 12.8 и их НЕТ на PyPI. Поэтому ставим их **отдельно** с `--index-url`, а весь остальной набор — **отдельно** с обычного PyPI. Если повесить `--index-url` на весь `requirements.txt`, pip уйдёт искать ВСЕ пакеты на индексе PyTorch и не найдёт там, например, `tables`, `timm`, `ruamel.yaml` — установка сломается.

### Шаг 1 — torch стек (CUDA 12.8 index)

```powershell
C:\dlc_live_env\Scripts\python.exe -m pip install `
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128
```

### Шаг 2 — остальное с PyPI

```powershell
C:\dlc_live_env\Scripts\python.exe -m pip install -r C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\requirements.txt
```

`requirements.txt` уже почищен и дедуплицирован; строки torch в нём закомментированы с пометкой про index-url, чтобы случайно не поставить их «не оттуда».

## 3. Автоматизация — готовый скрипт

Запустите `C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\install_python_env.ps1`. Он:
найдёт базовый Python 3.10 (через `py -3.10` или явный `-PythonExe`), создаст свежий venv в `C:\dlc_live_env`, обновит pip, выполнит **шаг 1** (torch +cu128), затем **шаг 2** (`requirements.txt`), запустит `pip check` и финальный smoke-тест (`torch.cuda`, `dlclive`, `cv2`, `numpy`).

```powershell
# из папки setup:
.\install_python_env.ps1
# при нестандартном расположении базового Python:
.\install_python_env.ps1 -PythonExe "C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe"
```

Скрипт намеренно НЕ ставит: драйвер NVIDIA, Galaxy SDK (gxipy), файлы модели и конфиг камеры — это отдельные разделы миграции.

## 4. Подводные камни на чистой Windows-машине (важно)

- **`opencv-python` против `opencv-python-headless` (ожидаемое предупреждение).** `deeplabcut-live==1.1.0` в метаданных требует `opencv-python-headless`, а в окружении стоит обычный `opencv-python==4.11.0.86`. Поэтому `pip check` ВЫДАЁТ предупреждение: `deeplabcut-live 1.1.0 requires opencv-python-headless, which is not installed`. **Это безвредно и так же на источнике** — оба пакета дают один и тот же модуль `cv2`, а live-мост использует `cv2` для отрисовки/окна, которое headless-сборка не умеет. НЕ «чините» это добавлением headless-пакета рядом: два провайдера `cv2` в одном venv конфликтуют. Оставляйте только `opencv-python`.
- **`tables` (PyTables) и `blosc2`.** На Windows исторически капризны при сборке из исходников. Здесь спасает закреплённая версия `tables==3.10.1` + `blosc2==4.1.2` для `cp310` — pip берёт готовое колесо, компилятор НЕ нужен. Поэтому строго держите Python 3.10 (для другой минорной версии колеса `cp310` не подойдут, и pip может попытаться собирать из исходников → ошибки про HDF5/компилятор).
- **`ruamel.yaml==0.19.1`.** Тянет за собой нативный акселератор; на 3.10 ставится колесом без проблем. Версия закреплена намеренно — у `ruamel.yaml` бывают ломающие изменения API между минорными версиями, а `deeplabcut-live` от него зависит.
- **`numpy` строго `1.26.4`.** Это NumPy 1.x. Не давайте pip подтянуть NumPy 2.x — часть стека (tables/numexpr/совместимость с собранными колёсами) рассчитана на 1.26. В `requirements.txt` он закреплён.
- **gxipy — это НЕ pip-пакет.** Его нет в `requirements.txt` и не должно быть. Он поставляется внутри Daheng Galaxy SDK (папка `Samples/Python SDK`) и добавляется в `sys.path` в рантайме кодом (`rt_dlc_live._prepare_sdk_environment`). Ставится установкой SDK (отдельный раздел миграции).
- **`torch.cuda.is_available() == False` после установки.** Колёса правильные (`+cu128`), значит причина внешняя: устаревший драйвер NVIDIA (нужен Blackwell-capable) либо запуск под виртуалкой/без GPU. Обновите драйвер.
- **Не ставьте `setuptools`/`wheel` вручную «на всякий случай».** На источнике их нет, весь lean-стек — чистые колёса. Лишние пакеты только уводят окружение от эталонного freeze.

## 5. Если позже понадобится офлайн `run_dlc.py` (обучение/анализ)

Текущее окружение — ТОЛЬКО для live-инференса. Для офлайн-пайплайна (`run_dlc.py`: train / analyze) нужен ПОЛНЫЙ пакет `deeplabcut` (PyTorch-движок), которого здесь нет. Рекомендация: ставить его в **ОТДЕЛЬНЫЙ** venv, а не в `C:\dlc_live_env`, чтобы не сломать выверенный набор live-зависимостей (полный `deeplabcut` тянет десятки дополнительных пакетов и может конфликтовать по версиям numpy/opencv/tables). Точные пины для офлайн-окружения в этом снапшоте отсутствуют — их нужно зафиксировать отдельно (это открытый вопрос).

## 6. Проверка результата

После установки ожидаемый вывод smoke-теста (его печатает и скрипт):

```
torch        : 2.10.0+cu128
cuda runtime : 12.8
cuda avail   : True
device       : NVIDIA GeForce RTX 5070
capability   : (12, 0)   # sm_120 Blackwell
dlclive      : 1.1.0
cv2          : 4.11.0.86
numpy        : 1.26.4
```

`pip freeze` должен дать **57 строк**, совпадающих с `requirements.lock.txt`. Проверено: все 53 non-torch пина из `requirements.txt` точно совпадают с живым окружением → набор разрешим и воспроизводим без конфликтов.

### CHECKLIST
- [ ] Установить свежий NVIDIA-драйвер с поддержкой Blackwell/sm_120 (CUDA Toolkit НЕ нужен — torch несёт CUDA внутри)
- [ ] Установить Python 3.10.11 64-bit (именно ветка 3.10; отметить Add to PATH или использовать py -3.10)
- [ ] Создать свежий venv: py -3.10 -m venv C:\dlc_live_env (или запустить install_python_env.ps1)
- [ ] Обновить pip: C:\dlc_live_env\Scripts\python.exe -m pip install --upgrade pip
- [ ] Шаг 1 (torch): pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
- [ ] Шаг 2 (остальное): pip install -r C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\requirements.txt (БЕЗ --index-url)
- [ ] Выполнить pip check — единственное ожидаемое предупреждение про opencv-python-headless ИГНОРИРОВАТЬ (используется opencv-python)
- [ ] Smoke-тест: python -c для проверки torch.cuda.is_available()==True, capability (12,0), импорта dlclive/cv2/numpy
- [ ] Сверить pip freeze с requirements.lock.txt (должно быть ровно 57 пакетов, нулевой diff)
- [ ] НЕ ставить gxipy через pip — он придёт из Galaxy SDK (отдельный раздел миграции)
- [ ] При torch.cuda.is_available()==False — обновить драйвер NVIDIA, а не переустанавливать torch
- [ ] (Опционально) для офлайн run_dlc.py — создать ОТДЕЛЬНЫЙ venv и поставить полный deeplabcut, не трогая C:\dlc_live_env

### ARTIFACTS
- C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\requirements.txt — очищенный, дедуплицированный pip-устанавливаемый набор; строки torch закомментированы с пометкой про cu128 index-url; gxipy и полный deeplabcut явно исключены с пояснениями
- C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\install_python_env.ps1 — PowerShell-скрипт: проверяет Python 3.10, создаёт venv, ставит torch +cu128 отдельным шагом, затем requirements.txt с PyPI, гоняет pip check и smoke-тест torch.cuda/dlclive (синтаксис скрипта проверен Parser-ом — PARSE OK)

### OPEN QUESTIONS
- Какую именно сборку Python 3.10 ставить на target — exe-инсталлятор с python.org (рекомендуется, exact 3.10.11) или иной источник? Подтвердить наличие прав администратора для установки.
- Есть ли на target доступ в интернет к download.pytorch.org/whl/cu128 и PyPI? Если ПК изолирован — нужен офлайн-набор колёс (pip download на машине с сетью под cp310/win_amd64), это отдельная процедура.
- Точная версия драйвера NVIDIA на target и поддерживает ли он Blackwell sm_120 — проверить nvidia-smi после установки; при is_available()==False обновлять драйвер.
- Нужен ли на этой машине офлайн run_dlc.py (train/analyze)? Если да — требуется зафиксировать пины ПОЛНОГО deeplabcut в отдельном окружении; в текущем снапшоте этих пинов нет.
- Путь venv: оставляем хардкод C:\dlc_live_env (на него завязан live_profiles.py:11 PYTHON_EXE) или меняем? Если меняем — нужно править live_profiles.py.


==========================================================================================
# DOMAIN 1: NVIDIA driver / CUDA / cuDNN (GPU stack for the live DLC pipeline)
==========================================================================================

# GPU-стек: драйвер NVIDIA, CUDA и cuDNN на целевом ПК

## Главная мысль (прочитайте первой)

Колесо `torch==2.10.0+cu128` — это **«+cu128»**-сборка. Она **уже содержит внутри себя рантайм CUDA 12.8 и cuDNN 9.10.02**. Это проверено прямо на рабочей машине-источнике:

```
torch 2.10.0+cu128
torch.version.cuda     -> 12.8        # CUDA, с которой собран torch
cudnn version          -> 91002       # = cuDNN 9.10.02, лежит внутри пакета
torch.cuda.is_available() -> True
```

Практический вывод: **отдельный CUDA Toolkit устанавливать НЕ нужно**, чтобы ЗАПУСКАТЬ live-пайплайн. На машине-источнике действительно стоит CUDA Toolkit v13.0 (`CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`), но torch его **не использует** — torch скомпилирован под 12.8 и носит свой рантайм с собой. Это случайный «хвост» от других экспериментов, к нашему пайплайну он отношения не имеет. На целевой ПК его ставить не требуется.

Единственное, что **обязательно** должно быть на целевом ПК со стороны системы, — это **свежий драйвер NVIDIA**, который понимает архитектуру **Blackwell (sm_120)** и умеет CUDA-рантайм версии ≥ 12.8.

## Почему так устроено (драйвер vs. рантайм)

CUDA состоит из двух слоёв:

1. **Driver API / user-mode driver** — ставится только установщиком драйвера NVIDIA. Это единственный системный компонент, который тащит torch.
2. **CUDA runtime + библиотеки (cuDNN, cuBLAS и т.д.)** — torch несёт свою копию внутри колеса `+cu128`.

CUDA-рантайм работает поверх драйвера по правилу **forward-compatibility**: драйвер, который поддерживает CUDA версии X, корректно исполняет приложения, собранные под CUDA ≤ X. На источнике `nvidia-smi` показывает «CUDA Version: 13.1» — это **максимальная** версия рантайма, которую тянет драйвер, а не то, что у нас установлено. 13.1 ≥ 12.8, поэтому torch+cu128 запускается без проблем. На целевом ПК нужно лишь, чтобы драйвер давал «CUDA Version ≥ 12.8».

## GPU и архитектура

| | Источник | Цель |
|---|---|---|
| GPU | RTX 5070 **Laptop** GPU | RTX 5070 **Desktop** |
| Compute capability | **(12, 0) = sm_120** | **(12, 0) = sm_120** |
| Архитектура | Blackwell | Blackwell |

Архитектура **одинаковая (sm_120)**, разница только «ноутбук vs десктоп». Колесо `+cu128` содержит скомпилированные ядра под sm_120 (это подтверждается тем, что на источнике инференс реально работает на этой карте), поэтому **перекомпиляция не нужна** — те же бинарные колёса заведутся на десктопной 5070 без изменений.

## Что нужно на целевом ПК — драйвер

- **Минимум:** любой драйвер NVIDIA, официально поддерживающий RTX 5070 (Blackwell). Для семейства RTX 50xx это драйверы линейки **R570 и новее** (первые публичные драйверы с поддержкой Blackwell). Если `nvidia-smi` пишет «CUDA Version: 12.8» или выше — этого формально достаточно.
- **Рекомендуется:** поставить **последний** Game Ready / Studio драйвер с сайта NVIDIA (раздел GeForce RTX 50 Series → RTX 5070, Windows 11). Источник работает на драйвере **592.01** (репортит «CUDA 13.1»). Брать драйвер не ниже того, что нужен для CUDA 12.8; новее — лучше. Чистая установка (опция «Custom → Perform a clean installation») избавляет от старых хвостов.
- Желательно совпадение разрядности WDDM/режима — на источнике карта в режиме **WDDM**, это норма для десктопа с дисплеем.

## Порядок установки на целевом ПК

1. Скачать и установить **последний драйвер NVIDIA для RTX 5070 (Desktop), Windows 11** с nvidia.com (Game Ready или Studio — для нашей задачи без разницы). Рекомендуется «чистая установка».
2. Перезагрузить ПК.
3. **CUDA Toolkit НЕ ставить** (не нужен для запуска; когда он нужен — см. ниже).
4. Установить Python-окружение из `requirements.lock.txt` (это отдельный домен миграции — там приедет `torch==2.10.0+cu128` со своим CUDA/cuDNN внутри).

## Проверка (выполнить ровно эти команды)

**Шаг 1. Драйвер видит GPU и заявляет CUDA ≥ 12.8:**

```powershell
nvidia-smi
```

Ожидаем в шапке: имя `NVIDIA GeForce RTX 5070`, строку `Driver Version: …` и `CUDA Version: 12.8` (или выше — на источнике 13.1). Если карта видна и CUDA-версия ≥ 12.8 — драйвер пригоден.

**Шаг 2. Самопроверка torch (главный тест готовности GPU-стека):**

```powershell
& "C:\dlc_live_env\Scripts\python.exe" -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

**Должно напечатать:**

```
True NVIDIA GeForce RTX 5070 (12, 0)
```

- `True` — torch видит GPU через драйвер.
- имя карты — `NVIDIA GeForce RTX 5070` (на десктопе без «Laptop»).
- `(12, 0)` — это sm_120 Blackwell, ровно то, под что собраны ядра в колесе.

**(необязательно) Шаг 3. Подтвердить версии встроенного CUDA/cuDNN:**

```powershell
& "C:\dlc_live_env\Scripts\python.exe" -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('cudnn',torch.backends.cudnn.version())"
```

Ожидаем: `torch 2.10.0+cu128`, `cuda 12.8`, `cudnn 91002`. Это значения, проверенные на источнике, — если совпало, GPU-стек идентичен рабочему.

## Когда CUDA Toolkit ВСЁ-ТАКИ нужен (не наш случай)

Отдельный CUDA Toolkit (nvcc, заголовки, отдельные библиотеки) требуется **только** если вы собираетесь **компилировать** CUDA/C++-ядра. В нашем пайплайне такого нет:

- **Наш путь — `torch.compile(backend="cudagraphs")`**. Бэкенд `cudagraphs` использует только готовый CUDA-рантайм из torch и **НЕ требует ни компилятора C++, ни CUDA Toolkit, ни Triton**. Это чистый pip-путь без шагов сборки.
- CUDA Toolkit + компилятор понадобились бы, только если бы кто-то переключился на бэкенд **`inductor`** (он генерирует и компилирует Triton/CUDA-код) — но мы его не используем.
- Сборка C++-плагина Open Ephys (`DualDLCLiveBridge`) — это **отдельная** история (нужны Visual Studio/MSVC + CMake) и к Python-/GPU-части отношения не имеет. Python-сторона пайплайна шагов сборки не содержит вообще.

Итог: для запуска live-инференса CUDA Toolkit на целевом ПК **не нужен**.

## Подводные камни

- **Старый драйвер до поддержки Blackwell.** Если драйвер старше линейки с поддержкой RTX 50xx, будет одно из двух: `torch.cuda.is_available()` вернёт `False`, либо инференс упадёт с ошибками ядра/«no kernel image is available for execution on the device» (нет ядра под sm_120). Лечение: обновить драйвер NVIDIA до свежего.
- **`nvidia-smi` показывает CUDA < 12.8.** Это драйвер слишком старый для рантайма 12.8 → обновить драйвер. Цифра в `nvidia-smi` — это «максимально поддерживаемая драйвером CUDA», а не «установленный Toolkit».
- **Соблазн «доустановить CUDA Toolkit, чтобы заработало».** Это не лечит проблему совместимости и не нужно torch. Если `is_available()=False` — проблема в драйвере, а не в отсутствии Toolkit.
- **Несовпадение колеса torch и драйвера.** Не «обновляйте» torch на CPU-сборку или другую CUDA-версию случайно — нужен именно `+cu128`. Ставить строго из `requirements.lock.txt`.
- **Имя карты в самотесте.** На целевом ПК ожидается `NVIDIA GeForce RTX 5070` без слова `Laptop` — это нормально и не является ошибкой; ключевой признак готовности — `True` и `(12, 0)`.

### CHECKLIST
- [ ] Установить последний драйвер NVIDIA для RTX 5070 Desktop под Windows 11 с nvidia.com (Game Ready или Studio), опция Custom -> clean installation; не ниже линейки с поддержкой Blackwell / CUDA 12.8
- [ ] Перезагрузить ПК после установки драйвера
- [ ] CUDA Toolkit НЕ устанавливать (torch+cu128 несёт CUDA 12.8 и cuDNN 9.10 внутри; Toolkit нужен только для компиляции CUDA/C++ или backend=inductor, а у нас cudagraphs)
- [ ] Проверить драйвер: запустить `nvidia-smi` — должно показать `NVIDIA GeForce RTX 5070` и `CUDA Version: 12.8` или выше (на источнике 13.1)
- [ ] После установки venv выполнить самотест: `& "C:\dlc_live_env\Scripts\python.exe" -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"` — должно вывести `True NVIDIA GeForce RTX 5070 (12, 0)`
- [ ] (опц.) Подтвердить версии: torch 2.10.0+cu128, torch.version.cuda == 12.8, cudnn == 91002
- [ ] Если is_available()=False или ошибка 'no kernel image for sm_120' — обновить драйвер NVIDIA (НЕ ставить CUDA Toolkit)

### ARTIFACTS

### OPEN QUESTIONS
- Точную минимальную версию драйвера NVIDIA для desktop RTX 5070 лучше взять с актуальной страницы загрузки NVIDIA на момент установки (источник работает на 592.01 / CUDA 13.1; достаточно любого драйвера, дающего CUDA >= 12.8 и поддержку Blackwell sm_120) — подтвердить на целевой машине, что nvidia-smi показывает CUDA >= 12.8
- Подтвердить, что на целевом ПК не установлен конфликтующий/устаревший драйвер от другой GPU и что Secure Boot / режим WDDM не мешают установке драйвера
- Целевая desktop RTX 5070 имеет больше VRAM (обычно 12 ГБ) против 8 ГБ на ноутбучной — для live-инференса не критично, но стоит убедиться, что профиль модели влезает (отдельный домен)


==========================================================================================
# DOMAIN 2: Daheng Galaxy SDK + камера MER2-230-168U3C + gxipy
==========================================================================================

# Установка камерного стека: Daheng Galaxy SDK + gxipy

Этот раздел описывает перенос камерной части на целевой ПК. Камера — **Daheng MER2-230-168U3C** (интерфейс **USB3 Vision**). Python-обёртка `gxipy` НЕ ставится через pip — она входит в состав SDK и подключается к `sys.path` во время выполнения. Поэтому единственный обязательный шаг — корректно установить **тот же** Galaxy SDK по ожидаемому пути.

## 1. Что именно ставит Galaxy SDK и зачем

Источник: `Galaxy SDK 1.18.2208.9301` (сборка от 2022-08-30), установлен в `C:\Program Files\Daheng Imaging\GalaxySDK`. Версия подтверждена в `...\GalaxySDK\ReleaseNote.txt` (строка `VERSION:1.18.2208.9301`).

Инсталлятор SDK выполняет всё, что нужно камере, одним действием:

- **Драйвер USB3 Vision** для камеры (без него камера не появится в системе как устройство Daheng).
- **GenTL-продюсеры** (транспортный слой GenICam) — файлы `.cti` в `...\GalaxySDK\GenTL\Win64`. На источнике их три: `GxU3VTL.cti` (USB3 Vision — используется этой камерой), `GxUSBTL.cti`, `GxGVTL.cti` (GigE). Для USB3-камеры критичен `GxU3VTL.cti`.
- **GenICam runtime** — `...\GalaxySDK\GenICam\bin\Win64_x64`.
- **API-библиотеки** — `...\GalaxySDK\APIDll\Win64`.
- **Python SDK + пакет `gxipy`** — `...\GalaxySDK\Samples\Python SDK\gxipy` (модули: `__init__.py`, `gxiapi.py`, `gxidef.py`, `gxwrapper.py`, `dxwrapper.py`).
- **Приложение Galaxy Viewer (GalaxyView)** — GUI для проверки камеры (ставится в меню «Пуск»).
- **Системные переменные окружения и записи PATH** (через служебный `...\GalaxySDK\Tools\Win64\UpdateEnv.exe`).

Системные переменные, которые инсталлятор выставляет на уровне **Machine** (проверено на источнике именно как Machine-уровень):

| Переменная | Значение на источнике |
|---|---|
| `GENICAM_GENTL64_PATH` | `C:\Program Files\Daheng Imaging\GalaxySDK\GenTL\Win64` |
| `GENICAM_GENTL32_PATH` | `C:\Program Files\Daheng Imaging\GalaxySDK\GenTL\Win32` |
| `GALAXY_GENICAM_ROOT` | `C:\Program Files\Daheng Imaging\GalaxySDK\GenICam` |
| `GALAXY_GENICAM_CACHE` | `C:\ProgramData\Galaxy\xml\cache` |
| `GALAXY_GENICAM_LOG_CONFIG` | `C:\Program Files\Daheng Imaging\GalaxySDK\GenICam\log\config\DebugLogging.properties` |
| `DAHENG_IMAVISION_MER_SETUP_PATH` | `C:\Program Files\Daheng Imaging\GalaxySDK\unins000.exe` |

Плюс в системный `PATH` добавляются `...\GenICam\bin\Win64_x64` и `...\APIDll\Win64`.

**Важно (пояс + ремень):** код всё равно сам пере-добавляет эти DLL-каталоги и переменные на старте через `rt_dlc_live._prepare_sdk_environment` (`rt_dlc_live.py:247-276`): он кладёт в начало `PATH` каталоги `APIDll\Win64`, `GenTL\Win64`, `GenICam\bin\Win64_x64`, вызывает `os.add_dll_directory(...)` для каждого, переопределяет `GENICAM_GENTL64_PATH`/`GENICAM_GENTL32_PATH` и вставляет `...\Samples\Python SDK` в `sys.path`. То есть **даже если переменные окружения по какой-то причине не подхватились, Python-путь camera поднимется**, если SDK лежит по ожидаемому пути. Но Galaxy Viewer (GUI) полагается на системные переменные — для него корректная установка переменных всё равно нужна.

## 2. Установка на целевом ПК (точные шаги)

1. Скачать **ровно ту же** версию: `Galaxy SDK (Windows) 1.18.2208.9301`, разрядность **x64**, с официального сайта Daheng Imaging (China Daheng Group / Imavision), раздел Download → Machine Vision → Galaxy SDK → Windows. Имя инсталлятора вида `Galaxy_Windows_EN_x86_x64_xxx.exe` (один инсталлятор ставит и x86, и x64). Если на сайте доступна только более новая версия — это, как правило, обратносовместимо для данной камеры, но для гарантированно идентичного поведения брать именно 1.18.2208.9301 (можно запросить у Daheng support, если в свежем списке её нет).

   > Открытый вопрос: дистрибутив 1.18.2208.9301 на целевой ПК нужно либо скачать заново, либо перенести с исходного ПК. Самого установщика на исходной машине в проверенных каталогах нет — есть только распакованный SDK. Уточнить у пользователя, где лежит инсталлятор.

2. Запустить инсталлятор **от администратора**. Оставить **путь установки по умолчанию** `C:\Program Files\Daheng Imaging\GalaxySDK` — код и все конфиги ожидают именно его (`config_rt_dlc_live.py:34-36`, дефолт `GALAXY_SDK_ROOT`). При установке отметить компоненты: USB3 driver, GenTL, Python SDK / Samples.

3. После установки **перезагрузить ПК** (или хотя бы выйти/войти в сессию). Это нужно, чтобы новые системные переменные окружения и PATH применились во всех новых процессах, и чтобы USB3-драйвер встал начисто.

4. Скопировать камерные конфиги: создать на целевом ПК каталог `C:\config_daheng` и положить туда все `.txt` из `C:\config_daheng` источника (там 25 файлов профилей; развёрнутый одиночный профиль — `Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt`). Путь `C:\config_daheng` жёстко прописан в `config_dual_rt_dlc_live.py:15,20` и в дефолте `config_rt_dlc_live.py:39-44`.

5. Подключить камеру **в порт USB 3.0** (синий разъём, либо помеченный «SS»). Кабель — родной USB3 Vision. См. пункт о подводных камнях ниже.

## 3. Если SDK нельзя поставить по пути по умолчанию

Если по каким-то причинам SDK ставится в другой каталог, не надо править код — есть env-override:

```powershell
[Environment]::SetEnvironmentVariable('DLC_LIVE_GALAXY_SDK_ROOT', 'D:\Daheng\GalaxySDK', 'Machine')
```

`config_rt_dlc_live.py:34-36` читает `DLC_LIVE_GALAXY_SDK_ROOT` и от него вычисляет все подпути (`APIDll\Win64`, `GenTL\Win64`, `GenICam\bin\Win64_x64`, `Samples\Python SDK`). Аналогично можно переопределить путь к конфигу камеры через `DLC_LIVE_GALAXY_CONFIG_PATH`. После установки переменной — **новый шелл / перезагрузка**, старые процессы её не увидят.

## 4. Проверка (verification)

**Шаг A — GalaxyView (GUI):**
Закрыть всё, что могло захватить камеру. Запустить **Galaxy Viewer (GalaxyView)** из меню «Пуск». В списке устройств должна появиться камера `MER2-230-168U3C` с серийником (на источнике используются SN `FDE22070173`, `FDE22070174`, `FDE22070175`). Открыть устройство, запустить захват — должно идти живое видео. Если камера видна и стримит — драйвер, GenTL и USB3 в порядке. **После проверки обязательно закрыть GalaxyView** (см. подводные камни — он держит камеру эксклюзивно).

**Шаг B — Python smoke-test (`gxipy` импортируется и видит камеру):**
SDK сам по себе не добавляет `Samples\Python SDK` в `sys.path`, это делает код проекта. Поэтому в одноразовом тесте путь добавляем вручную:

```powershell
C:\dlc_live_env\Scripts\python.exe -c "import sys; sys.path.insert(0, r'C:\Program Files\Daheng Imaging\GalaxySDK\Samples\Python SDK'); import gxipy as gx; dm = gx.DeviceManager(); n, info = dm.update_device_list(); print('cameras found:', n); print(info)"
```

Ожидаемо: `cameras found: 1` (или больше, если камер несколько) и список с моделью `MER2-230-168U3C` и серийником. Если `cameras found: 0` при том, что GalaxyView камеру видит — почти наверняка камера занята другим процессом (тот же GalaxyView) либо не подхватились GenTL-переменные/DLL (нужен свежий шелл после перезагрузки).

Более «боевая» проверка — запустить штатный семпл из SDK `...\Samples\Python SDK\GxSingleCamColor.py` (камера цветная, Bayer → RGB).

## 5. Особенность парсера конфигов (`import_config_file`)

Бридж применяет настройки камеры так: сначала пробует `cam.import_config_file(path, verify=False)` (`rt_dlc_live.py:282`), а при ошибке — падает на ручной разбор ключевых полей `_apply_core_config_file` (`rt_dlc_live.py:294-327`). Флаг `GALAXY_CONFIG_VERIFY = False` (`config_rt_dlc_live.py:46`) и `GALAXY_FALLBACK_APPLY_CONFIG = True` (строка 47) специально стоят, потому что:

- Файлы конфигов — это экспорт GenApi persistence (формат `# GenApi persistence file (version 3.0.0)`), пары `Feature<TAB>Value`. Парсер **чувствителен к мусору в строках**: посторонние inline-комментарии `#` в строках значений, а также **хвостовые пробелы** ломают применение/верификацию. В реальном файле источника такое есть — например, строка `AAROIWidth\t1920 ` (с пробелом в конце) и `AutoExposureTimeMax\t1e+006`. Поэтому держим `verify=False` и при необходимости опираемся на ручной fallback, который выставляет только критичные поля (PixelFormat, ROI: Width/Height/OffsetX/OffsetY, AcquisitionFrameRate, ExposureTime, Trigger*).
- **Правило при правке конфигов:** держать строки чистыми — `Feature<TAB>Value`, без inline `#`-комментариев и без хвостовых пробелов. Шапки-разделители вида `</-- ... -->` и строки, начинающиеся с `#` в начале строки (заголовок persistence-файла), парсер пропускает штатно — их трогать не нужно.

Содержательно деплойный профиль `Rat_TREDMILL_Left_...`: `PixelFormat BayerRG8`, ROI `Width 1920 / Height 220 / OffsetX 0 / OffsetY 510`, `AcquisitionFrameRate 100`, `ExposureTime 4000` (мкс), `TriggerMode Off` (свободный ран), `Gain 10`, `BalanceWhiteAuto Continuous`. Цветовое преобразование Bayer→RGB делает сам SDK (`GALAXY_OUTPUT_COLOR = "rgb"`, `config_dual_rt_dlc_live.py:45`).

## 6. Подводные камни (pitfalls)

- **SDK не по дефолтному пути.** Код ждёт `C:\Program Files\Daheng Imaging\GalaxySDK`. Если поставили иначе и не задали `DLC_LIVE_GALAXY_SDK_ROOT` — `gxipy` не импортируется, ошибка `Cannot import Daheng 'gxipy'. Check GALAXY_SDK_ROOT and Galaxy SDK installation` (`rt_dlc_live.py:164`).
- **USB2 вместо USB3.** Камера USB3 Vision: в порт USB 2.0 она либо не поднимется на нужной полосе, либо не выдаст 100 FPS на ROI 1920×220. Только синий/SS-порт USB 3.0; желательно прямой порт на материнской плате, а не через дешёвый хаб. В конфиге `DeviceLinkThroughputLimit 300000000` (300 МБ/с) — USB2 столько не даст.
- **GalaxyView держит камеру.** USB3 Vision — эксклюзивный доступ. Если открыт GalaxyView (или любой другой захватчик), Python-бридж камеру не откроет (или `update_device_list` вернёт 0 / `open_device` упадёт). Перед запуском бриджа закрыть GalaxyView. В коде это прямо отмечено: «GalaxyView should be closed or not acquiring while this runs» (`config_rt_dlc_live.py:33`).
- **Переменные окружения не обновились.** После установки SDK или после ручной правки `DLC_LIVE_GALAXY_SDK_ROOT` нужен **новый процесс/шелл, лучше перезагрузка**. Уже запущенные шеллы и службы видят старое окружение. GalaxyView особенно зависит от системных GenTL-переменных.
- **Разрядность.** Ставить **x64**-компоненты, venv тоже x64 (Python 3.10.11 x64). Код жёстко тянет `Win64`-каталоги (`APIDll\Win64`, `GenTL\Win64`, `GenICam\bin\Win64_x64`).
- **Серийник в дефолте.** Дефолтный `GALAXY_SN = "FDE22070173"` и `GALAXY_CONFIG_PATH` указывают на профиль Top-камеры (`config_rt_dlc_live.py:37,42`). Если на целевом ПК камера с другим серийником — задать `DLC_LIVE_GALAXY_SN` / `DLC_LIVE_GALAXY_CONFIG_PATH` или выбрать камеру по индексу (`DLC_LIVE_GALAXY_INDEX`). Имена `.txt`-конфигов содержат серийник в скобках `(FDE220701xx)` — он специфичен для конкретной физической камеры; на новой камере состав ROI/экспозиции переносится, а соответствие профиль↔серийник нужно пересопоставить.
- **gxipy и pip.** Не пытаться `pip install gxipy` — это не тот пакет/не существует в нужном виде. Источник `gxipy` всегда — папка SDK.

### CHECKLIST
- [ ] Скачать Galaxy SDK для Windows x64, версия 1.18.2208.9301 (China Daheng / Imavision, раздел Download → Galaxy SDK → Windows). Если этой версии нет в списке — запросить у Daheng support или перенести инсталлятор с исходного ПК.
- [ ] Запустить инсталлятор от администратора, оставить путь по умолчанию C:\Program Files\Daheng Imaging\GalaxySDK, выбрать компоненты USB3 driver + GenTL + Python SDK/Samples.
- [ ] Перезагрузить ПК, чтобы применились системные переменные окружения (GENICAM_GENTL64_PATH, GALAXY_GENICAM_ROOT и др.) и записи PATH, и встал USB3-драйвер.
- [ ] Проверить переменные окружения: в PowerShell вывести [Environment]::GetEnvironmentVariable('GENICAM_GENTL64_PATH','Machine') и убедиться, что указывает на ...\GalaxySDK\GenTL\Win64.
- [ ] Создать C:\config_daheng и скопировать туда все .txt-профили камеры из C:\config_daheng исходного ПК (25 файлов).
- [ ] Подключить камеру MER2-230-168U3C в порт USB 3.0 (синий/SS), родным кабелем USB3 Vision.
- [ ] Открыть Galaxy Viewer (GalaxyView) из меню Пуск, убедиться, что камера MER2-230-168U3C энумерируется и стримит живое видео; затем закрыть GalaxyView.
- [ ] Python smoke-test: C:\dlc_live_env\Scripts\python.exe -c "import sys; sys.path.insert(0, r'C:\Program Files\Daheng Imaging\GalaxySDK\Samples\Python SDK'); import gxipy as gx; dm=gx.DeviceManager(); n,info=dm.update_device_list(); print(n, info)" — ожидать n>=1 и модель MER2-230-168U3C.
- [ ] Если SDK ставится не по дефолтному пути — задать [Environment]::SetEnvironmentVariable('DLC_LIVE_GALAXY_SDK_ROOT','<путь>','Machine') и открыть новый шелл.
- [ ] При правке любых .txt-конфигов держать строки в виде Feature<TAB>Value без inline #-комментариев и без хвостовых пробелов; verify оставить выключенным (GALAXY_CONFIG_VERIFY=False).
- [ ] Сопоставить серийник целевой камеры с нужным профилем; при необходимости переопределить DLC_LIVE_GALAXY_SN / DLC_LIVE_GALAXY_CONFIG_PATH / DLC_LIVE_GALAXY_INDEX.

### ARTIFACTS

### OPEN QUESTIONS
- Где находится сам установщик Galaxy SDK 1.18.2208.9301? На исходном ПК распакованный SDK есть, но файла инсталлятора в проверенных каталогах не обнаружено — нужно скачать заново с сайта Daheng или перенести с исходной машины.
- Доступна ли версия 1.18.2208.9301 в текущем списке загрузок Daheng? Если нет — согласовать с пользователем, брать ли более новую (обычно совместима) или запрашивать архивную у поддержки.
- Серийный номер камеры на целевой установке: совпадает ли с FDE22070173/74/75 (дефолты и имена .txt-конфигов завязаны на эти SN), или камера новая и нужно пересопоставить профиль↔серийник.
- Сколько камер будет на целевом ПК — одна (single-best профиль) или две (dual). От этого зависит, какие профили из C:\config_daheng реально нужны и какие серийники указывать.


==========================================================================================
# DOMAIN 3: Перенос кода, ассетов и правка путей
==========================================================================================

# Перенос кода, ассетов и правка путей

Этот раздел описывает, как перенести **рабочее состояние** проекта DLC-Live + Open Ephys на новый ПК так, чтобы команда запуска заработала без правок. Главный риск здесь — **git устарел** (см. ниже), поэтому переносить надо не клон, а реальное рабочее дерево.

## 0. КРИТИЧЕСКИЙ ПУНКТ: git устарел — `git clone` даёт сломанную версию

Проверено на исходной машине (`git status`):

- **12 отслеживаемых файлов изменены и НЕ закоммичены**, в т.ч. ключевой `rt_dlc_live.py` (огромный диф), `config_rt_dlc_live.py`, `run_dlc.py`, `README*`.
- **Все рабочие Gen3-файлы НЕ добавлены в git (untracked)**: `single_rt_dlc_live_bridge.py`, `dual_rt_dlc_live.py`, `config_dual_rt_dlc_live.py`, `live_profiles.py`, `run_live_profile.py`, `send_dual_dlc_bridge_test.py`, `check_online_buffering.py`, вся папка `optimization/`, `setup/`, и все свежие `.md`-доки.

Вывод: **`git clone https://github.com/nikaabigail/DLC_Spinal_cord_stimulation.git` на целевом ПК даст старую, нерабочую версию без файла, который вы запускаете (`single_rt_dlc_live_bridge.py`).** Поэтому миграция ОБЯЗАНА переносить рабочее дерево как архив, а не через clone.

После переноса — **зафиксировать рабочее дерево в git**, чтобы оно перестало быть устаревшим (см. раздел 6).

## 1. Что переносим: рабочее дерево `C:\dlc\DLC_OBS_Spinal_cord_stimulation`

Полный размер папки 163 МБ, но почти всё — мусорные артефакты (логи, .mp4, .csv-бенчмарки, снапшоты). Их переносить НЕ нужно.

**Заархивировать всю папку, ИСКЛЮЧИВ тяжёлые/генерируемые артефакты:**

Исключить:
- `*.mp4` (например `rt_dlc_live_output.mp4` 116 МБ, `rt_dlc_output.mp4` 30 МБ),
- `*.log` (`rt_dlc_debug.log` 12 МБ, `dual_rt_dlc_live_debug.log` 0.75 МБ и др.),
- `*_benchmark.csv` (`rt_dlc_benchmark.csv` 1.5 МБ и т.д.),
- `debug_snapshots/` (4.9 МБ),
- `__pycache__/`,
- `optimization/dump_*.csv` (250-килобайтные дампы, генерируемые).

Включить ОБЯЗАТЕЛЬНО:
- все `.py` (особенно untracked Gen3-файлы), включая `optimization/*.py`,
- все `config_*.py`, `live_profiles.py`, `run_live_profile.py`,
- `setup/requirements.lock.txt` (57 пинов),
- `.gitignore`, `requirements.txt`, все `.md`-доки.

Команда упаковки на источнике (PowerShell, без захвата артефактов):
```powershell
$src = "C:\dlc\DLC_OBS_Spinal_cord_stimulation"
$dst = "C:\dlc\setup_bundle\DLC_OBS_Spinal_cord_stimulation.zip"
New-Item -ItemType Directory -Force "C:\dlc\setup_bundle" | Out-Null
$exclude = '\.mp4$|\.log$|_benchmark\.csv$|\\debug_snapshots\\|\\__pycache__\\|\\optimization\\dump_.*\.csv$'
$files = Get-ChildItem -Path $src -Recurse -File | Where-Object { $_.FullName -notmatch $exclude }
Compress-Archive -Path $files.FullName -DestinationPath $dst -Force
```
(Если важнее простота — можно заархивировать всю папку целиком и просто удалить лишнее после распаковки; на сеть это +160 МБ.)

На целевом ПК распаковать строго в `C:\dlc\DLC_OBS_Spinal_cord_stimulation` (см. раздел 4, почему именно сюда).

## 2. Внешние ассеты — что и куда копировать

Эти данные лежат ВНЕ папки проекта и в git НЕ входят (модель и видео отсечены `.gitignore`: `*.pt`, `*.h5`, `videos/`). Переносить вручную.

### (a) Модель — минимальный LIVE-набор (проверено)

LIVE-путь (`single_rt_dlc_live_bridge.py`) грузит модель напрямую: `DLCLive(model_path=config.MODEL_PATH, ...)`, где `MODEL_PATH` указывает на **один экспортированный самодостаточный файл** `.pt` (метаданные/bodyparts читаются из самого `.pt`). Отдельный `pytorch_config.yaml` рядом с экспортированной моделью НЕ требуется (его там и нет).

Минимальный набор для LIVE (всего 93 МБ):
```
C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\
  DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\
    DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt   (97 МБ на диске)
```
Скопировать в **тот же путь** на целевом ПК.

`config.yaml` проекта (`C:\dlc\project\r_tm_side-og-2024-10-25\config.yaml`, 4.6 КБ) для LIVE-запуска **не нужен** — он используется только офлайн-скриптами (`run_dlc.py`, `check_*.py`). Можно скопировать «для порядка», но на работу live-моста не влияет.

**Полный проект (2.3 ГБ)** — `C:\dlc\project\r_tm_side-og-2024-10-25` целиком — нужен ТОЛЬКО если планируется дообучение/переэкспорт. Для запуска стимуляции не нужен. Не тащить, если задача — только воспроизвести рабочую систему.

### (b) Конфиги камер Daheng

```
C:\config_daheng\*.txt
```
Деплоится `Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt` (используется в single-best как «left»; dual ещё берёт `...Right...(FDE22070175).txt`, а базовый конфиг — `...Top...(FDE22070173).txt`). Скопировать **всю папку** `C:\config_daheng\` (25 файлов, ~124 КБ) в тот же путь — дёшево и страхует от пропуска нужного профиля. ВАЖНО: серийники камер в именах файлов (FDE2207017x) — это серийники конкретных камер. На целевом ПК камеры должны быть теми же физическими устройствами (или конфиг переименовать/перепривязать под новые SN).

### (c) Видео (опционально, только для офлайн-прогона)

```
C:\dlc\videos\   (1.9 ГБ)
```
Нужны только для офлайн `USE_VIDEO_FILE=True` / `run_dlc.py analyze`. Для live-стимуляции НЕ нужны. Переносить выборочно или не переносить.

## 3. Полный перечень захардкоженных абсолютных путей (выгреб из .py)

Все ниже — реальные строки из деплоя. Сгруппированы по файлам. Помечено, есть ли env-override.

**`live_profiles.py` (строки 10-11):**
- `WORK_DIR = C:\dlc\DLC_OBS_Spinal_cord_stimulation` — без override; формирует команды запуска и пути логов профилей.
- `PYTHON_EXE = C:\dlc_live_env\Scripts\python.exe` — без override.

**`config_rt_dlc_live.py` (базовый конфиг live, наследуется dual и single):**
- `VIDEO_FILE_PATH` (19-21): `C:\dlc\videos\1_MER2-230-168U3C(FDE22070174)_20240604_152156.avi` — override `DLC_LIVE_VIDEO_PATH`.
- `GALAXY_SDK_ROOT` (34-36): `C:\Program Files\Daheng Imaging\GalaxySDK` — override `DLC_LIVE_GALAXY_SDK_ROOT`.
- `GALAXY_CONFIG_PATH` (39-43): `C:\config_daheng\Rat_TREDMILL_Top_1920px_340px_100Hz_(FDE22070173).txt` — override `DLC_LIVE_GALAXY_CONFIG_PATH`.
- `MODEL_PATH` (63-67): `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt` — override `DLC_LIVE_MODEL_PATH`.
- `OUTPUT_VIDEO_PATH` (136), `LOG_PATH` (143), `BENCHMARK_CSV_PATH` (147): все внутри `C:\dlc\DLC_OBS_Spinal_cord_stimulation\...` — без override, но это пути ВНУТРИ рабочей папки (создаются автоматически, безопасны).

**`config_dual_rt_dlc_live.py` (используется и single-мостом — `single_rt_dlc_live_bridge.py` импортирует именно его):**
- `DUAL_CAMERAS[0].config_path` (15): `C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt` — без override (хардкод в списке).
- `DUAL_CAMERAS[1].config_path` (20): `C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt` — без override.
- `DUAL_OUTPUT_LEFT_PATH` (81), `DUAL_OUTPUT_RIGHT_PATH` (82), `LOG_PATH` (89), `BENCHMARK_CSV_PATH` (90): внутри рабочей папки — безопасны.

**`config_rt_dlc.py` (старый OBS-конфиг, не live; нужен только если запускаете `rt_dlc_obs.py`):**
- `VIDEO_FILE_PATH` (15), `DLC_SNAPSHOT` (30-33, путь `dlc-models-pytorch\...\snapshot-best-380.pt`), `DLC_PYTORCH_CFG` (34-37, `pytorch_config.yaml`), `OUTPUT_VIDEO_PATH` (135), `LOG_PATH` (158), `BENCHMARK_CSV_PATH` (171), dual `VIDEO_FILE_PATHS` (178-180) — все в `C:\dlc\...`, без override. ВНИМАНИЕ: `DLC_SNAPSHOT` тут указывает на НЕэкспортированный снапшот внутри полного проекта `dlc-models-pytorch\...` — этого файла в LIVE-наборе НЕТ. Для live он не нужен.

**`run_dlc.py` (офлайн, для обучения/анализа — нужен полный проект):**
- `CONFIG_PATH` (11): `C:\dlc\project\r_tm_side-og-2024-10-25\config.yaml`, и далее `PROJECT_PATH`, `LABELED_DATA_DIR`, `pytorch_config.yaml` производятся от него. Без override.

**`check_dlc_dataset.py` (5), `check_dlc_shuffles.py` (4):**
- `PROJECT_PATH = C:\dlc\project\r_tm_side-og-2024-10-25` — без override.

**`optimization\run_live_cudagraphs.py` (11,14):** только в docstring-примерах `C:\dlc_live_env\Scripts\python.exe` — не исполняется.

**Galaxy SDK env-пути** в `rt_dlc_live.py:247-276` (`_prepare_sdk_environment`) строятся ОТНОСИТЕЛЬНО `GALAXY_SDK_ROOT` (`APIDll\Win64`, `GenTL\Win64`, `GenICam\bin\Win64_x64`, `Samples\Python SDK`). Если SDK стоит по тому же пути — править нечего.

## 4. Рекомендация: повторить те же пути `C:\` (НИЧЕГО не править) — самый простой вариант

Часть путей (особенно `DUAL_CAMERAS[*].config_path`, `WORK_DIR`, `PYTHON_EXE`) **не имеют env-override**. Поэтому **самый надёжный и быстрый путь — воссоздать на целевом ПК ту же файловую структуру `C:\`**, тогда правок в коде НОЛЬ:

| Источник | Цель (та же) |
| --- | --- |
| `C:\dlc\DLC_OBS_Spinal_cord_stimulation\` | то же |
| `C:\dlc_live_env\` (venv, Python 3.10.11) | то же |
| `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\...snapshot-best-380.pt` | то же |
| `C:\config_daheng\*.txt` | то же |
| `C:\Program Files\Daheng Imaging\GalaxySDK\` | то же (ставится инсталлятором SDK) |
| `C:\dlc\videos\` (опц.) | то же |

**Альтернатива (env-override)** — если разложить иначе, можно переопределить через переменные окружения: `DLC_LIVE_MODEL_PATH`, `DLC_LIVE_GALAXY_SDK_ROOT`, `DLC_LIVE_GALAXY_CONFIG_PATH`, `DLC_LIVE_VIDEO_PATH`, `DLC_LIVE_GALAXY_SN`, `DLC_LIVE_CAMERA_BACKEND`. **НО** override НЕ покрывают `DUAL_CAMERAS[*].config_path`, `WORK_DIR`, `PYTHON_EXE` и пути в `config_rt_dlc.py`/офлайн-скриптах — их пришлось бы править в коде руками. Поэтому env-вариант оправдан, только если по какой-то причине нельзя занять диск `C:\` теми же путями; иначе — повторяйте пути 1-в-1.

## 5. Конкретный чек-лист копирования

1. На источнике: упаковать рабочее дерево по разделу 1 → `DLC_OBS_Spinal_cord_stimulation.zip`.
2. Перенести также (отдельно): файл модели `.pt` (93 МБ), всю папку `C:\config_daheng\` (~124 КБ), `setup\requirements.lock.txt` (уже внутри архива). Опционально — `C:\dlc\videos\` и/или полный проект 2.3 ГБ для дообучения.
3. На целевом ПК создать каталоги: `C:\dlc`, `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5`, `C:\config_daheng`.
4. Распаковать архив в `C:\dlc\DLC_OBS_Spinal_cord_stimulation`.
5. Положить `.pt` точно в путь из `MODEL_PATH` (раздел 2a).
6. Скопировать конфиги камер в `C:\config_daheng`.
7. Проверить наличие venv `C:\dlc_live_env` и Galaxy SDK (это домены venv/SDK, но пути должны совпасть).
8. Smoke-тест путей одной командой (см. раздел 7) ДО реального запуска.
9. Запуск: `cd C:\dlc\DLC_OBS_Spinal_cord_stimulation ; C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --no-display`.

## 6. Зафиксировать рабочее дерево в git (чтобы перестало быть устаревшим)

Лучший вариант — **на ИСХОДНОЙ машине** добавить и закоммитить рабочее дерево, запушить в `main`, затем на целевом просто `git pull`. Если так не сделали и переносите архивом — на целевом ПК инициализировать/привязать репо и закоммитить:
```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
git add -A
git status   # убедиться, что Gen3-файлы и config_*.py попали в индекс
git commit -m "Snapshot working tree (Gen3 live bridge) from source machine"
git push origin main   # если remote доступен и это согласовано
```
`.gitignore` уже исключает `*.pt`, `*.mp4`, логи и `videos/` — модель и видео в репозиторий не попадут (это правильно; их переносим вручную).

## 7. Подводные камни

- **Не делайте `git clone` как способ переноса кода** — получите старую версию без `single_rt_dlc_live_bridge.py`. Только архив рабочего дерева.
- **single-мост зависит от `config_dual_rt_dlc_live.py`** (а тот — от `config_rt_dlc_live.py`). Если перенести только «single»-файлы и забыть dual-конфиг — импорт упадёт. Переносите ВСЕ `config_*.py` и `live_profiles.py`.
- **`DUAL_CAMERAS[*].config_path` не имеет env-override** — если путь `C:\config_daheng\...` не совпадёт, камеры не сконфигурируются. Это вторая причина держать пути 1-в-1.
- Имена конфигов камер содержат **серийники** (FDE2207017x). На целевом ПК должны быть те же камеры, либо конфиги перепривязать.
- `config_rt_dlc.py:DLC_SNAPSHOT` ссылается на снапшот внутри ПОЛНОГО проекта (`dlc-models-pytorch\...`), которого нет в LIVE-наборе — для live он не используется, не пугайтесь отсутствия файла.
- Кириллица в путях профиля Windows (`C:\Users\Владимир\...`) исходной машины НЕ влияет на live-путь: все рабочие пути — латиница (`C:\dlc`, `C:\config_daheng`, `C:\dlc_live_env`). На целевом ПК имя пользователя роли не играет, пока вы держите проект под `C:\dlc`.

Быстрая проверка всех LIVE-путей до запуска:
```powershell
$paths = @(
  "C:\dlc\DLC_OBS_Spinal_cord_stimulation\single_rt_dlc_live_bridge.py",
  "C:\dlc\DLC_OBS_Spinal_cord_stimulation\config_dual_rt_dlc_live.py",
  "C:\dlc\DLC_OBS_Spinal_cord_stimulation\live_profiles.py",
  "C:\dlc_live_env\Scripts\python.exe",
  "C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt",
  "C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt",
  "C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt",
  "C:\Program Files\Daheng Imaging\GalaxySDK"
)
$paths | ForEach-Object { "{0}  {1}" -f (Test-Path $_), $_ }
```
Все строки должны вернуть `True`.

### CHECKLIST
- [ ] На ИСХОДНОЙ машине упаковать рабочее дерево C:\dlc\DLC_OBS_Spinal_cord_stimulation в zip, ИСКЛЮЧИВ *.mp4, *.log, *_benchmark.csv, debug_snapshots/, __pycache__/, optimization/dump_*.csv, но ВКЛЮЧИВ все .py (особенно untracked Gen3: single_rt_dlc_live_bridge.py, dual_rt_dlc_live.py, config_dual_rt_dlc_live.py, live_profiles.py, run_live_profile.py, optimization/*.py), все config_*.py, setup/requirements.lock.txt, .gitignore, .md-доки
- [ ] НЕ использовать git clone для переноса кода — он даёт устаревшую сломанную версию без рабочих Gen3-файлов
- [ ] Скопировать файл модели (93 МБ): C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt
- [ ] Скопировать всю папку C:\config_daheng\*.txt (25 файлов, ~124 КБ) — конфиги камер Daheng
- [ ] (Опционально) C:\dlc\videos\ (1.9 ГБ) только для офлайн-анализа; полный проект 2.3 ГБ только для дообучения
- [ ] На ЦЕЛЕВОМ ПК создать каталоги C:\dlc, путь exported-models-pytorch\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5, C:\config_daheng
- [ ] Распаковать архив строго в C:\dlc\DLC_OBS_Spinal_cord_stimulation (тот же путь — иначе WORK_DIR/PYTHON_EXE/DUAL_CAMERAS не совпадут)
- [ ] Положить .pt в точный путь из MODEL_PATH; конфиги камер в C:\config_daheng
- [ ] Воспроизвести те же C:\-пути 1-в-1 (рекомендуется) — тогда правок в коде НОЛЬ; env-override (DLC_LIVE_MODEL_PATH/GALAXY_*/VIDEO) не покрывает DUAL_CAMERAS config_path, WORK_DIR, PYTHON_EXE
- [ ] Прогнать PowerShell smoke-тест Test-Path по всем LIVE-путям — все должны быть True
- [ ] Запустить: cd C:\dlc\DLC_OBS_Spinal_cord_stimulation ; C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --no-display
- [ ] После переноса зафиксировать рабочее дерево в git (git add -A; commit; при согласии push origin main) — или запушить с источника заранее, чтобы репо перестало быть устаревшим

### ARTIFACTS

### OPEN QUESTIONS
- Целевые камеры Daheng — те же физические устройства (серийники FDE22070174/175/173 зашиты в имена конфигов и в GALAXY_SN)? Если камеры другие — конфиги и SN надо перепривязать
- Допустимо ли занять на целевом ПК те же пути C:\dlc, C:\dlc_live_env, C:\config_daheng (рекомендуемый zero-edit вариант)? Если нет — потребуется ручная правка путей без env-override (DUAL_CAMERAS, WORK_DIR, PYTHON_EXE)
- Нужен ли на целевом ПК офлайн-функционал (run_dlc.py train/analyze, check_*.py)? Если да — тащить полный проект 2.3 ГБ + ставить полный пакет deeplabcut; если нужен только live — достаточно 93 МБ .pt
- Переносить ли видео C:\dlc\videos (1.9 ГБ)? Для live-стимуляции не нужны
- Есть ли доступ на push в репозиторий nikaabigail/DLC_Spinal_cord_stimulation, чтобы закоммитить рабочее дерево и снять устаревание git? Иначе коммитить локально


==========================================================================================
# DOMAIN 4: Open Ephys + C++ плагин DualDLCLiveBridge (вопрос Visual Studio)
==========================================================================================

# Open Ephys и C++ плагин DualDLCLiveBridge: перенос на целевой ПК

## Краткий ответ на вопрос про Visual Studio

**Для самого плагина Visual Studio нужна только в одном из двух сценариев — при пересборке из исходников.** Если получится переиспользовать уже собранный `DualDLCLiveBridge.dll`, Visual Studio на целевой машине **не нужна совсем**.

**Python-сторона НЕ требует Visual Studio ни при каком сценарии.** Это важно проговорить отдельно: бэкенд `torch.compile(backend="cudagraphs")` использует только CUDA-рантайм (он уже внутри колеса `torch==2.10.0+cu128`), компилятор C++ ему не нужен. Visual Studio понадобился бы только бэкенду `inductor` (Triton). Весь live-путь Python — это чистые pip-колёса без шагов сборки.

Итого по плагину — два пути:
- **Путь A (рекомендуется первым): скопировать готовый `DualDLCLiveBridge.dll`.** Самый быстрый, Visual Studio НЕ нужна. Работает только при совпадении ABI (см. ниже).
- **Путь B (запасной): пересобрать из исходников.** Требует Visual Studio (MSVC) + CMake + совпадающее дерево исходников `plugin-GUI`.

---

## 1. Что реально лежит на исходной машине (проверено)

### 1.1 Дерево сборки Open Ephys GUI

Open Ephys собран **из исходников** (не установщиком), каталог:
```
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
```

Внутри — ДВА дерева сборки:
- `Build\` — старое, через генератор Visual Studio/MSBuild (`open-ephys.vcxproj`, `open-ephys-GUI.slnx`), от декабря.
- `out\build\x64-Debug\` — **актуальное, рабочее**, генератор **Ninja**, конфигурация **Debug**. Здесь лежит живой `open-ephys.exe` (от 4 июня) и собранный плагин.

`GUI_VERSION = 1.0.1` (из корневого `CMakeLists.txt`).

**Версия плагинного API (критично для совместимости): `PLUGIN_API_VER = 10`** (из `Source\Processors\PluginManager\OpenEphysPlugin.h:40`).

### 1.2 Чем собрано (важно для ABI)

Из `out\build\x64-Debug\CMakeCache.txt`:
- Компилятор: **MSVC `cl.exe` версии 14.51.36231** из `Microsoft Visual Studio\18\Insiders` (то есть **Visual Studio 18 Insiders / preview-тулчейн**).
- Генератор: **Ninja**, `CMAKE_BUILD_TYPE = Debug`.
- CMake: ветка 3.31.

Это значит: DLL — **Debug-сборка**, собранная **preview-компилятором VS18 Insiders**. Это ключевой фактор для оценки совместимости в Пути A (см. раздел 4).

### 1.3 Собранный DLL (проверено, найден в двух местах, побайтово одинаковый)

```
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug\Plugins\DualDLCLiveBridge.dll
C:\tmp\Dual_DLC_live_plugin\dist\windows-x64-debug\DualDLCLiveBridge.dll
```
- Размер: **385 024 байт**, дата 4 июня 18:55.
- Экспортирует стандартные точки входа Open Ephys: `getLibInfo`, `getPluginInfo` (проверено по строкам в DLL). Внутренняя строка-имя: `Dual DLCLive Bridge`.
- Рядом лежат `.lib/.exp/.pdb/.ilk` — типичные артефакты Debug-сборки MSVC.

### 1.4 Исходники плагина (две копии, идентичны по составу)

```
C:\tmp\Dual_DLC_live_plugin\open_ephys_plugin\DualDLCLiveBridge\
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\Plugins\DualDLCLiveBridge\
```
Состав:
```
DualDLCLiveBridge.cpp        (~37 КБ, основная логика: UDP-приём, разбор DDLP/v1, фильтрация, угол, TTL)
DualDLCLiveBridge.h
DualDLCLiveBridgeEditor.cpp  (UI-редактор узла, статусная строка)
DualDLCLiveBridgeEditor.h
OpenEphysLib.cpp             (getLibInfo/getPluginInfo, тип PROCESSOR / UTILITY)
CMakeLists.txt               (add_sources + PluginRules.cmake)
check_plugin_load.py         (smoke-тест: грузит DLL и проверяет экспорты)
README.md
```
Сторонних C++-зависимостей у плагина нет — только Open Ephys Plugin-GUI/JUCE и Windows-сокеты.

### 1.5 Как плагин «вшит» в GUI

В `Plugins\CMakeLists.txt` строка 20: `add_subdirectory(DualDLCLiveBridge)` — плагин уже включён в общую сборку наравне со штатными (ArduinoOutput, BandpassFilter и т.д.). Поэтому при пересборке всего дерева DLL собирается автоматически.

### 1.6 Что делает узел (из исходников)

- Имя процессора в GUI: **`Dual DLCLive Bridge`**, тип **Utility** (`OpenEphysLib.cpp`).
- Параметры: `enabled` (вкл. UDP-слушатель), `udp_port` (по умолчанию **47000**).
- Сокет биндится строго на **loopback**: `socket->bindToPort(port, "127.0.0.1")` — слушает только `127.0.0.1`, удалённые пакеты не примет (это нормально, Python шлёт локально).
- Создаёт TTL-канал событий **`Dual DLCLive TTL`**.
- Протокол: бинарный, магия **`DDLP`** (4 байта), версия 1; разбор в `applyBinaryPosePacket`. Также понимает JSON-pose и legacy ttl_lines.
- Значение TTL-линий: 0 = валиден левый триплет, 1 = правый триплет, 2 = триггер по углу левой задней лапы, 3 = по углу правой (если включён `angle_trigger`).

### 1.7 Куда GUI ищет плагины (из `PluginManager.cpp`, проверено)

При старте `open-ephys.exe` плагины грузятся из путей (Windows):
1. `<папка_exe>\plugins` — для сборки из исходного дерева.
2. `%LOCALAPPDATA%\Open Ephys\plugins-api10` — **этот путь использует и установленный релиз Open Ephys** (суффикс `-api10` берётся из `PLUGIN_API_VER=10`).

Путь №2 добавляется только если путь к exe НЕ содержит `plugin-GUI\Build\`. То есть установленный из инсталлятора Open Ephys будет искать кастомные плагины именно в `%LOCALAPPDATA%\Open Ephys\plugins-api10`.

---

## 2. Целевая машина: что нужно понять в первую очередь

На целевом ПК «есть OpenEphys», но **скорее всего это стандартный релиз из официального установщика**, в котором кастомного плагина `DualDLCLiveBridge` нет. Кастомный плагин обязательно должен там появиться — иначе узел `Dual DLCLive Bridge` не возникнет в списке процессоров, и весь путь стимуляции работать не будет.

**Первое, что нужно сделать на целевой машине — определить версию установленного Open Ephys и его plugin-API:**

```powershell
# Найти исполняемый файл и спросить версию (в логе при старте печатается "Open Ephys GUI vX (Plugin API vN)")
Get-ChildItem "C:\Program Files\Open Ephys" -Recurse -Filter "open-ephys.exe" -ErrorAction SilentlyContinue
# Где релиз ищет кастомные плагины:
explorer "$env:LOCALAPPDATA\Open Ephys"
Test-Path "$env:LOCALAPPDATA\Open Ephys\plugins-api10"
```

Если установленный релиз — это **plugin-API 10** (тот же, что на исходной машине), Путь A имеет шанс. Если API другой (api8, api9, ...) — Путь A исключён, нужен Путь B на совпадающих исходниках.

---

## 3. ПУТЬ A — скопировать готовый DLL (быстро, без Visual Studio)

Это рекомендуемый первый шаг.

### 3.1 Действия

1. Скопировать с исходной машины файл:
   ```
   C:\tmp\Dual_DLC_live_plugin\dist\windows-x64-debug\DualDLCLiveBridge.dll
   ```
   (385 024 байта; идентичен сборочному `out\build\x64-Debug\Plugins\DualDLCLiveBridge.dll`).

2. Положить его на целевой машине в плагинную папку установленного Open Ephys:
   ```powershell
   $dst = "$env:LOCALAPPDATA\Open Ephys\plugins-api10"
   New-Item -ItemType Directory -Force $dst
   Copy-Item "<путь к скопированному>\DualDLCLiveBridge.dll" $dst
   ```
   (Если у установленного релиза другой суффикс api — подставить его: `plugins-apiN`. Также можно положить рядом с `open-ephys.exe` в подпапку `plugins`, если она используется.)

3. Запустить Open Ephys и проверить, что в списке процессоров (Filters/Utilities) появился узел **`Dual DLCLive Bridge`**.

### 3.2 Условия совместимости (обязательно проговорить)

Путь A сработает только если одновременно:
- **Совпадает версия plugin-API** установленного Open Ephys (`plugins-api10`). Open Ephys грузит только плагины своего API; иначе DLL молча игнорируется.
- **Совпадает (или близка) версия самого Open Ephys GUI / JUCE-ABI.** Плагины Open Ephys крайне чувствительны к ABI — DLL собран против конкретного дерева `plugin-GUI` (`GUI_VERSION 1.0.1`). Установленный релиз должен соответствовать этому же дереву.
- **На целевой машине есть нужный рантайм MSVC.** DLL собран компилятором **VS18 Insiders (MSVC 14.51), конфигурация Debug**.

### 3.3 Главный риск Пути A — это Debug-сборка preview-компилятором

DLL — **Debug** (зависит от отладочного CRT `ucrtbased.dll` / `vcruntime140d.dll`, которых нет на обычной машине без установленной Visual Studio), да ещё и собран **preview-тулчейном VS18 Insiders**. Установленный из официального инсталлятора Open Ephys — это всегда **Release**, собранный стабильным тулчейном. Смешивание Debug-плагина с Release-хостом по разным CRT — частый источник того, что «плагин не виден» или GUI падает при загрузке.

Поэтому реалистичная оценка: **Путь A с этим конкретным Debug-DLL, скорее всего, не заведётся против установленного Release Open Ephys.** Есть три варианта внутри Пути A:
- **A1:** перенести на целевую машину целиком рабочее дерево `out\build\x64-Debug` (вместе с `open-ephys.exe` Debug и его `plugins\DualDLCLiveBridge.dll`) и запускать именно этот exe. Тогда хост и плагин — одинаковые Debug одного тулчейна, совместимость гарантирована. Минус: тащить отладочный рантайм и весь билд, и установленный «штатный» Open Ephys при этом не используется.
- **A2:** попробовать положить Debug-DLL в `plugins-api10` установленного Release-релиза. Быстро проверить — но будь готов, что не загрузится из-за Debug-CRT/ABI.
- **A3 (чистый вариант):** на исходной машине пересобрать плагин в **Release** против того же дерева `plugin-GUI` и переносить именно Release-DLL. Это ближе к Пути B по требованиям, но сборку делаем на исходной машине, где тулчейн уже есть.

**Практическая рекомендация:** сначала быстро попробовать **A2** (минута работы, копирование одного файла). Если узел не появился — переходить к **A1** (перенести рабочее Debug-дерево целиком и запускать его `open-ephys.exe`) или к Пути B.

---

## 4. ПУТЬ B — пересборка из исходников (нужна Visual Studio)

Применяется, если Путь A не прошёл по ABI/API, либо нужна чистая Release-сборка под установленный Open Ephys.

### 4.1 Что установить на целевой машине

- **Visual Studio 2022 Community 17.x** (или Build Tools 2022/2026) с рабочей нагрузкой **«Разработка классических приложений на C++» (Desktop development with C++)**: компилятор MSVC x64, Windows 10/11 SDK.
  - Замечание: исходный DLL собран VS18 Insiders, но для пересборки это **не обязательно** — подойдёт стабильная VS 2022 17.x. Главное, чтобы плагин и `open-ephys.exe` целевой машины были собраны **одним и тем же** тулчейном и в одной конфигурации (оба Release или оба Debug).
- **CMake** (на исходной машине ветка 3.31; минимум из `CMakeLists.txt` плагина — 3.15). Достаточно любой современной 3.2x+.
- **Ninja** — если воспроизводить ту же схему (`out\build\x64-Debug`, генератор Ninja). Опционально: можно собрать и через генератор Visual Studio.
- **Git** — для клонирования исходников.

Проверить наличие VS, не полагаясь на видимость `cl.exe` в обычном PowerShell (он доступен только в Developer-консоли):
```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -all -products * -format json
```

### 4.2 Что собирать

Нужно **совпадающее дерево исходников `plugin-GUI`** той же версии, что хост на целевой машине (важно, чтобы `PLUGIN_API_VER` совпал). Внутрь него поместить исходники плагина и собрать.

1. Положить исходники плагина в дерево GUI:
   ```
   <plugin-GUI-root>\Plugins\DualDLCLiveBridge\   (5 файлов .cpp/.h + CMakeLists.txt + OpenEphysLib.cpp)
   ```
   Источник для копирования — `C:\tmp\Dual_DLC_live_plugin\open_ephys_plugin\DualDLCLiveBridge\`.

2. Убедиться, что `<plugin-GUI-root>\Plugins\CMakeLists.txt` содержит строку:
   ```cmake
   add_subdirectory(DualDLCLiveBridge)
   ```
   (на исходной машине это строка 20 — уже добавлено).

3. Сконфигурировать и собрать (пример с Ninja, конфигурация Debug — как на источнике; для установленного Release-хоста собирать Release):
   ```powershell
   # из Developer PowerShell for VS
   cmake -S <plugin-GUI-root> -B <plugin-GUI-root>\out\build\x64-Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DOE_DONT_CHECK_BUILD_PATH=TRUE
   cmake --build <plugin-GUI-root>\out\build\x64-Debug --target DualDLCLiveBridge
   ```
   Результат: `<...>\out\build\x64-Debug\Plugins\DualDLCLiveBridge.dll`.

4. Скопировать полученный DLL в плагинную папку нужного хоста (`<exe_dir>\plugins` или `%LOCALAPPDATA%\Open Ephys\plugins-apiN`).

### 4.3 Параметр OE_DONT_CHECK_BUILD_PATH

На исходной машине в `CMakeSettings.json` выставлен `OE_DONT_CHECK_BUILD_PATH=TRUE` — Open Ephys по умолчанию проверяет путь сборки; этот флаг отключает проверку. При сборке вне «канонического» пути его нужно повторить.

---

## 5. Проверка без камеры (плагин можно полностью протестировать отдельно)

Огромный плюс: проверить плагин можно **без камеры и без всего Python-инференса** — синтетическими UDP-пакетами. В репозитории есть готовый отправитель `send_dual_dlc_bridge_test.py` (он есть и в рабочем каталоге `C:\dlc\DLC_OBS_Spinal_cord_stimulation\`, и в `C:\tmp\Dual_DLC_live_plugin\python\`).

Порядок:
1. Запустить Open Ephys с загруженным плагином, перетащить узел **`Dual DLCLive Bridge`** в сигнальную цепочку.
2. В UI узла убедиться: `enabled` включён, `udp_port = 47000`.
3. Отправить тестовые пакеты:
   ```powershell
   C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\send_dual_dlc_bridge_test.py --host 127.0.0.1 --port 47000 --mode pose --wire-format binary
   ```
   **Важно про флаг:** правильное имя — `--wire-format binary` (в одном из migration-документов встречается ошибочное `--wire binary`). По умолчанию скрипт и так шлёт на `127.0.0.1:47000` бинарным DDLP, так что можно запустить и вовсе без аргументов. Параметр `--mode ttl` гоняет напрямую TTL-линии.

### Что должно произойти (признаки успеха)
- Счётчик принятых пакетов в статусной строке узла растёт.
- Статус сообщает про binary pose пакеты (и/или `listening on UDP 127.0.0.1:47000`).
- TTL-линии канала `Dual DLCLive TTL` меняются в зависимости от синтетических точек и порогов.

Дополнительно: smoke-тест самого DLL (без запуска GUI) — `check_plugin_load.py` рядом с исходниками: грузит DLL и проверяет, что экспортируются `getLibInfo`/`getPluginInfo`.

---

## 6. Типичные причины «плагин не виден» (из исходников и migration-доков)

- DLL положен не в ту плагинную папку (нужна `plugins-apiN`, совпадающая по N, или `<exe_dir>\plugins`).
- Несовпадение plugin-API / ABI Open Ephys (плагин собран против другой версии GUI).
- Отсутствует нужный рантайм Visual C++ (особенно отладочный — для Debug-сборки).
- В дереве исходников не добавлено `add_subdirectory(DualDLCLiveBridge)` (при Пути B).
- Запущен не тот `open-ephys.exe`, рядом с которым лежит собранный плагин.

---

## 7. Резюме и рекомендация

1. Python-сторона **не требует Visual Studio** — это отдельный факт, его стоит проговорить пользователю явно.
2. По плагину сначала определить **plugin-API установленного Open Ephys** на целевой машине (нужен `api10`).
3. Попробовать **Путь A2** (скопировать готовый DLL в `plugins-api10`) — это минута, Visual Studio не нужна. Но учесть: DLL — Debug-сборка preview-компилятором, против Release-релиза он, вероятно, не загрузится.
4. Если A2 не вышло — либо **A1** (перенести рабочее Debug-дерево `out\build\x64-Debug` целиком и запускать его `open-ephys.exe`), либо **Путь B** (пересборка из совпадающих исходников; тут Visual Studio + CMake обязательны), либо **A3** (собрать Release-DLL на исходной машине).
5. В любом случае финальная проверка — без камеры, через `send_dual_dlc_bridge_test.py ... --wire-format binary`: узел появился, счётчик пакетов растёт, TTL-линии реагируют.

### CHECKLIST
- [ ] На целевой машине определить версию установленного Open Ephys и его plugin-API (нужен api10): найти open-ephys.exe и проверить Test-Path "$env:LOCALAPPDATA\Open Ephys\plugins-api10"
- [ ] ПУТЬ A2 (сначала, без Visual Studio): скопировать C:\tmp\Dual_DLC_live_plugin\dist\windows-x64-debug\DualDLCLiveBridge.dll (385024 байта) в %LOCALAPPDATA%\Open Ephys\plugins-api10 на целевой машине
- [ ] Запустить Open Ephys и проверить, что в списке процессоров появился узел 'Dual DLCLive Bridge' (тип Utility)
- [ ] Если узел не появился (вероятно из-за того, что DLL — Debug-сборка VS18 Insiders, а релиз — Release): перейти к A1 — перенести рабочее дерево out\build\x64-Debug целиком и запускать его open-ephys.exe; либо к Пути B
- [ ] ПУТЬ B (если нужна пересборка): установить Visual Studio 2022 Community 17.x с нагрузкой 'Desktop development with C++' (MSVC x64 + Windows SDK), CMake 3.2x+, при необходимости Ninja и Git
- [ ] Проверить наличие VS через vswhere.exe (не полагаться на видимость cl.exe в обычном PowerShell)
- [ ] ПУТЬ B: получить совпадающее по версии дерево исходников plugin-GUI (совпадение PLUGIN_API_VER=10), положить плагин в <plugin-GUI-root>\Plugins\DualDLCLiveBridge\, убедиться в add_subdirectory(DualDLCLiveBridge) в Plugins\CMakeLists.txt
- [ ] ПУТЬ B: собрать через cmake -G Ninja -DOE_DONT_CHECK_BUILD_PATH=TRUE --target DualDLCLiveBridge (Release под установленный релиз, Debug — если воспроизводим источник), скопировать DLL в plugins-apiN целевого хоста
- [ ] Перетащить узел 'Dual DLCLive Bridge' в сигнальную цепочку, в UI выставить enabled и udp_port=47000
- [ ] Проверить плагин БЕЗ камеры: C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\send_dual_dlc_bridge_test.py --host 127.0.0.1 --port 47000 --mode pose --wire-format binary (внимание: флаг --wire-format, НЕ --wire)
- [ ] Убедиться: счётчик принятых пакетов в статусной строке растёт, статус сообщает про binary pose / listening on UDP 127.0.0.1:47000, TTL-линии канала 'Dual DLCLive TTL' реагируют
- [ ] Подтвердить пользователю отдельно: Python-сторона (cudagraphs) Visual Studio НЕ требует — нужны только CUDA-рантайм из колеса torch и чистые pip-колёса

### ARTIFACTS

### OPEN QUESTIONS
- Что именно установлено на целевой машине как 'OpenEphys': официальный релиз из установщика, дерево исходников plugin-GUI, или иной дистрибутив? От этого зависит выбор Пути A vs B.
- Какова версия plugin-API установленного Open Ephys на целевой машине — это api10 (как на источнике) или другой? Если не api10, Путь A исключён.
- Совпадает ли версия Open Ephys GUI / JUCE-ABI с деревом-источником (GUI_VERSION 1.0.1)? Плагины Open Ephys крайне чувствительны к ABI.
- Готовый DLL — это Debug-сборка preview-компилятором VS18 Insiders (MSVC 14.51); он зависит от отладочного рантайма (vcruntime140d/ucrtbased). Загрузится ли он в Release-хост на целевой машине, или нужна Release-пересборка (A3/Путь B)?
- Доступно ли на исходной машине дерево plugin-GUI для повторной Release-сборки, и есть ли на целевой машине совпадающие исходники plugin-GUI нужной версии для Пути B?
- Не занят ли локальный UDP-порт 47000 другим слушателем на целевой машине (плагин биндится строго на 127.0.0.1:47000)?


==========================================================================================
# DOMAIN 5: Скрипт проверки окружения (заготовка авто-инсталлятора)
==========================================================================================

# Проверка окружения целевого ПК: `check_environment.py`

## Зачем это нужно

Перенос системы DLC-Live (реалтайм-инференс позы крысы и отправка в Open Ephys) состоит из множества разнородных частей: Python 3.10 c точными версиями пакетов, CUDA-сборка PyTorch под Blackwell sm_120, драйвер NVIDIA, SDK камеры Daheng Galaxy с особым (не-pip) пакетом `gxipy`, экспортированная модель `.pt`, конфиги камеры и рабочий каталог скриптов. Любая из этих частей может быть не установлена или установлена «не той версии», и тогда `single_rt_dlc_live_bridge.py` молча упадёт с непонятной ошибкой.

Скрипт **`C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\check_environment.py`** делает **пре-флайт-проверку**: зондирует машину и печатает по одной строке-вердикту на каждое требование — `[  OK  ]` / `[MISSING]` / `[MISMATCH]` / `[ WARN ]`, затем итог `N OK / M MISSING / K MISMATCH` и пронумерованный TODO-список с готовыми командами. Скрипт намеренно **зависит только от stdlib**: все сторонние импорты (`torch`, `cv2`, `gxipy`, `importlib.metadata`) обёрнуты в `try/except`, поэтому он запускается даже на полупустой машине и сам подскажет, чего не хватает.

## Что именно проверяется (8 блоков)

1. **Python == 3.10.x** — сверяется `sys.version_info`. На исходной машине это `3.10.11`. Дополнительно WARN, если запущен НЕ из venv `C:\dlc_live_env` (иначе проверка версий пакетов покажет системный Python, а не боевой).
2. **Критичные pip-пакеты vs пины** — читается `requirements.lock.txt` (источник истины, 57 пинов) и сравнивается «установлено vs запинено» через `importlib.metadata` (без вызова pip). Проверяются: `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, `torchaudio==2.10.0+cu128`, `deeplabcut-live==1.1.0`, `numpy==1.26.4`, `opencv-python==4.11.0.86`, `scipy==1.15.3`, `pandas==2.3.3`, `pillow==12.1.1`, `PyYAML==6.0.3`, `ruamel.yaml==0.19.1`, `tables==3.10.1`, `timm==1.0.26`, `huggingface_hub==1.9.0`, `dlclibrary==0.0.11`, `networkx==3.4.2`, `safetensors==0.7.0`. Для каждого пакета не только сверяется версия, но и делается реальный `import` модуля — чтобы поймать «версия есть, а import падает» (битые native-DLL).
3. **torch + CUDA + GPU** — `torch.version.cuda` (должно быть `12.8`, рантайм вшит в колесо — отдельный CUDA Toolkit НЕ нужен), `torch.cuda.is_available()`, имя GPU, и `get_device_capability(0) == (12, 0)` = sm_120 Blackwell. В конце делается **реальная операция на GPU** (`torch.ones(8, device="cuda")*2`) — это ловит ситуацию «драйвер старый, арку sm_120 не понимает».
4. **Драйвер NVIDIA** — вызывается `nvidia-smi --query-gpu=name,driver_version`. На источнике: `592.01`.
5. **Daheng Galaxy SDK + gxipy** — наличие `C:\Program Files\Daheng Imaging\GalaxySDK` (или `DLC_LIVE_GALAXY_SDK_ROOT`), наличие папки `Samples\Python SDK`, затем **`gxipy` добавляется в `sys.path` и импортируется** (это НЕ pip-пакет!), и проверяется системная переменная `GENICAM_GENTL64_PATH`. Перед импортом скрипт добавляет `GenTL\Win64` через `os.add_dll_directory`, повторяя логику `rt_dlc_live._prepare_sdk_environment`.
6. **Модель** — наличие экспортированного снапшота `.pt` по пути из `DLC_LIVE_MODEL_PATH` или дефолтному `C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch\...\snapshot-best-380.pt` (на источнике ~93 МБ).
7. **Каталог конфигов камеры** — `C:\config_daheng`, считается число `.txt` (на источнике 25 файлов).
8. **Боевые скрипты в рабочем каталоге** — 13 файлов, включая `single_rt_dlc_live_bridge.py`, `dual_rt_dlc_live.py`, `config_dual_rt_dlc_live.py`, `config_rt_dlc_live.py`, `live_profiles.py`, `rt_dlc_live.py` и `optimization\*.py`. Здесь же напоминание: **`git clone` даёт СТАРОЕ/сломанное дерево** — переносить надо актуальное рабочее дерево целиком.

## Как запускать

Рекомендуемый способ — из боевого venv (чтобы проверки пакетов были осмысленными):

```
C:\dlc_live_env\Scripts\python.exe C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\check_environment.py
```

Либо через тонкую обёртку PowerShell, которая сама ищет интерпретатор (сначала venv `C:\dlc_live_env\Scripts\python.exe`, затем `$env:DLC_LIVE_PYTHON`, затем `python`/`py` из PATH — поэтому работает даже до создания venv):

```
powershell -ExecutionPolicy Bypass -File C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\check_environment.ps1
```

**Код возврата:** `0`, если нет ни одного `MISSING`/`MISMATCH`; иначе `1`. Это можно использовать в авто-инсталляторе (заготовка которого — этот скрипт): «прогнать проверку → если не 0, поставить из TODO → прогнать ещё раз».

## Фактический вывод на исходной (рабочей) машине

Скрипт прогнан на источнике через `C:\dlc_live_env\Scripts\python.exe`. Итог:

```
==========================================================================
 1. Python interpreter
==========================================================================
[  OK  ]   Python == 3.10.x
           -> running 3.10.11  (C:\dlc_live_env\Scripts\python.exe)

==========================================================================
 2. Critical pip packages (installed vs pinned)
==========================================================================
[  OK  ]   requirements.lock.txt readable  -> 57 pins loaded
[  OK  ]   torch==2.10.0+cu128            -> installed 2.10.0+cu128
[  OK  ]   torchvision==0.25.0+cu128      -> installed 0.25.0+cu128
[  OK  ]   deeplabcut-live==1.1.0         -> installed 1.1.0
[  OK  ]   numpy==1.26.4 / opencv 4.11.0.86 / scipy 1.15.3 / pandas 2.3.3 ... все OK

==========================================================================
 3. torch CUDA runtime & GPU
==========================================================================
[  OK  ]   import torch                   -> torch 2.10.0+cu128
[  OK  ]   torch bundled CUDA runtime     -> CUDA 12.8 (wheel-bundled, no Toolkit needed)
[  OK  ]   torch.cuda.is_available()      -> True
[  OK  ]   GPU device                     -> NVIDIA GeForce RTX 5070 Laptop GPU
[  OK  ]   compute capability == (12, 0)  -> sm_120 (Blackwell)
[  OK  ]   CUDA tensor op                 -> GPU allocation + compute succeeded

==========================================================================
 4. NVIDIA driver (nvidia-smi)
==========================================================================
[  OK  ]   nvidia-smi                     -> NVIDIA GeForce RTX 5070 Laptop GPU, 592.01

==========================================================================
 5. Daheng Galaxy SDK + gxipy
==========================================================================
[  OK  ]   Galaxy SDK root                -> C:\Program Files\Daheng Imaging\GalaxySDK
[  OK  ]   Galaxy Python SDK folder       -> ...\Samples\Python SDK
[  OK  ]   import gxipy                   -> imported from Samples/Python SDK
[  OK  ]   GENICAM_GENTL64_PATH env var   -> ...\GenTL\Win64

==========================================================================
 6/7/8. model / camera configs / deployed scripts -- все OK
==========================================================================
[  OK  ]   model snapshot present  -> ...snapshot-best-380.pt  (93 MB)
[  OK  ]   camera config dir       -> C:\config_daheng  (25 .txt)
[  OK  ]   all deployed scripts present -> 13 files

==========================================================================
 SUMMARY
==========================================================================
  46 OK / 0 MISSING / 0 MISMATCH / 0 WARN
  No action needed -- environment looks ready for the live bridge.
RESULT: READY (no MISSING / MISMATCH).      [exit code 0]
```

Обёртка `check_environment.ps1` тоже проверена: находит `C:\dlc_live_env\Scripts\python.exe`, печатает полный отчёт (115 строк) и завершается с кодом `0`.

## Что ожидать на ЧИСТОМ целевом ПК (и подводные камни)

- **GPU.** На целевом ПК стоит **десктопный RTX 5070** (тоже Blackwell sm_120) — блок 3 должен дать `compute capability == (12, 0)`. Имя GPU будет другим (`NVIDIA GeForce RTX 5070`, без «Laptop»), это нормально — проверяется именно capability. Если capability ≠ (12,0) → скрипт выдаст WARN, а не FAIL (модель может пойти, но пины тюнились под Blackwell).
- **Драйвер.** Самый вероятный реальный блокер: на свежей машине драйвер старый и `torch.cuda.is_available()` вернёт `False`, либо упадёт «CUDA tensor op» с ошибкой «no kernel image for sm_120». Нужен **Blackwell-совместимый драйвер** (на источнике 592.01, поддерживает CUDA 13.1; для рантайма torch достаточно драйвера под CUDA 12.8+). Поставить драйвер → перезагрузить → проверить `nvidia-smi`.
- **CUDA Toolkit ставить НЕ нужно.** Колесо `torch==2.10.0+cu128` несёт рантайм CUDA 12.8 + cuDNN внутри. На источнике параллельно стоит Toolkit v13.0 (`CUDA_PATH=...v13.0`), но torch его НЕ использует. На целевом ПК Toolkit не требуется для боевого инференса.
- **`gxipy` не ставится через pip.** Это ловушка: его НЕТ в `requirements.lock.txt`, потому что он подгружается из `<GalaxySDK>\Samples\Python SDK` (скрипт добавляет путь в `sys.path` и грузит DLL из `GenTL\Win64`). Если поставить только pip-пакеты — `import gxipy` упадёт. Лечится установкой **самого SDK Daheng Galaxy** (на источнике 1.18.2208.9301) с компонентом «Samples/Python SDK».
- **Системные переменные SDK.** Инсталлятор Galaxy выставляет `GENICAM_GENTL64_PATH`, `GALAXY_GENICAM_ROOT` и дополняет PATH. Если скрипт показал `[MISSING] GENICAM_GENTL64_PATH` — после установки SDK **нужна перезагрузка**, чтобы системные env-переменные подхватились новыми процессами.
- **git stale.** Блок 8 ловит главную опасность переноса кода: задеплоенные Gen3-файлы НЕ закоммичены, а в трекаемых — большие незакоммиченные диффы. Поэтому переносить надо **рабочее дерево целиком** (копией каталога), а не `git clone`. Если блок 8 показывает MISSING — значит притащили старую версию из git.
- **Жёстко зашитые пути.** Скрипт сверяет дефолтные абсолютные пути источника (`C:\dlc\project\...`, `C:\config_daheng`, `C:\dlc\DLC_OBS_Spinal_cord_stimulation`, `C:\dlc_live_env`). Если на целевом ПК структура другая — либо воспроизвести те же пути, либо задать env-override (`DLC_LIVE_MODEL_PATH`, `DLC_LIVE_GALAXY_SDK_ROOT`, `DLC_LIVE_GALAXY_CONFIG_PATH`); тогда скрипт проверит уже новые пути. (Сама `live_profiles.py:10-11` хардкодит `WORK_DIR` и `PYTHON_EXE` — их env-override не покрывает, при смене путей надо править файл.)
- **Open Ephys / C++ плагин — вне зоны этого скрипта.** Проверка покрывает только Python-сторону живого моста. Плагин `DualDLCLiveBridge` (приём UDP-позы на 127.0.0.1:47000) и его сборка через Visual Studio + CMake проверяются отдельно. Сама Python-сторона сборки не требует (только готовые pip-колёса; `cudagraphs`-бэкенд `torch.compile` требует лишь CUDA, не компилятор C++).

## Боевая команда после успешной проверки

Когда `check_environment.py` показал `RESULT: READY`:

```
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best
```

(добавить `--no-display`, если оконный вывод не нужен).


### CHECKLIST
- [ ] На целевом ПК скопировать рабочее дерево C:\dlc\DLC_OBS_Spinal_cord_stimulation целиком (НЕ git clone — git устарел), вместе с setup\check_environment.py, check_environment.ps1 и requirements.lock.txt
- [ ] Установить Python 3.10.x (на источнике 3.10.11) и создать venv C:\dlc_live_env
- [ ] Установить Blackwell-совместимый драйвер NVIDIA, перезагрузить, проверить вывод nvidia-smi
- [ ] Установить Daheng Galaxy SDK (1.18.2208.9301 или совместимый) с компонентом Samples/Python SDK (даёт gxipy), перезагрузить для применения системных env-переменных
- [ ] Скопировать экспортированную модель (snapshot-best-380.pt и её папку) в C:\dlc\project\...\exported-models-pytorch\ или задать DLC_LIVE_MODEL_PATH
- [ ] Скопировать .txt-конфиги камеры в C:\config_daheng
- [ ] Установить пакеты из requirements.lock.txt в venv (torch/torchvision/torchaudio брать с --index-url https://download.pytorch.org/whl/cu128)
- [ ] Запустить: C:\dlc_live_env\Scripts\python.exe setup\check_environment.py (или check_environment.ps1)
- [ ] Прочитать итог N OK / M MISSING / K MISMATCH; выполнить все пункты TODO; повторять до RESULT: READY (exit code 0)
- [ ] Запустить боевой мост: cd C:\dlc\DLC_OBS_Spinal_cord_stimulation ; C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best

### ARTIFACTS
- C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\check_environment.py — stdlib-only пре-флайт-проверка окружения (8 блоков: Python, pip-пины vs lock, torch/CUDA/GPU sm_120, nvidia-smi, Galaxy SDK+gxipy+GenTL env, модель .pt, конфиги камеры, боевые скрипты); печатает PASS/FAIL/TODO, exit 0/1
- C:\dlc\DLC_OBS_Spinal_cord_stimulation\setup\check_environment.ps1 — тонкая PowerShell-обёртка: находит python (venv C:\dlc_live_env → $env:DLC_LIVE_PYTHON → python/py из PATH) и запускает check_environment.py, пробрасывая код возврата

### OPEN QUESTIONS
- Установлен ли на целевом ПК Blackwell-совместимый драйвер NVIDIA (под CUDA 12.8+)? Это самый вероятный реальный блокер — проверяется блоками 3 и 4 скрипта.
- Будут ли на целевом ПК воспроизведены те же абсолютные пути (C:\dlc\project\..., C:\config_daheng, C:\dlc_live_env, C:\dlc\DLC_OBS_Spinal_cord_stimulation), или нужно использовать env-override? Учесть, что live_profiles.py:10-11 хардкодит WORK_DIR и PYTHON_EXE — env-override их не покрывает.
- Какая версия Daheng Galaxy SDK будет установлена на целевом ПК и точно ли в неё включён компонент Samples/Python SDK (без него gxipy не импортируется)?
- Десктопный RTX 5070 действительно рапортует compute capability (12,0)? Подтвердить блоком 3 на целевой машине (ожидается sm_120, как у Laptop-версии).
- Нужен ли на целевом ПК ещё и офлайн-путь (train/analyze через run_dlc.py с полным deeplabcut и проектом 2.3 ГБ), или достаточно lean live-инференса? Этот скрипт проверяет только живой мост.
