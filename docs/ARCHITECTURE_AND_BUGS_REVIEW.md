# Архитектура, уязвимости и анализ точности

_Дата: 2026-06-10. Разобраны обе половины системы: Python-раннтайм (`C:\dlc\DLC_OBS_Spinal_cord_stimulation`) и C++ плагин Open Ephys `DualDLCLiveBridge` (репозиторий github.com/nikaabigail/Dual_DLC_live_plugin, ветка main). Анализ read-only, в рабочий пайплайн ничего не переносилось, git не трогался._

---

## 1. Полная архитектура (end-to-end)

```
[Galaxy камера FDE22070174, 1920x220 @100Hz, выход RGB, NEWEST_ONLY low-latency]
        │  (reader: drain старых кадров, берётся самый свежий)
        ▼
[Python: single_rt_dlc_live_bridge.py  (рабочий режим single-best)]
   rt_dlc_live.GalaxyCameraSource.read()  → FramePacket(frame=RGB, capture_ts)
        ▼  dual.run_raw_inference():
   dlc_live.convert2rgb = CONVERT_TO_RGB(=False)   # НЕ переворачивает каналы
   dlc_live.process_frame(frame)                   # crop=None, resize=1.0
   dlc_live.runner.get_pose(...)                   # ResNet-50, FP32+TF32, CUDA
   postprocess_pose_without_processor()            # вернуть координаты (resize/crop undo)
   pose_to_compact_array()                         # (6,3) float32, NaN для невидимых
        ▼  OpenEphysBridge.send():  _build_binary_pose_payload()
   DDLP/v1 binary, 252 байта: header + left-блок(6 точек) + right-блок(6 точек)
   В single-режиме: left = реальные точки, right = 6×NaN
        ▼  UDP 127.0.0.1:47000  (non-blocking, fire-and-forget)
        ▼
[C++ плагин Open Ephys "Dual DLCLive Bridge"]
   run() UDP-поток → applyDatagram() → applyBinaryPosePacket()
     • парсинг с bounds-check, version==1, point_count==6
     • per-point: valid = isfinite(x)&&isfinite(y)&&isfinite(l)   ← NaN → invalid
   evaluateSidePosePoints() для каждой стороны-камеры:
     • filterPoint(): pcutoff(conf_use=0.20) → despike(150px) → median(3) → [hold off]
     • scoreTriplet(left_leg) vs scoreTriplet(right_leg) → берётся лучший
     • hasTriplet = hip/ankle/toes valid && likelihood>=conf_draw(0.15)
     • safeAngleDeg(hip,ankle,toes): угол В ВЕРШИНЕ ANKLE
   TTL word (8 бит):
     bit0=left триплет valid, bit1=right valid,
     bit2=left angle trigger (angle<=55° & enabled), bit3=right angle trigger
   shouldEmitAngleTrigger(): refractory gating (refractory_ms=0 → без рефрактерности)
   queueTtlWord(): кладёт в очередь ТОЛЬКО при изменении слова
        ▼  process() audio-callback → emitPendingTtlState() → setTTLState()
[event channel "Dual DLCLive TTL"] → downstream Open Ephys stim/output processor → стимуляция
```

Ключевой принцип: **Python = только зрение + упаковка точек; весь «мозг» стимуляции (фильтр, триплет, угол, порог, refractory, TTL) — в C++ плагине.** Сделано ради latency.

Разделение ответственности подтверждено в коде: в fast-режиме (`DUAL_FAST_POSE_ONLY=True`, `PACKET_MODE="pose"`) Python ставит `runtime.processor=None` и шлёт сырьё; плагин делает фильтрацию (та же логика, что Python-овский `OnlinePoseProcessor`).

---

## 2. Ответ про RGB: конвертация снята или нет?

**Короткий ответ: в рабочем режиме (single-best) внутренняя конвертация DLCLive ВЫКЛЮЧЕНА (`convert2rgb=False`), потому что камера настроена отдавать RGB напрямую (`GALAXY_OUTPUT_COLOR="rgb"`). Модель получает RGB — это правильно. Цвет НЕ является причиной падения точности.**

Полная трассировка по коду:

| Путь | GALAXY_OUTPUT_COLOR | CONVERT_TO_RGB | что отдаёт камера | что делает DLCLive | модель получает |
| --- | --- | --- | --- | --- | --- |
| **single-best / dual (рабочий)** | `rgb` | `False` | RGB (без свопа в `_raw_image_to_bgr`) | ничего (`convert2rgb=False`) | **RGB ✓** |
| `single-rgb-on` (fallback) | `bgr` | `True` | BGR | своп BGR→RGB | RGB ✓ |
| base `config_rt_dlc_live.py` (если запускать `rt_dlc_live.py` напрямую) | не задан → `bgr` | `True` | BGR | своп BGR→RGB | RGB ✓ |

Все три пути дают модели RGB. Раньше был «двойной» путь RGB→BGR→RGB; его убрали ради экономии CPU — камера теперь сразу отдаёт RGB, а DLCLive не переворачивает. Итог по цвету одинаковый.

**Где можно случайно сломать цвет (единственная опасная комбинация):** `GALAXY_OUTPUT_COLOR="rgb"` + `CONVERT_TO_RGB=True`. Тогда камера даёт RGB, DLCLive свопит в BGR → модель видит BGR → точность резко падает. Ни один профиль так не делает, НО:

- `run_raw_inference` (dual_rt_dlc_live.py:835) и batch-путь (:1022) каждый раз делают `dlc_live.convert2rgb = bool(getattr(config, "CONVERT_TO_RGB", True))` — **дефолт `True`**. Если конфиг/профиль забудет выставить `CONVERT_TO_RGB=False` при rgb-камере, цвет молча сломается.
- В базовом `config_rt_dlc_live.py:73` стоит `CONVERT_TO_RGB=True`, а `GALAXY_OUTPUT_COLOR` не задан (дефолт `bgr`). Это согласовано. Но если кто-то добавит `GALAXY_OUTPUT_COLOR="rgb"` в базовый конфиг, не тронув `CONVERT_TO_RGB`, получит сломанный цвет.

**Как проверить за 5 секунд:** в `single_rt_dlc_live_bridge_debug.log` смотри строки старта — должно быть `convert2rgb=False`:
```
Single bridge started. profile=single-best ... convert2rgb=False
DLC_MODEL_DEVICE_AFTER_INIT=cuda:0 ... precision=FP32 convert2rgb=False
```
В сегодняшнем логе именно так — значит цвет корректный.

---

## 3. Почему точность падает в real-time (ранжировано)

Цвет исключён (раздел 2). Реальные кандидаты, от вероятного к менее:

1. **Live-препроцессинг ≠ offline `analyze_videos` (самый вероятный и проверяемый).** В live `RESIZE=1.0`, `CROPPING=None`, `DYNAMIC_CROPPING=off` — модель кормится сырым 1920×220. Если offline-анализ DLC применял иной масштаб (в `pose_cfg.yaml` обычно есть `global_scale`, типично 0.8), сеть на инференсе видит другое эффективное разрешение, и координаты смещаются. Это ровно «Next step #2» из `HISTORY_DLC_LIVE.md`. **Тест:** сохранить сырой вывод DLCLive по записанному видео и сравнить с `run_dlc.py analyze` на тех же кадрах (по координатам и likelihood). Это даст точную величину расхождения.

2. **TF32 / FP16 — снижение численной точности.** В `single-best` включён `DUAL_TORCH_ALLOW_TF32=True` → matmul округляется до ~10 бит мантиссы → сдвиги координат на доли-единицы пикселей и иногда ниже likelihood. В `single-fp16`/`dual-fp16` хуже — логи сами отмечают «weaker likelihoods». **Эталон точности — профиль `single-strict` (FP32, TF32 off).** Прогони A/B: `single-strict` vs `single-best` vs `single-fp16` по `raw_visible` и по координатам на одной сцене.

3. **Смаз от экспозиции.** Левая камера: exposure=8000µs=8 мс при 100 Гц. Быстрые toes за 8 мс смазываются → низкий/скачущий likelihood (в логах toes 0.37–0.92, ankle/hip стабильнее). Это физика, не код: уменьшить экспозицию + добавить света/gain, чтобы убрать смаз на быстрых точках.

4. **Реальный infer-rate ≈25–37 Гц, а не 100.** Камера 100 Гц, но модель успевает ~25–37 инференсов/с (логи: `live_fps≈35`, `result_hz` до 37; dual ещё ниже). Значит между инференсами конечность проходит больше пути → despike(150px) и median(3) работают по «инференс-кадрам», а не по 10-мс кадрам, и сглаживание/отбраковка влияют сильнее, чем кажется. Если оптимизировать скорость (см. раздел 5), точность триплета вырастет сама.

5. **Что видишь ≠ что триггерит.** В single bridge OpenCV-оверлей рисует СЫРЫЕ точки (в fast-режиме Python не фильтрует), а TTL-решение принимает плагин по ОТФИЛЬТРОВАННЫМ (median+despike) точкам. Поэтому визуально точки могут «дрожать», а триггер идёт по сглаженным — и наоборот. Для честной оценки точности смотри не только окно, но и `L`/`R` angle и `ttl` в статус-строке плагина.

6. **Dual batch-путь численно отличается от single.** `run_batch_inference` вручную пересобирает препроцессинг (`runner.pose_transform` + `model()` + `get_predictions`), минуя `runner.get_pose`. Если в `get_pose` есть нормализация, которой нет в ручном пути, dual будет чуть менее точен, чем single. Рабочий режим — single, так что сейчас это не бьёт, но при возврате к dual проверь.

---

## 4. Баги и уязвимости

### 4.1. SAFETY-critical (замкнутый контур стимуляции)

- **[C++] Нет watchdog по возрасту пакета → TTL может залипнуть.** Плагин считает `getLastPacketAgeMs()` и показывает `age` в UI, но **нигде не использует его, чтобы погасить TTL**. `queueTtlWord` эмитит только при ИЗМЕНЕНИИ слова; `closeSocket` гасит линии лишь при выключении/смене порта. Если Python зависнет/упадёт в момент, когда line0 (valid) или line2 (angle trigger) = HIGH, **TTL останется HIGH бесконечно** → стимуляционный гейт залипнет. _Фикс:_ в `process()` форсировать все линии в 0, если `getLastPacketAgeMs() > N` мс (например 50–100 мс).

- **[Python] Нет staleness-gate на устаревший кадр.** `inference_loop` спаривает «последние» кадры по seq; `host_dt_ms` считается, но не проверяется. Зависшая камера → старая поза уходит в плагин со свежим `host_time`/`pair_index`. _Фикс:_ отклонять/занулять сторону, чей `capture_ts` старше нескольких периодов.

- **[Python] `single_rt_dlc_live_bridge.py` не валидирует конфиг.** `dual.main()` зовёт `validate_dual_config()` (проверяет порт, режим, модель и **совпадение `DUAL_USE_POINTS` с фиксированным binary-порядком**). Single bridge не зовёт ничего. Если порядок точек разойдётся, single молча отправит переставленные точки → плагин посчитает угол по НЕ тем точкам. _Фикс:_ вызвать `dual.validate_dual_config()` в начале single bridge.

- **[оба] Тихие потери UDP.** Сокет non-blocking, при переполнении буфера `sendto` бросает `BlockingIOError`, его ловят и логируют warning — пакет теряется. Счётчика потерь нет; если плагин не слушает, Python шлёт «в пустоту» без ошибки (ACK по умолчанию off). _Фикс:_ счётчик потерь + периодический warning; опционально heartbeat.

### 4.2. Логические/корректность

- **[Python] Сломанная не-fast ветка `run_inference` (dual_rt_dlc_live.py:931–951).** Две ошибки в одной ветке (`runtime.processor is not None`, т.е. при `DUAL_FAST_POSE_ONLY=False`):
  - стр. 938: `ure_ts=float(packet.capture_ts)` — опечатка, должно быть `capture_ts=` → `TypeError` (неизвестный аргумент) при `init_inference`;
  - стр. 941: `print_model_device_after_init(...)` не определена (есть `log_model_device_after_init`) → `NameError`.
  Обе уронят inference-поток на первом кадре. В fast-продакшене ветка мёртвая, но это «мина» при попытке вернуть Python-постобработку.

- **[C++] Семантика angle trigger — level, не edge.** При `refractory_ms=0` (дефолт) line2 = `(angle<=55°)` держится HIGH на протяжении всех пакетов, где условие истинно (`shouldEmitAngleTrigger` возвращает true и обновляет время каждый раз). Refractory лишь разносит rising edges. Нужно решить, чего ждёт downstream — фронт или уровень. Плюс направление: триггер при `angle <= threshold` (малый/согнутый угол) — подтвердить, что это нужная фаза шага.

- **[C++] Авто-выбор ноги внутри камеры.** `evaluateSidePosePoints` для КАЖДОГО пакета-камеры считает score обоих триплетов (left-нога и right-нога) и берёт лучший. В single-left обычно выигрывает левая нога. Но если точки дальней (правой) ноги случайно прыгнут выше порога, плагин может выбрать не ту ногу → не тот угол. line0 при этом значит «левая КАМЕРА дала какой-то валидный триплет», а не «левая нога».

- **[C++] `angle_trigger_enabled=false` по умолчанию.** Линии 2/3 (собственно стимуляционный триггер) не сработают из коробки — нужно явно включить флаг. Операционная ловушка (есть и в handoff).

### 4.3. Тайминг (для closed-loop важно)

- **[C++] Разрешение TTL = размер audio-блока.** Переходы эмитятся в `process()` и раскидываются `emitPendingTtlState` по сэмплам блока по ИНДЕКСУ (равномерно), а не по реальному времени прихода. Если за один блок пришло несколько пакетов, их фронты расставляются равномерно, а не по `host_time`. Латентность/джиттер ≈ один блок обработки. Это свойство Open Ephys, но его надо учитывать в бюджете задержки замкнутого контура.

- **[C++] `medianValue` сортирует deque на каждый вызов** — мелочь при window=3, но это O(n log n) на точку на пакет.

### 4.4. Что СДЕЛАНО правильно (чтобы не трогать)

- Бинарный парсер плагина — с проверкой границ (`readBinaryValue` сверяет offset+size ≤ numBytes), проверяет `version` и `point_count`. Переполнения нет.
- **NaN-обработка корректна и безопасна:** `valid = isfinite(x)&&isfinite(y)&&isfinite(likelihood)`. NaN-заглушка правой стороны → правый триплет невалиден → line1/line3 молчат. Это снимает мой прежний вопрос про sentinel — здесь именно `isfinite`.
- Потокобезопасность плагина разумна: `socketLock`/`poseStateLock`/`pendingLock` + atomic для линий и счётчиков.
- Фильтр плагина (pcutoff/despike/median/hold) численно повторяет Python `OnlinePoseProcessor` — поведение согласовано.
- Тест `send_dual_dlc_bridge_test.py` повторяет формат байт-в-байт (но см. ниже).

### 4.5. Тест-харнес — расхождения с боевым путём

- Тест собирает байты независимо от боевого упаковщика и **никогда не шлёт NaN** — то есть safety-критичный путь «пропущенная точка → NaN → invalid» end-to-end не проверяется. JSON-режим теста использует `allow_nan=True`, а боевой `_build_payload` — `allow_nan=False`. Стоит: (а) импортировать боевые упаковщики в тест, (б) добавить кейс с NaN-точками и убедиться, что плагин держит line1/line3 в 0.

---

## 5. Рекомендации под оптимизацию (скорость + точность)

Безопасность (сделать до боевых сессий стимуляции):
1. **[C++]** Watchdog по `age`: гасить все TTL-линии в `process()`, если последний пакет старше ~50–100 мс.
2. **[Python]** Staleness-gate на сторону по `capture_ts`; счётчик потерянных UDP-отправок.
3. **[Python]** Вызвать `validate_dual_config()` в `single_rt_dlc_live_bridge.py`.

Точность:
4. **Сравнить live DLCLive vs offline `analyze`** на одном видео (координаты + likelihood) — это вскроет, есть ли расхождение препроцессинга/масштаба. Проверить `global_scale`/входной размер в `pose_cfg.yaml` экспортированной модели.
5. **A/B по точности:** `single-strict` (эталон FP32) vs `single-best` (TF32) vs `single-fp16`. Если strict заметно точнее — для боевой стимуляции брать strict, а скорость добирать иначе (см. ниже).
6. **Камера:** снизить экспозицию (+ свет/gain), чтобы убрать смаз на toes; это поднимет likelihood быстрых точек напрямую.
7. **Правую камеру** физически навести на дорожку (offset-сканы в `debug_snapshots/` — попытка подобрать OffsetY); до этого dual не имеет смысла.

Скорость (поднять infer-rate с ~35 к ~100, тогда фильтры/триплет станут точнее):
8. Включить **batch** только когда обе камеры реально нужны; для single — путь уже минимальный.
9. Профилировщик стадий уже есть (`stage_profile`): сейчас узкое место — `inference` (~18–21 мс single, ~30–44 мс dual), `camera/read`~1–10 мс, `pack/send`~0.15 мс, `display`~4–6 мс. Для боевого — `--no-display` (убирает ~5–6 мс/кадр). Дальше резать только `inference`: меньший вход/резайз, или экспортировать модель под меньший вход, или TensorRT/FP16 (с обязательной проверкой точности по п.5).
10. **Опечатка/мина** в не-fast ветке (4.2) — починить или удалить ветку, чтобы случайно не словить креш при экспериментах с Python-фильтром.

---

## 6. Сводка источников истины

- Рабочий рантайм: `single-best` → `single_rt_dlc_live_bridge.py --profile single-best` (одна левая камера, RGB-вход, FP32+TF32).
- Цвет: модель получает RGB; `convert2rgb=False`; конвертация в DLCLive снята (камера сама даёт RGB). Не причина падения точности.
- Плагин: NaN-safe, фильтр зеркалит Python, угол в ankle, триггер при `angle<=55°` и только при `angle_trigger_enabled=true`, `refractory_ms=0` (уровень, без рефрактерности).
- Главные риски: залипание TTL при заморозке (нет watchdog ни в Python, ни в C++), отсутствие staleness-gate, single bridge без валидации, сломанная не-fast ветка, тихие потери UDP.
- Точность в real-time: проверять препроцессинг live-vs-offline, TF32/FP16, экспозицию/смаз, реальный infer-rate.

---

## 7. Судьба MVP-наработок и тюнинг фильтров

### 7.1. Что выжило, что удалено (проверено grep'ом по production-файлам)

| Компонент старого MVP (`rt_dlc_obs.py`/`config_rt_dlc.py`) | В боевой DLCLive-линии | Комментарий |
|---|---|---|
| Display/sync буфер (`DISPLAY_BUFFER_MS`, `MAX_FRAME_BUFFER`, `MAX_PRED_BUFFER`, матчинг по `frame_id`) | **Удалён** (0 вхождений) | Stage-2 идея, ухудшала latency. Для closed-loop правильно, что убрали. |
| Admission-гейты (`INFER_EVERY_N_FRAMES`, `skip_motion/duplicate/busy/fps`, `SUPPRESS_LOW_MOTION`, `INFER_QUEUE`) | **Удалён** (0 вхождений) | Заменено на «дроп старых кадров + инференс на каждом кадре». |
| Идея «не копить кадры на GPU» | **Выжила в упрощённом виде** | `NEWEST_ONLY` drain на камере + latest-only слот. |
| Онлайн-фильтр (pcutoff/despike/hold/median) | **Выжил, активен** | Переписан в `OnlinePoseProcessor` (Python, спит в fast) и в C++ `filterPoint` (работает). |

Production-файлы (`rt_dlc_live.py`, `dual_rt_dlc_live.py`, `single_rt_dlc_live_bridge.py`, оба live-конфига) **не импортируют** легаси-модули. Старая буферизация физически не выполняется — ломать ничего не может.

Концептуально: буфер не «даёт модели время на точность» (точность на кадр фиксирована сетью), он добавляет только задержку. Поэтому удаление правильное; для closed-loop нужен свежайший кадр.

### 7.2. Где сейчас живут фильтры (карта)

Поток в C++ плагине (`DualDLCLiveBridge.cpp::filterPoint`, для каждой точки на кадр):

```
raw point (x,y,likelihood)  ← из UDP; Python в fast-режиме НЕ фильтрует
   │
   ├─ pcutoff:  if enable_pcutoff && likelihood < conf_thresh_use(0.20) → точка невалидна
   │
   ├─ despike:  if enable_despike && jump>despike_threshold_px(150) && gap<=reset_gap(15) → невалидна
   │            (jump = расстояние от прошлой ПРИНЯТОЙ точки)
   │
   ├─ median:   координата = median(последние median_window(3) принятых)  ← ВНОСИТ ЛАГ
   │
   └─ hold:     if enable_hold(off) && точка пропала ≤ max_hold_frames(20) → держать прошлую
                   → далее: триплет → угол в ankle → порог 55° → refractory → TTL
```

Дубликат фильтра в Python (`OnlinePoseProcessor`, `rt_dlc_live.py:807`) в fast-режиме отключён (`processor=None`). **Не включать `DUAL_FAST_POSE_ONLY=False`** — иначе фильтрация удвоится (двойной лаг). Все ручки — в UI плагина (editor), меняются вживую без пересборки.

### 7.3. Тюнинг median/despike под целевую частоту

Оба эффекта зависят от РЕАЛЬНОГО infer-rate (~35 Гц сейчас), а не от 100 Гц камеры.

**`median_window` → лаг угла.** Медиана из N даёт лаг ≈ (N−1)/2 сэмплов:

| infer-rate | median=1 | median=3 | median=5 |
|---|---|---|---|
| ~35 Гц (сейчас) | 0 мс | **~28 мс** | ~57 мс |
| 60 Гц | 0 мс | ~17 мс | ~33 мс |
| 100 Гц | 0 мс | ~10 мс | ~20 мс |

Лаг угла = ошибка тайминга фазы шага для триггера. Вывод: **при текущих ~35 Гц снизить `median_window` с 3 до 1**; вернуть 3 только после выхода на 100 Гц (~10 мс терпимо).

**`despike_threshold_px` → отбраковка быстрых точек.** Порог = макс. допустимый сдвиг между инференсами, должен покрывать реальную скорость toes за Δt=1/rate. При росте частоты тот же 150px становится «с запасом» (реже режет — безопасно):

| infer-rate | «с запасом» (меньше дропов) | «активная отбраковка» |
|---|---|---|
| ~35 Гц | 150px (как сейчас) | ~120px |
| 60 Гц | 150px (теперь щедро) | ~85px |
| 100 Гц | 150px (очень щедро) | ~50px |

Точные числа лучше из данных: залогировать per-inference смещение каждой точки, порог = ~99-й перцентиль × 1.5. `despike_reset_gap_frames=15` тоже во «фреймах»: при росте частоты поднять до ~25–40, чтобы окно переразбора осталось ~300–400 мс.

**Приём для стимуляции:** ответственность за дребезг у порога переносить с `median` (лагает позицию) на `refractory_ms` (НЕ лагает позицию, лишь разносит фронты). То есть `median_window=1` + `refractory_ms=30–50`.

### 7.4. Рекомендованные стартовые значения (UI плагина)

| Параметр | Сейчас | При ~35 Гц | Цель 100 Гц |
|---|---|---|---|
| `enable_pcutoff` / `conf_thresh_use` | on / 0.20 | оставить | оставить |
| `conf_thresh_draw` | 0.15 | 0.15 (→0.30 если ложные триплеты) | то же |
| `enable_despike` / `despike_threshold_px` | on / 150 | 150 (не резать toes) | 85→50 если нужна отбраковка |
| `despike_reset_gap_frames` | 15 | 15 | 25–40 |
| `median_window` | 3 | **1** (убрать ~28 мс лага) | 3 (≤10 мс) |
| `enable_hold` / `max_hold_frames` | off / 20 | off | off |
| `refractory_ms` | 0 | **30–50** (де-дребезг вместо median) | 30–50 |
| `angle_trigger_enabled` | false | **true** (иначе нет стимуляции) | true |

Все значения — стартовые гипотезы; финально подтверждать A/B (likelihood + тайминг триггера) тем же способом, что и FP16-сравнение.

---

## 8. Задачи на оптимизацию (бриф для агента)

> Агент, читай это как рабочее задание. Цель — поднять скорость и точность real-time pose→TTL контура стимуляции спинного мозга. Система: Galaxy-камера (RGB, 1920×220@100Hz) → Python DLCLive/PyTorch (`single_rt_dlc_live_bridge.py`, профиль `single-best`) → 6 точек задней лапы → UDP binary `DDLP` → C++ плагин Open Ephys (`DualDLCLiveBridge`, фильтр+угол+порог+TTL). Реальный infer-rate сейчас ~25–37 Гц (не 100), инференс ~18–24 мс FP32 на RTX 5070 Laptop, видно только 3/6 точек (левая нога; правая камера не наведена).

### 8.0. Жёсткие правила (НЕ нарушать)
- **Git не трогать**, в рабочий пайплайн не коммитить. Любые эксперименты — в отдельных файлах/ветке рабочей копии, помечать.
- **Не менять бинарный формат `DDLP` и порядок 6 точек** (`hl_ankle_l, hl_ankle_r, hl_hip_l, hl_hip_r, hl_toes_l, hl_toes_r`) — иначе C++ плагин разложит точки не так и угол стимуляции исказится. Если меняешь точки — синхронно правь `BINARY_POSE_POINT_NAMES` в плагине и `DUAL_USE_POINTS`, плюс включи `validate_dual_config()`.
- **Любая оптимизация скорости проходит accuracy-gate**: A/B на ОДНОМ записанном видео (eager FP32 baseline vs изменение), метрики — координаты каждой точки (Δpx) и likelihood; расхождение медианы координат ключевых точек должно быть ≤ ~1px, иначе это уже потеря точности.
- **Каждое изменение проверять и на скорости, и на точности** (две оси независимы).
- Безопасность контура: не вносить залипание TTL, не ломать NaN-инвалидацию (правый NaN-блок должен оставаться invalid).

### 8.1. Контекст-файлы и точки входа (куда лезть)
- Хот-путь инференса: `dlc_live_env/Lib/site-packages/dlclive/pose_estimation_pytorch/runner.py` → `PyTorchRunner.get_pose` (строки ~174–230); ключевая строка `outputs = self.model(model_input)` (~210) и `self.model.get_predictions(outputs)["bodypart"]["poses"]` (~211). `self.model` — обычный `nn.Module` (`PoseModel`), грузится в `load_model` (~257–268).
- Препроцессинг/кроп: `dlclive/dlclive.py` → `process_frame` (239–296: cropping 253-254, resize 290-291, convert2rgb 293-294), восстановление координат `_post_process_pose` (349–371).
- Их обёртка инференса: `dual_rt_dlc_live.py` → `run_raw_inference` (830–858), `run_batch_inference` (1007–1082), `postprocess_pose_without_processor` (989–1004).
- Single-runtime: `single_rt_dlc_live_bridge.py` (рабочий main loop).
- Профили/точность: `live_profiles.py` (precision, TF32, threads).
- Фильтр (активный): C++ `open_ephys_plugin/DualDLCLiveBridge/DualDLCLiveBridge.cpp` → `filterPoint` + UI-параметры (`registerParameters`). Репозиторий: github.com/nikaabigail/Dual_DLC_live_plugin.
- Kalman-процессор (для задачи B7): `dlclive/processor/kalmanfilter.py` → `KalmanFilterPredictor(forward=…, fps=…)`.

---

### Bucket A — оптимизация скорости БЕЗ потери точности (accuracy-neutral)

**[A1] Headless для боевого режима.** `effort:S · risk:none`
- Где: `single_rt_dlc_live_bridge.py` display-блок (overlay→SHOW_SCALE resize→`cv2.imshow`); в dual — блок ~1436–1504.
- Что: гонять стим-режим с `--no-display`; убедиться, что `display` стадия профайлера → 0.
- Выигрыш: −5–6 мс/кадр (замерено). Координаты не меняются.
- Проверка: `stage_profile ... display=0.00`, live_fps вырос.

**[A2] Статический горизонтальный CROPPING в DLCLive.** `effort:M · risk:low (accuracy-safe)`
- Где: `config_rt_dlc_live.py:81` `CROPPING=None` → `[x1, x2, 0, 220]`. Восстановление координат уже есть (`_post_process_pose` 359–361). 
- Что: (1) сначала ИЗМЕРИТЬ реальный x-диапазон 6 точек за репрезентативную сессию — из offline-H5 (`C:\dlc\videos\...DLC_Resnet50_...best-380.h5`) или залогировать x вживую; (2) взять [min−margin, max+margin]; (3) **ширину кропа сделать кратной `pad_width_divisor`** из `cfg["data"]["inference"]["auto_padding"]` (иначе `AutoPadToDivisor` в runner.py:302–310 добавит свой паддинг). 
- Выигрыш: стоимость инференса ~линейна по пикселям; 1920→~900px ≈ ~2× (≈20→~10 мс). На тредмиле зверь держится на месте → допустимо.
- Не сломать: кроп должен ВСЕГДА содержать лапу при дрейфе животного; проверить по всей сессии, не только по одному кадру.
- Проверка: на записанном видео точки ВНУТРИ кропа дают те же координаты/likelihood (Δ≤1px), что и full-frame. Замерить infer_ms до/после.

**[A3] `torch.compile(runner.model)`.** `effort:S–M · risk:low`
- Где: `runner.py:load_model`, после `self.model.eval()` (~265): `self.model = torch.compile(self.model, mode="reduce-overhead")` (или внешне после `build_dlc_live`). torch 2.10 есть.
- Выигрыш: inductor, численно эквивалентно (точность не меняется), ×1.3–2.
- Не сломать: первый вызов компилируется (медленный warmup — прогреть до старта стимуляции); **статический размер входа обязателен** — НЕ совмещать с динамическим кропом (рекомпиляции). `get_predictions` остаётся как есть (компилируется только forward).
- Проверка: вывод bit-близкий к eager, infer_ms ↓, нет спама рекомпиляций (`TORCH_LOGS=recompiles`).

**[A4] `channels_last` + явный TF32-контроль.** `effort:S · risk:low`
- Где: после загрузки модели — `self.model.to(memory_format=torch.channels_last)`, вход `model_input.to(memory_format=torch.channels_last)` (runner.py ~206). `cudnn.benchmark` уже True.
- Выигрыш: ускоряет свёртки на тензор-кор, раскладка памяти — точность не меняется. (TF32 — отдельная ось точности, см. B4; здесь только не регрессировать.)
- Проверка: infer_ms ↓; координаты те же.

**[A5] Перенести `pose_transform` на GPU.** `effort:M · risk:low`
- Где: `runner.py:get_pose` — сейчас `pose_transform` (ToDtype/Normalize/AutoPad, собран в `load_model` 302–313) применяется к CPU-тензору ДО `.to(device)` (200–206). Перенести нормализацию/ToDtype на устройство (сначала `.to(device)`, потом transform на GPU-тензоре).
- Выигрыш: снимает CPU-препроцессинг с горячего пути (актуально для dual, где он ×2). Численно эквивалентно при аккуратном порядке.
- Проверка: координаты идентичны (Δ≈0), CPU-время `preprocess` ↓.

**[A6] TensorRT (FP32) на фиксированном размере.** `effort:L · risk:med (валидировать)`
- Где: компилировать forward `runner.model` через Torch-TensorRT (`torch_tensorrt.compile(module, inputs=[torch_tensorrt.Input((1,3,H,W))], enabled_precisions={torch.float32})`), `get_predictions` оставить в Python (это отдельный метод, не часть трассируемого forward; forward, вероятно, отдаёт dict — компилировать тензор-выдающую часть). Встроенного TRT в DLCLive PyTorch НЕТ (`tensorrt` model_type — только TF). 
- Зависит от: [A2] фиксированный статический размер (TRT требует статический shape; динамический кроп несовместим).
- Выигрыш: крупнейший единичный рычаг по `inference`. В FP32 точность сохраняется (подтвердить).
- Не сломать: FP16/INT8 НЕ включать без A/B (FP16 уже измеримо роняет likelihood — см. B4). Прогрев/сборка engine до старта.
- Проверка: позы совпадают с eager FP32 (Δ≤1px) на записанном видео; infer_ms ↓.

**[A7] Микро-чистка горячего пути.** `effort:S · risk:none`
- `dual_rt_dlc_live.py:1018` — `import torch` вынести из `run_batch_inference` наружу. `:835/:1022` — `dlc_live.convert2rgb=...` ставить один раз при инициализации, а не на каждый кадр. Pinned memory + `non_blocking=True` для H2D (в батч-пути уже есть). Подтвердить отсутствие скрытых RGB↔BGR конверсий в боевом headless-пути.
- Проверка: координаты те же; мелкое снижение CPU.

---

### Bucket B — оптимизация для УВЕЛИЧЕНИЯ точности в real-time

**[B1] Убрать лаг медианы: `median_window` 3→1 + `refractory_ms` 30–50.** `effort:S · risk:low`
- Где: UI плагина (`median_window`, `refractory_ms`). Логика — `filterPoint` (median) и `shouldEmitAngleTrigger` (refractory).
- Почему: при ~35 Гц median=3 вносит ~28 мс лага на угле, который триггерит стим → ошибка тайминга фазы шага. Де-дребезг у порога переносим с median (лагает позицию) на refractory (не лагает). 
- Проверка: тайминг rising-edge TTL относительно реального события фазы; стабильность угла без роста ложных срабатываний.

**[B2] Поднять реальный infer-rate (включает Bucket A).** `effort:зависит · risk:low`
- Почему: ~35 Гц → редкая временная выборка быстрых toes → рваный триплет, despike чаще режет. Рост до 60–100 Гц = плотнее выборка = выше валидность триплета и меньше ложных дропов. Это главный системный enabler точности.
- Зависит от: A1–A6.
- Проверка: `raw_visible` и доля валидного триплета ↑ на той же сцене.

**[B3] Экспозиция/свет камеры (самый прямой рычаг по toes).** `effort:S · risk:low (физика)`
- Где: Galaxy `.txt` конфиг, `ExposureTime` (лево 8000µs=8мс → смаз быстрых toes; toes — самая слабая точка, likelihood скачет 0.37–0.92). Снизить экспозицию + добавить света/gain.
- Проверка: A/B по per-point likelihood (особенно toes) и доле валидного триплета. Это меняет ДАННЫЕ, не код.

**[B4] FP32 vs TF32 vs FP16 — зафиксировать эталон точности.** `effort:S · risk:low`
- Почему: измерено — FP16 роняет `hl_ankle_l` 0.79→0.48, валидность триплета 100%→78%. `single-best` использует TF32 (не bit-exact FP32). 
- Что: A/B `single-strict` (FP32, TF32 off) vs `single-best` (TF32) vs `single-fp16` по координатам/likelihood. Для боевой стимуляции взять самый точный, скорость добирать через Bucket A. 
- Проверка: таблица Δpx и Δlikelihood по 6 точкам.

**[B5] Аудит live-vs-offline препроцессинга.** `effort:M · risk:low`
- Почему: live `RESIZE=1.0/CROPPING=None`; если offline `analyze` применял `global_scale`/иной вход (см. `pose_cfg`/`cfg["data"]` экспортированной модели), сеть на инференсе видит иное разрешение → систематический сдвиг координат.
- Что: сохранить сырые позы DLCLive по записанному клипу и сравнить с `run_dlc.py analyze` на тех же кадрах; если есть систематический offset/scale — устранить. (HISTORY next-step #2.)
- Проверка: попарный diff поз live vs offline → near-zero после фикса.

**[B6] Тюнинг порогов решения (`conf_thresh_draw`).** `effort:S · risk:low`
- Почему: 0.15 может пропускать маргинальные точки → неверный триплет/угол. A/B 0.15 / 0.25 / 0.30 по корректности триплета и стабильности угла. Опционально — пер-точечные пороги (toes слабее) в плагине.
- Проверка: доля ложных/верных триплетов на размеченном клипе.

**[B7] Kalman вместо median: сгладить БЕЗ лага и компенсировать latency.** `effort:L · risk:med`
- Где: `dlclive/processor/kalmanfilter.py` `KalmanFilterPredictor(forward=…, fps=…)` — constant-velocity Kalman с forward-предсказанием. Два пути: (а) перенести фильтрацию обратно в Python (`DUAL_FAST_POSE_ONLY=False`, processor=Kalman) — НО тогда плагин не должен фильтровать второй раз (иначе двойной фильтр), и теряется батч; (б) портировать предиктивный фильтр (CV-Kalman) в C++ плагин, заменив median.
- Почему: median лагает на ~filter-window; Kalman с `forward` ПРЕДСКАЗЫВАЕТ позицию на ~infer+transport latency вперёд → угол на момент стимуляции точнее по времени, без median-лага. Это одновременно +точность и −латентность для триггера.
- Не сломать: настроить `forward` под реальную end-to-end задержку; не допускать «улёта» при низкой likelihood (есть `lik_thresh`). Валидировать на реальной походке.
- Проверка: тайминг и величина угла на rising-edge vs ground-truth фаза; отсутствие овершутов.

**[B8] Навести правую камеру (вернуть 3/6 → 6/6).** `effort:S–M · risk:low (физика)`
- Где: offset-сканы в `debug_snapshots/right_offset_*.png` — подбор OffsetY дорожки для правой камеры. Пока правый likelihood ~0.02–0.09.
- Почему: даёт вторую ногу → избыточность/устойчивость триплета и работу dual.
- Проверка: правые точки likelihood ↑, `right_triplet` валиден.

**[B9] (Глубокий трек) Fine-tune модели под live-домен.** `effort:L · risk:med`
- Почему: если live-кадры (экспозиция, смаз, свет) отличаются от обучающих — модель недообучена под боевые условия. Дообучить на размеченных live-кадрах (особенно где теряются toes).
- Зависит от: B3/B5 (сначала исключить препроцессинг/смаз как причину). Это отдельный длинный трек, не трогает real-time код.

---

### 8.2. Порядок выполнения (рекомендованный)
1. Точность-эталон: [B4] FP32/TF32 A/B + [B5] live-vs-offline аудит → знать «истинную» точность и нет ли скрытого сдвига.
2. Дешёвая латентность: [A1] headless + [B1] median→1/refractory + [B3] экспозиция.
3. Скорость без потери точности: [A2] статический кроп → [A3] torch.compile → [A4]/[A5]/[A7] → при необходимости [A6] TensorRT FP32.
4. С ростом infer-rate [B2]: вернуть умеренный median или внедрить [B7] Kalman-предиктор; [B6] пороги.
5. Параллельно физика: [B8] правая камера; долгий трек [B9].

### 8.3. Валидационный харнесс (как мерить каждое изменение)
- **Точность (offline-эталон):** прогнать ОДНО записанное видео через изменённый и эталонный путь, сравнить координаты (Δpx по каждой точке) и likelihood; ключевые `hl_ankle_l/hl_toes_l/hl_hip_l`. Порог «без потери точности»: медианный Δ ≤ ~1px.
- **Скорость:** `stage_profile` (`inference`, `preprocess`, `display`, `pack/send`), `live_fps`, `result_hz` до/после.
- **Real-time точность:** доля валидного левого триплета (%кадров, все 3 точки ≥ `conf_thresh_draw`), per-point likelihood, тайминг rising-edge TTL.
- **Источник live-метрик:** `single_rt_dlc_live_bridge_debug.log` (frame=…, `raw_visible`, `p=`per-point, `stage_profile`). Угол/TTL/age — в UI плагина (в лог Python не пишутся; при необходимости включить ACK или логировать угол — отдельная мелкая правка).

### 8.4. Известные ловушки (не наступать при правках)
- `dlc_live.get_pose()` (метод DLCLive, `dlclive.py:342-343`) ФОРСИТ `convert2rgb=True`. Боевой путь использует `process_frame`+`runner.get_pose` и этого избегает; если перепишешь на `dlc_live.get_pose()` при `GALAXY_OUTPUT_COLOR="rgb"` — получишь BGR на входе модели и просадку точности.
- Дефолт `getattr(config,"CONVERT_TO_RGB",True)` (`run_raw_inference:835`) — при rgb-камере забыть выставить `False` = сломанный цвет.
- Не включать `DUAL_FAST_POSE_ONLY=False` без снятия фильтра в плагине — двойная фильтрация/лаг.
- Динамический кроп DLCLive (`DYNAMIC_CROPPING=(True,…)`) отключает dual-batch и несовместим со статическим TRT — выбирать одно (для TRT нужен фиксированный кроп).
- Сломанная не-fast ветка `run_inference` (`dual_rt_dlc_live.py:938` `ure_ts=` опечатка, `:941` `print_model_device_after_init` не определена) — починить, если будешь экспериментировать с Python-постобработкой/Kalman, иначе креш inference-потока.
