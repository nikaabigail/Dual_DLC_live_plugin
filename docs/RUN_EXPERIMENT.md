# Как поставить эксперимент

Пошаговый запуск замкнутого контура **камеры → DLCLive → UDP → Open Ephys плагин → TTL → стимуляция**. Предполагается, что установка уже выполнена по [`INSTALL_AND_RUN.md`](INSTALL_AND_RUN.md) (Python-окружение, Galaxy SDK, модель, конфиги камер, плагин собран и лежит в папке плагинов OE).

Архитектуру и формат пакетов см. в [`../README.md`](../README.md).

---

## 0. Перед запуском (чек-лист)

- [ ] GalaxyView **закрыт** (иначе Python не откроет камеры).
- [ ] Камеры Left `FDE22070174` и Right `FDE22070175` в портах **USB 3.0**.
- [ ] Если эксперимент идёт по аппаратному кадровому триггеру (Line2) — линия заведена. По умолчанию `DUAL_FORCE_TRIGGER_OFF = False`, т.е. соблюдается trigger-настройка из `.txt` камеры. Для прогона без триггера на столе временно поставить `DUAL_FORCE_TRIGGER_OFF = True` в `python/config_dual_rt_dlc_live.py`.
- [ ] Папка для записей существует: `C:\dlc\DLC_OBS_Spinal_cord_stimulation\recordings`.

---

## 1. Запустить Open Ephys и добавить узел

1. Запустить **Open Ephys** из `C:\Users\NeuroLab\Desktop\Open Ephys\oe_20260420\` (портативная сборка Release, plugin API 10; плагин уже в `oe_20260420\plugins\DualDLCLiveBridge.dll`).
2. В signal chain добавить узел **`Dual DLCLive Bridge`** и за ним — downstream processor стимуляции/выхода, подключённый к event-каналу `Dual DLCLive TTL`.

## 2. Параметры узла плагина

Базовые (в сетке редактора OE):

| Параметр | Значение для эксперимента |
|---|---|
| `enabled` | `true` |
| `udp_port` | `47000` |
| `angle_trigger_enabled` | `true` — если нужны угловые линии 2/3 для стимуляции (по умолчанию `false`) |
| `angle_threshold_deg` | рабочий порог угла голеностопа (триггер срабатывает при `угол ≤ порога`) |
| `refractory_ms` | `>0` рекомендуется при включённом угловом триггере — антидребезг фронта TTL |
| `watchdog_timeout_ms` | `100` (по умолчанию). Если пакеты не приходят дольше этого, плагин **принудительно гасит все TTL-линии**. Не выключать на живом животном: без сторожа падение Python с поднятой линией оставляет стимуляционный гейт открытым навсегда. `0` отключает |

Фильтр точек (`use_filter`, `conf_thresh_use=0.20`, `conf_thresh_draw=0.15`, `enable_despike`, `despike_threshold_px=150`, `median_window=3`, …) — оставить по умолчанию, пере-настроить на живой крысе при необходимости.

## 3. Правила достоверности кинематики (валидность)

Плагин гасит **угловой триггер (линии 2/3)** на кадрах с физически невозможной позой — до того, как они уйдут в стимуляцию. Гейты **только подавляют** триггер, никогда его не создают и **не трогают** линии 0/1 (видимость триплета).

Значения по умолчанию (широкие потолки — ловят телепорты/ошибки детекции, не реальную походку):

| Правило | Параметр(ы) | По умолчанию |
|---|---|---|
| Полоса угла | `enable_angle_plausibility`, `angle_min_deg`, `angle_max_deg` | вкл, 30°, 170° |
| Скорость угла | `enable_angle_delta`, `angle_max_delta_deg` | вкл, 25°/кадр |
| Смещение точки (доля сегмента hip-ankle) | `enable_disp_segment`, `disp_frac_toe/ankle/hip`, `disp_seg_min_px` | вкл, 1.0 / 0.5 / 0.3, 30px |
| Мастер-переключатель | `enable_validity_gates` | вкл |

> ⚠️ В текущей сборке эти параметры **не выведены в сетку редактора** (она заполнена) — действуют со значениями по умолчанию. Чтобы временно отключить всё их влияние, выключите `angle_trigger_enabled` (гейты влияют только на линии 2/3). Изменение самих порогов пока требует пересборки плагина (`docs/BUILD_PLUGIN.md`) — отдельная панель тюнинга в редакторе запланирована.
>
> Пороги калибровались по эмпирике левой ноги; **перепроверить и при необходимости ужесточить на реальной записи живой крысы** (измерить px/cm камеры, охарактеризовать правую ногу).

## 4. Запись видео + keypoints

В `python/config_dual_rt_dlc_live.py`:

- `DUAL_RECORD_ENABLED = True`, `SINGLE_RECORD_ENABLED = True` — пишет сырое видео и keypoints с обеих камер (`dual_<ts>_left.*`, `dual_<ts>_right.*`) в `C:\dlc\DLC_OBS_Spinal_cord_stimulation\recordings`, в фоновом потоке (без влияния на контур).
- `SINGLE_RECORD_VIDEO_CODEC = "FFV1"` — lossless `.avi` (расширение выбирается автоматически). Для более лёгких файлов с потерями — `"MJPG"`. Если рекордер не успевает на 100 fps, он логирует дроп кадров (контур при этом не страдает).
- Keypoints: `SINGLE_KP_FORMAT = "csv"` (или `"binary"` → `.dlckp`, конвертер `scripts/kp_to_csv.py`). Координаты — полнокадровые; окно ROI `[x1,x2]` пишется в каждой строке.

## 5. Проверка плагина без камер (опционально)

```powershell
cd C:\dlc\Dual_DLC_live_plugin\python
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
# ожидается: acked 5/5
```

## 6. Запуск live-контура

**Две камеры:**
```powershell
cd C:\dlc\Dual_DLC_live_plugin\python
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```
**Одна камера (боевой режим):**
```powershell
C:\dlc_live_env\Scripts\python.exe single_rt_dlc_live_bridge.py --profile single-best --no-display
```

В логе ждать: `Open Ephys bridge enabled ...`, `Opened left/right ... fps=100.0`, `CUDA_CHECK ... cuda=True`, `stage_profile ... result_hz=...`.

## 7. Убедиться, что контур работает

| Признак (UI плагина / лог) | Что значит |
|---|---|
| растёт `pkts` | плагин получает live-пакеты по UDP |
| `mode bin` | бинарный DDLP-путь (рабочий) |
| меняется `ttl 0x..` | плагин меняет TTL-слово |
| `L` / `R` показывают угол (число, не `-`) | триплет валиден и угол посчитан |
| линии 2/3 в TTL при `угол ≤ порога` и `angle_trigger_enabled=true` | угловой триггер уходит на стимуляцию |

Без животного нормально: `L -`, `R -`, `ttl 0x00` (pipeline жив, валидных точек нет).

## 8. Провести запись и остановить

1. В Open Ephys включить acquisition и запись (Record) — события `Dual DLCLive TTL` пишутся в сессию OE.
2. Видео/keypoints пишутся параллельно Python-рекордером в `...\recordings`.
3. По завершении: остановить запись/acquisition в OE, затем остановить Python-скрипт (Ctrl+C — рекордер дописывает остаток и закрывает файлы).

## 9. Диагностика

`missing ack`, камеры не открываются, точки не детектятся — см. [`../README.md`](../README.md) разделы «Как понять, что всё работает» и «Типичные проблемы». Сборка/деплой плагина — [`BUILD_PLUGIN.md`](BUILD_PLUGIN.md).
