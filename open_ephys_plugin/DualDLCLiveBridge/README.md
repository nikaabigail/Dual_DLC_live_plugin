# Dual DLCLive Bridge для Open Ephys

## Current protocol

Default Python bridge mode sends compact binary UDP pose packets (`DDLP`/v1).
These packets contain raw pose points and timing metadata, not ready-made TTL
decisions. JSON `dual_dlc_live.pose.v1` packets are still supported as fallback
when `DUAL_OE_BRIDGE_WIRE_FORMAT = "json"`.

This plugin computes:

- online point filter: p-cutoff, despike, median and optional hold;
- left/right triplet validity;
- hind angle at ankle;
- TTL state word;
- optional line 2/3 angle triggers when `angle_trigger_enabled` is enabled.

TTL map:

```text
line 0 = left valid triplet
line 1 = right valid triplet
line 2 = left angle trigger
line 3 = right angle trigger
line 4..7 = reserved
```

The old `dual_dlc_live.v1`/`ttl_lines` packet is still supported as legacy
input. Use `send_dual_dlc_bridge_test.py --mode pose --wire-format binary` for
the current synthetic test, `--wire-format json` for JSON pose compatibility,
and `--mode ttl` for legacy compatibility.

Important current behavior:

- Default pose mode receives raw pose points, not `ttl_lines`.
- The plugin computes filter, triplet validity, angle, TTL word and refractory
  gating.
- `angle_trigger_enabled` and `angle_threshold_deg` in the plugin UI control
  lines `2` and `3`.
- Python `DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` is only for legacy `ttl` mode.
- In Python binary fast mode, points are packed directly from a compact
  `raw_pose_array` NumPy array; `raw_points` dictionaries are only built for JSON
  fallback or local Python overlay.
- Python stage profiler lines named `stage_profile` show camera/read,
  preprocess, inference, pack/send and display timing before packets reach this
  plugin.

Этот документ описывает, как запускать связку:

```text
2 Daheng USB3 камеры -> dual_rt_dlc_live.py -> UDP localhost -> Open Ephys plugin -> TTL events -> stimulation/output processor
```

Плагин называется `Dual DLCLive Bridge`. Он не запускает DLCLive и не читает
камеры. Камерами и нейросетью управляет отдельный Python-процесс
`dual_rt_dlc_live.py`, а Open Ephys получает от него только маленькие UDP
binary packets с raw pose points и metadata. TTL-состояние считается внутри plugin.

## Где лежат файлы

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation
  dual_rt_dlc_live.py
  config_dual_rt_dlc_live.py
  send_dual_dlc_bridge_test.py

C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
  out\build\x64-Debug\open-ephys.exe
  Plugins\DualDLCLiveBridge
  out\build\x64-Debug\plugins\DualDLCLiveBridge.dll
```

Камеры в текущем конфиге:

| Side | Serial | Galaxy config |
| --- | --- | --- |
| left | `FDE22070174` | `C:\config_daheng\Rat_TREDMILL_Left_1920px_220px_100Hz_(FDE22070174).txt` |
| right | `FDE22070175` | `C:\config_daheng\Rat_TREDMILL_Right_1920px_220px_100Hz_(FDE22070175).txt` |

ROI уже находится в `.txt` конфигах камер. В Python поэтому стоит
`CROPPING = None`, дополнительный software crop не нужен.

## Какие TTL-линии формируются

Python формирует массив `ttl_lines` длиной 8:

| Line | Что означает | Для чего использовать |
| --- | --- | --- |
| `0` | left: найден валидный triplet `hip/ankle/toes` | quality/gate, не основной trigger |
| `1` | right: найден валидный triplet `hip/ankle/toes` | quality/gate, не основной trigger |
| `2` | left: угол задней лапы `<= DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` | trigger для стимуляции по левой стороне |
| `3` | right: угол задней лапы `<= DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` | trigger для стимуляции по правой стороне |
| `4..7` | зарезервировано | можно добавить новые условия позже |

Если `DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG = None`, линии `2` и `3` всегда
остаются `False`. Чтобы включить стимуляционный trigger, задай число, например:

```python
DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG = 55.0
```

Тогда линия `2` станет `True`, когда левая сторона валидна и угол `<= 55.0`.
Линия `3` работает так же для правой стороны.

TTL word внутри плагина считается как битовая маска:

```text
ttl = line0 * 1 + line1 * 2 + line2 * 4 + line3 * 8 + ...
```

Примеры:

| Активные линии | `ttl` в UI |
| --- | --- |
| none | `0x00` |
| line 0 + line 1 | `0x03` |
| line 0 + line 1 + line 2 | `0x07` |
| line 1 + line 3 | `0x0A` |

Плагин отправляет TTL state changes, а не постоянный поток одинаковых событий.
Если условие уже `True`, повторные одинаковые пакеты не создают новые rising
edges. Для стимуляции обычно нужно в downstream processor выбрать rising edge
линии `2` или `3`. Если нужен повторяющийся pulse train, пока условие остается
истинным, это лучше делать отдельной логикой в стимуляционном processor или
добавить pulse-mode в bridge.

## Настройки Python bridge

Файл:

```text
C:\dlc\DLC_OBS_Spinal_cord_stimulation\config_dual_rt_dlc_live.py
```

Основные параметры:

```python
DUAL_OE_BRIDGE_ENABLED = True
DUAL_OE_BRIDGE_HOST = "127.0.0.1"
DUAL_OE_BRIDGE_PORT = 47000
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS = 1
DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG = None
```

Что они делают:

| Параметр | Значение |
| --- | --- |
| `DUAL_OE_BRIDGE_ENABLED` | включает отправку UDP в Open Ephys |
| `DUAL_OE_BRIDGE_HOST` | адрес Open Ephys bridge, обычно `127.0.0.1` |
| `DUAL_OE_BRIDGE_PORT` | UDP port, должен совпадать с `udp_port` в plugin UI |
| `DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS` | отправлять каждый N-й результат DLCLive |
| `DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG` | порог угла для линий `2` и `3` |

Для минимальной задержки:

```python
DUAL_PROCESS_EVERY_N_PAIRS = 1
DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS = 1
DUAL_LOW_LATENCY = True
DUAL_STREAM_BUFFER_HANDLING_MODE = "NEWEST_ONLY"
```

## Настройки Open Ephys плагина

В UI у `Dual DLCLive Bridge` есть параметры:

| Параметр | Значение |
| --- | --- |
| `enabled` | включает/выключает UDP socket |
| `udp_port` | локальный UDP port, по умолчанию `47000` |

Статусная строка:

```text
pkts 128 | pair 128 | ttl 0x03 | q 0 | age 42ms
```

Расшифровка:

| Поле | Значение |
| --- | --- |
| `pkts` | сколько валидных UDP пакетов принял C++ plugin |
| `pair` | последний `pair_index` из Python |
| `ttl` | текущая битовая маска линий `0..7` |
| `q` | сколько TTL words ждут следующего Open Ephys processing callback |
| `age` | сколько миллисекунд прошло с последнего принятого пакета |

Если `pkts` растет, UDP-связь Python -> Open Ephys работает.

## Порядок запуска для реальной работы

### 1. Подготовить камеры

1. Подключи обе Daheng камеры через USB3.
2. Закрой GalaxyView, если он открыт. GalaxyView может держать камеры и мешать
   Python открыть их.
3. Проверь, что серийники соответствуют текущему конфигу:
   `FDE22070174` слева, `FDE22070175` справа.

### 2. Запустить Open Ephys

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main\out\build\x64-Debug
.\open-ephys.exe
```

В signal chain должен быть процессор `Dual DLCLive Bridge`.

Если его нет:

1. Добавь source node, который будет давать Open Ephys processing clock. Для
   теста можно использовать `File Reader`, для реальной записи - вашу обычную
   acquisition source.
2. Добавь processor `Dual DLCLive Bridge`.
3. Убедись, что `enabled = true`.
4. Убедись, что `udp_port = 47000`.

Важно: UDP socket может работать уже после загрузки processor, но TTL events
нормально попадают дальше по цепочке только когда Open Ephys acquisition/processing
запущен. Для стимуляции запускай acquisition до запуска live Python.

### 3. Проверить UDP handshake без камер

В отдельном PowerShell:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
python send_dual_dlc_bridge_test.py --count 5 --interval 0.025 --wait-ack
```

Ожидаемый результат:

```text
sent pair=1 ttl=[False, True, False, False, False, False, False, False]
ack pair=1 from=127.0.0.1:47000 dual_dlc_live.ack pair=1 ttl=0x02
...
acked 5/5
```

Если `acked 5/5` есть, значит Python packet дошел до C++ plugin, JSON распарсен,
TTL word сформирован, plugin ответил.

Обычный `dual_rt_dlc_live.py` ACK не запрашивает. ACK используется только для
диагностики через `send_dual_dlc_bridge_test.py --wait-ack`.

### 4. Запустить реальный dual DLCLive

В PowerShell с активированным окружением:

```powershell
& C:\dlc_live_env\Scripts\Activate.ps1
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
python dual_rt_dlc_live.py
```

Или без активации, напрямую через Python из окружения:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe dual_rt_dlc_live.py
```

После запуска смотри:

1. В окне Python - нет ли ошибок открытия камер/Galaxy config.
2. В окне Open Ephys bridge - растет ли `pkts`.
3. Меняется ли `ttl`, когда поза валидна или когда угол пересекает threshold.
4. Для стимуляции - downstream output processor должен видеть line `2` или `3`.

## Как формируется output для стимуляции

Формирование идет в два этапа.

### Этап 1: Python решает, какие линии активны

В `dual_rt_dlc_live.py` после инференса двух камер создается пакет:

```json
{
  "schema": "dual_dlc_live.v1",
  "pair_index": 123,
  "host_time": 1780000000.0,
  "host_dt_ms": 2.1,
  "camera_dt_ms": null,
  "left": {"valid": true, "angle_deg": 62.4},
  "right": {"valid": true, "angle_deg": 58.1},
  "ttl_lines": [true, true, false, false, false, false, false, false]
}
```

Главное поле для Open Ephys - `ttl_lines`. Все остальные поля полезны для
debug/логов.

Логика линий:

```python
ttl_lines[0] = left_has_triplet
ttl_lines[1] = right_has_triplet
ttl_lines[2] = left_has_triplet and left_angle <= threshold
ttl_lines[3] = right_has_triplet and right_angle <= threshold
```

### Этап 2: C++ plugin превращает линии в Open Ephys TTL events

`Dual DLCLive Bridge` принимает UDP packet, читает `ttl_lines`, считает битовую
маску `ttl`, кладет изменение в очередь и в `process()` вызывает:

```cpp
setTTLState(sampleIndex, line, state);
```

Event channel называется:

```text
Dual DLCLive TTL
```

Если несколько TTL changes пришли между двумя Open Ephys callbacks, plugin
распределяет их по sample indices внутри текущего блока, а не ставит весь пакет
на sample `0`. Это помогает не потерять короткие переходы. Но нужно помнить:
точное время события все равно привязано к Open Ephys processing block, а не к
аппаратному timestamp камеры.

Практически для стимуляции:

1. В downstream stimulation/output processor выбери event channel
   `Dual DLCLive TTL`.
2. Для левой стороны используй line `2`.
3. Для правой стороны используй line `3`.
4. Trigger mode лучше ставить `rising edge`, чтобы стимуляция происходила при
   входе в условие, а не на каждом пакете.
5. Если нужно, добавь gate по line `0` или `1`, чтобы стимулировать только при
   валидной позе.

## Быстрая проверка, что Open Ephys слушает порт

Когда `Dual DLCLive Bridge` загружен и `enabled = true`, в PowerShell:

```powershell
netstat -ano -p udp | Select-String ':47000'
```

Пример:

```text
UDP    127.0.0.1:47000        *:*        22000
```

PID должен быть PID процесса `open-ephys.exe`.

## Сборка плагина после изменений C++

Перед сборкой закрой Open Ephys. Пока GUI открыт, Windows держит
`DualDLCLiveBridge.dll` загруженным, и линкер не сможет заменить файл.

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
cmd.exe /s /c "`"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`" -arch=x64 && cmake --build out\build\x64-Debug --target DualDLCLiveBridge --config Debug"
```

Проверить, что DLL экспортирует entrypoints Open Ephys:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
python -B Plugins\DualDLCLiveBridge\check_plugin_load.py
```

Ожидаемый результат:

```text
PLUGIN_MAP_OK
EXPORT_OK getLibInfo
EXPORT_OK getPluginInfo
PLUGIN_EXPORTS_OK
```

## Частые проблемы

### `missing ack`

Причины:

- Open Ephys не запущен.
- `Dual DLCLive Bridge` не добавлен в signal chain.
- `enabled = false`.
- Порт в Open Ephys не совпадает с Python `DUAL_OE_BRIDGE_PORT`.
- DLL старая и собрана до добавления diagnostic ACK.

### `pkts` не растет

Проверь:

```powershell
netstat -ano -p udp | Select-String ':47000'
```

Потом запусти:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
python send_dual_dlc_bridge_test.py --count 5 --wait-ack
```

Если handshake проходит, но live Python не двигает `pkts`, значит проблема на
стороне `dual_rt_dlc_live.py` или `DUAL_OE_BRIDGE_ENABLED`.

### Камера не открывается или GalaxySDK ругается

Проверь:

- GalaxyView закрыт.
- Камеры не заняты другой программой.
- USB3 hub не перегружен.
- Серийники в `DUAL_CAMERAS` совпадают с физическими камерами.
- `.txt` конфиги лежат по путям из `config_dual_rt_dlc_live.py`.

### TTL events есть, но стимуляции нет

`Dual DLCLive Bridge` создает только внутренние Open Ephys TTL events. Проверь
настройки downstream stimulation/output processor:

- выбран event channel `Dual DLCLive TTL`;
- выбрана правильная line `2` или `3`;
- включен trigger on rising edge;
- физический output device подключен и разрешен в Open Ephys;
- acquisition запущен.

## Минимальный checklist перед экспериментом

1. Камеры подключены, GalaxyView закрыт.
2. В `config_dual_rt_dlc_live.py` задан нужный
   `DUAL_OE_BRIDGE_ANGLE_THRESHOLD_DEG`.
3. Open Ephys запущен.
4. `Dual DLCLive Bridge` есть в signal chain, `enabled = true`, `udp_port = 47000`.
5. Handshake test дает `acked 5/5`.
6. Acquisition в Open Ephys запущен.
7. Downstream stimulation/output processor слушает `Dual DLCLive TTL`, line `2`
   или `3`.
8. Запущен `python dual_rt_dlc_live.py`.
9. В UI bridge растут `pkts`, меняется `pair`, `ttl`, `age` маленький.
