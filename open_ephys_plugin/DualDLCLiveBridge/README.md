# Dual DLCLive Bridge для Open Ephys

`Dual DLCLive Bridge` - это C++-процессор для Open Ephys. Он принимает
UDP-пакеты от `dual_rt_dlc_live.py`, берет сырые точки позы DLCLive и превращает
их в изменения TTL-состояний Open Ephys.

Плагин не открывает камеры и не запускает нейросеть. Камеры и DLCLive находятся
на стороне Python.

```text
dual_rt_dlc_live.py
  -> UDP 127.0.0.1:47000
  -> Dual DLCLive Bridge
  -> event channel "Dual DLCLive TTL"
  -> следующий processor стимуляции/output
```

## Навигация

| Раздел | Для чего нужен |
| --- | --- |
| Роль плагина | Что именно перенесено из Python в C++. |
| Установка | Где лежит plugin source и DLL. |
| UI параметры | Все параметры плагина и их смысл. |
| Входные UDP форматы | Binary `DDLP`, JSON pose, legacy TTL. |
| Порядок обработки | От UDP datagram до TTL event. |
| TTL lines | Какая линия что значит. |
| ACK | Как проверить, что плагин реально получил пакет. |
| Строка статуса | Как читать строку статуса в GUI. |
| Сборка | Как пересобрать DLL. |
| Диагностика | Synthetic tests и типичные проблемы. |

## Роль плагина

В рабочем режиме Python отправляет только сырые точки позы. Плагин сам делает:

- разбор UDP-пакета;
- проверку формата пакета;
- чтение left/right points;
- likelihood cutoff;
- despike rejection;
- median smoothing;
- optional hold последней хорошей точки;
- выбор side triplet;
- проверку валидности hip/ankle/toes;
- расчет hind angle;
- сравнение угла с threshold;
- refractory gating;
- сборку TTL word;
- отправку изменений TTL-состояний в Open Ephys.

Плагин поддерживает три входных режима:

| Вход | Статус | Когда нужен |
| --- | --- | --- |
| Binary `DDLP` pose v1 | Рабочий режим по умолчанию | Минимум CPU и allocations в Python. |
| JSON `dual_dlc_live.pose.v1` | Fallback/debug | Когда нужны имена точек или custom point set. |
| JSON `dual_dlc_live.v1` + `ttl_lines` | Legacy | Старый режим, где Python уже посчитал TTL. |

## Установка

Open Ephys tree:

```text
C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
  Plugins\DualDLCLiveBridge
  out\build\x64-Debug\plugins\DualDLCLiveBridge.dll
```

Репозиторий:

```text
C:\tmp\Dual_DLC_live_plugin\open_ephys_plugin\DualDLCLiveBridge
```

В `Plugins/CMakeLists.txt` должен быть подключен plugin:

```cmake
add_subdirectory(DualDLCLiveBridge)
```

## UI параметры

| Параметр | Default | Что делает |
| --- | --- | --- |
| `enabled` | `true` | Включает UDP listener. Если `false`, пакеты не принимаются. |
| `udp_port` | `47000` | Локальный UDP port, куда Python отправляет пакеты. |
| `angle_trigger_enabled` | `false` | Разрешает TTL lines `2` и `3` для angle trigger. |
| `angle_threshold_deg` | `55.0` | Порог hind angle для trigger. |
| `conf_thresh_use` | `0.20` | Порог likelihood до фильтрации. |
| `conf_thresh_draw` | `0.15` | Порог likelihood для валидного triplet. |
| `use_filter` | `true` | Включает C++ online filter. |
| `enable_pcutoff` | `true` | Отбрасывает точки ниже `conf_thresh_use`. |
| `enable_despike` | `true` | Отбрасывает резкие скачки точки. |
| `despike_threshold_px` | `150.0` | Максимальный разрешенный скачок точки. |
| `despike_reset_gap_frames` | `15` | Через сколько кадров разрешить reacquire после пропажи. |
| `median_window` | `3` | Размер медианного окна. |
| `enable_hold` | `false` | Удерживать последнюю хорошую точку при короткой пропаже. |
| `max_hold_frames` | `20` | Сколько кадров можно удерживать точку. |
| `refractory_ms` | `0` | Минимальный интервал между rising edges angle trigger. |

Для стимуляции по углу нужно явно включить:

```text
angle_trigger_enabled = true
```

Без этого lines `2` и `3` не будут активироваться.

## Входные UDP форматы

### Binary pose `DDLP` v1

Это рабочий формат.

На стороне Python:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "binary"
```

Пакет начинается с:

```text
magic = "DDLP"
version = 1
```

Высокоуровневая структура:

```text
packet header
  magic
  version
  flags
  pair_index
  host_time
  host_dt_ms
  camera_dt_ms
  point_count

left side block
  frame_id
  source_frame_id
  capture_ts
  infer_ms
  drops
  raw_visible
  6 points: x, y, likelihood

right side block
  frame_id
  source_frame_id
  capture_ts
  infer_ms
  drops
  raw_visible
  6 points: x, y, likelihood
```

Порядок точек фиксированный:

```text
hl_ankle_l
hl_ankle_r
hl_hip_l
hl_hip_r
hl_toes_l
hl_toes_r
```

Binary packet не передает имена точек. Плагин знает этот порядок заранее и
раскладывает значения в `PosePointMap`.

### JSON pose `dual_dlc_live.pose.v1`

Fallback режим:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "pose"
DUAL_OE_BRIDGE_WIRE_FORMAT = "json"
```

JSON несет имена точек явно:

```json
{
  "schema": "dual_dlc_live.pose.v1",
  "pair_index": 12,
  "tracked_points": ["hl_ankle_l", "..."],
  "left": {
    "frame_id": 101,
    "raw_points": {
      "hl_hip_l": {"x": 100.0, "y": 50.0, "likelihood": 0.9}
    }
  },
  "right": {
    "frame_id": 102,
    "raw_points": {}
  }
}
```

Этот режим медленнее binary, но удобнее для custom point names.

### Legacy TTL `dual_dlc_live.v1`

Legacy режим:

```python
DUAL_OE_BRIDGE_PACKET_MODE = "ttl"
```

Python сам считает TTL states:

```json
{
  "schema": "dual_dlc_live.v1",
  "pair_index": 12,
  "ttl_lines": [true, true, false, false, false, false, false, false]
}
```

Плагин в этом режиме не считает angle/triplet по точкам, а переносит готовые
`ttl_lines` в Open Ephys. Текущий рабочий path сейчас не такой.

## Порядок обработки внутри плагина

### 1. Socket listener

Когда `enabled = true`, `ensureSocket()` открывает UDP socket на `udp_port`.

В отдельном thread `run()` плагин читает datagrams. Каждый datagram передается в:

```text
applyDatagram(data, numBytes, ackMessage)
```

### 2. Выбор parser path

`applyDatagram` смотрит первые байты:

```text
если data[0:4] == "DDLP"
  -> applyBinaryPosePacket(...)
иначе
  -> String::fromUTF8(...)
  -> applyMessage(JSON)
```

Для JSON:

```text
schema empty или "dual_dlc_live.v1"
  -> applyTtlMessage(...)
schema "dual_dlc_live.pose.v1"
  -> applyPoseMessage(...)
```

### 3. Binary packet parsing

`applyBinaryPosePacket` читает:

- header;
- `version`;
- `flags`;
- `pair_index`;
- timing metadata;
- `point_count`;
- left side block;
- right side block.

Если `version != 1` или `point_count != 6`, пакет отклоняется.

Для каждой стороны плагин собирает:

```text
PosePointMap
  point name -> PosePoint(valid, x, y, likelihood)
```

Если `x`, `y` или `likelihood` не finite, точка считается невалидной.

### 4. Оценка стороны

Для binary и JSON pose path вызывается:

```text
evaluateSidePosePoints(...)
```

Он делает:

1. Берет сырые точки стороны.
2. Применяет `filterPoint` к каждой нужной точке.
3. Собирает triplet `hip`, `ankle`, `toes`.
4. Проверяет confidence.
5. Считает score triplet.
6. Считает hind angle, если triplet валиден.

### 5. Фильтрация точки

`filterPoint` работает так:

1. Проверяет `valid` и finite координаты.
2. Если `enable_pcutoff = true`, отбрасывает likelihood ниже `conf_thresh_use`.
3. Если `enable_despike = true`, сравнивает скачок с `despike_threshold_px`.
4. Если скачок слишком большой, но gap больше `despike_reset_gap_frames`, разрешает reacquire.
5. Добавляет точку в median buffer.
6. Возвращает медианную координату за `median_window`.
7. Если точка пропала и `enable_hold = true`, временно возвращает последнюю хорошую точку до `max_hold_frames`.

Если `use_filter = false`, фильтр отключается, но confidence/triplet проверка
все равно важна для TTL.

### 6. Расчет угла

Для валидного triplet:

```text
hip -> ankle -> toes
```

угол считается в ankle:

```text
angle = acos(dot(hip-ankle, toes-ankle) / (norm1 * norm2))
```

Если хотя бы одна точка невалидна или вектор почти нулевой, angle не считается.

### 7. Angle trigger

Линия angle trigger активируется, если:

```text
angle_trigger_enabled = true
triplet валиден
angle_deg <= angle_threshold_deg
refractory_ms разрешает новый rising edge
```

`refractory_ms` нужен, чтобы не давать слишком частые повторные rising edges.

### 8. Сборка TTL word

Плагин собирает 8-bit word:

```text
bit 0 -> line 0 -> left triplet valid
bit 1 -> line 1 -> right triplet valid
bit 2 -> line 2 -> left angle trigger
bit 3 -> line 3 -> right angle trigger
bit 4..7 reserved
```

Пример:

```text
left valid = true
right valid = true
left trigger = false
right trigger = false
TTL word = 0b00000011 = 0x03
```

### 9. Очередь TTL изменений

После разбора пакета:

```text
queueTtlWord(ttlWord)
```

Если новый `ttlWord` такой же, как прошлый, он не добавляется в очередь. Если
изменился, плагин кладет новое state word в `pendingTtlWords`.

### 10. Передача в Open Ephys

В audio/process callback:

```text
process(buffer)
  -> emitPendingTtlState(buffer.getNumSamples())
```

`emitPendingTtlState` берет все pending words и для каждой TTL line вызывает:

```text
setTTLState(sampleIndex, line, nextState)
```

Так Open Ephys получает изменения event-state на channel:

```text
Dual DLCLive TTL
```

Дальше физическая стимуляция зависит от downstream Open Ephys processors/output.
Сам `Dual DLCLive Bridge` не управляет физическим портом напрямую.

## TTL lines

| Line | Bit | Значение | Типичное применение |
| --- | --- | --- | --- |
| `0` | `0x01` | Left triplet valid. | Gate/quality сигнал. |
| `1` | `0x02` | Right triplet valid. | Gate/quality сигнал. |
| `2` | `0x04` | Left angle trigger. | Rising edge для левой стимуляции. |
| `3` | `0x08` | Right angle trigger. | Rising edge для правой стимуляции. |
| `4..7` | `0x10..0x80` | Reserved. | Будущие условия. |

Если мыши нет, обычно:

```text
ttl=0x00
left_angle=nan
right_angle=nan
```

Это нормально: плагин работает, но точки невалидны.

## ACK

ACK нужен для тестов, не для обычного high-rate live режима.

Python/test sender просит ACK:

- binary: ставит `flags & 0x01`;
- JSON: ставит `"ack": true` или `"request_ack": true`.

Плагин отвечает:

```text
dual_dlc_live.ack pair=5 mode=binary ttl=0x03 left_angle=135.00 right_angle=135.00
dual_dlc_live.ack pair=5 mode=pose ttl=0x03 left_angle=135.00 right_angle=135.00
dual_dlc_live.ack pair=5 mode=ttl ttl=0x03
```

`mode=binary` значит принят `DDLP`-пакет. Это основной рабочий путь.

## Строка статуса в GUI

В editor плагина строка выглядит примерно так:

```text
pkts 120 | mode bin | pair 120 | ttl 0x03 | L 135.0 | R 135.0 | q 0 | age 4ms
```

| Поле | Значение |
| --- | --- |
| `pkts` | Сколько UDP-пакетов принято. |
| `mode` | `bin`, `pose`, `ttl` или `-`. |
| `pair` | Последний `pair_index`. |
| `ttl` | Последний TTL word в hex. |
| `L` | Последний левый angle, если был. |
| `R` | Последний правый angle, если был. |
| `q` | Сколько TTL words ожидает emission. |
| `age` | Сколько ms прошло с последнего пакета. |

Если `pkts` растет, плагин получает UDP. Если `ttl` не меняется, либо точки
невалидны, либо условия trigger не выполняются.

## Сборка и установка

Закрыть Open Ephys перед сборкой, иначе DLL может быть занята.

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
cmd.exe /s /c "`"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`" -arch=x64 && cmake --build out\build\x64-Debug --target DualDLCLiveBridge --config Debug"
```

Smoke-test DLL:

```powershell
cd C:\Users\Владимир\Desktop\plugin-GUI-main\plugin-GUI-main
C:\dlc_live_env\Scripts\python.exe -B Plugins\DualDLCLiveBridge\check_plugin_load.py
```

Если DLL нужна как artifact, она также хранится в repo:

```text
dist/windows-x64-debug/DualDLCLiveBridge.dll
```

## Диагностика без камер

Запустить Open Ephys, добавить `Dual DLCLive Bridge`, поставить:

```text
enabled = true
udp_port = 47000
```

Binary test:

```powershell
cd C:\dlc\DLC_OBS_Spinal_cord_stimulation
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --interval 0.025 --wait-ack
```

Ожидаемо:

```text
acked 5/5
dual_dlc_live.ack ... mode=binary ...
```

Резервный JSON pose:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode pose --wire-format json --count 5 --interval 0.025 --wait-ack
```

Legacy TTL:

```powershell
C:\dlc_live_env\Scripts\python.exe send_dual_dlc_bridge_test.py --mode ttl --count 5 --interval 0.025 --wait-ack
```

## Диагностика с камерами

В live log хорошо, если есть:

```text
Open Ephys bridge enabled: UDP 127.0.0.1:47000 mode=pose wire=binary
Opened left sn=FDE22070174 ...
Opened right sn=FDE22070175 ...
stage_profile ... pack/send=...
```

Для точного подтверждения приема live-пакетов можно временно включить ACK и
прочитать ответы. В нормальной конфигурации ACK выключен, поэтому `pack/send`
показывает отправку, а plugin UI `pkts` показывает прием.

## Типичные проблемы

### `missing ack`

Причины:

- Open Ephys не запущен;
- plugin не добавлен в signal chain;
- `enabled = false`;
- порт не совпадает с `DUAL_OE_BRIDGE_PORT`;
- запущен не тот Open Ephys build.

### `pkts` не растет

Проверить:

- Python пишет `Open Ephys bridge enabled: UDP 127.0.0.1:47000 mode=pose wire=binary`;
- synthetic test дает `acked 5/5`;
- firewall не блокирует local UDP;
- в Open Ephys открыт именно этот plugin.

### `ttl=0x00`

Это не обязательно ошибка. `0x00` нормален, если:

- мыши нет;
- likelihood точек ниже threshold;
- triplet невалиден;
- `angle_trigger_enabled=false`;
- угол выше threshold.

### Lines `2` и `3` не появляются

Проверить:

- `angle_trigger_enabled = true`;
- `angle_threshold_deg` подходит под ожидаемый угол;
- triplet lines `0`/`1` валидны;
- `refractory_ms` не слишком большой;
- downstream processor слушает channel `Dual DLCLive TTL`.

### TTL в plugin есть, но стимуляции нет

Плагин создает Open Ephys event states, но физический выход делает downstream
chain. Нужно проверить:

- acquisition/processing в Open Ephys запущены;
- downstream output processor подключен к `Dual DLCLive TTL`;
- нужные TTL lines выбраны downstream;
- физический кабель/устройство стимуляции подключены.
