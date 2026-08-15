"""
Авторазметка фаз локомоторного цикла (stance / swing) задней конечности крысы
на ТРЕДМИЛЕ по кейпоинтам DeepLabCut.

Назначение: получить per-frame ground truth (stance / swing / unknown + события
touch_down / lift_off) для обучения классификатора фаз (кинематика, в перспективе + ЭМГ).
Это НЕ переобучение DLC: фаза - временное состояние кадра, а не ключевая точка.

МЕТОД: BFR (belt-frame residual), два этапа.
  Ключевая физика тредмила: в опоре (stance) лапа сцеплена с лентой и едет НАЗАД
  со скоростью ленты; в переносе (swing) - летит ВПЕРЁД. Поэтому решающий сигнал:

      R = (v_toe_x - v_belt) / |v_belt|

  stance -> R ~ 0, swing -> R ~ 1/(1-D) ~ 3 (D - duty factor).
  ВАЖНО: наивное правило "stance = |v| ~ 0" на движущейся ленте НЕВЕРНО -
  нулевая скорость в кадре достигается ровно в момент отрыва (lift_off).

  Этап A (детекция): сглаженный R + гистерезис Шмитта + минимальные длительности.
                     Робастно определяет, ЕСТЬ ли фаза.
  Этап B (тайминг):  двухсегментная регрессия по НЕсглаженной позиции вокруг
                     перехода. Точно определяет, КОГДА событие (убирает сдвиг фильтра).

Запуск:
    python gait_phase_labeler.py --csv <DLC_filtered.csv> [--raw-csv <DLC.csv>]
                                 [--fps 100] [--leg auto|l|r] [--out <prefix>]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- параметры --

@dataclass
class Params:
    fps: float = 100.0
    # likelihood (согласовано с config_rt_dlc_live.py)
    p_use: float = 0.20          # ниже - кадр невалиден
    p_trust: float = 0.60        # "золотые" кадры: только они идут в оценки
    # геометрия / валидность
    max_jump_frac: float = 0.35  # скачок точки > 0.35 * L(hip-ankle) = телепорт
    roi_margin_px: float = 5.0   # клиппинг по краю ROI (кадр 1920x220)
    frame_h: float = 220.0
    # пропуски
    gap_interp_max: int = 5      # <= 5 кадров - интерполируем, больше - unknown
    guard_frames: int = 3        # защитная кайма вокруг пропуска/шва
    # детекция эпизодов локомоции (bouts)
    bout_win: int = 50           # окно 0.5 с
    bout_std_px: float = 8.0     # порог std(toe - body) для "шагает"
    bout_min_frames: int = 60    # минимум 0.6 с - иначе не эпизод
    # фазы
    r_hi: float = 1.6            # Шмитт: R выше -> swing
    r_lo: float = 0.8            # Шмитт: R ниже -> stance
    min_stance: int = 12         # минимальные длительности фаз (кадры)
    min_swing: int = 8
    # верхние пределы: у крысы длительность swing почти инвариантна (~120 мс),
    # варьирует stance. Фаза длиннее предела - артефакт (крыса встала / потеря трека).
    max_swing: int = 25          # 250 мс
    max_stance: int = 90         # 900 мс
    smooth_win: int = 5          # сглаживание для ЭТАПА A (нечётное)
    refine_win: int = 12         # полуокно двухсегментной регрессии (этап B)
    phase_from_events: bool = False   # строить фазы по уточнённым событиям (этап B)
    # Калибровка запаздывания детектора. Измерена на эталонной разметке m38np
    # (480 циклов, 2 крысы, 10 и 20 см/с, ровно и наклон 25%): оба события
    # детектируются систематически ПОЗДНО - сглаживание + гистерезис. Поправка
    # задана в мс (не в кадрах), поэтому переносится между 100 и 200 fps.
    # После коррекции медиана |ошибки|: touch_down 10 -> 5 мс, lift_off 5 -> 0 мс.
    # Побочный эффект, который нам нужен: разметка выравнивается под то же
    # соглашение о событиях, на котором обучена EMG-модель (она училась на m38np).
    # Проверка правдоподобия результата по ноге. Дальняя от камеры нога
    # перекрыта телом: DLC её теряет, и разметка молча вырождается - на 12 видео
    # мы видели swing 200-210 мс (норма 110-130) и stance МЕНЬШЕ swing, что
    # физически невозможно. Такие ноги надо отбраковывать, а не выдавать за данные.
    q_min_cycles: int = 10
    q_swing_ms: tuple = (80.0, 180.0)
    q_duty: tuple = (0.50, 0.90)
    q_min_labeled_frac: float = 0.02
    keep_rejected: bool = False
    td_bias_ms: float = -10.0
    lo_bias_ms: float = -5.0
    apply_bias: bool = True


PHASE_STANCE, PHASE_SWING, PHASE_UNKNOWN = "stance", "swing", "unknown"


# ------------------------------------------------------------------ загрузка --

def load_dlc(csv_path: Path) -> pd.DataFrame:
    """DLC CSV -> DataFrame с MultiIndex (bodypart, coord)."""
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples([(a, b) for a, b in df.columns])
    return df


def pick_leg(df: pd.DataFrame, p_trust: float) -> str:
    """Выбор ноги по доле надёжных кадров: toes весит больше ankle."""
    best, best_score = None, -1.0
    for leg in ("r", "l"):
        need = [f"hl_toes_{leg}", f"hl_ankle_{leg}", f"hl_hip_{leg}"]
        if any((b, "likelihood") not in df.columns for b in need):
            continue
        toes = (df[(f"hl_toes_{leg}", "likelihood")].values >= p_trust).mean()
        ankle = (df[(f"hl_ankle_{leg}", "likelihood")].values >= p_trust).mean()
        score = toes + 0.5 * ankle
        if score > best_score:
            best, best_score = leg, score
    if best is None:
        raise SystemExit("В CSV нет точек hl_toes/hl_ankle/hl_hip - это не тот датасет.")
    return best


def body_ref(df: pd.DataFrame, leg: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Опора тела для снятия дрейфа: iliac надёжнее hip, если он есть."""
    for name in (f"hl_iliac_{leg}", f"hl_hip_{leg}"):
        if (name, "x") in df.columns:
            return df[(name, "x")].values, df[(name, "likelihood")].values, name
    raise SystemExit("Нет ни iliac, ни hip - не от чего считать движение тела.")


# ----------------------------------------------------------------- валидность --

def validity_mask(toe_x, toe_y, toe_p, ank_x, ank_y, ank_p, body_p, L_ref, prm: Params):
    """Маска валидных кадров + причины отбраковки."""
    n = len(toe_x)
    valid = np.ones(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    def kill(mask, tag):
        newly = mask & valid
        valid[newly] = False
        reason[newly] = tag

    kill((toe_p < prm.p_use) | ~np.isfinite(toe_x) | ~np.isfinite(toe_y), "low_p_toe")
    kill(body_p < prm.p_use, "low_p_body")
    # клиппинг по краю узкого ROI: точка "прилипла" к границе кадра
    kill((toe_y < prm.roi_margin_px) | (toe_y > prm.frame_h - prm.roi_margin_px), "clipped")
    # анатомия: сегмент ankle->toes почти жёсткий
    with np.errstate(invalid="ignore"):
        d_at = np.hypot(toe_x - ank_x, toe_y - ank_y)
    ok = np.isfinite(d_at) & (ank_p >= prm.p_trust) & (toe_p >= prm.p_trust)
    if ok.sum() > 30:
        med = np.median(d_at[ok])
        mad = 1.4826 * np.median(np.abs(d_at[ok] - med)) + 1e-6
        kill(np.isfinite(d_at) & (np.abs(d_at - med) > max(4 * mad, 0.25 * L_ref)), "anat_break")
    # телепорт: физически невозможный скачок за кадр
    jump = np.full(n, np.nan)
    jump[1:] = np.hypot(np.diff(toe_x), np.diff(toe_y))
    kill(np.isfinite(jump) & (jump > prm.max_jump_frac * L_ref), "teleport")
    return valid, reason


def fill_short_gaps(x: np.ndarray, valid: np.ndarray, prm: Params):
    """Короткие пропуски интерполируем, длинные оставляем дырами. Возвращает (x, usable)."""
    s = pd.Series(np.where(valid, x, np.nan))
    filled = s.interpolate(method="pchip", limit=prm.gap_interp_max,
                           limit_area="inside").values
    usable = np.isfinite(filled)
    # защитные каймы вокруг длинных дыр: производная через шов недостоверна
    holes = ~valid
    long_hole = np.zeros_like(holes)
    i = 0
    while i < len(holes):
        if holes[i]:
            j = i
            while j < len(holes) and holes[j]:
                j += 1
            if (j - i) > prm.gap_interp_max:
                long_hole[max(0, i - prm.guard_frames): min(len(holes), j + prm.guard_frames)] = True
            i = j
        else:
            i += 1
    usable &= ~long_hole
    return filled, usable


# -------------------------------------------------------- эпизоды локомоции --

def find_bouts(rel: np.ndarray, usable: np.ndarray, prm: Params) -> np.ndarray:
    """Крыса стоит большую часть записи. Размечаем только эпизоды реального шага."""
    s = pd.Series(np.where(usable, rel, np.nan))
    std = s.rolling(prm.bout_win, center=True, min_periods=prm.bout_win // 2).std().values
    moving = usable & np.isfinite(std) & (std > prm.bout_std_px)
    bout_id = np.zeros(len(rel), dtype=int)
    bid, i = 0, 0
    while i < len(moving):
        if moving[i]:
            j = i
            while j < len(moving) and moving[j]:
                j += 1
            if (j - i) >= prm.bout_min_frames:
                bid += 1
                bout_id[i:j] = bid
            i = j
        else:
            i += 1
    return bout_id


def estimate_belt_velocity(v: np.ndarray, dead: float = 60.0, bins: int = 181) -> float:
    """
    Скорость ленты = мода "опорного" кластера скоростей.

    Опора занимает ~70% времени и идёт ровно со скоростью ленты, перенос - вдвое
    быстрее и в обратную сторону. Значит опора это самый населённый пик
    гистограммы скоростей, и его положение и есть v_belt.

    БЕЗ ПРИВЯЗКИ К НАПРАВЛЕНИЮ. Прежняя версия брала медиану НИЖНЕЙ половины
    распределения, то есть молча предполагала, что лента везёт носок в -x. На
    камере с другой стороны дорожки крыса на кадре зеркальна, в опоре носок едет
    в +x, и оценка цепляла кластер переноса: на записи 4 давала -28 px/с вместо
    +439. Разметка при этом не падала, а выдавала правдоподобный мусор (57
    "постановок" вместо 244, цикл 685 мс вместо 390). Для двусторонней съёмки,
    где одна из камер всегда зеркальна, это тихая порча данных.

    Проверка: на паре синхронных камер модуль оценки сходится (439 и 447 px/с,
    отношение 0.98), знаки противоположны - как и должно быть при съёмке с
    разных сторон.

    dead - окрестность нуля, которую выбрасываем: паузы, когда крыса стоит на
    ленте, дают большой пик на v~0 и утягивают на себя любую оценку по квантилям.
    """
    v = v[np.isfinite(v)]
    v = v[np.abs(v) > dead]
    if v.size < 50:
        return np.nan
    lo, hi = np.percentile(v, [1, 99])
    if not np.isfinite(lo) or hi - lo < 1e-6:
        return float(np.median(v))
    h, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centre = 0.5 * (edges[:-1] + edges[1:])[int(np.argmax(h))]
    sel = v[np.abs(v - centre) <= (hi - lo) / bins * 3]
    return float(np.median(sel)) if sel.size else float(centre)


# ------------------------------------------------- этап A: детекция фаз по R --

def schmitt_phases(R: np.ndarray, usable: np.ndarray, prm: Params) -> np.ndarray:
    """Гистерезис Шмитта + минимальные длительности -> метка фазы."""
    n = len(R)
    ph = np.array([PHASE_UNKNOWN] * n, dtype=object)
    state = None
    for i in range(n):
        if not usable[i] or not np.isfinite(R[i]):
            state = None
            continue
        if state is None:
            state = PHASE_SWING if R[i] >= prm.r_hi else PHASE_STANCE
        elif state == PHASE_STANCE and R[i] >= prm.r_hi:
            state = PHASE_SWING
        elif state == PHASE_SWING and R[i] <= prm.r_lo:
            state = PHASE_STANCE
        ph[i] = state
    # подавляем слишком короткие фазы (дребезг)
    i = 0
    while i < n:
        if ph[i] in (PHASE_STANCE, PHASE_SWING):
            j = i
            while j < n and ph[j] == ph[i]:
                j += 1
            is_stance = ph[i] == PHASE_STANCE
            need = prm.min_stance if is_stance else prm.min_swing
            cap = prm.max_stance if is_stance else prm.max_swing
            if (j - i) < need or (j - i) > cap:
                ph[i:j] = PHASE_UNKNOWN
            i = j
        else:
            i += 1
    return ph


# ------------------------------------- этап B: уточнение момента события ----

def refine_event(x_raw: np.ndarray, usable: np.ndarray, k: int, prm: Params) -> float:
    """
    Двухсегментная регрессия по НЕсглаженной позиции вокруг перехода k.
    Ищем излом: до него наклон ~ v_belt, после - наклон swing (или наоборот).
    Возвращает уточнённый индекс (float, sub-frame).
    """
    a, b = max(0, k - prm.refine_win), min(len(x_raw), k + prm.refine_win + 1)
    idx = np.arange(a, b)
    m = usable[a:b] & np.isfinite(x_raw[a:b])
    if m.sum() < 8:
        return float(k)
    t, y = idx[m].astype(float), x_raw[a:b][m]
    best_sse, best_bp = np.inf, float(k)
    for bp in range(int(t[0]) + 3, int(t[-1]) - 2):
        left, right = t <= bp, t > bp
        if left.sum() < 3 or right.sum() < 3:
            continue
        sse = 0.0
        for msk in (left, right):
            p = np.polyfit(t[msk], y[msk], 1)
            sse += float(np.sum((y[msk] - np.polyval(p, t[msk])) ** 2))
        if sse < best_sse:
            best_sse, best_bp = sse, float(bp)
    return best_bp


def extract_events(ph: np.ndarray, x_raw: np.ndarray, usable: np.ndarray, prm: Params):
    """Переходы фаз -> события touch_down / lift_off с уточнённым таймингом."""
    events = []
    for i in range(1, len(ph)):
        prev, cur = ph[i - 1], ph[i]
        if prev == cur or PHASE_UNKNOWN in (prev, cur):
            continue
        name = "touch_down" if cur == PHASE_STANCE else "lift_off"
        bias_ms = prm.td_bias_ms if name == "touch_down" else prm.lo_bias_ms
        shift = (bias_ms / 1000.0 * prm.fps) if prm.apply_bias else 0.0
        events.append({"frame": i, "event": name,
                       "frame_refined": refine_event(x_raw, usable, i, prm),
                       "frame_corrected": i + shift})
    return events


def rebuild_phases_from_events(ph: np.ndarray, events: list, bout_id: np.ndarray,
                               usable: np.ndarray) -> np.ndarray:
    """
    ЭТАП B, вторая половина: пересобрать метки фаз по УТОЧНЁННЫМ границам событий.

    Гистерезис (этап A) срабатывает с задержкой в 1-2 кадра на каждом фронте и
    систематически подрезает swing, завышая duty factor. Уточнённые события
    (излом в позиционной области) свободны от этого сдвига, поэтому финальные
    метки строим именно по ним: [touch_down, lift_off) = stance,
    [lift_off, touch_down) = swing.
    """
    if not events:
        return ph
    out = np.array([PHASE_UNKNOWN] * len(ph), dtype=object)
    ev = sorted(events, key=lambda e: e["frame_refined"])
    for a, b in zip(ev, ev[1:]):
        if a["event"] == b["event"]:
            continue                                  # разрыв (между ними unknown)
        i0 = int(np.ceil(a["frame_refined"]))
        i1 = int(np.ceil(b["frame_refined"]))
        if i1 <= i0 or i1 - i0 > 200:
            continue
        if bout_id[i0] == 0 or bout_id[i0] != bout_id[max(i0, i1 - 1)]:
            continue                                  # интервал вне эпизода/через границу
        phase = PHASE_STANCE if a["event"] == "touch_down" else PHASE_SWING
        seg = slice(i0, i1)
        out[seg] = np.where(usable[seg], phase, PHASE_UNKNOWN)
    return out


# ----------------------------------------------------------------- пайплайн --

@dataclass
class LegResult:
    """Результат разметки одной ноги."""
    leg: str
    phase: np.ndarray
    events: list
    bout_id: np.ndarray
    R: np.ndarray
    valid: np.ndarray
    reason: np.ndarray
    toe_p: np.ndarray
    belt: dict
    L_ref: float
    body_name: str
    quality: dict = field(default_factory=dict)


def phase_runs_ms(phase: np.ndarray, name: str, fps: float) -> np.ndarray:
    runs, cur = [], 0
    for p in phase:
        if p == name:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return np.array(runs) / fps * 1000.0


def cycles_from_events(events: list, fps: float):
    """Полные циклы td -> lo -> td. Возвращает (длительности мс, duty)."""
    if not events:
        return np.array([]), np.array([])
    ev = sorted(events, key=lambda e: e["frame_corrected"])
    dur, duty = [], []
    for i in range(len(ev) - 2):
        a, b, c = ev[i], ev[i + 1], ev[i + 2]
        if a["event"] == "touch_down" and b["event"] == "lift_off" and c["event"] == "touch_down":
            cyc = c["frame_corrected"] - a["frame_corrected"]
            if 0.15 * fps < cyc < 1.2 * fps:
                dur.append(cyc / fps * 1000.0)
                duty.append((b["frame_corrected"] - a["frame_corrected"]) / cyc)
    return np.array(dur), np.array(duty)


def assess_quality(res: "LegResult", prm: Params) -> dict:
    """
    Физиологическая проверка результата по ноге.

    Длительность swing у крысы почти инвариантна (110-130 мс) и ВСЕГДА меньше
    stance. Нарушение этого означает, что размечалась перекрытая телом нога, а
    не реальная походка: на 12 видео мы наблюдали swing 200-210 мс и stance
    МЕНЬШЕ swing, что физически невозможно. Такие ноги отбраковываются, иначе
    мусор молча попадёт в обучающую выборку.
    """
    n = len(res.phase)
    sw = phase_runs_ms(res.phase, PHASE_SWING, prm.fps)
    st = phase_runs_ms(res.phase, PHASE_STANCE, prm.fps)
    dur, duty = cycles_from_events(res.events, prm.fps)
    labeled = int(np.sum((res.phase == PHASE_STANCE) | (res.phase == PHASE_SWING)))
    m = {
        "cycles": int(dur.size),
        "labeled": labeled,
        "labeled_frac": labeled / max(1, n),
        "swing_ms": float(np.median(sw)) if sw.size else float("nan"),
        "stance_ms": float(np.median(st)) if st.size else float("nan"),
        "cycle_ms": float(np.median(dur)) if dur.size else float("nan"),
        "duty": float(np.median(duty)) if duty.size else float("nan"),
    }
    bad = []
    if m["cycles"] < prm.q_min_cycles:
        bad.append(f"циклов {m['cycles']} < {prm.q_min_cycles}")
    if m["labeled_frac"] < prm.q_min_labeled_frac:
        bad.append(f"размечено {100 * m['labeled_frac']:.1f}% < {100 * prm.q_min_labeled_frac:.0f}%")
    if np.isfinite(m["swing_ms"]) and not (prm.q_swing_ms[0] <= m["swing_ms"] <= prm.q_swing_ms[1]):
        bad.append(f"swing {m['swing_ms']:.0f} мс вне {prm.q_swing_ms[0]:.0f}-{prm.q_swing_ms[1]:.0f}")
    if np.isfinite(m["stance_ms"]) and np.isfinite(m["swing_ms"]) and m["stance_ms"] <= m["swing_ms"]:
        bad.append(f"stance {m['stance_ms']:.0f} <= swing {m['swing_ms']:.0f} (физически невозможно)")
    if np.isfinite(m["duty"]) and not (prm.q_duty[0] <= m["duty"] <= prm.q_duty[1]):
        bad.append(f"duty {m['duty']:.2f} вне {prm.q_duty[0]:.2f}-{prm.q_duty[1]:.2f}")
    m["verdict"] = "ok" if not bad else "rejected"
    m["reasons"] = bad
    return m


def process_leg(df: pd.DataFrame, raw_df, leg: str, prm: Params):
    """Полная разметка одной ноги."""
    need = [f"hl_toes_{leg}", f"hl_ankle_{leg}", f"hl_hip_{leg}"]
    if any((b, "x") not in df.columns for b in need):
        return None

    def g(b, c):
        return df[(b, c)].values.astype(float)

    toe_x, toe_y, toe_p = g(f"hl_toes_{leg}", "x"), g(f"hl_toes_{leg}", "y"), g(f"hl_toes_{leg}", "likelihood")
    ank_x, ank_y, ank_p = g(f"hl_ankle_{leg}", "x"), g(f"hl_ankle_{leg}", "y"), g(f"hl_ankle_{leg}", "likelihood")
    hip_x, hip_y, hip_p = g(f"hl_hip_{leg}", "x"), g(f"hl_hip_{leg}", "y"), g(f"hl_hip_{leg}", "likelihood")
    body_x, body_p, body_name = body_ref(df, leg)

    ok = (ank_p >= prm.p_trust) & (hip_p >= prm.p_trust)
    L_ref = float(np.median(np.hypot(hip_x[ok] - ank_x[ok], hip_y[ok] - ank_y[ok]))) if ok.sum() > 30 else 78.0

    valid, reason = validity_mask(toe_x, toe_y, toe_p, ank_x, ank_y, ank_p, body_p, L_ref, prm)
    toe_f, usable = fill_short_gaps(toe_x, valid, prm)
    body_f, body_usable = fill_short_gaps(body_x, body_p >= prm.p_use, prm)
    usable &= body_usable

    x_raw = toe_f
    if raw_df is not None and (f"hl_toes_{leg}", "x") in raw_df.columns:
        cand = raw_df[(f"hl_toes_{leg}", "x")].values.astype(float)
        if len(cand) == len(toe_x):
            x_raw = cand

    bout_id = find_bouts(toe_f - body_f, usable, prm)
    phase = np.array([PHASE_UNKNOWN] * len(toe_x), dtype=object)
    R_all = np.full(len(toe_x), np.nan)
    belt = {}
    k = max(3, prm.smooth_win | 1)
    for bid in range(1, bout_id.max() + 1):
        sel = np.where(bout_id == bid)[0]
        if sel.size < prm.bout_min_frames:
            continue
        s, e = sel[0], sel[-1] + 1
        xs = pd.Series(toe_f[s:e]).rolling(k, center=True, min_periods=1).median().values
        v = np.gradient(xs) * prm.fps
        v_belt = estimate_belt_velocity(v[usable[s:e]])
        if not np.isfinite(v_belt) or abs(v_belt) < 20:
            continue
        belt[bid] = v_belt
        sgn = -1.0 if v_belt > 0 else 1.0
        R = sgn * (v - v_belt) / abs(v_belt)
        R_all[s:e] = R
        phase[s:e] = schmitt_phases(R, usable[s:e], prm)

    events = extract_events(phase, x_raw, usable, prm)
    if prm.phase_from_events:
        phase = rebuild_phases_from_events(
            phase, [dict(ev) for ev in events], bout_id, usable)
    elif prm.apply_bias:
        shifted = [dict(ev, frame_refined=ev["frame_corrected"]) for ev in events]
        phase = rebuild_phases_from_events(phase, shifted, bout_id, usable)

    res = LegResult(leg=leg, phase=phase, events=events, bout_id=bout_id, R=R_all,
                    valid=valid, reason=reason, toe_p=toe_p, belt=belt,
                    L_ref=L_ref, body_name=body_name)
    res.quality = assess_quality(res, prm)
    return res


def interlimb(res_a, res_b, prm: Params) -> dict:
    """
    Межконечностная координация.

    При нормальной чередующейся походке сдвиг фаз между ногами ~0.5 цикла.
    Отклонение от 0.5 и асимметрия длительностей опоры - прямые показатели
    односторонней патологии, что и нужно при гемисекции.
    """
    def tds(r):
        return np.array(sorted(e["frame_corrected"] for e in r.events if e["event"] == "touch_down"))

    A, B = tds(res_a), tds(res_b)
    out = {"phase_lag": float("nan"), "lag_n": 0, "symmetry_index": float("nan")}
    if A.size < 3 or B.size < 3:
        return out
    lags = []
    for i in range(len(A) - 1):
        cyc = A[i + 1] - A[i]
        if not (0.15 * prm.fps < cyc < 1.2 * prm.fps):
            continue
        nxt = B[(B > A[i]) & (B < A[i + 1])]
        if nxt.size:
            lags.append((nxt[0] - A[i]) / cyc)
    if lags:
        out["phase_lag"] = float(np.median(lags))
        out["lag_n"] = len(lags)
    sa = res_a.quality.get("stance_ms", float("nan"))
    sb = res_b.quality.get("stance_ms", float("nan"))
    if np.isfinite(sa) and np.isfinite(sb) and (sa + sb) > 0:
        out["symmetry_index"] = float(abs(sa - sb) / ((sa + sb) / 2) * 100.0)
    return out


def run(csv_path: Path, raw_csv, prm: Params, leg_opt: str, out_prefix: Path):
    """Двусторонняя разметка: обе ноги + проверка правдоподобия + координация."""
    df = load_dlc(csv_path)
    raw_df = load_dlc(raw_csv) if (raw_csv is not None and Path(raw_csv).exists()) else None
    legs = ("l", "r") if leg_opt in ("auto", "both") else (leg_opt,)

    results = {}
    for leg in legs:
        r = process_leg(df, raw_df, leg, prm)
        if r is not None:
            results[leg] = r
    if not results:
        raise SystemExit("не удалось разметить ни одну ногу")

    n = len(df)
    out = pd.DataFrame({"frame": np.arange(n), "time_s": np.round(np.arange(n) / prm.fps, 4)})
    ev_rows = []
    for leg, r in results.items():
        rejected = r.quality["verdict"] != "ok"
        ph = r.phase.copy()
        if rejected and not prm.keep_rejected:
            ph = np.array([PHASE_UNKNOWN] * n, dtype=object)
        ev_col = np.array([""] * n, dtype=object)
        if (not rejected) or prm.keep_rejected:
            for e in r.events:
                ev_col[e["frame"]] = e["event"]
                ev_rows.append({"leg": leg, "frame": e["frame"], "event": e["event"],
                                "frame_corrected": e["frame_corrected"],
                                "frame_refined": e["frame_refined"],
                                "time_s": round(e["frame_corrected"] / prm.fps, 4)})
        out[f"phase_{leg}"] = ph
        out[f"event_{leg}"] = ev_col
        out[f"R_{leg}"] = np.round(r.R, 4)
        out[f"bout_{leg}"] = r.bout_id
        out[f"toe_p_{leg}"] = np.round(r.toe_p, 4)
        out[f"quality_{leg}"] = r.quality["verdict"]

    inter = {}
    good = [leg for leg, r in results.items() if r.quality["verdict"] == "ok"]
    if len(good) == 2:
        inter = interlimb(results[good[0]], results[good[1]], prm)

    out_csv = out_prefix.with_suffix(".phases.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8")
    ev_df = pd.DataFrame(ev_rows)
    ev_csv = out_prefix.with_suffix(".events.csv")
    ev_df.to_csv(ev_csv, index=False, encoding="utf-8")

    print("=" * 74)
    print(f"кадров: {n} ({n / prm.fps:.1f} с)")
    for leg, r in results.items():
        q = r.quality
        mark = "OK" if q["verdict"] == "ok" else "ОТБРАКОВАНА"
        print(f"\nнога {leg.upper()} [{mark}]  опора тела: {r.body_name}, L={r.L_ref:.1f} px")
        print(f"   циклов {q['cycles']}, размечено {q['labeled']} ({100 * q['labeled_frac']:.1f}%), "
              f"эпизодов {int(r.bout_id.max())}")
        print(f"   duty {q['duty']:.3f} | цикл {q['cycle_ms']:.0f} мс | "
              f"swing {q['swing_ms']:.0f} мс | stance {q['stance_ms']:.0f} мс")
        for why in q["reasons"]:
            print(f"   причина отбраковки: {why}")
    if inter:
        print(f"\nкоординация: сдвиг фаз {inter['phase_lag']:.3f} цикла (норма ~0.5, "
              f"n={inter['lag_n']}), асимметрия stance {inter['symmetry_index']:.1f}%")
    elif len(results) > 1:
        print("\nкоординация не считалась: пригодна только одна нога")
    print(f"\nзаписано: {out_csv}\n          {ev_csv}")
    print("=" * 74)
    return out, ev_df


def main(argv=None):
    ap = argparse.ArgumentParser(description="Двусторонняя авторазметка stance/swing на тредмиле")
    ap.add_argument("--csv", required=True, type=Path, help="DLC CSV (лучше _filtered)")
    ap.add_argument("--raw-csv", type=Path, default=None, help="несглаженный DLC CSV для тайминга")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--leg", choices=["auto", "both", "l", "r"], default="both")
    ap.add_argument("--r-hi", type=float, default=None)
    ap.add_argument("--r-lo", type=float, default=None)
    ap.add_argument("--no-bias", action="store_true", help="отключить калибровку запаздывания")
    ap.add_argument("--phase-from-events", action="store_true")
    ap.add_argument("--keep-rejected", action="store_true",
                    help="не стирать метки ноги, не прошедшей проверку правдоподобия")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    prm = Params(fps=a.fps)
    if a.r_hi is not None:
        prm.r_hi = a.r_hi
    if a.r_lo is not None:
        prm.r_lo = a.r_lo
    prm.phase_from_events = bool(a.phase_from_events)
    prm.apply_bias = not bool(a.no_bias)
    prm.keep_rejected = bool(a.keep_rejected)
    run(a.csv, a.raw_csv, prm, a.leg, a.out or a.csv.with_suffix(""))


if __name__ == "__main__":
    sys.exit(main())
