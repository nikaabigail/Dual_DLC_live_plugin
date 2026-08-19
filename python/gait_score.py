"""
Счёт качества походки: одно число за пробу, которое максимизирует оптимизатор.

ЗАЧЕМ. Байесовской оптимизации нужен скаляр после каждой пробы. Сумму ЭМГ
брать нельзя (июльский разбор параметров): она растёт и от бесполезной
коактивации. Здесь счёт строится по КИНЕМАТИКЕ, которая у нас и так считается.

ЛОГИКА. Мы знаем, как выглядит нормальный шаг, потому что ИЗМЕРИЛИ его на
интактных животных: межконечностный сдвиг 0.486 цикла, асимметрия опоры 3.4%,
перенос 105 мс. Счёт - это мера близости к этой норме. Чем ближе, тем лучше
настройка стимуляции.

ЕДИНИЦЫ. Всё нормируется на длину сегмента бедро-голеностоп (L_ref), которую
уже считает gait_phase_labeler. Поэтому счёт не зависит от масштаба камеры и
сравним между записями и животными. В миллиметры не переводим: калибровки нет,
а для оптимизатора важен только порядок, а не абсолют.

ЧЕГО СЧЁТ НЕ ДЕЛАЕТ. Не заменяет ЭМГ-метрики из июльского документа
(селективность, коактивация антагонистов, латентности). Он про результат
движения, а не про то, какие мышцы включились. Когда ЭМГ появится в контуре,
его слагаемые добавляются сюда же с весами.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# норма, измеренная на интактных животных
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IntactNorm:
    """
    Референс интактной походки. Значения измерены на записи 4 (синхронная пара
    камер, 238 циклов), см. docs/REPORT_2026-08-19.md.

    clearance и step_length заданы в долях L_ref и уточняются по своим
    записям функцией measure_norm().
    """

    interlimb_phase: float = 0.486        # доля цикла между ногами
    stance_asymmetry: float = 0.034       # |L-R| / среднее
    swing_ms: float = 105.0
    swing_sd_ms: float = 15.0
    clearance_lref: float = 0.20          # подъём носка над опорой, доли L_ref
    step_length_lref: float = 1.60        # размах носка отн. iliac, доли L_ref
    cycle_cv: float = 0.21                # изменчивость длительности цикла у интакта


@dataclass(frozen=True)
class ScoreWeights:
    """
    Веса слагаемых. Сумма положительных = 1, штрафы отдельно, чтобы счёт
    читался как "доля от нормы минус штрафы".
    """

    clearance: float = 0.30
    step_length: float = 0.25
    interlimb: float = 0.25
    symmetry: float = 0.20
    variability_penalty: float = 0.30
    dose_penalty: float = 0.20


# --------------------------------------------------------------------------- #
# покадровые метрики
# --------------------------------------------------------------------------- #

@dataclass
class GaitMetrics:
    """Сырые кинематические показатели пробы, до сворачивания в счёт."""

    n_cycles: int
    clearance_lref: float
    step_length_lref: float
    swing_ms: float
    stance_ms: float
    cycle_cv: float                       # изменчивость длительности цикла
    step_cv: float                        # изменчивость длины шага
    interlimb_phase: Optional[float] = None
    stance_asymmetry: Optional[float] = None
    cycle_yield: float = 1.0              # доля шагов, распознанных как цикл

    def as_dict(self) -> dict:
        return {k: (None if v is None else float(v)) for k, v in self.__dict__.items()}


def _cycles(td: np.ndarray, lo: np.ndarray) -> list[tuple[int, int, int]]:
    """(постановка, отрыв, следующая постановка) в кадрах."""
    td, lo = np.sort(np.asarray(td, float)), np.sort(np.asarray(lo, float))
    out = []
    for i in range(len(td) - 1):
        a, c = td[i], td[i + 1]
        mid = lo[(lo > a) & (lo < c)]
        if mid.size != 1:
            continue
        ia, ib, ic = int(round(a)), int(round(mid[0])), int(round(c))
        if 10 < ib - ia < 200 and 5 < ic - ib < 200:
            out.append((ia, ib, ic))
    return out


def measure(toe_x: np.ndarray, toe_y: np.ndarray,
            body_x: np.ndarray, body_y: np.ndarray,
            td: Sequence[float], lo: Sequence[float],
            l_ref: float, fps: float = 100.0,
            other_td: Optional[Sequence[float]] = None,
            other_stance_ms: Optional[float] = None) -> GaitMetrics:
    """
    Кинематика одной ноги за пробу.

    toe/body - координаты носка и iliac в пикселях кадра; l_ref - длина
    сегмента бедро-голеностоп в тех же пикселях. other_td нужен только для
    межконечностного сдвига (доступен на синхронной паре камер).
    """
    td_arr = np.asarray(td, float)
    cyc = _cycles(td_arr, np.asarray(lo))
    # Доля шагов, которые вообще удалось распознать как цикл. Если ритм
    # разъехался, часть циклов бракуется - и без этого слагаемого счёт
    # считался бы по выжившим, то есть по самым чистым шагам. Проверка
    # монотонности ловила именно это: при сильном дрожании счёт РОС.
    expected = max(1, len(td_arr) - 1)
    yield_ = float(min(1.0, len(cyc) / expected))
    if len(cyc) < 3 or not np.isfinite(l_ref) or l_ref <= 0:
        raise ValueError(f"мало циклов ({len(cyc)}) или плохой l_ref ({l_ref})")

    relx = np.asarray(toe_x, float) - np.asarray(body_x, float)
    rely = np.asarray(toe_y, float) - np.asarray(body_y, float)

    clear, length, sw, st = [], [], [], []
    for ia, ib, ic in cyc:
        stance_y = rely[ia:ib]
        swing_y = rely[ib:ic]
        if not (np.isfinite(stance_y).any() and np.isfinite(swing_y).any()):
            continue
        # ось Y в кадре растёт ВНИЗ: подъём носка = уменьшение y относительно
        # уровня опоры, поэтому берём медиану опоры минус минимум переноса
        clear.append((np.nanmedian(stance_y) - np.nanmin(swing_y)) / l_ref)
        seg = relx[ia:ic]
        if np.isfinite(seg).any():
            length.append((np.nanmax(seg) - np.nanmin(seg)) / l_ref)
        sw.append((ic - ib) / fps * 1000.0)
        st.append((ib - ia) / fps * 1000.0)

    if len(clear) < 3:
        raise ValueError("не удалось посчитать ни одного полного цикла")

    cyc_len = np.array([ic - ia for ia, _, ic in cyc], float)
    interlimb = None
    if other_td is not None and len(other_td) >= 3:
        o = np.sort(np.asarray(other_td, float))
        lags = []
        for ia, _, ic in cyc:
            nxt = o[(o > ia) & (o < ic)]
            if nxt.size:
                lags.append((nxt[0] - ia) / (ic - ia))
        if lags:
            interlimb = float(np.median(lags))

    asym = None
    if other_stance_ms is not None and np.isfinite(other_stance_ms):
        m = np.median(st)
        if m + other_stance_ms > 0:
            asym = float(abs(m - other_stance_ms) / ((m + other_stance_ms) / 2))

    return GaitMetrics(
        n_cycles=len(clear),
        clearance_lref=float(np.median(clear)),
        step_length_lref=float(np.median(length)) if length else float("nan"),
        swing_ms=float(np.median(sw)),
        stance_ms=float(np.median(st)),
        cycle_cv=float(np.std(cyc_len) / np.mean(cyc_len)),
        step_cv=float(np.std(length) / np.mean(length)) if len(length) > 2 else float("nan"),
        interlimb_phase=interlimb,
        stance_asymmetry=asym,
        cycle_yield=yield_,
    )


# --------------------------------------------------------------------------- #
# свёртка в счёт
# --------------------------------------------------------------------------- #

def _closeness(value: float, target: float, scale: float) -> float:
    """
    Близость к целевому значению, 1 в точке и мягко падает.
    Мягкое падение вместо порога: оптимизатору нужен градиент, а не ступенька.
    """
    if not np.isfinite(value):
        return 0.0
    return float(np.exp(-abs(value - target) / max(scale, 1e-9)))


def _ratio(value: float, target: float) -> float:
    """Доля от нормы, обрезанная сверху: больше нормы не лучше нормы."""
    if not np.isfinite(value) or target <= 0:
        return 0.0
    return float(min(value / target, 1.0))


def score(m: GaitMetrics, norm: IntactNorm = IntactNorm(),
          w: ScoreWeights = ScoreWeights(), dose: float = 0.0) -> float:
    """
    Счёт пробы. Больше - лучше. Примерно 1.0 = походка как у интактного
    животного при нулевой дозе; 0 = полный дефицит.

    dose - нормированная доза из stim_params.normalized_dose(): при равном
    результате побеждает более щадящая стимуляция.
    """
    good = (w.clearance * _ratio(m.clearance_lref, norm.clearance_lref)
            + w.step_length * _ratio(m.step_length_lref, norm.step_length_lref))

    # межконечностный сдвиг: важно отклонение в любую сторону от 0.486
    if m.interlimb_phase is not None:
        good += w.interlimb * _closeness(m.interlimb_phase, norm.interlimb_phase, 0.12)
    if m.stance_asymmetry is not None:
        good += w.symmetry * _closeness(m.stance_asymmetry, norm.stance_asymmetry, 0.15)

    # если второй ноги нет, нормируем на доступные веса, иначе счёт занижен
    available = w.clearance + w.step_length
    if m.interlimb_phase is not None:
        available += w.interlimb
    if m.stance_asymmetry is not None:
        available += w.symmetry
    good = good / available if available > 0 else 0.0

    # Изменчивость штрафуется ОТНОСИТЕЛЬНО нормы, а не в абсолюте: у интакта
    # свой ненулевой разброс (крыса делает паузы), и абсолютный штраф просто
    # сдвигал бы шкалу, почти не реагируя на ухудшение ритма.
    excess = [max(0.0, (v - norm.cycle_cv) / max(norm.cycle_cv, 1e-9))
              for v in (m.cycle_cv, m.step_cv) if np.isfinite(v)]
    var = float(np.mean(excess)) if excess else 0.0

    # Нераспознанный шаг - это тоже дефицит, а не повод его не считать.
    raw = good - w.variability_penalty * var - w.dose_penalty * float(dose)
    return float(raw * m.cycle_yield)


def measure_norm(metrics: Sequence[GaitMetrics]) -> IntactNorm:
    """
    Собрать норму по интактным пробам, а не брать значения по умолчанию.
    Так счёт привязывается к своей установке и своему животному.
    """
    if not metrics:
        raise ValueError("нет проб для оценки нормы")
    med = lambda f: float(np.nanmedian([getattr(x, f) for x in metrics]))  # noqa: E731
    il = [x.interlimb_phase for x in metrics if x.interlimb_phase is not None]
    asym = [x.stance_asymmetry for x in metrics if x.stance_asymmetry is not None]
    return IntactNorm(
        interlimb_phase=float(np.median(il)) if il else IntactNorm.interlimb_phase,
        stance_asymmetry=float(np.median(asym)) if asym else IntactNorm.stance_asymmetry,
        swing_ms=med("swing_ms"),
        swing_sd_ms=float(np.nanstd([x.swing_ms for x in metrics])),
        clearance_lref=med("clearance_lref"),
        step_length_lref=med("step_length_lref"),
        cycle_cv=med("cycle_cv"),
    )
