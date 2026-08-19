"""
Где геометрия обязана выиграть у таймера: когда события начинают теряться.

ЧТО ВЫЯСНИЛОСЬ РАНЬШЕ. На интактной крысе на тредмиле геометрический процент
(доля пройденного пути) и временной (доля прошедшего времени) расходятся не
более чем на 5%, и разброс позы в момент поджига у них одинаковый. Причина
физическая: в опоре лента тащит стопу с ПОСТОЯННОЙ скоростью, поэтому равные
интервалы времени это равные отрезки пути, тождественно. Выбор определения на
интакте ничего не меняет.

ТОГДА В ЧЁМ РАЗНИЦА. В том, что нужно каждому методу В МОМЕНТ РАБОТЫ:
  таймер    - свежее событие КАЖДЫЙ цикл. Нет отрыва - нет импульса;
  геометрия - только эталонная петля. Она строится раз в N циклов, а дальше
              процент считается проекцией текущей точки, событий не требуется.

Это и есть смысл предложения Горского: развязать оценку фазы и детекцию событий.
На интакте выгоды нет, потому что события детектируются с recall 99%. После
гемисекции шаг разваливается, события начинают теряться, и вот тогда разница
становится решающей.

Здесь мы это моделируем: выбиваем долю событий и смотрим, что происходит с
каждым методом.

Запуск:
    python phase_robustness.py --csv <DLC_filtered.csv> --events <*.events.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gait_phase_labeler as G  # noqa: E402

NPTS = 100
REF_CYCLES = 10


def cycles_of(td, lo):
    td, lo = np.asarray(sorted(td), float), np.asarray(sorted(lo), float)
    out = []
    for i in range(len(td) - 1):
        a, c = td[i], td[i + 1]
        mid = lo[(lo > a) & (lo < c)]
        if mid.size != 1:
            continue
        b = mid[0]
        ia, ib, ic = int(round(a)), int(round(b)), int(round(c))
        if 10 < ib - ia < 200 and 5 < ic - ib < 200:
            out.append((ia, ib, ic))
    return out


def truth_percent(n, cyc):
    pct, cid = np.full(n, np.nan), np.full(n, -1)
    for k, (ia, ib, ic) in enumerate(cyc):
        pct[ia:ib] = 100.0 * (np.arange(ia, ib) - ia) / (ib - ia)
        pct[ib:ic] = 100.0 + 100.0 * (np.arange(ib, ic) - ib) / (ic - ib)
        cid[ia:ic] = k
    return pct, cid


def cycle_blocks(relx, vel, cyc):
    """Каждый цикл -> петля из 2*NPTS точек. Ресемплинг по времени внутри фазы."""
    out = []
    for ia, ib, ic in cyc:
        seg = []
        for s, e in ((ia, ib), (ib, ic)):
            src = np.linspace(0, 1, e - s)
            dst = np.linspace(0, 1, NPTS, endpoint=False)
            seg.append(np.column_stack([np.interp(dst, src, relx[s:e]),
                                        np.interp(dst, src, vel[s:e])]))
        blk = np.vstack(seg)
        out.append(blk if np.isfinite(blk).all() else None)
    return out


def build_loop(relx, vel, cyc):
    """Замороженный эталон: первые REF_CYCLES циклов, дальше не обновляется."""
    acc = [b for b in cycle_blocks(relx, vel, cyc[:REF_CYCLES]) if b is not None]
    return np.median(np.stack(acc), axis=0) if acc else None


def rolling_loops(relx, vel, cyc, n_frames):
    """
    Скользящий эталон: после КАЖДОГО завершённого цикла пересобираем петлю по
    последним REF_CYCLES циклам. Возвращает (петли, карта кадр -> индекс петли).

    Событий требует только сборка петли (нужны границы опора/перенос), и не
    каждый цикл: если часть событий потеряна, петля просто соберётся по тем
    циклам, что распознались. Сама оценка процента в рантайме событий не просит.
    """
    blocks = cycle_blocks(relx, vel, cyc)
    loops, which = [], np.full(n_frames, -1, dtype=int)
    have = []
    for k, (ia, ib, ic) in enumerate(cyc):
        if blocks[k] is not None:
            have.append(blocks[k])
            if len(have) > REF_CYCLES:
                have.pop(0)
        if len(have) < 3:
            continue
        loops.append(np.median(np.stack(have), axis=0))
        # эта петля действует со следующего цикла и до появления следующей
        nxt = cyc[k + 1][2] if k + 1 < len(cyc) else n_frames
        which[ic:nxt] = len(loops) - 1
    return loops, which


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--target", type=float, default=145.0)
    ap.add_argument("--latency-ms", type=float, default=28.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rolling", action="store_true")
    ap.add_argument("--eval-from", type=int, default=0,
                    help="считать попадания только с этого кадра (для честной сверки с потоком)")
    ap.add_argument("--eval-to", type=int, default=0, help="0 = до конца")
    a = ap.parse_args(argv)

    df = G.load_dlc(a.csv)
    tx = df[(f"hl_toes_{a.leg}", "x")].values.astype(float)
    ix = df[(f"hl_iliac_{a.leg}", "x")].values.astype(float)
    relx = tx - ix
    vel = np.gradient(pd.Series(relx).rolling(3, min_periods=1).median().values) * a.fps

    ev = pd.read_csv(a.events)
    col = "frame_corrected" if "frame_corrected" in ev.columns else "frame"
    if "leg" in ev.columns:
        ev = ev[ev.leg == a.leg]
    td = ev.loc[ev.event == "touch_down", col].values.astype(float)
    lo = ev.loc[ev.event == "lift_off", col].values.astype(float)
    cyc = cycles_of(td, lo)
    n = len(relx)
    truth, cid = truth_percent(n, cyc)
    lat = int(round(a.latency_ms / 1000.0 * a.fps))

    loop = build_loop(relx, vel, cyc)
    sc = np.array([np.ptp(loop[:, 0]), np.ptp(loop[:, 1])])
    R = loop / sc
    start = cyc[REF_CYCLES][2] if len(cyc) > REF_CYCLES else n

    r_loops, r_which = rolling_loops(relx, vel, cyc, n)
    Rs = [L / np.array([np.ptp(L[:, 0]), np.ptp(L[:, 1])]) for L in r_loops]
    scs = [np.array([np.ptp(L[:, 0]), np.ptp(L[:, 1])]) for L in r_loops]

    # ---- каузальная геометрическая фаза: НИ ОДНОГО события в рантайме -------
    def project_all(rolling: bool):
        e = np.full(n, np.nan)
        for i in range(start, n):
            if rolling:
                w = r_which[i]
                if w < 0:
                    continue
                Ri, si = Rs[w], scs[w]
            else:
                Ri, si = R, sc
            c = np.array([relx[i], vel[i]]) / si
            if np.isfinite(c).all():
                e[i] = float(np.argmin(((c - Ri) ** 2).sum(1))) * (200.0 / len(Ri))
        return e

    est_frozen = project_all(False)
    est_rolling = project_all(True)
    est = est_rolling if a.rolling else est_frozen

    # скорость движения ВДОЛЬ ПЕТЛИ, раздельно для опоры и переноса,
    # оценивается по самой же оценке фазы - события не нужны
    def rates_of(e):
        rate = {0: [], 1: []}
        hist_r = np.full((n, 2), np.nan)
        for i in range(start + 1, n):
            p0, p1 = e[i - 1], e[i]
            if np.isfinite(p0) and np.isfinite(p1):
                d = (p1 - p0 + 100.0) % 200.0 - 100.0
                if 0 < d < 40:                      # разумный шаг вперёд
                    k = 0 if p1 < 100.0 else 1
                    rate[k].append(d)
                    if len(rate[k]) > 200:
                        rate[k].pop(0)
            for k in (0, 1):
                hist_r[i, k] = np.median(rate[k]) if len(rate[k]) > 20 else np.nan
        return hist_r

    def extrapolate(p, k, r_st, r_sw):
        left = float(k)
        while left > 0:
            r = r_st if p < 100.0 else r_sw
            if not np.isfinite(r) or r <= 0:
                return np.nan
            edge = 100.0 if p < 100.0 else 200.0
            need = (edge - p) / r
            if need > left:
                return p + r * left
            left -= need
            p = edge % 200.0
        return p

    def fire_geometry(e, hist_r):
        out, fired = [], set()
        for i in range(start, n - lat):
            c = cid[i + lat]
            if c < 0 or c in fired or not np.isfinite(e[i]):
                continue
            p = extrapolate(e[i], lat, hist_r[i, 0], hist_r[i, 1])
            # Поджиг на ПОЛШАГА раньше пересечения. На 100 Гц один кадр в
            # переносе это ~10% фазы, и правило "первый кадр, где прогноз
            # перевалил за цель" систематически перелетает на полкадра, то есть
            # на +5%. Центрируем: жжём, когда цель ближе к текущему прогнозу,
            # чем к следующему.
            r = hist_r[i, 1] if (np.isfinite(e[i]) and e[i] >= 100.0) else hist_r[i, 0]
            half = 0.5 * r if np.isfinite(r) else 0.0
            if np.isfinite(p) and a.target - half <= p < a.target + 40:
                fired.add(c)
                j = i + lat
                if j < a.eval_from or (a.eval_to and j >= a.eval_to):
                    continue                      # вне окна сверки
                if np.isfinite(truth[j]):
                    out.append(truth[j] - a.target)
        return np.asarray(out), len(fired)

    def fire_timer(keep_mask):
        """Таймер: ждёт отрыв. Если событие потеряно, импульса в цикле нет."""
        out, fired = [], set()
        sw = np.median([ic - ib for ia, ib, ic in cyc])
        for k, (ia, ib, ic) in enumerate(cyc):
            if ib < start or not keep_mask[k]:
                continue
            f = max(ib + lat, int(round(ib + (a.target - 100.0) / 100.0 * sw)))
            if f >= n or cid[f] < 0:
                continue
            fired.add(cid[f])
            if np.isfinite(truth[f]):
                out.append(truth[f] - a.target)
        return np.asarray(out), len(fired)

    rng = np.random.default_rng(a.seed)
    n_cyc = len(cyc) - REF_CYCLES
    ef, nf_ = fire_geometry(est_frozen, rates_of(est_frozen))
    eg, ng = fire_geometry(est_rolling, rates_of(est_rolling))

    print(f"циклов после эталона: {n_cyc}, цель {a.target:.0f}%, "
          f"задержка {a.latency_ms:.0f} мс\n")
    print(f"{'потеря событий':>16} | {'ТАЙМЕР':>28} | {'ГЕОМ. заморож.':>28} | "
          f"{'ГЕОМ. скользящая':>28}")
    print(f"{'':>16} | {'смещ.':>8} {'p90':>7} {'имп.':>10} | "
          f"{'смещ.':>8} {'p90':>7} {'имп.':>10} | {'смещ.':>8} {'p90':>7} {'имп.':>10}")
    print("-" * 118)
    for drop in (0.0, 0.1, 0.25, 0.5, 0.75):
        keep = rng.random(len(cyc)) >= drop
        et, nt = fire_timer(keep)
        line = f"{100*drop:>15.0f}% |"
        for e, nf in ((et, nt), (ef, nf_), (eg, ng)):
            if len(e) < 5:
                line += f" {'нет импульсов':>28} |"
                continue
            line += (f" {np.median(e):>+7.1f}% {np.percentile(np.abs(e),90):>6.1f}% "
                     f"{100*nf/n_cyc:>9.0f}% |")
        print(line)
    print("\nгеометрия не зависит от потери событий: её колонка постоянна по построению")
    print("импульсов = доля циклов, в которые удалось положить импульс")


if __name__ == "__main__":
    main()
