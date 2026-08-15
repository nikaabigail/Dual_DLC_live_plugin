"""
Геометрический процент фазы против временного: что надёжнее возвращает в одну
и ту же точку движения.

ЗАЧЕМ. Прошлое сравнение было нечестным: эталонный процент я определял как долю
ПРОШЕДШЕГО ВРЕМЕНИ внутри фазы, и с ним же сравнивал таймер. Таймер выиграл по
построению. Геометрия при временном эталоне обязана показывать расхождение
всюду, где носок идёт по пути неравномерно, а он идёт неравномерно всегда.

ЧЕСТНЫЙ КРИТЕРИЙ. Смысл задавать процент в том, чтобы импульс каждый раз
попадал в ОДНО И ТО ЖЕ состояние конечности. Значит мерить надо так: зажигаем на
фиксированном проценте много шагов подряд и смотрим, насколько РАЗБРОСАНА
реальная поза ноги в момент поджига. Меньше разброс - лучше определение. Ни одно
из определений при этом не считается истиной.

ДВА ОПРЕДЕЛЕНИЯ (оба по одной и той же усреднённой петле):
  время     - процент это доля прошедшего времени внутри фазы;
  геометрия - процент это доля ПРОЙДЕННОГО ПУТИ вдоль петли (длина дуги).
Они не совпадают: в опоре носок ползёт медленно по длинной дуге, в переносе
проскакивает большой путь быстро.

Запуск:
    python phase_geometry_vs_time.py --csv <DLC_filtered.csv> --events <*.events.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gait_phase_labeler as G  # noqa: E402

NPTS = 100          # точек на фазу в эталонной петле


def cycles_of(td, lo):
    """Список (постановка, отрыв, следующая постановка) в кадрах."""
    td = np.asarray(sorted(td), float)
    lo = np.asarray(sorted(lo), float)
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


def build_loop(relx, rely, cyc):
    """Средняя петля: опора и перенос ресемплируются по NPTS точек (по времени)."""
    acc = []
    for ia, ib, ic in cyc:
        seg = []
        for s, e in ((ia, ib), (ib, ic)):
            src = np.linspace(0, 1, e - s)
            dst = np.linspace(0, 1, NPTS, endpoint=False)
            seg.append(np.column_stack([np.interp(dst, src, relx[s:e]),
                                        np.interp(dst, src, rely[s:e])]))
        blk = np.vstack(seg)
        if np.isfinite(blk).all():
            acc.append(blk)
    return (np.median(np.stack(acc), axis=0), len(acc)) if acc else (None, 0)


def arclen_percent(loop):
    """
    Процент по ДЛИНЕ ПУТИ. Внутри опоры путь распределяется на 0..100%,
    внутри переноса на 100..200%: соглашение о границах фаз сохраняем,
    меняем только правило внутри фазы.
    """
    pct = np.zeros(len(loop))
    for k, (s, e, base) in enumerate(((0, NPTS, 0.0), (NPTS, 2 * NPTS, 100.0))):
        seg = loop[s:e]
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        pct[s:e] = base + 100.0 * d / d[-1] if d[-1] > 0 else base
    return pct


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--targets", default="120,135,145,160,180,30,50,70")
    a = ap.parse_args(argv)

    df = G.load_dlc(a.csv)
    tx = df[(f"hl_toes_{a.leg}", "x")].values.astype(float)
    ty = df[(f"hl_toes_{a.leg}", "y")].values.astype(float)
    ix = df[(f"hl_iliac_{a.leg}", "x")].values.astype(float)
    iy = df[(f"hl_iliac_{a.leg}", "y")].values.astype(float)
    relx, rely = tx - ix, ty - iy

    ev = pd.read_csv(a.events)
    col = "frame_corrected" if "frame_corrected" in ev.columns else "frame"
    if "leg" in ev.columns:
        ev = ev[ev.leg == a.leg]
    td = ev.loc[ev.event == "touch_down", col].values.astype(float)
    lo = ev.loc[ev.event == "lift_off", col].values.astype(float)
    cyc = cycles_of(td, lo)

    loop, n_used = build_loop(relx, rely, cyc)
    if loop is None:
        raise SystemExit("не удалось построить петлю")
    geo = arclen_percent(loop)
    tim = np.r_[np.linspace(0, 100, NPTS, endpoint=False),
                np.linspace(100, 200, NPTS, endpoint=False)]

    print(f"петля по {n_used} циклам, циклов всего {len(cyc)}")
    print(f"размах петли: X {np.ptp(loop[:,0]):.0f} px, Y {np.ptp(loop[:,1]):.0f} px")
    d_st = np.sum(np.hypot(*np.diff(loop[:NPTS], axis=0).T))
    d_sw = np.sum(np.hypot(*np.diff(loop[NPTS:], axis=0).T))
    print(f"длина пути: опора {d_st:.0f} px, перенос {d_sw:.0f} px\n")

    print("НАСКОЛЬКО РАСХОДЯТСЯ ОПРЕДЕЛЕНИЯ (одна и та же точка петли)")
    dd = geo - tim
    for name, s, e in (("опора", 0, NPTS), ("перенос", NPTS, 2 * NPTS)):
        seg = dd[s:e]
        print(f"  {name:8s}: медиана {np.median(seg):+6.1f}%, "
              f"максимум {seg[np.argmax(np.abs(seg))]:+6.1f}%")
    print()

    targets = [float(x) for x in a.targets.split(",")]
    print(f"{'цель':>6} | {'ПО ВРЕМЕНИ':>27} | {'ПО ГЕОМЕТРИИ':>27}")
    print(f"{'':>6} | {'разброс позы':>14} {'разброс t':>12} | "
          f"{'разброс позы':>14} {'разброс t':>12}")
    print("-" * 70)

    for T in targets:
        row = f"{T:>5.0f}% |"
        for pct_map in (tim, geo):
            # для каждого цикла: кадр, где определение впервые достигает T
            j = int(np.argmin(np.abs(pct_map - T)))
            frac = (j % NPTS) / NPTS          # доля фазы по ЭТОМУ определению
            in_swing = j >= NPTS
            xs, ys, ts = [], [], []
            for ia, ib, ic in cyc:
                s, e = (ib, ic) if in_swing else (ia, ib)
                f = s + frac * (e - s)
                k = int(round(f))
                if 0 <= k < len(relx) and np.isfinite(relx[k]) and np.isfinite(rely[k]):
                    xs.append(relx[k]); ys.append(rely[k])
                    ts.append((k - ib) / a.fps * 1000.0)   # мс от отрыва
            if len(xs) < 10:
                row += f" {'мало данных':>27} |"
                continue
            xs, ys, ts = np.array(xs), np.array(ys), np.array(ts)
            # разброс позы = средний радиус облака точек носка вокруг медианы
            r = np.hypot(xs - np.median(xs), ys - np.median(ys))
            row += f" {np.percentile(r,90):>11.1f} px {np.percentile(np.abs(ts-np.median(ts)),90):>9.0f} мс |"
        print(row)

    print("\nразброс позы = p90 отклонения носка от медианной точки, пиксели")
    print("разброс t    = p90 отклонения момента от медианного, мс от отрыва")
    print("\nМЕНЬШЕ РАЗБРОС ПОЗЫ = определение надёжнее возвращает в ту же точку движения")


if __name__ == "__main__":
    main()
