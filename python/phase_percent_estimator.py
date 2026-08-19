"""
Процент фазы шага через проекцию на усреднённую траекторию.

Задача (постановка О.В. Горского): подавать стимул не «в опору/перенос», а в
конкретную точку цикла, например на 146%. Соглашение: 0..100% - опора,
100..200% - перенос. Бинарный детектор такого не выражает.

Метод:
  1) по последним N циклам строится СРЕДНЯЯ траектория носка относительно iliac
     (относительно тела - чтобы не зависеть от движения ленты и дрейфа крысы);
  2) опора и перенос ресемплируются по 100 точек каждая -> каждой точке эталонной
     петли соответствует свой процент фазы;
  3) текущий процент = процент ближайшей точки эталона.

Проверенные на данных особенности, из-за которых наивная реализация не работает:
  * проекция ПО ОДНОЙ КООРДИНАТЕ неоднозначна: одно значение X встречается за
    цикл дважды (носок проходит ту же точку назад в опоре и вперёд в переносе);
  * размах по X 148 px против 46 px по Y, поэтому без нормировки осей X задавит Y
    и мы фактически вернёмся к одномерному случаю;
  * минимум X совпадает с отрывом (-5 мс), а вот максимум X НЕ совпадает с
    постановкой (разброс 410 мс) - носок достигает крайней передней точки в
    середине переноса. Поэтому границы цикла берутся из детектора событий,
    а не из экстремумов траектории.

Запуск:
    python phase_percent_estimator.py --csv <DLC_filtered.csv> --gt <*.phases.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gait_phase_labeler as G  # noqa: E402

N_STANCE = 100          # точек на опору  -> 0..100%
N_SWING = 100           # точек на перенос -> 100..200%


def build_reference(relx, rely, vel, td, lo, n_cycles=None):
    """Средняя петля: опора и перенос ресемплируются по 100 точек каждая."""
    cycles = []
    td = np.asarray(sorted(td), dtype=float)
    lo = np.asarray(sorted(lo), dtype=float)
    for i in range(len(td) - 1):
        a, c = td[i], td[i + 1]
        mid = lo[(lo > a) & (lo < c)]
        if mid.size != 1:
            continue
        b = mid[0]
        ia, ib, ic = int(round(a)), int(round(b)), int(round(c))
        if not (10 < ib - ia < 200 and 5 < ic - ib < 200):
            continue
        seg = []
        for (s, e, n) in ((ia, ib, N_STANCE), (ib, ic, N_SWING)):
            if e - s < 3:
                seg = []
                break
            src = np.linspace(0, 1, e - s)
            dst = np.linspace(0, 1, n, endpoint=False)
            part = np.column_stack([np.interp(dst, src, relx[s:e]),
                                    np.interp(dst, src, rely[s:e]),
                                    np.interp(dst, src, vel[s:e])])
            seg.append(part)
        if len(seg) == 2 and np.isfinite(np.vstack(seg)).all():
            cycles.append(np.vstack(seg))
    if not cycles:
        return None, 0
    arr = np.stack(cycles)
    if n_cycles:
        arr = arr[-n_cycles:]
    return np.median(arr, axis=0), len(arr)


def gt_percent(n, td, lo):
    """Эталонный процент по событиям: линейно внутри опоры и внутри переноса."""
    pct = np.full(n, np.nan)
    td = np.asarray(sorted(td), dtype=float)
    lo = np.asarray(sorted(lo), dtype=float)
    for i in range(len(td) - 1):
        a, c = td[i], td[i + 1]
        mid = lo[(lo > a) & (lo < c)]
        if mid.size != 1:
            continue
        b = mid[0]
        ia, ib, ic = int(round(a)), int(round(b)), int(round(c))
        if not (10 < ib - ia < 200 and 5 < ic - ib < 200):
            continue
        pct[ia:ib] = 100.0 * (np.arange(ia, ib) - a) / (b - a)
        pct[ib:ic] = 100.0 + 100.0 * (np.arange(ib, ic) - b) / (c - b)
    return pct


def project(ref, cur, cols, scale):
    """Ближайшая точка эталона -> её процент. cols - какие столбцы используем."""
    R = ref[:, cols] / scale[cols]
    C = cur[:, cols] / scale[cols]
    out = np.full(len(C), np.nan)
    ok = np.isfinite(C).all(axis=1)
    if ok.any():
        d = ((C[ok][:, None, :] - R[None, :, :]) ** 2).sum(-1)
        out[ok] = np.argmin(d, axis=1) * (200.0 / len(R))
    return out


def circ_err(a, b, period=200.0):
    d = (a - b + period / 2) % period - period / 2
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--n-cycles", type=int, default=0, help="строить эталон по последним N циклам (0 = все)")
    a = ap.parse_args(argv)

    df = G.load_dlc(a.csv)
    tx = df[(f"hl_toes_{a.leg}", "x")].values.astype(float)
    ty = df[(f"hl_toes_{a.leg}", "y")].values.astype(float)
    ix = df[(f"hl_iliac_{a.leg}", "x")].values.astype(float)
    iy = df[(f"hl_iliac_{a.leg}", "y")].values.astype(float)
    relx, rely = tx - ix, ty - iy
    vel = np.gradient(pd.Series(relx).rolling(3, min_periods=1).median().values) * a.fps

    ev = pd.read_csv(a.gt.with_suffix("").with_suffix("").parent /
                     (a.gt.stem.replace(".phases", "") + ".events.csv"))
    col = "frame_corrected" if "frame_corrected" in ev.columns else "frame"
    if "leg" in ev.columns:
        ev = ev[ev.leg == a.leg]
    td = ev.loc[ev.event == "touch_down", col].values
    lo = ev.loc[ev.event == "lift_off", col].values

    ref, n_used = build_reference(relx, rely, vel, td, lo, a.n_cycles or None)
    if ref is None:
        raise SystemExit("не удалось построить эталонную траекторию")
    gtp = gt_percent(len(relx), td, lo)
    valid = np.isfinite(gtp)
    print(f"эталон построен по {n_used} циклам, точек в петле {len(ref)}")
    print(f"кадров с эталонным процентом: {valid.sum()}\n")

    cur = np.column_stack([relx, rely, vel])
    rng = np.array([np.ptp(ref[:, 0]), np.ptp(ref[:, 1]), np.ptp(ref[:, 2])])
    print(f"размах эталона: X {rng[0]:.0f} px, Y {rng[1]:.0f} px, скорость {rng[2]:.0f} px/с")
    ones = np.ones(3)

    variants = [
        ("только X (1D)", [0], ones),
        ("X+Y без нормировки", [0, 1], ones),
        ("X+Y с нормировкой", [0, 1], rng),
        ("X+скорость, норм.", [0, 2], rng),
        ("X+Y+скорость, норм.", [0, 1, 2], rng),
    ]
    print(f"\n{'вариант':<24} {'|ошибка| медиана':>17} {'p90':>8} {'в мс':>8} {'скачков назад':>15}")
    print("-" * 78)
    for name, cols, sc in variants:
        est = project(ref, cur, np.array(cols), sc)
        e = circ_err(est[valid], gtp[valid])
        med, p90 = np.median(np.abs(e)), np.percentile(np.abs(e), 90)
        # 100% фазы = половина цикла; переводим в мс через медианный цикл
        cyc_ms = np.median(np.diff(sorted(td))) / a.fps * 1000
        ms = med / 200.0 * cyc_ms
        d = np.diff(est[valid])
        back = 100 * np.mean(circ_err(d, 0) < -2)
        print(f"{name:<24} {med:>15.1f}% {p90:>7.1f}% {ms:>7.1f} {back:>14.1f}%")
    print(f"\n(медианный цикл {cyc_ms:.0f} мс; 1% фазы = {cyc_ms/200:.2f} мс)")


if __name__ == "__main__":
    main()
