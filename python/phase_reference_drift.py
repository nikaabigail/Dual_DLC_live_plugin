"""
Нужно ли ПЕРЕСТРАИВАТЬ эталонную траекторию на ходу, или хватает построить раз.

Горский просил "усреднённую за последние N шагов траекторию". У нас эталон
строится один раз по первым N циклам и замораживается. Разница проявится только
если траектория меняется: при восстановлении, под стимуляцией, при смене
скорости ленты.

МЕРА БЕЗ ДОПУЩЕНИЙ. Не сравниваем ни с временем, ни с геометрией: меряем
НЕВЯЗКУ ПРОЕКЦИИ - расстояние от текущей точки носка до ближайшей точки эталона,
в пикселях. Если эталон описывает текущее движение, невязка мала. Если эталон
устарел, она растёт. Никакого определения "истинного процента" тут не нужно.

Два вопроса:
  1) внутри одной записи - расходится ли замороженный эталон со временем;
  2) между записями - годится ли эталон, снятый в другом состоянии. Это прокси
     для "походка изменилась": у записей разный цикл (390 против 520 мс).

Запуск:
    python phase_reference_drift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gait_phase_labeler as G  # noqa: E402

NPTS = 100
VID = Path(r"C:\dlc\videos")
PH = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\phases")

REC = [
    ("4", "r", "4_MER2-230-168U3C(FDE22070175)_20240604_164748DLCLive_resnet50_snapshot_best_380"),
    ("5", "r", "5_MER2-230-168U3C(FDE22070174)_20240604_165132DLCLive_resnet50_snapshot_best_380"),
    ("6", "r", "6_MER2-230-168U3C(FDE22070174)_20240604_170308DLCLive_resnet50_snapshot_best_380"),
    ("13", "r", "13_MER2-230-168U3C(FDE22070174)_20240604_181719DLCLive_resnet50_snapshot_best_380"),
]


def load(base, leg):
    df = G.load_dlc(VID / f"{base}_filtered.csv")
    relx = (df[(f"hl_toes_{leg}", "x")].values.astype(float)
            - df[(f"hl_iliac_{leg}", "x")].values.astype(float))
    rely = (df[(f"hl_toes_{leg}", "y")].values.astype(float)
            - df[(f"hl_iliac_{leg}", "y")].values.astype(float))
    ev = pd.read_csv(PH / f"{base}.events.csv")
    col = "frame_corrected" if "frame_corrected" in ev.columns else "frame"
    if "leg" in ev.columns:
        ev = ev[ev.leg == leg]
    td = np.sort(ev.loc[ev.event == "touch_down", col].values.astype(float))
    lo = np.sort(ev.loc[ev.event == "lift_off", col].values.astype(float))
    cyc = []
    for i in range(len(td) - 1):
        a, c = td[i], td[i + 1]
        mid = lo[(lo > a) & (lo < c)]
        if mid.size != 1:
            continue
        ia, ib, ic = int(round(a)), int(round(mid[0])), int(round(c))
        if 10 < ib - ia < 200 and 5 < ic - ib < 200:
            cyc.append((ia, ib, ic))
    return relx, rely, cyc


def loop_from(relx, rely, cyc):
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
    return np.median(np.stack(acc), axis=0) if acc else None


def residual(loop, relx, rely, cyc):
    """Медианное расстояние точки носка до ближайшей точки эталона, px."""
    idx = np.concatenate([np.arange(ia, ic) for ia, _, ic in cyc]) if cyc else np.array([], int)
    idx = idx[(idx >= 0) & (idx < len(relx))]
    P = np.column_stack([relx[idx], rely[idx]])
    ok = np.isfinite(P).all(1)
    if ok.sum() < 100:
        return np.nan
    d = np.sqrt(((P[ok][:, None, :] - loop[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d))


data = {k: load(b, l) for k, l, b in REC}

print("1. ВНУТРИ ЗАПИСИ: замороженный эталон против скользящего\n")
print(f"{'запись':>7} {'циклов':>7} {'цикл, мс':>9} | {'заморож. (первые 10)':>21} "
      f"{'скольз. (последние 10)':>23}")
print("-" * 78)
for k, leg, base in REC:
    relx, rely, cyc = data[k]
    if len(cyc) < 40:
        continue
    cyc_ms = np.median([ic - ia for ia, _, ic in cyc]) * 10
    frozen = loop_from(relx, rely, cyc[:10])
    # оцениваем на ПОСЛЕДНЕЙ трети записи, где заморозка успела бы устареть
    tail = cyc[-len(cyc) // 3:]
    r_frozen = residual(frozen, relx, rely, tail)
    rolling = loop_from(relx, rely, cyc[-len(cyc) // 3 - 10:-len(cyc) // 3])
    r_roll = residual(rolling, relx, rely, tail)
    print(f"{k:>7} {len(cyc):>7} {cyc_ms:>9.0f} | {r_frozen:>18.1f} px "
          f"{r_roll:>20.1f} px")

print("\n\n2. МЕЖДУ ЗАПИСЯМИ: годится ли эталон из другого состояния\n")
loops = {}
for k, leg, base in REC:
    relx, rely, cyc = data[k]
    loops[k] = loop_from(relx, rely, cyc[:10])
keys = [k for k, _, _ in REC]
print("невязка, px       эталон построен на:")
print(f"{'проверен на':>14} | " + " ".join(f"{k:>9}" for k in keys))
print("-" * (17 + 10 * len(keys)))
for kt in keys:
    relx, rely, cyc = data[kt]
    row = f"{kt:>14} |"
    for ks in keys:
        row += f" {residual(loops[ks], relx, rely, cyc):>9.1f}"
    print(row + ("   <- свой эталон на диагонали" if kt == keys[0] else ""))

print("\nневязка = медианное расстояние носка до ближайшей точки эталона")
print("диагональ = свой эталон; вне диагонали = чужой")
