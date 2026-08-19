"""
Проверка счёта походки: падает ли он монотонно при ухудшении, и сколько шагов
нужно, чтобы отличить два соседних режима.

ЗАЧЕМ. Байесовский оптимизатор верит счёту слепо. Если счёт шумит сильнее, чем
эффект стимуляции, оптимизатор будет гоняться за шумом, и по графику это будет
выглядеть как работа. Поэтому счёт надо проверить ДО того, как он что-то решает.

КАК. Постинъюрных данных нет, поэтому дефицит синтезируем из интактной записи:
  drag        - носок хуже поднимается в переносе (волочение);
  short       - укорочен размах шага;
  irregular   - разъезжается ритм;
  combined    - всё сразу.
Тяжесть 0.0 (интакт) ... 0.8. Счёт обязан падать монотонно.

ЧУВСТВИТЕЛЬНОСТЬ. Циклы режутся на блоки (одна "проба"), по каждому считается
счёт, и по разбросу между блоками оценивается, сколько проб нужно, чтобы
отличить уровень от интакта. Формула n на группу для мощности 0.8 при alpha
0.05: n ~ 16 / d^2, где d - Cohen's d.

Запуск:
    python validate_gait_score.py --csv <DLC_filtered.csv> --events <*.events.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gait_phase_labeler as G  # noqa: E402
from gait_score import GaitMetrics, IntactNorm, measure, measure_norm, score  # noqa: E402

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]


def load(csv: Path, events: Path, leg: str):
    df = G.load_dlc(csv)
    g = lambda b, c: df[(b, c)].values.astype(float)  # noqa: E731
    tx, ty = g(f"hl_toes_{leg}", "x"), g(f"hl_toes_{leg}", "y")
    bx, by = g(f"hl_iliac_{leg}", "x"), g(f"hl_iliac_{leg}", "y")
    ax, ay = g(f"hl_ankle_{leg}", "x"), g(f"hl_ankle_{leg}", "y")
    hx, hy = g(f"hl_hip_{leg}", "x"), g(f"hl_hip_{leg}", "y")
    ap = g(f"hl_ankle_{leg}", "likelihood")
    hp = g(f"hl_hip_{leg}", "likelihood")
    ok = (ap >= 0.6) & (hp >= 0.6)
    l_ref = float(np.median(np.hypot(hx[ok] - ax[ok], hy[ok] - ay[ok])))

    ev = pd.read_csv(events)
    col = "frame_corrected" if "frame_corrected" in ev.columns else "frame"
    if "leg" in ev.columns:
        ev = ev[ev.leg == leg]
    td = np.sort(ev.loc[ev.event == "touch_down", col].values.astype(float))
    lo = np.sort(ev.loc[ev.event == "lift_off", col].values.astype(float))
    return tx, ty, bx, by, td, lo, l_ref


def degrade(kind: str, sev: float, tx, ty, bx, by, td, lo, cycles, rng):
    """Синтетический дефицит. Возвращает изменённые координаты и события."""
    tx, ty, td, lo = tx.copy(), ty.copy(), td.copy(), lo.copy()

    if kind in ("drag", "combined"):
        # носок хуже поднимается: подъём в переносе сжимается к уровню опоры
        rely = ty - by
        for ia, ib, ic in cycles:
            base = np.nanmedian(rely[ia:ib])           # уровень опоры
            seg = rely[ib:ic]
            ty[ib:ic] = by[ib:ic] + base - (base - seg) * (1.0 - sev)

    if kind in ("short", "combined"):
        # укорочен размах: сжимаем горизонтальную экскурсию к её середине
        relx = tx - bx
        for ia, _, ic in cycles:
            seg = relx[ia:ic]
            mid = np.nanmean(seg)
            tx[ia:ic] = bx[ia:ic] + mid + (seg - mid) * (1.0 - 0.6 * sev)

    if kind in ("irregular", "combined"):
        # разъезжается ритм: дрожание границ цикла
        med = float(np.median(np.diff(td))) if len(td) > 2 else 40.0
        jit = 0.30 * sev * med
        td = np.sort(td + rng.normal(0, jit, td.size))
        lo = np.sort(lo + rng.normal(0, jit, lo.size))

    return tx, ty, td, lo


def blocks_of(cyc, per_block):
    return [cyc[i:i + per_block] for i in range(0, len(cyc) - per_block + 1, per_block)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--cycles-per-trial", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    tx, ty, bx, by, td, lo, l_ref = load(a.csv, a.events, a.leg)
    from gait_score import _cycles
    cyc = _cycles(td, lo)
    rng = np.random.default_rng(a.seed)
    print(f"циклов {len(cyc)}, L_ref {l_ref:.1f} px, блок {a.cycles_per_trial} циклов\n")

    # норма по самой записи, а не по умолчанию
    base = measure(tx, ty, bx, by, td, lo, l_ref, a.fps)
    norm = measure_norm([base])
    print(f"норма записи: клиренс {norm.clearance_lref:.3f} L_ref, "
          f"шаг {norm.step_length_lref:.3f} L_ref, перенос {norm.swing_ms:.0f} мс\n")

    print(f"{'дефицит':<11} {'тяжесть':>8} {'счёт':>8} {'клиренс':>9} {'шаг':>7} "
          f"{'CV цикла':>9} {'блоков':>7} {'n на группу':>12}")
    print("-" * 82)

    results = {}
    for kind in ("drag", "short", "irregular", "combined"):
        per_level = {}
        for sev in LEVELS:
            dx, dy, dtd, dlo = degrade(kind, sev, tx, ty, bx, by, td, lo, cyc, rng)
            dcyc = _cycles(dtd, dlo)
            vals = []
            for blk in blocks_of(dcyc, a.cycles_per_trial):
                lo_i, hi_i = blk[0][0], blk[-1][2]
                sel_td = dtd[(dtd >= lo_i) & (dtd <= hi_i)]
                sel_lo = dlo[(dlo >= lo_i) & (dlo <= hi_i)]
                try:
                    m = measure(dx, dy, bx, by, sel_td, sel_lo, l_ref, a.fps)
                except ValueError:
                    continue
                vals.append(score(m, norm))
            if not vals:
                continue
            v = np.asarray(vals)
            per_level[sev] = v
            m_all = measure(dx, dy, bx, by, dtd, dlo, l_ref, a.fps)
            # сколько блоков нужно, чтобы отличить от интакта
            if sev == 0.0:
                n_txt = "-"
            else:
                ref = per_level[0.0]
                sd = np.sqrt((ref.var(ddof=1) + v.var(ddof=1)) / 2)
                d = abs(ref.mean() - v.mean()) / sd if sd > 0 else np.inf
                n_txt = "1" if d == np.inf else f"{max(1, int(np.ceil(16 / d ** 2)))}"
            print(f"{kind:<11} {sev:>8.1f} {v.mean():>8.3f} {m_all.clearance_lref:>9.3f} "
                  f"{m_all.step_length_lref:>7.3f} {m_all.cycle_cv:>9.3f} "
                  f"{len(v):>7} {n_txt:>12}")
        results[kind] = per_level
        print()

    print("=" * 82)
    print("МОНОТОННОСТЬ (счёт обязан падать с ростом тяжести)")
    ok_all = True
    for kind, per in results.items():
        means = [per[s].mean() for s in LEVELS if s in per]
        drops = np.diff(means)
        mono = bool(np.all(drops <= 1e-9))
        ok_all &= mono
        arrow = " -> ".join(f"{m:.3f}" for m in means)
        print(f"  {kind:<11} {'ОК' if mono else 'НАРУШЕНА'}   {arrow}")
    print()
    print("вывод:", "счёт пригоден как целевая функция" if ok_all
          else "СЧЁТ НЕПРИГОДЕН: где-то ухудшение не снижает счёт")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
