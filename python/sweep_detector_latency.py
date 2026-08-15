"""
Размен «задержка против дребезга» для потокового детектора фазы.

Задержка потока складывается из двух управляемых слагаемых:
  * скользящая НАЗАД медиана окна k -> лаг ~(k-1)/2 кадров;
  * подтверждение перехода confirm кадров -> ещё confirm кадров.
Уменьшать их можно, но платой будет дребезг: лишние переключения фазы и
ложные события. Здесь оба эффекта меряются на одних и тех же координатах.

Почему без GPU: поза уже посчитана офлайн, перебираются только параметры
детектора. Тот же CSV служит и источником координат, и эталоном, поэтому
сравнение изолирует именно детектор, без шума инференса.

Запуск:
    python sweep_detector_latency.py --csv <DLC_filtered.csv> --gt <*.phases.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from realtime_phase_sim import StreamingPhaseDetector, STANCE, SWING  # noqa: E402
import gait_phase_labeler as G  # noqa: E402


def stream(toe_x, toe_p, body_x, body_p, fps, med_win, confirm):
    det = StreamingPhaseDetector(fps=fps, med_win=med_win, confirm=confirm)
    n = len(toe_x)
    phase = np.empty(n, dtype=object)
    events = []
    for i in range(n):
        ph, ev = det.update(toe_x[i], toe_p[i], body_x[i], body_p[i])
        phase[i] = ph
        if ev:
            events.append((i, ev))
    return phase, events


def score(phase, events, gt_phase, gt_events, fps):
    m = min(len(phase), len(gt_phase))
    p, g = phase[:m], gt_phase[:m]
    both = ((g == STANCE) | (g == SWING)) & ((p == STANCE) | (p == SWING))
    raw = 100 * np.mean(p[both] == g[both]) if both.sum() else np.nan

    best_lag, best_acc = 0, -1.0
    for sh in range(0, 13):
        pp = p[sh:]
        gg = g[:len(g) - sh] if sh else g
        k = min(len(pp), len(gg))
        mk = ((gg[:k] == STANCE) | (gg[:k] == SWING)) & ((pp[:k] == STANCE) | (pp[:k] == SWING))
        if mk.sum() > 100:
            acc = 100 * np.mean(pp[:k][mk] == gg[:k][mk])
            if acc > best_acc:
                best_lag, best_acc = sh, acc

    out = {"raw_acc": raw, "lag_frames": best_lag, "lag_ms": best_lag / fps * 1000,
           "acc_at_lag": best_acc}
    for name in ("touch_down", "lift_off"):
        D = np.array([f for f, e in events if e == name], dtype=float)
        T = np.asarray(gt_events.get(name, []), dtype=float)
        if not D.size or not T.size:
            out[f"{name}_recall"] = np.nan
            out[f"{name}_extra"] = np.nan
            out[f"{name}_off"] = np.nan
            continue
        errs, used = [], set()
        for t in T:
            d = np.abs(D - t)
            j = int(np.argmin(d))
            if d[j] <= 0.15 * fps and j not in used:
                used.add(j)
                errs.append(D[j] - t)
        errs = np.array(errs) / fps * 1000
        out[f"{name}_recall"] = 100 * len(errs) / len(T)
        out[f"{name}_extra"] = D.size / T.size          # >1 = дребезг
        out[f"{name}_off"] = float(np.median(errs)) if errs.size else np.nan
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="DLC _filtered.csv (координаты)")
    ap.add_argument("--gt", required=True, type=Path, help="*.phases.csv (эталон)")
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    a = ap.parse_args(argv)

    df = G.load_dlc(a.csv)
    toe_x = df[(f"hl_toes_{a.leg}", "x")].values.astype(float)
    toe_p = df[(f"hl_toes_{a.leg}", "likelihood")].values.astype(float)
    body_x = df[(f"hl_iliac_{a.leg}", "x")].values.astype(float)
    body_p = df[(f"hl_iliac_{a.leg}", "likelihood")].values.astype(float)

    gt = pd.read_csv(a.gt)
    pcol = f"phase_{a.leg}" if f"phase_{a.leg}" in gt.columns else "phase"
    ecol = f"event_{a.leg}" if f"event_{a.leg}" in gt.columns else "event"
    gt_phase = gt[pcol].astype(str).values
    ev_col = gt[ecol].fillna("").astype(str).values
    gt_events = {n: np.where(ev_col == n)[0] for n in ("touch_down", "lift_off")}
    print(f"кадров {len(toe_x)}, эталонных событий: "
          f"td={len(gt_events['touch_down'])}, lo={len(gt_events['lift_off'])}\n")

    print(f"{'медиана':>8} {'confirm':>8} | {'лаг':>10} | {'совп.сырое':>11} {'совп.@лаг':>10} | "
          f"{'recall td':>10} {'лишних td':>10} | {'recall lo':>10} {'лишних lo':>10}")
    print("-" * 108)
    rows = []
    for med in (1, 3, 5, 7):
        for conf in (1, 2, 3):
            ph, ev = stream(toe_x, toe_p, body_x, body_p, a.fps, med, conf)
            s = score(ph, ev, gt_phase, gt_events, a.fps)
            s.update(med_win=med, confirm=conf)
            rows.append(s)
            print(f"{med:>8} {conf:>8} | {s['lag_frames']:>3} к = {s['lag_ms']:>3.0f} мс | "
                  f"{s['raw_acc']:>10.1f}% {s['acc_at_lag']:>9.1f}% | "
                  f"{s['touch_down_recall']:>9.0f}% {s['touch_down_extra']:>10.2f}x | "
                  f"{s['lift_off_recall']:>9.0f}% {s['lift_off_extra']:>10.2f}x")
    d = pd.DataFrame(rows)
    out = a.gt.parent / "detector_latency_sweep.csv"
    d.to_csv(out, index=False, encoding="utf-8")

    ok = d[(d.touch_down_recall >= 90) & (d.lift_off_recall >= 90) &
           (d.touch_down_extra <= 1.15) & (d.lift_off_extra <= 1.15)]
    print("\nдопустимые (recall >= 90% и лишних <= 1.15x):")
    if len(ok):
        best = ok.loc[ok.lag_ms.idxmin()]
        for _, r in ok.sort_values("lag_ms").iterrows():
            print(f"   медиана {int(r.med_win)}, confirm {int(r.confirm)} -> лаг {r.lag_ms:.0f} мс, "
                  f"совп.@лаг {r.acc_at_lag:.1f}%")
        print(f"\nМИНИМАЛЬНЫЙ ЛАГ: медиана {int(best.med_win)}, confirm {int(best.confirm)} "
              f"-> {best.lag_ms:.0f} мс (сейчас 40 мс)")
    else:
        print("   нет комбинаций, проходящих порог")
    print(f"\nзаписано: {out}")


if __name__ == "__main__":
    main()
