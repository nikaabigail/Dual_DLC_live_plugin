"""
Пакетная разметка фаз по всем видео + сводная статистика.

Прогоняет gait_phase_labeler по каждому DLC-CSV, собирает per-video метрики
(эпизоды, размеченные кадры, duty factor, длительности фаз, число циклов) и
общий пул циклов для распределений.

Запуск:
    python aggregate_gait_stats.py --videos "C:/dlc/videos" --out "stats"
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gait_phase_labeler as G  # noqa: E402


def phase_runs(phase: np.ndarray, name: str, fps: float) -> np.ndarray:
    """Длительности непрерывных участков фазы, мс."""
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


def cycles_from_events(ev: pd.DataFrame, fps: float):
    """Полные циклы: td -> lo -> td. Возвращает (длительность_мс, duty)."""
    if ev.empty:
        return np.array([]), np.array([])
    e = ev.sort_values("frame_corrected" if "frame_corrected" in ev else "frame")
    col = "frame_corrected" if "frame_corrected" in ev else "frame"
    arr = e[[col, "event"]].values
    dur, duty = [], []
    for i in range(len(arr) - 2):
        if arr[i][1] == "touch_down" and arr[i + 1][1] == "lift_off" and arr[i + 2][1] == "touch_down":
            c = arr[i + 2][0] - arr[i][0]
            if 0.15 * fps < c < 1.2 * fps:            # 150..1200 мс
                dur.append(c / fps * 1000.0)
                duty.append((arr[i + 1][0] - arr[i][0]) / c)
    return np.array(dur), np.array(duty)


def process(csv: Path, raw: Path | None, out_dir: Path, fps: float):
    prm = G.Params(fps=fps)
    prefix = out_dir / csv.stem.replace("_filtered", "")
    with contextlib.redirect_stdout(io.StringIO()):
        ph, ev = G.run(csv, raw, prm, "auto", prefix)
    lab = ph[ph.phase.isin(["stance", "swing"])]
    dur, duty = cycles_from_events(ev, fps)
    sw = phase_runs(ph.phase.values, "swing", fps)
    st = phase_runs(ph.phase.values, "stance", fps)
    return {
        "video": csv.stem.split("DLC")[0][:34],
        "leg": ph["leg"].iloc[0] if len(ph) else "?",
        "frames": len(ph),
        "sec": round(len(ph) / fps, 1),
        "bouts": int(ph.bout_id.max()),
        "labeled": len(lab),
        "labeled_%": round(100 * len(lab) / max(1, len(ph)), 1),
        "cycles": len(dur),
        "D": round(float(np.median(duty)), 3) if duty.size else np.nan,
        "cycle_ms": round(float(np.median(dur))) if dur.size else np.nan,
        "swing_ms": round(float(np.median(sw))) if sw.size else np.nan,
        "stance_ms": round(float(np.median(st))) if st.size else np.nan,
        "events": len(ev),
    }, dur, duty, sw, st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path(r"C:\dlc\videos"))
    ap.add_argument("--out", type=Path, default=Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation\phases"))
    ap.add_argument("--fps", type=float, default=100.0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    csvs = sorted(p for p in a.videos.glob("*_filtered.csv"))
    if not csvs:
        raise SystemExit("не найдено *_filtered.csv - сначала прогони batch_offline_pose.py")

    rows, D_all, C_all, SW_all, ST_all = [], [], [], [], []
    for i, c in enumerate(csvs, 1):
        raw = Path(str(c).replace("_filtered.csv", ".csv"))
        print(f"[{i}/{len(csvs)}] {c.stem[:50]}", flush=True)
        try:
            r, dur, duty, sw, st = process(c, raw if raw.exists() else None, a.out, a.fps)
            rows.append(r)
            C_all.append(dur); D_all.append(duty); SW_all.append(sw); ST_all.append(st)
            print(f"      нога {r['leg'].upper()}, циклов {r['cycles']}, D={r['D']}, "
                  f"размечено {r['labeled']} ({r['labeled_%']}%)", flush=True)
        except Exception as e:
            print(f"      ОШИБКА {type(e).__name__}: {e}", flush=True)

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "summary_by_video.csv", index=False, encoding="utf-8")

    cyc = np.concatenate([x for x in C_all if x.size]) if any(x.size for x in C_all) else np.array([])
    duty = np.concatenate([x for x in D_all if x.size]) if any(x.size for x in D_all) else np.array([])
    sw = np.concatenate([x for x in SW_all if x.size]) if any(x.size for x in SW_all) else np.array([])
    st = np.concatenate([x for x in ST_all if x.size]) if any(x.size for x in ST_all) else np.array([])

    print("\n" + "=" * 96)
    print(df.to_string(index=False))
    print("=" * 96)
    print(f"\nвидео: {len(df)}   суммарно {df.sec.sum() / 60:.1f} мин")
    print(f"размечено кадров: {df.labeled.sum()} ({df.labeled.sum() / 100 / 60:.1f} мин локомоции)")
    print(f"эпизодов: {df.bouts.sum()}   полных циклов: {len(cyc)}")
    if duty.size:
        print(f"\nduty factor: медиана {np.median(duty):.3f}  IQR {np.percentile(duty, 25):.3f}-{np.percentile(duty, 75):.3f}")
        print(f"цикл, мс:    медиана {np.median(cyc):.0f}  IQR {np.percentile(cyc, 25):.0f}-{np.percentile(cyc, 75):.0f}")
    if sw.size:
        print(f"swing, мс:   медиана {np.median(sw):.0f}  IQR {np.percentile(sw, 25):.0f}-{np.percentile(sw, 75):.0f}   [норма 110-130]")
    if st.size:
        print(f"stance, мс:  медиана {np.median(st):.0f}  IQR {np.percentile(st, 25):.0f}-{np.percentile(st, 75):.0f}")
    np.savez(a.out / "pooled_distributions.npz", cycle_ms=cyc, duty=duty, swing_ms=sw, stance_ms=st)
    print(f"\nзаписано: {a.out / 'summary_by_video.csv'}")


if __name__ == "__main__":
    main()
