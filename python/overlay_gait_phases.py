"""
Видео-оверлей разметки фаз: наложение состояния (STANCE / SWING / UNKNOWN),
ключевых точек, сигнала R и событий touch_down / lift_off на исходное видео.

Нужен для покадровой проверки глазами: совпадает ли метка с тем, что реально
делает лапа на кадре.

Запуск:
    python overlay_gait_phases.py --video <.avi> --phases <*.phases.csv> \
        [--dlc-csv <DLC_filtered.csv>] [--start 0] [--dur 10] [--out out.mp4]

Примечание: OpenCV не умеет кириллицу в putText, поэтому подписи латиницей.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PANEL_H = 150                      # высота информационной панели снизу
COL = {                            # BGR
    "stance": (171, 225, 159),
    "swing": (117, 199, 250),
    "unknown": (228, 232, 232),
    "td": (117, 158, 29),
    "lo": (48, 90, 216),
    "txt": (40, 40, 40),
    "muted": (130, 130, 130),
    "line": (200, 200, 200),
}


def load_points(dlc_csv: Path, leg: str):
    """Координаты hip/ankle/toes выбранной ноги для рисовки скелета."""
    df = pd.read_csv(dlc_csv, header=[1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples([(a, b) for a, b in df.columns])
    out = {}
    for part in ("hip", "ankle", "toes"):
        name = f"hl_{part}_{leg}"
        if (name, "x") in df.columns:
            out[part] = (df[(name, "x")].values,
                         df[(name, "y")].values,
                         df[(name, "likelihood")].values)
    return out


def build_static_panel(ph_slice, w, r_span=(-1.0, 6.0)):
    """
    Статичная часть панели (таймлайн фаз + график R) строится ОДИН раз.
    Иначе попиксельная отрисовка через pandas.iloc на каждом кадре съедает всё время.
    """
    panel = np.full((PANEL_H, w, 3), 255, np.uint8)
    tl_x0, tl_x1 = 620, w - 20
    tl_y, tl_h = 22, 26
    n = len(ph_slice)
    if n < 2 or tl_x1 <= tl_x0:
        return panel, (tl_x0, tl_x1, tl_y, tl_h)

    phases = ph_slice["phase"].astype(str).to_numpy()
    Rv = pd.to_numeric(ph_slice["R"], errors="coerce").to_numpy(dtype=float)
    idx = ((np.arange(tl_x0, tl_x1) - tl_x0) / (tl_x1 - tl_x0) * (n - 1)).astype(int)

    for k, px in enumerate(range(tl_x0, tl_x1)):
        cv2.line(panel, (px, tl_y), (px, tl_y + tl_h),
                 COL.get(phases[idx[k]], COL["unknown"]), 1)
    cv2.putText(panel, "phase timeline", (tl_x0, tl_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["muted"], 1, cv2.LINE_AA)

    gy0, gh = tl_y + tl_h + 16, 58
    cv2.rectangle(panel, (tl_x0, gy0), (tl_x1, gy0 + gh), COL["line"], 1)
    lo, hi = r_span
    prev = None
    for k, px in enumerate(range(tl_x0, tl_x1)):
        v = Rv[idx[k]]
        if not np.isfinite(v):
            prev = None
            continue
        yy = int(gy0 + gh - gh * (np.clip(v, lo, hi) - lo) / (hi - lo))
        if prev is not None:
            cv2.line(panel, prev, (px, yy), (219, 138, 55), 1, cv2.LINE_AA)
        prev = (px, yy)
    for lvl, c in ((0.0, COL["muted"]), (1.6, (48, 90, 216))):
        yy = int(gy0 + gh - gh * (lvl - lo) / (hi - lo))
        cv2.line(panel, (tl_x0, yy), (tl_x1, yy), c, 1)
    cv2.putText(panel, "R", (tl_x0 - 16, gy0 + gh // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["muted"], 1, cv2.LINE_AA)
    return panel, (tl_x0, tl_x1, tl_y, tl_h)


def draw_panel(canvas, ph_slice, i_local, i_global, fps, static_panel, geom):
    """Динамическая часть: бейдж состояния, числа, бегунок по таймлайну."""
    h, w = canvas.shape[:2]
    y0 = h - PANEL_H
    canvas[y0:h] = static_panel
    cv2.line(canvas, (0, y0), (w, y0), COL["line"], 1)

    row = ph_slice.iloc[i_local]
    phase = str(row["phase"])
    # --- крупный бейдж состояния ---
    badge = phase.upper()
    cv2.rectangle(canvas, (14, y0 + 14), (250, y0 + 66), COL.get(phase, COL["unknown"]), -1)
    cv2.putText(canvas, badge, (28, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL["txt"], 2, cv2.LINE_AA)

    # --- числа ---
    rv = row["R"]
    info = [
        f"frame {int(row['frame'])}   t={row['time_s']:.2f}s   leg={row['leg']}",
        f"R = {rv:.2f}" if np.isfinite(rv) else "R = -",
        f"bout {int(row['bout_id'])}   p(toe)={row['toe_likelihood']:.2f}",
    ]
    for k, s in enumerate(info):
        cv2.putText(canvas, s, (268, y0 + 32 + k * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COL["txt"], 1, cv2.LINE_AA)

    ev = row["event"]
    if isinstance(ev, str) and ev:
        c = COL["td"] if ev == "touch_down" else COL["lo"]
        cv2.putText(canvas, ev.upper(), (268, y0 + 104), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, c, 2, cv2.LINE_AA)

    # --- бегунок по заранее отрисованному таймлайну ---
    tl_x0, tl_x1, tl_y_rel, tl_h = geom
    n = len(ph_slice)
    if n > 1 and tl_x1 > tl_x0:
        tl_y = y0 + tl_y_rel
        cur = int(tl_x0 + (tl_x1 - tl_x0) * i_local / max(1, n - 1))
        cv2.line(canvas, (cur, tl_y - 4), (cur, tl_y + tl_h + 4), (0, 0, 0), 2)
    return canvas


def run(video: Path, phases_csv: Path, dlc_csv: Path | None, start_s: float,
        dur_s: float, out_path: Path, only_bout: int | None):
    ph = pd.read_csv(phases_csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"не открылось видео: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if only_bout:
        sel = ph.index[ph["bout_id"] == only_bout]
        if not len(sel):
            raise SystemExit(f"эпизод {only_bout} не найден")
        f0, f1 = int(sel[0]), int(sel[-1]) + 1
    else:
        f0 = int(start_s * fps)
        f1 = int(min(len(ph), f0 + dur_s * fps))
    ph_slice = ph.iloc[f0:f1].reset_index(drop=True)

    leg = str(ph_slice["leg"].iloc[0])
    pts = load_points(dlc_csv, leg) if dlc_csv and dlc_csv.exists() else {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H + PANEL_H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    static_panel, geom = build_static_panel(ph_slice, W)

    written = 0
    for i in range(len(ph_slice)):
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.full((H + PANEL_H, W, 3), 255, np.uint8)
        canvas[:H] = frame
        gi = f0 + i
        phase = str(ph_slice.iloc[i]["phase"])

        # цветная рамка кадра по состоянию
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), COL.get(phase, COL["unknown"]), 6)

        # скелет hip-ankle-toes
        if pts:
            xy = {}
            for part, (X, Y, P) in pts.items():
                if gi < len(X) and P[gi] >= 0.2 and np.isfinite(X[gi]):
                    xy[part] = (int(X[gi]), int(Y[gi]))
            for a, b in (("hip", "ankle"), ("ankle", "toes")):
                if a in xy and b in xy:
                    cv2.line(canvas, xy[a], xy[b], (60, 60, 60), 2, cv2.LINE_AA)
            for part, p in xy.items():
                c = (48, 90, 216) if part == "toes" else (120, 120, 120)
                cv2.circle(canvas, p, 5 if part == "toes" else 4, c, -1, cv2.LINE_AA)

        # вспышка на событии
        ev = ph_slice.iloc[i]["event"]
        if isinstance(ev, str) and ev:
            c = COL["td"] if ev == "touch_down" else COL["lo"]
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), c, 12)

        draw_panel(canvas, ph_slice, i, gi, fps, static_panel, geom)
        writer.write(canvas)
        written += 1

    cap.release()
    writer.release()
    n_st = (ph_slice["phase"] == "stance").sum()
    n_sw = (ph_slice["phase"] == "swing").sum()
    n_ev = ph_slice["event"].astype(str).str.len().gt(0).sum() - (ph_slice["event"].isna()).sum()
    print(f"кадров записано: {written}  ({written / fps:.1f} с)")
    print(f"stance={n_st}, swing={n_sw}, unknown={written - n_st - n_sw}, событий={max(0, n_ev)}")
    print(f"готово: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Видео-оверлей фаз шага для визуальной проверки")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--phases", required=True, type=Path, help="*.phases.csv от gait_phase_labeler")
    ap.add_argument("--dlc-csv", type=Path, default=None, help="DLC CSV для рисовки точек")
    ap.add_argument("--start", type=float, default=0.0, help="начало, с")
    ap.add_argument("--dur", type=float, default=10.0, help="длительность, с")
    ap.add_argument("--bout", type=int, default=None, help="взять эпизод целиком по номеру")
    ap.add_argument("--out", type=Path, default=Path("overlay.mp4"))
    a = ap.parse_args(argv)
    run(a.video, a.phases, a.dlc_csv, a.start, a.dur, a.out, a.bout)


if __name__ == "__main__":
    main()
