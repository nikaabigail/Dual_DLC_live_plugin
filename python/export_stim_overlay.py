"""
Видео-выгрузка контура стимуляции по проценту фазы: что система видит, где она
считает себя по циклу, и в какой момент уходит импульс.

Зачем именно видео: цифры (смещение +5%, разброс 12 мс) не дают понять, куда
физически попадает импульс. Здесь это видно глазами, покадрово.

Что на панели:
  1) ЭТАЛОННАЯ ПЕТЛЯ - усреднённая за последние N шагов траектория носка
     относительно iliac. Скользящее окно, перестраивается после каждого цикла.
     Жёлтая точка - где носок сейчас, белая - её проекция на петлю. Процент
     фазы это и есть номер белой точки вдоль петли.
  2) ШКАЛА 0..200% - опора и перенос. Красная черта - цель, бегунок - текущая
     фаза, полупрозрачный бегунок - прогноз на задержку контура.
  3) ПЛАШКА СТИМ - загорается в кадре, куда физически лёг бы импульс.

Импульс поджигается ЗАРАНЕЕ: решение принимается за latency-ms до цели, потому
что столько идёт от физического события до реакции контура. Поэтому бегунок
прогноза всегда впереди текущего.

Видео пишется замедленным (--out-fps), иначе перенос длиной 95 мс на 100 Гц
проскакивает за 9 кадров и ничего не разглядеть.

Запуск:
    python export_stim_overlay.py --video <.avi> --csv <DLC_filtered.csv> \
        --events <*.events.csv> --leg r --target 145 --start-frame 3000 --dur 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gait_phase_labeler as G  # noqa: E402

NPTS = 100
REF_CYCLES = 10

# BGR
C_BG = (250, 248, 245)
C_STANCE = (117, 158, 29)
C_SWING = (48, 90, 216)
C_TXT = (60, 60, 60)
C_DIM = (150, 150, 150)
C_FIRE = (60, 40, 235)
C_LOOP = (170, 170, 170)


def cycles_of(td, lo):
    td, lo = np.asarray(sorted(td), float), np.asarray(sorted(lo), float)
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


def blocks_of(cols, cyc):
    """Каждый цикл -> петля 2*NPTS точек в пространстве заданных сигналов."""
    out = []
    for ia, ib, ic in cyc:
        seg = []
        for s, e in ((ia, ib), (ib, ic)):
            src = np.linspace(0, 1, e - s)
            dst = np.linspace(0, 1, NPTS, endpoint=False)
            seg.append(np.column_stack([np.interp(dst, src, c[s:e]) for c in cols]))
        blk = np.vstack(seg)
        out.append(blk if np.isfinite(blk).all() else None)
    return out


def rolling(cols, cyc, n):
    """Скользящий эталон: после каждого цикла пересборка по последним REF_CYCLES."""
    bl = blocks_of(cols, cyc)
    loops, which, have = [], np.full(n, -1, int), []
    for k, (ia, ib, ic) in enumerate(cyc):
        if bl[k] is not None:
            have.append(bl[k])
            if len(have) > REF_CYCLES:
                have.pop(0)
        if len(have) < 3:
            continue
        loops.append(np.median(np.stack(have), axis=0))
        nxt = cyc[k + 1][2] if k + 1 < len(cyc) else n
        which[ic:nxt] = len(loops) - 1
    return loops, which


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--target", type=float, default=145.0)
    ap.add_argument("--latency-ms", type=float, default=28.0)
    ap.add_argument("--start-frame", type=int, default=-1,
                    help="-1 = выбрать самый плотный по шагам участок автоматически")
    ap.add_argument("--dur", type=float, default=12.0, help="секунд записи")
    ap.add_argument("--out-fps", type=float, default=25.0, help="замедление")
    ap.add_argument("--out", type=Path, default=Path("stim_overlay.mp4"))
    a = ap.parse_args(argv)

    df = G.load_dlc(a.csv)
    g = lambda b, c: df[(b, c)].values.astype(float)  # noqa: E731
    tx, ty, tp = g(f"hl_toes_{a.leg}", "x"), g(f"hl_toes_{a.leg}", "y"), g(f"hl_toes_{a.leg}", "likelihood")
    ix, iy, ip = g(f"hl_iliac_{a.leg}", "x"), g(f"hl_iliac_{a.leg}", "y"), g(f"hl_iliac_{a.leg}", "likelihood")
    relx, rely = tx - ix, ty - iy
    vel = np.gradient(pd.Series(relx).rolling(3, min_periods=1).median().values) * a.fps
    n = len(relx)

    ev = pd.read_csv(a.events)
    col = "frame_corrected" if "frame_corrected" in ev.columns else "frame"
    if "leg" in ev.columns:
        ev = ev[ev.leg == a.leg]
    cyc = cycles_of(ev.loc[ev.event == "touch_down", col].values,
                    ev.loc[ev.event == "lift_off", col].values)

    # Участок выбираем не наугад: крыса на записи регулярно останавливается,
    # и в паузе показывать нечего. Берём окно с максимальным числом циклов.
    if a.start_frame < 0:
        want = int(a.dur * a.fps)
        st = np.array([c0 for c0, _, _ in cyc])
        en = np.array([c2 for _, _, c2 in cyc])
        best = (0, -1)
        for s0 in range(0, max(1, n - want), 25):
            k = int(((st >= s0) & (en <= s0 + want)).sum())
            if k > best[1]:
                best = (s0, k)
        a.start_frame = best[0]
        print(f"участок выбран автоматически: кадр {best[0]}, циклов в нём {best[1]}")

    # петля для ПРОЕКЦИИ (x + скорость) и для ПОКАЗА (x + y) - индексы общие
    L_prj, which = rolling((relx, vel), cyc, n)
    L_dsp, _ = rolling((relx, rely), cyc, n)
    lat = int(round(a.latency_ms / 1000.0 * a.fps))

    # ---- каузальная фаза проекцией, событий в рантайме не требуется ----------
    est = np.full(n, np.nan)
    proj = np.full(n, -1, int)
    for i in range(n):
        w = which[i]
        if w < 0 or not (np.isfinite(relx[i]) and np.isfinite(vel[i])):
            continue
        R = L_prj[w]
        sc = np.array([np.ptp(R[:, 0]), np.ptp(R[:, 1])])
        k = int(np.argmin((((np.array([relx[i], vel[i]]) - R) / sc) ** 2).sum(1)))
        proj[i] = k
        est[i] = k * (200.0 / len(R))

    # темп движения вдоль петли, раздельно опора/перенос
    rate, hist = {0: [], 1: []}, np.full((n, 2), np.nan)
    for i in range(1, n):
        if np.isfinite(est[i - 1]) and np.isfinite(est[i]):
            d = (est[i] - est[i - 1] + 100.0) % 200.0 - 100.0
            if 0 < d < 40:
                k = 0 if est[i] < 100.0 else 1
                rate[k].append(d)
                if len(rate[k]) > 200:
                    rate[k].pop(0)
        for k in (0, 1):
            hist[i, k] = np.median(rate[k]) if len(rate[k]) > 20 else np.nan

    def extrap(p, k, r0, r1):
        left = float(k)
        while left > 0:
            r = r0 if p < 100.0 else r1
            if not np.isfinite(r) or r <= 0:
                return np.nan
            edge = 100.0 if p < 100.0 else 200.0
            need = (edge - p) / r
            if need > left:
                return p + r * left
            left -= need
            p = edge % 200.0
        return p

    pred = np.array([extrap(est[i], lat, hist[i, 0], hist[i, 1]) if np.isfinite(est[i])
                     else np.nan for i in range(n)])

    # ---- поджиг: решение в кадре i, импульс ложится в i+lat -----------------
    cid = np.full(n, -1, int)
    for k, (ia, _, ic) in enumerate(cyc):
        cid[ia:ic] = k
    fire = np.zeros(n, bool)        # кадр, куда физически лёг импульс
    decide = np.zeros(n, bool)      # кадр, где было принято решение
    done = set()
    for i in range(n - lat):
        c = cid[i + lat]
        if c < 0 or c in done or not np.isfinite(pred[i]):
            continue
        # полшага упреждения: на 100 Гц кадр в переносе это ~10% фазы, и
        # правило "первый кадр за целью" систематически перелетает на +5%
        r = hist[i, 1] if est[i] >= 100.0 else hist[i, 0]
        half = 0.5 * r if np.isfinite(r) else 0.0
        if a.target - half <= pred[i] < a.target + 40:
            done.add(c)
            decide[i] = True
            fire[i + lat] = True

    # ---- рендер -------------------------------------------------------------
    cap = cv2.VideoCapture(str(a.video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    PANEL = 230
    wr = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"),
                         a.out_fps, (W, H + PANEL))

    LX, LY, LW, LH = 24, H + 44, 340, 130          # петля
    BX, BY, BW, BH = 430, H + 96, 1030, 44         # шкала
    SX, SY, SW, SH = 1540, H + 26, 350, 176        # плашка

    n_frames = int(a.dur * a.fps)
    lit, pend, count = 0, 0, 0
    for f in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        i = a.start_frame + f
        if i >= n:
            break
        canvas = np.full((H + PANEL, W, 3), C_BG, np.uint8)
        canvas[:H] = frame

        p = est[i]
        in_sw = np.isfinite(p) and p >= 100.0
        ph_col = C_SWING if in_sw else C_STANCE
        if decide[i]:
            pend = lat
        if fire[i]:
            lit = 6
            count += 1

        # --- кадр: точки и рамка фазы
        for (x, y, pp, c, r) in ((ix[i], iy[i], ip[i], C_DIM, 5),
                                 (tx[i], ty[i], tp[i], ph_col, 7)):
            if pp >= 0.2 and np.isfinite(x):
                cv2.circle(canvas, (int(x), int(y)), r, c, -1, cv2.LINE_AA)
        if np.isfinite(ix[i]) and np.isfinite(tx[i]) and min(ip[i], tp[i]) >= 0.2:
            cv2.line(canvas, (int(ix[i]), int(iy[i])), (int(tx[i]), int(ty[i])), C_DIM, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), ph_col if np.isfinite(p) else C_DIM, 4)
        if lit > 0:
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), C_FIRE, 12)

        # --- панель 1: эталонная петля
        cv2.putText(canvas, "ETALON: srednyaya traektoriya noska za 10 shagov (otn. iliac)",
                    (LX, LY - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_TXT, 1, cv2.LINE_AA)
        w = which[i]
        if w >= 0:
            D = L_dsp[w]
            x0, x1 = D[:, 0].min(), D[:, 0].max()
            y0, y1 = D[:, 1].min(), D[:, 1].max()
            # Оси масштабируем НЕЗАВИСИМО: реальная петля плоская (размах по X
            # втрое больше, чем по Y), при равном масштабе она вырождается в
            # полоску и ничего не видно. Растяжение подписано, чтобы не вводить
            # в заблуждение - на решение оно не влияет, проекция считается по
            # исходным координатам.
            sx = LW / max(1e-6, x1 - x0)
            sy = LH / max(1e-6, y1 - y0)
            px = (LX + (D[:, 0] - x0) * sx).astype(int)
            py = (LY + LH - (D[:, 1] - y0) * sy).astype(int)
            for k in range(len(D) - 1):
                c = C_STANCE if k < NPTS else C_SWING
                cv2.line(canvas, (px[k], py[k]), (px[k + 1], py[k + 1]), c, 2, cv2.LINE_AA)
            cv2.line(canvas, (px[-1], py[-1]), (px[0], py[0]), C_SWING, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Y rastyanut x{sy / sx:.1f}", (LX, LY - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)
            # точка цели на самой петле: сюда должен попасть импульс
            kt = int(round(a.target / 200.0 * len(D))) % len(D)
            cv2.circle(canvas, (px[kt], py[kt]), 8, C_FIRE, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{a.target:.0f}%", (px[kt] - 16, py[kt] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_FIRE, 1, cv2.LINE_AA)
            # текущая точка и её проекция
            if np.isfinite(relx[i]) and np.isfinite(rely[i]):
                cur = (int(LX + (relx[i] - x0) * sx), int(LY + LH - (rely[i] - y0) * sy))
                if proj[i] >= 0:
                    pr = (px[proj[i]], py[proj[i]])
                    cv2.line(canvas, cur, pr, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.circle(canvas, pr, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pr, 6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(canvas, cur, 6, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, "zhyoltyy = nosok seychas, belyy = ego proekciya",
                        (LX, LY + LH + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)

        # --- панель 2: шкала 0..200%
        cv2.rectangle(canvas, (BX, BY), (BX + BW // 2, BY + BH), C_STANCE, -1)
        cv2.rectangle(canvas, (BX + BW // 2, BY), (BX + BW, BY + BH), C_SWING, -1)
        cv2.putText(canvas, "OPORA 0-100%", (BX + 12, BY + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "PERENOS 100-200%", (BX + BW // 2 + 12, BY + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # цель
        tg = int(BX + BW * a.target / 200.0)
        cv2.line(canvas, (tg, BY - 14), (tg, BY + BH + 14), C_FIRE, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"CEL {a.target:.0f}%", (tg - 40, BY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_FIRE, 2, cv2.LINE_AA)
        # прогноз (куда попадём через задержку) и текущая фаза
        if np.isfinite(pred[i]):
            xq = int(BX + BW * pred[i] / 200.0)
            cv2.line(canvas, (xq, BY + 4), (xq, BY + BH - 4), (255, 255, 255), 2, cv2.LINE_AA)
        if np.isfinite(p):
            xp = int(BX + BW * p / 200.0)
            cv2.line(canvas, (xp, BY - 8), (xp, BY + BH + 8), (30, 30, 30), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"{p:.0f}%", (xp - 22, BY + BH + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "chernyy = faza seychas   belyy = prognoz na +"
                    f"{a.latency_ms:.0f} ms (zaderzhka kontura)",
                    (BX, BY - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TXT, 1, cv2.LINE_AA)

        # --- панель 3: плашка стимула
        on, waiting = lit > 0, pend > 0 and lit == 0
        bg = C_FIRE if on else ((90, 170, 240) if waiting else (228, 228, 228))
        fg = (255, 255, 255) if (on or waiting) else (165, 165, 165)
        cv2.rectangle(canvas, (SX, SY), (SX + SW, SY + SH), bg, -1)
        cv2.rectangle(canvas, (SX, SY), (SX + SW, SY + SH), (120, 120, 120), 2)
        cv2.putText(canvas, "STIM", (SX + 108, SY + 68), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, fg, 3, cv2.LINE_AA)
        sub = "IMPULS" if on else ("reshenie prinyato" if waiting else "ozhidanie")
        cv2.putText(canvas, sub, (SX + 16, SY + 104), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, fg, 2, cv2.LINE_AA)
        if waiting:
            cv2.putText(canvas, f"impuls cherez {pend * 1000 / a.fps:.0f} ms",
                        (SX + 16, SY + 134), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"vsego impulsov: {count}", (SX + 16, SY + 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    fg if (on or waiting) else C_TXT, 1, cv2.LINE_AA)
        lit = max(0, lit - 1)
        pend = max(0, pend - 1)

        cv2.putText(canvas, f"kadr {i}  t={f / a.fps:.2f}s  "
                    f"zamedlenie x{a.fps / a.out_fps:.0f}",
                    (LX, H + PANEL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TXT, 1, cv2.LINE_AA)
        wr.write(canvas)

    cap.release()
    wr.release()
    print(f"импульсов в отрезке: {count}, кадров: {f + 1}")
    print(f"видео записано: {a.out}")


if __name__ == "__main__":
    main()
