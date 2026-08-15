"""
Процент фазы В РЕАЛЬНОМ ПОТОКЕ: инференс DLC, каузальная геометрия, поджиг.

Кадр идёт с видео как с камеры, поза считается на GPU, процент фазы определяется
проекцией на скользящий эталон. Всё каузально, ни одного отсчёта из будущего.

АРХИТЕКТУРА (постановка О.В. Горского):
  * эталон = усреднённая за последние N шагов траектория носка ОТНОСИТЕЛЬНО
    iliac, поэтому движение дорожки и дрейф крысы по кадру не мешают;
  * каждой точке эталона приписан процент: опора 0..100, перенос 100..200;
  * текущий процент = процент ближайшей точки эталона.

ГДЕ ТУТ СОБЫТИЯ. Они нужны только чтобы РАЗРЕЗАТЬ запись на циклы при сборке
эталона, то есть раз в цикл. Сама оценка процента событий не требует вовсе.
Этим геометрия и отличается от таймера, которому свежее событие нужно каждый
цикл без исключения.

ТРИ ПОПРАВКИ, БЕЗ КОТОРЫХ НЕ РАБОТАЕТ (все найдены измерением, см. историю):
  1) лаг событий. Детектор сообщает об отрыве с опозданием на кадр. Геометрия
     петли при этом снимается верно, а ЯРЛЫКИ на ней съезжают: точка "100%"
     оказывается там, где перенос уже начался. Смещение выходило +9.5%, то есть
     ровно кадр (в переносе кадр это ~10% фазы). Относим события назад;
  2) полшага упреждения. Прогноз идёт скачками по ~10% за кадр, и правило
     "первый кадр за целью" систематически перелетает на половину кадра;
  3) НЕ сглаживать координату. Каждый кадр медианы добавляет свой лаг, который
     никто не компенсирует: p90 растёт 9.8 -> 17.6 -> 24.0% при окне 1/3/5.

ПОДЖИГ ТОЛЬКО НА ХОДУ. Полный прогон показал 272 импульса на 238 циклов: в
паузах проекция продолжает выдавать процент (петля-то никуда не делась), фаза
бродит через цель и система стреляет по стоящему животному. Для стимуляции это
недопустимо, поэтому поджиг закрыт признаком движения: разброс положения носка
относительно тела за последнее окно. Признак каузальный и событий не требует.

Запуск:
    python realtime_phase_percent.py --video <.avi> --events <*.events.csv> \
        --leg r --target 145 --save-video out.mp4 --render-from 7375
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from realtime_phase_sim import (BODYPARTS, LegRoiTracker, StreamingPhaseDetector,
                                build_live)

NPTS = 100

# BGR
C_BG = (250, 248, 245)
C_STANCE = (117, 158, 29)
C_SWING = (48, 90, 216)
C_TXT = (60, 60, 60)
C_DIM = (150, 150, 150)
C_FIRE = (60, 40, 235)
C_WAIT = (90, 170, 240)


class StreamingPhasePercent:
    """
    Каузальный процент фазы: проекция на скользящий эталон.

    Петля хранится в трёх столбцах (relx, rely, скорость relx). Проекция идёт по
    (relx, скорость) - это пространство даёт 0.2% откатов назад против 3.2% у
    (relx, rely). Столбец rely нужен только для отрисовки, на решение не влияет.

    Стоимость на кадр: argmin по 200 точкам. Замерена отдельно, ~0.26 мс.
    """

    def __init__(self, fps=100.0, ref_cycles=10, target=145.0, latency_ms=28.0,
                 p_use=0.20, med_win=1, event_lag=1, bout_win=50, bout_std=8.0,
                 min_gap=100.0):
        self.fps = fps
        self.target = float(target)
        self.lat = max(1, int(round(latency_ms / 1000.0 * fps)))
        self.p_use = p_use
        self.ev_lag = int(event_lag)
        self.med = deque(maxlen=max(1, med_win))
        self.hist = deque(maxlen=ref_cycles)
        self.loop = None                          # (2*NPTS, 3)
        self.scale = None                         # масштаб столбцов 0 и 2
        self.buf = []                             # текущий цикл: список строк
        self.lo_at = None
        self.prev = None                          # (relx, rely)
        self.rate = {0: deque(maxlen=200), 1: deque(maxlen=200)}
        self.prev_pct = None
        # Рефрактерность по НАКОПЛЕННОЙ фазе вместо флага "взведён".
        #
        # Это УКРЕПЛЕНИЕ, а не исправление ошибки: двойных срабатываний в данных
        # не нашлось ни одного (проверено разнесением импульсов по номерам шагов;
        # три подозрительных случая оказались артефактом счётчика циклов, который
        # не увеличивался на паузах, и импульсы там разделены 0.5-21 секундой).
        #
        # Зачем менять то, что работает: прежний взвод снимался ОДНИМ кадром с
        # pct < 50, то есть держался на том, что проекция не выбрасывает
        # одиночных выбросов. На интактной крысе не выбрасывает. На разъехавшейся
        # после гемисекции траектории рассчитывать на это нельзя. Накопитель к
        # выбросам нечувствителен по построению: он суммирует только
        # правдоподобные приращения (0 < d < 40), те же, что идут в оценку темпа.
        #
        # min_gap = 100 это половина цикла. Развёртка: при 100 покрытие 233/238
        # (столько же, сколько у прежнего взвода), при 150 уже 231, при 190 - 223.
        # Смысла затягивать нет, всё, что ближе полушага, и так заведомо ложное.
        self.total = 0.0
        self.last_fire_total = -1e9
        self.min_gap = min_gap
        self.n_rebuild = 0
        self.walk = deque(maxlen=bout_win)
        self.bout_std = float(bout_std)
        self.walking = False

    # ---- сборка эталона ---------------------------------------------------
    def _close_cycle(self):
        if self.lo_at is None or self.lo_at < 3 or len(self.buf) - self.lo_at < 3:
            return
        A = np.asarray(self.buf, dtype=float)
        seg = []
        for s, e in ((0, self.lo_at), (self.lo_at, len(A))):
            src = np.linspace(0, 1, e - s)
            dst = np.linspace(0, 1, NPTS, endpoint=False)
            seg.append(np.column_stack([np.interp(dst, src, A[s:e, c]) for c in range(3)]))
        blk = np.vstack(seg)
        if not np.isfinite(blk).all():
            return
        self.hist.append(blk)
        if len(self.hist) >= 3:
            loop = np.median(np.stack(self.hist), axis=0)
            sc = np.array([np.ptp(loop[:, 0]), np.ptp(loop[:, 2])])
            if np.all(sc > 1e-6):
                self.loop, self.scale = loop, sc
                self.n_rebuild += 1

    # ---- экстраполяция ----------------------------------------------------
    def _rates(self):
        r0 = np.median(self.rate[0]) if len(self.rate[0]) > 20 else np.nan
        r1 = np.median(self.rate[1]) if len(self.rate[1]) > 20 else np.nan
        return r0, r1

    def _extrapolate(self, p, k, r0, r1):
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

    # ---- кадр -------------------------------------------------------------
    def update(self, toe, body, event):
        """toe/body = (x, y, likelihood). -> (процент, прогноз, поджиг)."""
        tx, ty, tp = toe
        bx, by, bp = body
        if not np.isfinite(tx) or tp < self.p_use or bp < self.p_use:
            return np.nan, np.nan, False
        self.med.append(float(tx))
        relx = float(np.median(self.med)) - float(bx)
        rely = float(ty) - float(by)
        v = np.nan if self.prev is None else (relx - self.prev[0]) * self.fps
        self.prev = (relx, rely)
        if not np.isfinite(v):
            return np.nan, np.nan, False

        # шагает ли: разброс положения носка относительно тела за окно
        self.walk.append(relx)
        self.walking = (len(self.walk) == self.walk.maxlen
                        and float(np.std(self.walk)) >= self.bout_std)

        # накопление цикла, границы относим назад на лаг детектора
        if event == "lift_off":
            self.lo_at = max(1, len(self.buf) - self.ev_lag)
        elif event == "touch_down":
            tail = self.buf[len(self.buf) - self.ev_lag:] if self.ev_lag else []
            if self.ev_lag:
                self.buf = self.buf[:len(self.buf) - self.ev_lag]
            self._close_cycle()
            self.buf, self.lo_at = list(tail), None
        self.buf.append((relx, rely, v))
        if len(self.buf) > 400:                    # крыса встала, цикл не закроется
            self.buf, self.lo_at = [], None

        if self.loop is None:
            return np.nan, np.nan, False

        # проекция по (relx, скорость)
        R = self.loop[:, (0, 2)] / self.scale
        c = np.array([relx, v]) / self.scale
        k = int(np.argmin(((c - R) ** 2).sum(1)))
        pct = k * (200.0 / len(self.loop))
        self.proj_idx = k

        if self.prev_pct is not None:
            d = (pct - self.prev_pct + 100.0) % 200.0 - 100.0
            if 0 < d < 40:                       # правдоподобный шаг вперёд
                self.rate[0 if pct < 100.0 else 1].append(d)
                self.total += d                  # то же условие держит накопитель
        self.prev_pct = pct

        r0, r1 = self._rates()
        pred = self._extrapolate(pct, self.lat, r0, r1)

        fire = False
        ready = (self.total - self.last_fire_total) >= self.min_gap
        if ready and self.walking and np.isfinite(pred):
            r = r1 if pct >= 100.0 else r0
            half = 0.5 * r if np.isfinite(r) else 0.0
            if self.target - half <= pred < self.target + 40.0:
                fire = True
                self.last_fire_total = self.total
        return pct, pred, fire


# --------------------------------------------------------------------------- #
def truth_percent(n, td, lo):
    pct = np.full(n, np.nan)
    td, lo = np.sort(np.asarray(td, float)), np.sort(np.asarray(lo, float))
    for i in range(len(td) - 1):
        a, c = td[i], td[i + 1]
        mid = lo[(lo > a) & (lo < c)]
        if mid.size != 1:
            continue
        ia, ib, ic = int(round(a)), int(round(mid[0])), int(round(c))
        if not (10 < ib - ia < 200 and 5 < ic - ib < 200):
            continue
        pct[ia:ib] = 100.0 * (np.arange(ia, ib) - a) / (ib - ia)
        pct[ib:ic] = 100.0 + 100.0 * (np.arange(ib, ic) - ib) / (ic - ib)
    return pct


def render(video, out_path, rows, loops, first_frame, lo_idx, hi_idx,
           target, lat, fps, out_fps):
    """Видео из ЖИВОГО прогона: всё, что рисуется, посчитано в потоке."""
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo_idx)
    PANEL = 230
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         out_fps, (W, H + PANEL))
    LX, LY, LW, LH = 24, H + 44, 340, 130
    BX, BY, BW, BH = 430, H + 96, 1030, 44
    SX, SY, SW, SH = 1540, H + 26, 350, 176

    fire_at = np.zeros(len(rows), bool)
    for k, r in enumerate(rows):
        if r["fire"] and k + lat < len(rows):
            fire_at[k + lat] = True

    lit = pend = count = 0
    for i in range(lo_idx, hi_idx):
        ok, frame = cap.read()
        if not ok:
            break
        k = i - first_frame
        if k < 0 or k >= len(rows):
            continue
        r = rows[k]
        canvas = np.full((H + PANEL, W, 3), C_BG, np.uint8)
        canvas[:H] = frame
        p, q = r["pct"], r["pred"]
        walking = r["walk"]
        in_sw = np.isfinite(p) and p >= 100.0
        ph_col = (C_SWING if in_sw else C_STANCE) if np.isfinite(p) else C_DIM
        if r["fire"]:
            pend = lat
        if fire_at[k]:
            lit, count = 6, count + 1

        # --- кадр: ROI, точки, рамка
        if r["roi"] is not None:
            x1, x2, y1, y2 = r["roi"]
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (90, 90, 90), 1)
        for (x, y, pp, col, rad) in ((r["body"][0], r["body"][1], r["body"][2], C_DIM, 5),
                                     (r["toe"][0], r["toe"][1], r["toe"][2], ph_col, 7)):
            if pp >= 0.2 and np.isfinite(x):
                cv2.circle(canvas, (int(x), int(y)), rad, col, -1, cv2.LINE_AA)
        if min(r["toe"][2], r["body"][2]) >= 0.2:
            cv2.line(canvas, (int(r["body"][0]), int(r["body"][1])),
                     (int(r["toe"][0]), int(r["toe"][1])), C_DIM, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), ph_col, 4)
        if lit > 0:
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), C_FIRE, 12)

        # --- петля
        cv2.putText(canvas, "ETALON: srednyaya traektoriya noska za 10 shagov (otn. iliac)",
                    (LX, LY - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_TXT, 1, cv2.LINE_AA)
        D = loops[r["loop"]] if r["loop"] >= 0 else None
        if D is not None:
            x0, x1_ = D[:, 0].min(), D[:, 0].max()
            y0, y1_ = D[:, 1].min(), D[:, 1].max()
            sx = LW / max(1e-6, x1_ - x0)
            sy = LH / max(1e-6, y1_ - y0)
            px = (LX + (D[:, 0] - x0) * sx).astype(int)
            py = (LY + LH - (D[:, 1] - y0) * sy).astype(int)
            for t in range(len(D) - 1):
                cv2.line(canvas, (px[t], py[t]), (px[t + 1], py[t + 1]),
                         C_STANCE if t < NPTS else C_SWING, 2, cv2.LINE_AA)
            cv2.line(canvas, (px[-1], py[-1]), (px[0], py[0]), C_SWING, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Y rastyanut x{sy / sx:.1f}", (LX, LY - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)
            kt = int(round(target / 200.0 * len(D))) % len(D)
            cv2.circle(canvas, (px[kt], py[kt]), 8, C_FIRE, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{target:.0f}%", (px[kt] - 16, py[kt] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_FIRE, 1, cv2.LINE_AA)
            if r["proj"] >= 0 and np.isfinite(r["relx"]):
                cur = (int(LX + (r["relx"] - x0) * sx),
                       int(LY + LH - (r["rely"] - y0) * sy))
                pr = (px[r["proj"]], py[r["proj"]])
                cv2.line(canvas, cur, pr, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(canvas, pr, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pr, 6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(canvas, cur, 6, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "zhyoltyy = nosok seychas, belyy = ego proekciya",
                    (LX, LY + LH + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)

        # --- шкала
        cv2.rectangle(canvas, (BX, BY), (BX + BW // 2, BY + BH), C_STANCE, -1)
        cv2.rectangle(canvas, (BX + BW // 2, BY), (BX + BW, BY + BH), C_SWING, -1)
        cv2.putText(canvas, "OPORA 0-100%", (BX + 12, BY + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "PERENOS 100-200%", (BX + BW // 2 + 12, BY + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tg = int(BX + BW * target / 200.0)
        cv2.line(canvas, (tg, BY - 14), (tg, BY + BH + 14), C_FIRE, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"CEL {target:.0f}%", (tg - 40, BY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_FIRE, 2, cv2.LINE_AA)
        if np.isfinite(q):
            xq = int(BX + BW * q / 200.0)
            cv2.line(canvas, (xq, BY + 4), (xq, BY + BH - 4), (255, 255, 255), 2, cv2.LINE_AA)
        if np.isfinite(p):
            xp = int(BX + BW * p / 200.0)
            cv2.line(canvas, (xp, BY - 8), (xp, BY + BH + 8), (30, 30, 30), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"{p:.0f}%", (xp - 22, BY + BH + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "chernyy = faza seychas   belyy = prognoz na +"
                    f"{lat * 1000 / fps:.0f} ms (zaderzhka kontura)",
                    (BX, BY - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TXT, 1, cv2.LINE_AA)
        # индикатор движения
        cv2.rectangle(canvas, (BX, BY + BH + 44), (BX + 250, BY + BH + 78),
                      (117, 158, 29) if walking else (185, 185, 185), -1)
        cv2.putText(canvas, "SHAGAET" if walking else "STOIT - podzhig zakryt",
                    (BX + 10, BY + BH + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # --- плашка
        on, waiting = lit > 0, pend > 0 and lit == 0
        bg = C_FIRE if on else (C_WAIT if waiting else (228, 228, 228))
        fg = (255, 255, 255) if (on or waiting) else (165, 165, 165)
        cv2.rectangle(canvas, (SX, SY), (SX + SW, SY + SH), bg, -1)
        cv2.rectangle(canvas, (SX, SY), (SX + SW, SY + SH), (120, 120, 120), 2)
        cv2.putText(canvas, "STIM", (SX + 108, SY + 68), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, fg, 3, cv2.LINE_AA)
        sub = "IMPULS" if on else ("reshenie prinyato" if waiting else "ozhidanie")
        cv2.putText(canvas, sub, (SX + 16, SY + 104), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, fg, 2, cv2.LINE_AA)
        if waiting:
            cv2.putText(canvas, f"impuls cherez {pend * 1000 / fps:.0f} ms",
                        (SX + 16, SY + 134), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"vsego impulsov: {count}", (SX + 16, SY + 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg if (on or waiting) else C_TXT,
                    1, cv2.LINE_AA)
        lit, pend = max(0, lit - 1), max(0, pend - 1)

        cv2.putText(canvas, f"REAL-TIME  kadr {i}  infer {r['ms']:.1f} ms  "
                    f"zamedlenie x{fps / out_fps:.0f}",
                    (LX, H + PANEL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TXT, 1, cv2.LINE_AA)
        wr.write(canvas)
    cap.release()
    wr.release()
    print(f"импульсов в отрезке: {count}")
    print(f"видео записано: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--target", type=float, default=145.0)
    ap.add_argument("--latency-ms", type=float, default=28.0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=15687)
    ap.add_argument("--roi-width", type=int, default=256)
    ap.add_argument("--event-lag", type=int, default=1)
    ap.add_argument("--pct-med-win", type=int, default=1)
    ap.add_argument("--min-gap", type=float, default=100.0,
                    help="рефрактерность в процентах накопленной фазы")
    ap.add_argument("--no-walk-gate", action="store_true",
                    help="снять запрет поджига на стоянке (для сравнения)")
    ap.add_argument("--save-video", type=Path, default=None)
    ap.add_argument("--render-from", type=int, default=-1, help="-1 = плотный участок")
    ap.add_argument("--render-dur", type=float, default=15.0)
    ap.add_argument("--out-fps", type=float, default=25.0)
    a = ap.parse_args(argv)

    idx = {b: i for i, b in enumerate(BODYPARTS)}
    i_toe, i_body = idx[f"hl_toes_{a.leg}"], idx[f"hl_iliac_{a.leg}"]

    cap = cv2.VideoCapture(str(a.video))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    n = min(n_total - a.start_frame, a.max_frames)

    live = build_live(False)
    det = StreamingPhaseDetector(fps=a.fps)
    pct_est = StreamingPhasePercent(fps=a.fps, target=a.target, latency_ms=a.latency_ms,
                                    event_lag=a.event_lag, med_win=a.pct_med_win,
                                    bout_std=0.0 if a.no_walk_gate else 8.0,
                                    min_gap=a.min_gap)
    leg_idx = [i for i, nm in enumerate(BODYPARTS) if nm.startswith("hl_")]
    tracker = LegRoiTracker(fw, fh, leg_idx, width=a.roi_width)
    print(f"кадр {fw}x{fh}, ROI {a.roi_width}px, старт {a.start_frame}, кадров {n}",
          flush=True)

    t_read, t_inf, t_det, t_pct = [], [], [], []
    fires = np.zeros(n, bool)
    n_fire_standing = 0        # импульсы, выданные при выключенном признаке движения
    rows, loops, loop_id = [], [], -1
    save = a.save_video is not None
    inited = False
    for i in range(n):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            n = i
            break
        w = tracker.window()
        if w is not None:
            live.cropping = w
        if not inited:
            live.init_inference(frame)
            inited = True
        pose = np.asarray(live.get_pose(frame), dtype=np.float32)
        tracker.update(pose)
        w = tracker.window()
        if w:
            pose = pose.copy()
            pose[:, 0] += w[0]
        t2 = time.perf_counter()
        _, ev = det.update(pose[i_toe, 0], pose[i_toe, 2], pose[i_body, 0], pose[i_body, 2])
        t3 = time.perf_counter()
        nb = pct_est.n_rebuild
        p, q, f = pct_est.update(tuple(pose[i_toe]), tuple(pose[i_body]), ev)
        t4 = time.perf_counter()
        fires[i] = f
        if f and not pct_est.walking:
            n_fire_standing += 1
        if save:
            if pct_est.n_rebuild != nb and pct_est.loop is not None:
                loops.append(pct_est.loop[:, :2].copy())
                loop_id = len(loops) - 1
            rows.append({
                "roi": w, "loop": loop_id, "pct": p, "pred": q, "fire": bool(f),
                "walk": pct_est.walking, "ms": (t2 - t1) * 1000.0,
                "proj": getattr(pct_est, "proj_idx", -1) if np.isfinite(p) else -1,
                "relx": pct_est.prev[0] if pct_est.prev else np.nan,
                "rely": pct_est.prev[1] if pct_est.prev else np.nan,
                "toe": (float(pose[i_toe, 0]), float(pose[i_toe, 1]), float(pose[i_toe, 2])),
                "body": (float(pose[i_body, 0]), float(pose[i_body, 1]), float(pose[i_body, 2])),
            })
        t_read.append(t1 - t0); t_inf.append(t2 - t1)
        t_det.append(t3 - t2); t_pct.append(t4 - t3)
    cap.release()

    rd, inf = np.array(t_read[:n]) * 1000, np.array(t_inf[:n]) * 1000
    dt, pc = np.array(t_det[:n]) * 1000, np.array(t_pct[:n]) * 1000
    tot = inf + dt + pc
    print("=" * 74)
    print(f"кадров обработано: {n}, эталон пересобран {pct_est.n_rebuild} раз")
    print(f"поджиг на стоянке: {'РАЗРЕШЁН' if a.no_walk_gate else 'закрыт признаком движения'}")
    print("\n--- задержка по стадиям, мс (медиана / p95) ---")
    for nm, arr in (("чтение кадра", rd), ("инференс DLC", inf),
                    ("детектор событий", dt), ("ПРОЦЕНТ ФАЗЫ", pc), ("ИТОГО контур", tot)):
        print(f"  {nm:<17}: {np.median(arr):7.3f} / {np.percentile(arr, 95):7.3f}")
    print(f"\nдостижимая частота: {1000.0 / np.median(tot):.0f} к/с при целевых {a.fps:.0f}")

    ev_df = pd.read_csv(a.events)
    col = "frame_corrected" if "frame_corrected" in ev_df.columns else "frame"
    if "leg" in ev_df.columns:
        ev_df = ev_df[ev_df.leg == a.leg]
    td = ev_df.loc[ev_df.event == "touch_down", col].values.astype(float)
    lo = ev_df.loc[ev_df.event == "lift_off", col].values.astype(float)
    truth = truth_percent(n_total, td, lo)[a.start_frame:a.start_frame + n]

    lat = pct_est.lat
    land, off = [], 0
    for i in np.where(fires)[0]:
        j = i + lat
        if j >= n:
            continue
        if np.isfinite(truth[j]):
            land.append((truth[j] - a.target + 100.0) % 200.0 - 100.0)
        else:
            off += 1                       # импульс вне размеченного шага
    land = np.asarray(land)
    n_cyc = int(np.sum((truth[:-1] > 150) & (truth[1:] < 50)))
    swing_ms = float(np.median([np.min(td[td > x]) - x for x in lo
                                if (td > x).any() and np.min(td[td > x]) - x < 40])) * 1000 / a.fps
    print("\n--- попадание в цель ---")
    print(f"  импульсов выдано : {int(fires.sum())}, циклов в записи: {n_cyc}")
    # Две РАЗНЫЕ вещи, которые я сначала перепутал:
    #   "на стоянке" - импульс при выключенном признаке движения. Вот это дефект;
    #   "без офлайн-метки" - импульс там, где РАЗМЕТЧИК не дал цикла. Он не даёт
    #   его на 38% записи, и в 62% таких кадров крыса движется, то есть это в
    #   основном отвергнутые разметчиком шаги, а не стояние.
    print(f"  НА СТОЯНКЕ       : {n_fire_standing}")
    print(f"  без офлайн-метки : {off} ({100 * off / max(1, int(fires.sum())):.1f}%)")
    if len(land):
        print(f"  смещение медиана : {np.median(land):+.1f}% "
              f"({np.median(land) * swing_ms / 100:+.1f} мс)")
        print(f"  разброс p90      : {np.percentile(np.abs(land), 90):.1f}% "
              f"({np.percentile(np.abs(land), 90) * swing_ms / 100:.1f} мс)")
        print(f"  покрытие циклов  : {100 * len(land) / max(1, n_cyc):.0f}%")
    print("=" * 74, flush=True)

    if save:
        if a.render_from < 0:
            want = int(a.render_dur * a.fps)
            w_ = np.array([r["walk"] for r in rows], bool)
            best, bi = -1, a.start_frame
            for s in range(0, max(1, len(w_) - want), 25):
                k = int(w_[s:s + want].sum())
                if k > best:
                    best, bi = k, a.start_frame + s
            a.render_from = bi
            print(f"участок выбран автоматически: кадр {bi}")
        hi = min(a.start_frame + n, a.render_from + int(a.render_dur * a.fps))
        render(a.video, a.save_video, rows, loops, a.start_frame,
               a.render_from, hi, a.target, lat, a.fps, a.out_fps)


if __name__ == "__main__":
    main()
