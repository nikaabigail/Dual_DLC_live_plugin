"""
Симуляция реального времени: видео подаётся как поток «с камеры», фаза шага
определяется КАУЗАЛЬНО, результат сверяется с офлайн-разметкой (ground truth).

Зачем: офлайн-разметчик (gait_phase_labeler) использует будущее - центрированные
окна сглаживания, интерполяцию пропусков, уточнение событий по обе стороны от
перехода и калибровочный сдвиг события НАЗАД во времени. В реальном контуре
ничего этого нет. Поэтому здесь отдельный потоковый детектор, а разница с
офлайн-метками и есть цена каузальности.

Что меряется:
  1) задержка по стадиям: чтение кадра -> инференс DLC -> обновление детектора;
  2) достижимая частота и сколько кадров пришлось бы отбросить при 100 Гц
     (режим NEWEST_ONLY боевого контура);
  3) точность: совпадение фазы с офлайн-разметкой и смещение событий.

Запуск:
    python realtime_phase_sim.py --video <.avi> --gt <*.phases.csv> [--max-frames N]
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_PATH = (r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch"
              r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5"
              r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt")
BODYPARTS = ["nose", "eye_l", "eye_r", "fl_toes_l", "fl_toes_r",
             "hl_toes_l", "hl_ankle_l", "hl_hip_l", "hl_iliac_l",
             "hl_toes_r", "hl_ankle_r", "hl_hip_r", "hl_iliac_r", "spine", "tail"]

STANCE, SWING, UNKNOWN = "stance", "swing", "unknown"


class StreamingPhaseDetector:
    """
    Каузальный детектор фазы: ни одного отсчёта из будущего.

    Отличия от офлайн-версии и их цена:
      * медиана СКОЛЬЗЯЩАЯ НАЗАД (не центрированная) - даёт лаг ~(k-1)/2 кадров;
      * скорость ленты оценивается по накопленному окну, а не по всему эпизоду;
      * переход подтверждается N кадрами - иначе дребезг, но это добавляет N
        кадров задержки (та же плата, что у FSM в ЭМГ-тракте);
      * калибровочный сдвиг события назад (-10 мс) ПРИНЦИПИАЛЬНО неприменим:
        сдвинуть событие в прошлое в реальном времени нельзя.
    """

    # Значения med_win / confirm подобраны развёрткой (sweep_detector_latency.py).
    #
    # confirm=1: подтверждение перехода оказалось вредным. Сигнал R почти
    # бинарный (крутые фронты), дребезга на переходах нет, поэтому confirm не
    # фильтровал ложные события, а терял настоящие ("лишних" падало НИЖЕ 1.0).
    #
    # med_win=1 (было 5, потом 3): каждый лишний кадр медианы это лишние 10 мс
    # лага, а качество он не добавляет. Проверено на ПЯТИ записях, med_win 1
    # против 3, совпадение после компенсации лага:
    #     запись 5: 96.7 / 96.7      запись 6: 90.7 / 90.7
    #     запись 13: 92.2 / 93.0     запись 8: 95.3 / 95.6
    #     запись 17: 95.7 / 95.7
    # recall и доля "лишних" событий совпадают везде; сырое совпадение (без
    # компенсации лага) у med_win=1 ВЫШЕ на 4-5 п.п., потому что метка не
    # запаздывает. Максимальная потеря 0.8 п.п. на одной записи, цена - 10 мс.
    #
    # Зачем эти 10 мс: задержка контура задаёт, в какой самый ранний процент
    # фазы мы способны положить импульс. 39 мс это 41% переноса, то есть всё
    # раньше 137% недостижимо. 28 мс опускают порог до ~129% (fire_at_percent.py).
    def __init__(self, fps=100.0, med_win=1, belt_win_s=3.0, bout_win=50,
                 bout_std=8.0, r_hi=1.6, r_lo=0.8, confirm=1,
                 min_stance=12, min_swing=8, p_use=0.20):
        self.fps = fps
        self.med = deque(maxlen=med_win)
        self.vel = deque(maxlen=int(belt_win_s * fps))
        self.rel = deque(maxlen=bout_win)
        self.bout_std = bout_std
        self.r_hi, self.r_lo = r_hi, r_lo
        self.confirm = confirm
        self.min_stance, self.min_swing = min_stance, min_swing
        self.p_use = p_use
        self.prev_x = None
        self.state = None
        self.dwell = 0
        self.pending = None
        self.pending_n = 0
        self.n = 0
        self._belt = np.nan
        self._belt_age = 10 ** 9

    def _belt_velocity(self):
        """
        Скорость ленты по накопленному окну, БЕЗ ПРИВЯЗКИ К НАПРАВЛЕНИЮ.

        Опора занимает ~70% времени и идёт ровно со скоростью ленты, значит это
        самый населённый пик гистограммы скоростей. Прежняя версия брала медиану
        нижней половины окна и тем самым молча предполагала, что носок в опоре
        едет в -x. При двусторонней съёмке одна камера всегда зеркальна, там
        оценка цеплялась за кластер переноса (на записи 4: -28 px/с вместо +439)
        и детектор выдавал правдоподобный мусор, а не отказ.

        Пересчитываем не каждый кадр: скорость ленты меняется медленно, а окно
        3 с. Раз в 10 кадров хватает, и стоимость гистограммы размазывается.
        """
        if self._belt_age < 10 and np.isfinite(self._belt):
            self._belt_age += 1
            return self._belt
        arr = np.asarray(self.vel)
        arr = arr[np.isfinite(arr)]
        arr = arr[np.abs(arr) > 60.0]              # выбрасываем стояние на месте
        if arr.size < 50:
            return np.nan
        lo, hi = np.percentile(arr, [1, 99])
        if hi - lo < 1e-6:
            self._belt, self._belt_age = float(np.median(arr)), 0
            return self._belt
        h, edges = np.histogram(arr, bins=61, range=(lo, hi))
        c = 0.5 * (edges[:-1] + edges[1:])[int(np.argmax(h))]
        sel = arr[np.abs(arr - c) <= (hi - lo) / 61 * 3]
        self._belt = float(np.median(sel)) if sel.size else float(c)
        self._belt_age = 0
        return self._belt

    def update(self, toe_x, toe_p, body_x, body_p):
        """Один кадр -> (фаза, событие или None). Только прошлое."""
        self.n += 1
        event = None
        if not np.isfinite(toe_x) or toe_p < self.p_use or body_p < self.p_use:
            return (self.state or UNKNOWN), None

        self.med.append(float(toe_x))
        xs = float(np.median(self.med))
        self.rel.append(xs - float(body_x))

        v = np.nan if self.prev_x is None else (xs - self.prev_x) * self.fps
        self.prev_x = xs
        if not np.isfinite(v):
            return (self.state or UNKNOWN), None
        self.vel.append(v)

        # шагает ли: разброс положения носка относительно тела за последнее окно
        if len(self.rel) < self.rel.maxlen or np.std(self.rel) < self.bout_std:
            self.state, self.dwell, self.pending, self.pending_n = None, 0, None, 0
            return UNKNOWN, None

        if len(self.vel) < self.fps:
            return UNKNOWN, None
        v_belt = self._belt_velocity()
        if not np.isfinite(v_belt) or abs(v_belt) < 20:
            return UNKNOWN, None

        sgn = -1.0 if v_belt > 0 else 1.0
        R = sgn * (v - v_belt) / abs(v_belt)

        target = None
        if self.state is None:
            target = SWING if R >= self.r_hi else STANCE
        elif self.state == STANCE and R >= self.r_hi:
            target = SWING
        elif self.state == SWING and R <= self.r_lo:
            target = STANCE

        if target is None or target == self.state:
            self.pending, self.pending_n = None, 0
            self.dwell += 1
            return (self.state or UNKNOWN), None

        need = self.min_stance if self.state == STANCE else self.min_swing
        if self.state is not None and self.dwell < need:
            return self.state, None                      # минимальная выдержка

        if self.pending != target:
            self.pending, self.pending_n = target, 1
        else:
            self.pending_n += 1
        if self.pending_n < self.confirm:
            return (self.state or UNKNOWN), None          # подтверждение

        prev = self.state
        self.state, self.dwell = target, 0
        self.pending, self.pending_n = None, 0
        if prev is not None:
            event = "touch_down" if target == STANCE else "lift_off"
        return self.state, event


class LegRoiTracker:
    """
    Скользящий ROI ФИКСИРОВАННОЙ ширины - копия боевой реализации из
    single_rt_dlc_live_bridge.py (origin/main). Отличие от стокового dynamic
    в DLCLive принципиальное: ширина окна НИКОГДА не меняется и полного кадра
    не бывает, поэтому форма входа модели постоянна и cudagraphs остаётся жив.
    При потере точек окно HOLD-ит на месте ~1 с (развороты/окклюзии, где ноги
    пропадают и появляются там же), и только потом SWEEP-ит по кадру.
    """

    def __init__(self, frame_w, frame_h, leg_indices, width=256,
                 detect_thresh=0.30, hold_frames=100, center_ema=0.35):
        self.fw, self.fh = int(frame_w), int(frame_h)
        self.leg_idx = list(leg_indices)
        self.width = max(16, min(int(width), self.fw))
        self.thresh = float(detect_thresh)
        self.hold_frames = int(hold_frames)
        self.ema = float(center_ema)
        self.cx = self.fw / 2.0
        self.misses = 0
        self.n_hold = 0
        self.n_sweep = 0

    def window(self):
        if self.cx is None:
            return None
        half = self.width // 2
        x1 = int(round(self.cx)) - half
        x1 = max(0, min(x1, self.fw - self.width))
        return [x1, x1 + self.width, 0, self.fh]

    def update(self, pose):
        if not self.leg_idx:
            return
        pts = np.asarray(pose)[self.leg_idx]
        visible = pts[pts[:, 2] >= self.thresh]
        if len(visible) >= 1:
            cx_new = float(np.mean(visible[:, 0]))
            if self.cx is None or self.misses > 0:
                self.cx = cx_new
            else:
                self.cx = self.ema * cx_new + (1.0 - self.ema) * self.cx
            self.misses = 0
        else:
            self.misses += 1
            if self.misses <= self.hold_frames:
                self.n_hold += 1
            else:
                self.n_sweep += 1
                base = self.cx if self.cx is not None else self.width / 2.0
                self.cx = base + self.width
                if self.cx > self.fw:
                    self.cx = self.width / 2.0


PAL = {"stance": (171, 225, 159), "swing": (117, 199, 250), "unknown": (228, 232, 232)}


def render_video(video, out_path, rows, gt_phase, gt_event, fps, roi_w, start=0):
    """
    Видео того, что видит контур: окно ROI, точки, решение ПОТОКА и решение
    офлайн-эталона рядом. Нужно, чтобы оценить работу глазами до эксперимента.
    """
    cap = cv2.VideoCapture(str(video))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PANEL = 132
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + PANEL))
    n = len(rows)
    # заранее отрисованные полосы фаз (поток и эталон)
    bx = 470
    strip_w = W - bx - 24
    idx = (np.arange(strip_w) / max(1, strip_w - 1) * (n - 1)).astype(int)
    band_s = np.zeros((16, strip_w, 3), np.uint8)
    band_g = np.zeros((16, strip_w, 3), np.uint8)
    for k, j in enumerate(idx):
        band_s[:, k] = PAL.get(str(rows[j]["phase"]), PAL["unknown"])
        band_g[:, k] = PAL.get(str(gt_phase[j]) if j < len(gt_phase) else "unknown", PAL["unknown"])

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.full((H + PANEL, W, 3), 255, np.uint8)
        canvas[:H] = frame
        r = rows[i]
        # окно ROI - то, что реально подаётся в модель
        if r["roi"] is not None:
            x1, x2, y1, y2 = r["roi"]
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (48, 90, 216), 2)
            cv2.putText(canvas, f"ROI {roi_w}px", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (48, 90, 216), 1, cv2.LINE_AA)
        for key, col, rad in (("body", (120, 120, 120), 4), ("toe", (48, 90, 216), 5)):
            x, y, p = r[key]
            if p >= 0.2 and np.isfinite(x):
                cv2.circle(canvas, (int(x), int(y)), rad, col, -1, cv2.LINE_AA)
        # рамка кадра по решению ПОТОКА
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), PAL.get(str(r["phase"]), PAL["unknown"]), 5)
        if r["event"]:
            c = (117, 158, 29) if r["event"] == "touch_down" else (48, 90, 216)
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), c, 11)

        y0 = H
        gp = str(gt_phase[i]) if i < len(gt_phase) else "unknown"
        ge = str(gt_event[i]) if i < len(gt_event) else ""
        for row, (lbl, ph, ev, band) in enumerate((
                ("POTOK (real-time)", str(r["phase"]), r["event"] or "", band_s),
                ("ETALON (offline)", gp, ge, band_g))):
            yy = y0 + 14 + row * 56
            cv2.rectangle(canvas, (12, yy), (250, yy + 40), PAL.get(ph, PAL["unknown"]), -1)
            cv2.putText(canvas, ph.upper(), (22, yy + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.putText(canvas, lbl, (262, yy + 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (110, 110, 110), 1, cv2.LINE_AA)
            if ev:
                cv2.putText(canvas, ev.upper(), (262, yy + 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (48, 90, 216) if "lift" in ev else (117, 158, 29), 2, cv2.LINE_AA)
            canvas[yy + 12: yy + 28, bx: bx + strip_w] = band
            cur = int(bx + strip_w * i / max(1, n - 1))
            cv2.line(canvas, (cur, yy + 8), (cur, yy + 32), (0, 0, 0), 2)
        cv2.putText(canvas, f"frame {i}  t={i / fps:.2f}s  loop={r['ms']:.1f} ms",
                    (W - 300, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        wr.write(canvas)
    cap.release()
    wr.release()
    print(f"видео записано: {out_path}", flush=True)


def build_live(dynamic=False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    from dlclive import DLCLive
    return DLCLive(model_path=MODEL_PATH, model_type="pytorch", precision="FP32",
                   single_animal=True, device="cuda", cropping=None,
                   dynamic=((True, 0.5, 10) if dynamic else (False, 0.5, 10)),
                   resize=1.0, processor=None,
                   convert2rgb=True, display=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Симуляция реального времени + сверка с офлайн-разметкой")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path, help="*.phases.csv от офлайн-разметчика")
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--confirm", type=int, default=1)
    ap.add_argument("--med-win", type=int, default=3)
    ap.add_argument("--save-video", type=Path, default=None,
                    help="отрендерить видео: окно ROI, точки, поток против эталона")
    ap.add_argument("--dynamic", action="store_true",
                    help="стоковый dynamic из DLCLive (НЕ боевой вариант)")
    ap.add_argument("--leg-roi", action="store_true",
                    help="боевой скользящий ROI фиксированной ширины (LegRoiTracker)")
    ap.add_argument("--roi-width", type=int, default=256)
    a = ap.parse_args(argv)

    gt = pd.read_csv(a.gt)
    pcol = f"phase_{a.leg}" if f"phase_{a.leg}" in gt.columns else "phase"
    ecol = f"event_{a.leg}" if f"event_{a.leg}" in gt.columns else "event"
    gt_phase = gt[pcol].astype(str).values
    gt_event = gt[ecol].fillna("").astype(str).values

    idx = {b: i for i, b in enumerate(BODYPARTS)}
    i_toe, i_body = idx[f"hl_toes_{a.leg}"], idx[f"hl_iliac_{a.leg}"]

    cap = cv2.VideoCapture(str(a.video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
        gt_phase = gt_phase[a.start_frame:]
        gt_event = gt_event[a.start_frame:]
    n = min(n_total - a.start_frame, a.max_frames) if a.max_frames else n_total - a.start_frame
    live = build_live(a.dynamic)
    if a.dynamic:
        print("режим: динамический ROI (как в боевом контуре)")
    det = StreamingPhaseDetector(fps=a.fps, confirm=a.confirm, med_win=a.med_win)
    tracker = None
    if a.leg_roi:
        leg_idx = [i for i, nm in enumerate(BODYPARTS) if nm.startswith("hl_")]
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracker = LegRoiTracker(fw, fh, leg_idx, width=a.roi_width)
        print(f"режим: боевой скользящий ROI {a.roi_width}px, кадр {fw}x{fh}, "
              f"точек-якорей {len(leg_idx)}", flush=True)

    t_read, t_infer, t_det = [], [], []
    rec_rows = []                      # для видеовыгрузки
    pred_phase = np.array([UNKNOWN] * n, dtype=object)
    pred_events = []
    inited = False
    for i in range(n):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        if tracker is not None:
            w = tracker.window()
            if w is not None:
                live.cropping = w
        if not inited:
            live.init_inference(frame)
            inited = True
        pose = np.asarray(live.get_pose(frame), dtype=np.float32)
        if tracker is not None:
            tracker.update(pose)
        t2 = time.perf_counter()
        ph, ev = det.update(pose[i_toe, 0], pose[i_toe, 2], pose[i_body, 0], pose[i_body, 2])
        t3 = time.perf_counter()
        pred_phase[i] = ph
        if ev:
            pred_events.append((i, ev))
        if a.save_video is not None:
            rec_rows.append({
                "roi": (tracker.window() if tracker is not None else None),
                "toe": (float(pose[i_toe, 0]), float(pose[i_toe, 1]), float(pose[i_toe, 2])),
                "body": (float(pose[i_body, 0]), float(pose[i_body, 1]), float(pose[i_body, 2])),
                "phase": ph, "event": ev, "ms": (t3 - t1) * 1000.0,
            })
        t_read.append(t1 - t0); t_infer.append(t2 - t1); t_det.append(t3 - t2)
    cap.release()

    rd, inf, dt = np.array(t_read) * 1000, np.array(t_infer) * 1000, np.array(t_det) * 1000
    tot = rd + inf + dt
    print("=" * 72, flush=True)
    print(f"кадров обработано: {len(tot)} из {n_total}")
    if tracker is not None:
        print(f"ROI: кадров в HOLD {tracker.n_hold}, в SWEEP {tracker.n_sweep} "
              f"({100 * tracker.n_sweep / max(1, len(tot)):.1f}% времени в поиске)")
    print("\n--- задержка по стадиям, мс (медиана / p95) ---")
    print(f"  чтение кадра : {np.median(rd):6.2f} / {np.percentile(rd, 95):6.2f}")
    print(f"  инференс DLC : {np.median(inf):6.2f} / {np.percentile(inf, 95):6.2f}")
    print(f"  детектор фазы: {np.median(dt):6.3f} / {np.percentile(dt, 95):6.3f}")
    print(f"  ИТОГО        : {np.median(tot):6.2f} / {np.percentile(tot, 95):6.2f}")
    fps_eff = 1000.0 / np.median(tot)
    print(f"\nдостижимая частота: {fps_eff:.1f} к/с при целевых {a.fps:.0f}")
    if fps_eff < a.fps:
        drop = 100 * (1 - fps_eff / a.fps)
        print(f"  -> при подаче {a.fps:.0f} Гц пришлось бы отбрасывать {drop:.0f}% кадров (NEWEST_ONLY)")

    m = min(len(pred_phase), len(gt_phase))
    p, g = pred_phase[:m], gt_phase[:m]
    lab = (g == STANCE) | (g == SWING)
    both = lab & ((p == STANCE) | (p == SWING))
    print("\n--- точность против офлайн-разметки ---")
    print(f"  кадров с эталонной меткой: {lab.sum()} ({100 * lab.mean():.1f}%)")
    if both.sum():
        acc = 100 * np.mean(p[both] == g[both])
        print(f"  совпадение фазы там, где размечены оба: {acc:.1f}% (n={both.sum()})")
    only_gt = lab & ~((p == STANCE) | (p == SWING))
    print(f"  эталон есть, поток молчит: {only_gt.sum()} кадров ({100 * only_gt.sum() / max(1, lab.sum()):.1f}%)")
    # Сколько несовпадений - просто постоянный лаг, а сколько настоящая ошибка?
    best = (0, -1.0)
    for sh in range(0, 13):
        pp, gg = p[sh:], g[:len(g) - sh] if sh else g
        k = min(len(pp), len(gg))
        mk = ((gg[:k] == STANCE) | (gg[:k] == SWING)) & ((pp[:k] == STANCE) | (pp[:k] == SWING))
        if mk.sum() > 100:
            acc_s = 100 * np.mean(pp[:k][mk] == gg[:k][mk])
            if acc_s > best[1]:
                best = (sh, acc_s)
    print(f"  после компенсации лага {best[0]} кадров ({best[0] / a.fps * 1000:.0f} мс): {best[1]:.1f}%")

    gt_ev = {"touch_down": np.where(gt_event[:m] == "touch_down")[0],
             "lift_off": np.where(gt_event[:m] == "lift_off")[0]}
    print("\n--- события: смещение потока относительно офлайн-разметки ---")
    for name in ("touch_down", "lift_off"):
        D = np.array([f for f, e in pred_events if e == name and f < m], dtype=float)
        T = gt_ev[name].astype(float)
        if not D.size or not T.size:
            print(f"  {name:11s}: поток {D.size}, эталон {T.size}")
            continue
        errs, used = [], set()
        for t in T:
            d = np.abs(D - t)
            j = int(np.argmin(d))
            if d[j] <= 0.15 * a.fps and j not in used:
                used.add(j)
                errs.append(D[j] - t)
        errs = np.array(errs) / a.fps * 1000
        if errs.size:
            print(f"  {name:11s}: поток {D.size} / эталон {int(T.size)} | recall {100 * len(errs) / len(T):.0f}% | "
                  f"смещение {np.median(errs):+.1f} мс (IQR {np.percentile(errs, 25):+.0f}..{np.percentile(errs, 75):+.0f})")
    if a.save_video is not None and rec_rows:
        render_video(a.video, a.save_video, rec_rows, gt_phase, gt_event, a.fps,
                     a.roi_width if a.leg_roi else 0, a.start_frame)

    print("=" * 72)


if __name__ == "__main__":
    main()
