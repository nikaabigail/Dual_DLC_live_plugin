"""
Попадание стимула в заданный процент фазы (постановка О.В. Горского).

Соглашение (уточнено): процент считается ВНУТРИ КАЖДОЙ ФАЗЫ отдельно.
Опора это 0..100%, перенос это 100..200%. Шкала не линейна по времени: при
опоре 290 мс и переносе 100 мс один процент опоры стоит 2.9 мс, а один процент
переноса 1.0 мс. Значит требования к точности в переносе втрое жёстче.

Схема, которую проверяем: часы это ЗДОРОВАЯ нога, стимул уходит на паретичную.
Если паретичная нога не шагает, своей фазы у неё нет, и другой опоры нет.

Задержка контура 39 мс (инференс 17-22 мс + детектор + лаг медианы). Значит
решение нужно принимать ЗАРАНЕЕ: в момент t мы предсказываем фазу на t+39 мс и
поджигаем, когда предсказание пересекает цель. Меряем, где импульс оказался на
самом деле относительно цели.

Сравниваем три стратегии:
  event   - ждём отрыва и отсчитываем фиксированное время. Так работает FSM;
  phase   - непрерывная фаза без упреждения (поджиг по факту пересечения);
  predict - непрерывная фаза + экстраполяция на задержку контура.

Запуск:
    python fire_at_percent.py --csv <DLC_filtered.csv> --events <*.events.csv> --target 146
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gait_phase_labeler as G  # noqa: E402

N_STANCE = N_SWING = 100


def circ(d, period=200.0):
    """Ошибка по кольцу фазы: 199% и 1% различаются на 2%, а не на 198%."""
    return (d + period / 2) % period - period / 2
REF_CYCLES = 10          # эталон насыщается на 10 циклах (проверено развёрткой)


def gt_percent(n, td, lo):
    """Эталонный процент по событиям, нормировка ВНУТРИ каждой фазы."""
    pct = np.full(n, np.nan)
    cyc = np.full(n, -1)
    td = np.asarray(sorted(td), float)
    lo = np.asarray(sorted(lo), float)
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
        cyc[ia:ic] = i
    return pct, cyc


def build_reference(relx, vel, td, lo, upto_frame):
    """Средняя петля по циклам, ЗАКОНЧИВШИМСЯ до upto_frame (только прошлое)."""
    td = np.asarray(sorted(td), float)
    lo = np.asarray(sorted(lo), float)
    cycles = []
    for i in range(len(td) - 1):
        a, c = td[i], td[i + 1]
        if c > upto_frame:
            break
        mid = lo[(lo > a) & (lo < c)]
        if mid.size != 1:
            continue
        b = mid[0]
        ia, ib, ic = int(round(a)), int(round(b)), int(round(c))
        if not (10 < ib - ia < 200 and 5 < ic - ib < 200):
            continue
        seg = []
        for (s, e, k) in ((ia, ib, N_STANCE), (ib, ic, N_SWING)):
            src = np.linspace(0, 1, e - s)
            dst = np.linspace(0, 1, k, endpoint=False)
            seg.append(np.column_stack([np.interp(dst, src, relx[s:e]),
                                        np.interp(dst, src, vel[s:e])]))
        blk = np.vstack(seg)
        if np.isfinite(blk).all():
            cycles.append(blk)
    if len(cycles) < 3:
        return None, 0
    arr = np.stack(cycles[-REF_CYCLES:])
    return np.median(arr, axis=0), len(arr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--leg", default="r")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--target", type=float, default=146.0)
    ap.add_argument("--latency-ms", type=float, default=39.0)
    ap.add_argument("--sweep", action="store_true", help="карта достижимых процентов")
    ap.add_argument("--targets", default=None, help="свой список целей через запятую")
    a = ap.parse_args(argv)

    df = G.load_dlc(a.csv)
    tx = df[(f"hl_toes_{a.leg}", "x")].values.astype(float)
    ix = df[(f"hl_iliac_{a.leg}", "x")].values.astype(float)
    relx = tx - ix
    vel = np.gradient(pd.Series(relx).rolling(3, min_periods=1).median().values) * a.fps

    ev = pd.read_csv(a.events)
    col = "frame_corrected" if "frame_corrected" in ev.columns else "frame"
    if "leg" in ev.columns:
        ev = ev[ev.leg == a.leg]
    td = ev.loc[ev.event == "touch_down", col].values.astype(float)
    lo = ev.loc[ev.event == "lift_off", col].values.astype(float)

    n = len(relx)
    truth, cyc_id = gt_percent(n, td, lo)
    lat = int(round(a.latency_ms / 1000.0 * a.fps))

    td_s, lo_s = np.sort(td), np.sort(lo)
    stance_ms = np.median([lo_s[(lo_s > t) & (lo_s < t + 200)][0] - t
                           for t in td_s[:-1]
                           if len(lo_s[(lo_s > t) & (lo_s < t + 200)])]) / a.fps * 1000
    cycle_ms = float(np.median(np.diff(td_s))) / a.fps * 1000
    swing_ms = cycle_ms - stance_ms
    per_pct = swing_ms / 100.0 if a.target > 100 else stance_ms / 100.0
    print(f"цикл {cycle_ms:.0f} мс, опора {stance_ms:.0f} мс, перенос {swing_ms:.0f} мс")
    print(f"цель {a.target:.0f}% -> {a.target-100:.0f}% переноса = {(a.target-100)*per_pct:.0f} мс после отрыва")
    print(f"1% в этой зоне = {per_pct:.2f} мс, задержка контура {a.latency_ms:.0f} мс = "
          f"{a.latency_ms/per_pct:.0f}% фазы\n")

    # эталон по первым REF_CYCLES циклам, дальше идём каузально
    start = int(td_s[REF_CYCLES + 1]) if len(td_s) > REF_CYCLES + 1 else n
    ref, n_ref = build_reference(relx, vel, td, lo, start)
    if ref is None:
        raise SystemExit("не хватило циклов на эталон")
    sc = np.array([np.ptp(ref[:, 0]), np.ptp(ref[:, 1])])
    R = ref / sc
    print(f"эталон по {n_ref} циклам, каузальный прогон с кадра {start}\n")

    # ---- каузальная оценка фазы -------------------------------------------
    est = np.full(n, np.nan)
    hist = []
    for i in range(start, n):
        c = np.array([relx[i], vel[i]]) / sc
        if not np.isfinite(c).all():
            hist.append(np.nan)
            continue
        p = float(np.argmin(((c - R) ** 2).sum(1))) * (200.0 / len(R))
        hist.append(p)
        est[i] = p
    hist = np.asarray(hist)

    # Скорость фазы НЕЛЬЗЯ мерить скользящим окном по оценке: на отрыве она
    # скачком меняется втрое (100% опоры за 295 мс против 100% переноса за 95),
    # и трейлинговое окно тащит в перенос скорость опоры. Отсюда систематическое
    # опоздание. Берём скорость из длительностей ПОСЛЕДНИХ циклов, раздельно по
    # фазам, и экстраполируем с учётом перехода через границу 100%.
    def recent_rates(frame, k=10):
        """(%/кадр в опоре, %/кадр в переносе) по последним k завершённым циклам."""
        st, sw = [], []
        for t0 in td_s[td_s < frame][-k - 1:]:
            nx = lo_s[(lo_s > t0) & (lo_s < t0 + 200)]
            nt = td_s[(td_s > t0) & (td_s < t0 + 250)]
            if nx.size and nt.size:
                st.append(nx[0] - t0)
                sw.append(nt[0] - nx[0])
        if len(st) < 3:
            return np.nan, np.nan
        return 100.0 / np.median(st), 100.0 / np.median(sw)

    def extrapolate(p, k, r_st, r_sw):
        """Фаза через k кадров, с переходом опора -> перенос -> следующий цикл."""
        left = float(k)
        while left > 0:
            r = r_st if p < 100.0 else r_sw
            edge = 100.0 if p < 100.0 else 200.0
            need = (edge - p) / r
            if need > left:
                return p + r * left
            left -= need
            p = edge % 200.0
        return p

    # ---- три стратегии поджига --------------------------------------------
    # Решение принимается в кадре i, импульс ложится в кадр i+lat (сенсорная
    # задержка уже учтена тем, что est[i] это знание о кадре i). Ошибка - истинная
    # фаза в кадре приземления минус цель.
    def landings(mode, target):
        out, fired = [], set()
        for i in range(start, n - lat):
            c = cyc_id[i + lat]
            if c < 0 or c in fired:
                continue
            p = est[i]
            if not np.isfinite(p):
                continue
            if mode == "predict":
                r_st, r_sw = recent_rates(i)
                if not np.isfinite(r_st):
                    continue
                p = extrapolate(p, lat, r_st, r_sw)
            # полшага упреждения, см. phase_robustness.py: иначе поджиг
            # систематически перелетает цель на половину кадра
            if mode == "predict":
                rr = r_sw if est[i] >= 100.0 else r_st
                half = 0.5 * rr if np.isfinite(rr) else 0.0
            else:
                half = 0.0
            if target - half <= p < target + 40:
                fired.add(c)
                t = truth[i + lat]
                if np.isfinite(t):
                    out.append(circ(t - target))
        return np.asarray(out), len(fired)

    # Базовая стратегия FSM: дождаться отрыва и отсчитать фиксированную выдержку.
    # Отрыв становится известен через lat кадров после того, как случился.
    def landings_event(target):
        out, fired, late = [], set(), 0
        anchor = lo_s if target >= 100.0 else td_s
        for t0 in anchor:
            L = int(round(t0))
            know = L + lat                       # когда мы узнали о событии
            r_st, r_sw = recent_rates(L)
            if not np.isfinite(r_sw):
                continue
            off = (target - 100.0) / r_sw if target >= 100.0 else target / r_st
            want = L + off                       # когда надо положить импульс
            if want < know:
                late += 1                        # цель прошла раньше, чем узнали
            fire = max(know, int(round(want)))
            if fire >= n or fire < start:
                continue
            c = cyc_id[fire]
            if c < 0 or c in fired:
                continue
            fired.add(c)
            if np.isfinite(truth[fire]):
                out.append(circ(truth[fire] - target))
        return np.asarray(out), len(fired)

    n_cyc = len(np.unique(cyc_id[cyc_id >= 0]))
    if a.targets:
        targets = [float(x) for x in a.targets.split(",")]
    elif a.sweep:
        targets = [10, 30, 50, 70, 90, 105, 110, 120, 130, 146, 160, 180, 195]
    else:
        targets = [a.target]

    print(f"{'цель':>6} {'мс от события':>14} | {'по событию (FSM)':>28} | "
          f"{'по фазе + упреждение':>28}")
    print(f"{'':>6} {'':>14} | {'смещ.':>8} {'p90':>7} {'циклов':>10} | "
          f"{'смещ.':>8} {'p90':>7} {'циклов':>10}")
    print("-" * 96)
    for tg in targets:
        pp = swing_ms / 100.0 if tg >= 100 else stance_ms / 100.0
        off_ms = (tg - 100) * pp if tg >= 100 else tg * pp
        line = f"{tg:>5.0f}% {off_ms:>13.0f} |"
        for fn, mode in ((landings_event, None), (landings, "predict")):
            e, nf = fn(tg) if mode is None else fn(mode, tg)
            if len(e) < 5:
                line += f" {'не достижимо':>28} |"
                continue
            med, p90 = np.median(e), np.percentile(np.abs(e), 90)
            line += (f" {med:>+7.1f}% {p90:>6.1f}% {100*nf/n_cyc:>9.0f}% |")
        print(line)
    print(f"\nмс от события = сколько ждать после отрыва (цель>=100%) или "
          f"постановки (цель<100%)")
    print(f"смещ. = медиана (где импульс лёг минус цель), плюс = опоздали; "
          f"p90 = разброс |ошибки|")
    print(f"циклов = доля циклов, в которые удалось положить импульс "
          f"(всего циклов {n_cyc})")
    print(f"задержка контура {a.latency_ms:.0f} мс = {a.latency_ms/(swing_ms/100):.0f}% "
          f"переноса или {a.latency_ms/(stance_ms/100):.0f}% опоры")


if __name__ == "__main__":
    main()
