"""
Сколько проб влезает в сессию и каким должен быть размер блока.

ЗАЧЕМ. Медленная петля планировалась «около 60 проб на животное», но проба - это
не абстракция, а блок из N подряд идущих шагов. Значит вопрос упирается не в
арифметику «10 шагов по 390 мс = 4 секунды», а в то, ходит ли животное 10 шагов
подряд и сколько раз за сессию. Это видно прямо в записях.

ЧТО СЧИТАЕТСЯ. По каждой записи: эпизоды непрерывной локомоции (bout_id
разметчика), сколько в каждом полных циклов, сколько складывается блоков по N,
какие паузы животное делает само. И отдельно - фактический разброс счёта между
блоками, то есть тот шум, против которого работает оптимизатор.

ЧТО ВЫШЛО на 12 записях (31 мин, 1996 циклов, интактное животное):

  * эпизод локомоции: медиана 7 циклов, 75-й перцентиль 20, максимум 194.
    До десяти шагов подряд дотягивают только 42% эпизодов - вот настоящее
    ограничение, а не длительность самой пробы;
  * блоков по 10 шагов набирается 158, то есть 5.1 пробы на минуту записи.
    Шестьдесят проб - это около 12 минут ходьбы, а не час;
  * паузы животного: медиана 2.1 с. Отдых между пробами не надо назначать,
    животное берёт его само;
  * разброс между записями огромный: от 4 блоков (запись 1, 14 эпизодов по 4
    цикла) до 25 (запись 5). Планировать надо по худшим.

РАЗБРОС СЧЁТА ОКАЗАЛСЯ НЕ ТАКИМ, КАК ЗАЛОЖЕНО. Стенд оптимизатора считает шум
0.06 на блоке из 10 шагов и пересчитывает его как корень из числа шагов.
Измерение на записях:

    шагов   разброс счёта   формула 0.06*sqrt(10/n)
        4       0.047            0.095
        6       0.037            0.077
        8       0.037            0.067
       10       0.033            0.060
       15       0.028            0.049
       20       0.029            0.042

Два вывода. Во-первых, фактический разброс вдвое меньше заложенного, то есть
стенд был пессимистичен. Во-вторых, шум НЕ падает как корень: от 4 шагов к 20
он снижается в 1.6 раза вместо 2.2. Значит есть слагаемое, которое усреднением
не убирается - медленный дрейф внутри записи и разница между эпизодами. То же
самое, что моделирует validate_stim_drift.py.

ОТСЮДА РАЗМЕР БЛОКА. Короткий блок шумнее, но проб даёт больше. Побеждает не
меньший шум и не большее число проб, а меньшее произведение. С измеренным шумом
(30 прогонов, время до 80% успеха):

    шагов   проб/мин   проб до 80%   минут записи
        4      14.9         44           3.0
        6       9.3         40           4.3
        8       6.7         40           6.0
       10       5.1         45           8.8

Формально выигрывает блок из четырёх шагов, но на нём счёт считается ровно по
трём циклам - это минимум, который принимает gait_score.measure, и один
забракованный цикл ломает пробу целиком. Рабочая точка - ШЕСТЬ шагов: почти так
же быстро и с запасом по числу циклов.

Запуск:
    python session_budget.py --videos C:/dlc/videos --labeler C:/dlc/DLC_OBS_Spinal_cord_stimulation
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

SIZES = (4, 6, 8, 10, 15, 20)


def cycles_with_frames(ev: pd.DataFrame, fps: float) -> np.ndarray:
    """Полные циклы td -> lo -> td: (кадр начала, длительность мс)."""
    if ev.empty:
        return np.empty((0, 2))
    col = "frame_corrected" if "frame_corrected" in ev else "frame"
    arr = ev.sort_values(col)[[col, "event"]].values
    out = []
    for i in range(len(arr) - 2):
        if (arr[i][1] == "touch_down" and arr[i + 1][1] == "lift_off"
                and arr[i + 2][1] == "touch_down"):
            c = arr[i + 2][0] - arr[i][0]
            if 0.15 * fps < c < 1.2 * fps:
                out.append((arr[i][0], c / fps * 1000.0))
    return np.asarray(out) if out else np.empty((0, 2))


def blocks_of(td, lo, n: int, fps: float, max_gap_ms: float):
    """
    Блоки из n подряд идущих циклов, не перепрыгивая паузы.

    Проверка на разрыв обязательна: без неё блок склеивает шаги до и после
    остановки, и разброс счёта получается завышенным - меряется не шум пробы,
    а факт паузы посередине.
    """
    td = np.asarray(td, float)
    lo = np.asarray(lo, float)
    out, i = [], 0
    while i + n < len(td):
        seg = td[i:i + n + 1]
        if np.all(np.diff(seg) / fps * 1000.0 < max_gap_ms):
            sl = lo[(lo > seg[0]) & (lo < seg[-1])]
            if len(sl) >= n - 1:
                out.append((seg, sl))
            i += n
        else:
            i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True,
                    help="папка с *_filtered.csv от batch_offline_pose")
    ap.add_argument("--labeler", type=Path, required=True,
                    help="папка с gait_phase_labeler.py и gait_score.py")
    ap.add_argument("--out", type=Path, default=Path("session_budget"))
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--block", type=int, default=10, help="шагов в пробе")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(a.labeler))
    import gait_phase_labeler as G                      # noqa: E402
    from gait_score import measure, score               # noqa: E402
    from validate_gait_score import load as load_rec    # noqa: E402

    a.out.mkdir(parents=True, exist_ok=True)
    rows, bouts_all, gaps_all, noise = [], [], [], {n: [] for n in SIZES}

    for csv in sorted(a.videos.glob("*_filtered.csv")):
        raw = Path(str(csv).replace("_filtered.csv", ".csv"))
        prefix = a.out / csv.stem[:18]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ph, ev = G.run(csv, raw if raw.exists() else None,
                               G.Params(fps=a.fps), "auto", prefix)
        except Exception as e:
            print(f"  пропуск {csv.stem[:22]}: {type(e).__name__}: {e}", flush=True)
            continue
        if ev.empty:
            continue

        # Разметчик отдаёт обе ноги колонками phase_l/bout_l и phase_r/bout_r.
        # Берём ту, у которой событий больше: это ближняя к камере нога.
        leg = ev.leg.value_counts().idxmax()
        ev_leg = ev[ev.leg == leg]
        phl = ph.rename(columns={f"phase_{leg}": "phase", f"bout_{leg}": "bout_id"})

        cyc = cycles_with_frames(ev_leg, a.fps)
        if not len(cyc):
            continue
        bout_of = phl.set_index("frame")["bout_id"].to_dict()
        bid = np.array([bout_of.get(int(f), -1) for f in cyc[:, 0]])
        per_bout = np.array([int((bid == b).sum()) for b in np.unique(bid[bid >= 0])])
        blocks = int(sum(per_bout // a.block))

        # паузы животного: сколько подряд кадров без локомоции
        lab = phl.phase.isin(["stance", "swing"]).values
        gaps, run = [], 0
        for v in lab:
            if not v:
                run += 1
            else:
                if run:
                    gaps.append(run / a.fps)
                run = 0
        gaps = np.array([g for g in gaps if g > 0.5])

        # фактический разброс счёта между блоками
        evp = prefix.with_suffix("")
        evp = Path(str(evp) + ".events.csv")
        line = ""
        if evp.exists():
            try:
                tx, ty, bx, by, td, lo, l_ref = load_rec(csv, evp, leg)
                cyc_ms = float(np.median(np.diff(np.asarray(td, float)))) / a.fps * 1000
                parts = []
                for n in SIZES:
                    sc = []
                    for seg, sl in blocks_of(td, lo, n, a.fps, 2.5 * cyc_ms):
                        try:
                            sc.append(score(measure(tx, ty, bx, by, seg, sl,
                                                    l_ref, a.fps)))
                        except Exception:
                            pass
                    if len(sc) >= 4:
                        sd = float(np.std(sc, ddof=1))
                        noise[n].append((sd, len(sc)))
                        parts.append(f"n{n}:{sd:.3f}")
                line = "  " + " ".join(parts)
            except Exception as e:
                line = f"  счёт не посчитан: {type(e).__name__}"

        rows.append(dict(
            video=csv.stem.split("DLC")[0][:22], leg=leg,
            sec=round(len(phl) / a.fps, 1), cycles=len(cyc), bouts=len(per_bout),
            bout_med=int(np.median(per_bout)) if per_bout.size else 0,
            bout_max=int(per_bout.max()) if per_bout.size else 0,
            long_bouts=int((per_bout >= a.block).sum()), blocks=blocks,
            gap_med=round(float(np.median(gaps)), 1) if gaps.size else np.nan))
        bouts_all.append(per_bout)
        if gaps.size:
            gaps_all.append(gaps)
        print(f"  {rows[-1]['video']:<24} циклов {len(cyc):>4}, эпизодов "
              f"{len(per_bout):>3}, блоков по {a.block}: {blocks:>3}{line}",
              flush=True)

    if not rows:
        raise SystemExit("ничего не разметилось")
    df = pd.DataFrame(rows)
    total_min = df.sec.sum() / 60
    print("\n" + "=" * 104)
    print(df.to_string(index=False))
    print("=" * 104)

    pb = np.concatenate(bouts_all)
    gp = np.concatenate(gaps_all) if gaps_all else np.array([])
    print(f"\nзаписей {len(df)}, суммарно {total_min:.1f} мин, циклов {df.cycles.sum()}")
    print(f"эпизод локомоции, циклов: медиана {np.median(pb):.0f}, "
          f"75-й {np.percentile(pb, 75):.0f}, максимум {pb.max():.0f}")
    for n in (5, 10, 15, 20):
        print(f"  эпизодов >= {n:>2} шагов: {100 * np.mean(pb >= n):>4.0f}%  "
              f"({int((pb >= n).sum())} из {pb.size})")
    if gp.size:
        print(f"паузы животного, с: медиана {np.median(gp):.1f}, "
              f"IQR {np.percentile(gp, 25):.1f}-{np.percentile(gp, 75):.1f}")
    print(f"\nблоков по {a.block} шагов: {df.blocks.sum()} за {total_min:.1f} мин "
          f"= {df.blocks.sum() / total_min:.1f} проб в минуту")

    print("\n" + "=" * 72)
    print(f"{'шагов':>6} {'проб за сессию':>15} {'проб/мин':>9} "
          f"{'разброс счёта':>14} {'формула':>10}")
    print("-" * 72)
    for n in SIZES:
        blk = int(sum(int(b) // n for b in pb))
        sd = np.median([x[0] for x in noise[n]]) if noise[n] else np.nan
        print(f"{n:>6} {blk:>15} {blk / total_min:>9.1f} {sd:>14.3f} "
              f"{0.06 * (10.0 / n) ** 0.5:>10.3f}")
    print("\nформула - это допущение стенда (шум как корень из числа шагов).")
    print("Расхождение с измерением означает слагаемое, которое усреднением не")
    print("убирается: дрейф внутри записи и разница между эпизодами.")

    df.to_csv(a.out / "session_budget.csv", index=False, encoding="utf-8")
    print(f"\nзаписано: {a.out / 'session_budget.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
