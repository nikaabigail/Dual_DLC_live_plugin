"""
Пакетный ОФЛАЙН-инференс позы по видео через DLCLive.

Зачем не deeplabcut.analyze_videos: полного DeepLabCut в рабочем окружении нет
(стоит только deeplabcut-live), а ставить его в закреплённый боевой env нельзя -
он потянет свои зависимости и может сломать пины torch. Модель уже
экспортирована (.pt), поэтому для инференса достаточно DLCLive - и это ровно та
же модель и тот же путь, что в realtime-контуре.

Результат пишется в формате DLC CSV (та же трёхстрочная шапка scorer/bodyparts/
coords), поэтому gait_phase_labeler.py и overlay_gait_phases.py читают его без
изменений. Дополнительно пишется *_filtered.csv (скользящая медиана) - аналог
deeplabcut.filterpredictions.

Запуск:
    python batch_offline_pose.py --videos "C:/dlc/videos" --out "C:/dlc/videos"
    python batch_offline_pose.py --videos <dir> --limit 1 --max-frames 500   # смоук
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_PATH = (r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch"
              r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5"
              r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt")
SCORER = "DLCLive_resnet50_snapshot_best_380"
MEDIAN_WIN = 5          # окно медианы для *_filtered (аналог filterpredictions)


def build_dlclive():
    import torch
    # TF32 - тот же режим, что в боевом профиле single-best: ~1.3x к скорости
    # при пренебрежимом влиянии на координаты.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    from dlclive import DLCLive
    live = DLCLive(
        model_path=MODEL_PATH,
        model_type="pytorch",
        precision="FP32",
        single_animal=True,
        device="cuda",
        cropping=None,
        dynamic=(False, 0.5, 10),
        resize=1.0,
        processor=None,
        convert2rgb=True,
        display=False,
    )
    return live


def bodyparts_from_model(live) -> list[str]:
    for holder in (getattr(live, "cfg", None), getattr(live, "model_cfg", None)):
        if isinstance(holder, dict):
            md = holder.get("metadata", {})
            if "bodyparts" in md:
                return list(md["bodyparts"])
            if "bodyparts" in holder:
                return list(holder["bodyparts"])
    return []


def to_dlc_frame(pose: np.ndarray, bodyparts: list[str]) -> pd.DataFrame:
    """(T, K, 3) -> DataFrame с MultiIndex (bodypart, coord), как у DLC."""
    cols = pd.MultiIndex.from_product([bodyparts, ["x", "y", "likelihood"]])
    flat = pose.reshape(pose.shape[0], -1)
    return pd.DataFrame(flat, columns=cols)


def write_dlc_csv(df: pd.DataFrame, path: Path, scorer: str):
    """Трёхстрочная шапка DLC: scorer / bodyparts / coords."""
    bodyparts = [c[0] for c in df.columns]
    coords = [c[1] for c in df.columns]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("scorer," + ",".join([scorer] * len(df.columns)) + "\n")
        f.write("bodyparts," + ",".join(bodyparts) + "\n")
        f.write("coords," + ",".join(coords) + "\n")
        vals = df.to_numpy()
        for i in range(len(df)):
            row = ",".join("" if not np.isfinite(v) else f"{v:.5f}" for v in vals[i])
            f.write(f"{i},{row}\n")


def median_filter_df(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """Медиана по x/y (likelihood не трогаем) - аналог filterpredictions."""
    out = df.copy()
    for col in df.columns:
        if col[1] in ("x", "y"):
            out[col] = df[col].rolling(win, center=True, min_periods=1).median()
    return out


def analyze_video(live, video: Path, bodyparts: list[str], out_dir: Path,
                  max_frames: int | None, log_every: int = 2000,
                  pause_every: int = 0, pause_sec: float = 0.0):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  !! не открылось: {video.name}")
        return None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = min(n_total, max_frames) if max_frames else n_total
    poses, inited = [], False
    t0 = time.time()

    for i in range(n_target):
        ok, frame = cap.read()
        if not ok:
            break
        if not inited:
            live.init_inference(frame)
            inited = True
            p = live.get_pose(frame)
        else:
            p = live.get_pose(frame)
        poses.append(np.asarray(p, dtype=np.float32))
        # Охлаждающая пауза: RTX 5070 Laptop уходит в тепловой троттлинг, а при
        # длительной 100% нагрузке машина может выключиться (проверено).
        if pause_every and pause_sec and (i + 1) % pause_every == 0:
            time.sleep(pause_sec)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    {i + 1}/{n_target}  {(i + 1) / el:.0f} к/с  осталось ~{(n_target - i - 1) / ((i + 1) / el) / 60:.1f} мин",
                  flush=True)
    cap.release()

    if not poses:
        return None
    arr = np.stack(poses)                    # (T, K, 3)
    if arr.shape[1] != len(bodyparts):
        bodyparts = [f"bp{k}" for k in range(arr.shape[1])]
    df = to_dlc_frame(arr, bodyparts)

    stem = video.stem
    raw_path = out_dir / f"{stem}{SCORER}.csv"
    flt_path = out_dir / f"{stem}{SCORER}_filtered.csv"
    write_dlc_csv(df, raw_path, SCORER)
    write_dlc_csv(median_filter_df(df, MEDIAN_WIN), flt_path, SCORER)

    el = time.time() - t0
    print(f"  готово: {len(df)} кадров за {el / 60:.1f} мин ({len(df) / el:.0f} к/с) -> {raw_path.name}",
          flush=True)
    return {"video": video.name, "frames": len(df), "sec": el, "csv": str(raw_path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Пакетный офлайн-инференс позы через DLCLive")
    ap.add_argument("--videos", required=True, type=Path, help="папка с .avi")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="сколько видео обработать")
    ap.add_argument("--max-frames", type=int, default=None, help="ограничить кадры (смоук-тест)")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--pause-every", type=int, default=3000,
                    help="охлаждающая пауза каждые N кадров (0 = без пауз)")
    ap.add_argument("--pause-sec", type=float, default=8.0,
                    help="длительность охлаждающей паузы, с")
    ap.add_argument("--cool-between", type=float, default=45.0,
                    help="пауза между видео, с")
    a = ap.parse_args(argv)

    out_dir = a.out or a.videos
    out_dir.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in a.videos.glob("*.avi"))

    todo = []
    for v in vids:
        # пропускаем те, где уже есть готовый DLC-CSV (любого scorer)
        if a.skip_existing and list(a.videos.glob(f"{v.stem}*_filtered.csv")):
            continue
        todo.append(v)
    if a.limit:
        todo = todo[:a.limit]

    print(f"видео всего: {len(vids)}, к обработке: {len(todo)}")
    if not todo:
        print("нечего делать (у всех уже есть CSV)")
        return

    print("инициализация DLCLive...", flush=True)
    live = build_dlclive()
    bps = bodyparts_from_model(live)
    if not bps:
        bps = ["nose", "eye_l", "eye_r", "fl_toes_l", "fl_toes_r",
               "hl_toes_l", "hl_ankle_l", "hl_hip_l", "hl_iliac_l",
               "hl_toes_r", "hl_ankle_r", "hl_hip_r", "hl_iliac_r", "spine", "tail"]
        print("  bodyparts взяты из config.yaml проекта (в модели метаданных нет)")
    print(f"  точек: {len(bps)}")

    stats = []
    for k, v in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {v.name}", flush=True)
        r = analyze_video(live, v, bps, out_dir, a.max_frames,
                          pause_every=a.pause_every, pause_sec=a.pause_sec)
        if r:
            stats.append(r)
        if a.cool_between and k < len(todo):
            print(f"  пауза на охлаждение {a.cool_between:.0f} с...", flush=True)
            time.sleep(a.cool_between)

    if stats:
        tot_f = sum(s["frames"] for s in stats)
        tot_t = sum(s["sec"] for s in stats)
        print(f"\nИТОГО: {len(stats)} видео, {tot_f} кадров, {tot_t / 60:.1f} мин "
              f"({tot_f / tot_t:.0f} к/с)")


if __name__ == "__main__":
    sys.exit(main())
