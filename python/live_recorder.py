"""Background recorder for the single-camera bridge: raw video + keypoints.

Runs entirely on a worker thread so encoding / disk IO never adds latency to the
camera -> pose -> UDP loop. The hot loop only copies the frame and enqueues it.

Design notes
------------
* Each queue item carries the raw frame AND its keypoints, so the two streams
  stay FRAME-ALIGNED: if the queue overflows both are dropped together, and every
  keypoint record stores its absolute ``frame_index`` so alignment survives drops.
* Keypoint coordinates are FULL-FRAME. The live pipeline already restores the
  sliding-ROI crop offset (``postprocess_pose_without_processor``), so what we
  store is full-stripe (1920x220) coords -- no window distortion. The ROI window
  ``[x1, x2]`` the model actually saw is stored per frame too, so a crop-local
  coordinate for fine-tuning is just ``x - roi_x1``.
* Video is the RAW camera frame (no overlay) -- "just the video as DLC runs".

Keypoint file formats
----------------------
* ``binary`` (default, fastest): a ``.dlckp`` stream --
    header : b"DLCKP1\\n", int32 n_bodyparts, int32 names_len, names (utf-8, "\\n"-joined)
    record : int64 frame_index, float64 capture_ts, int32 roi_x1, int32 roi_x2,
             then n_bodyparts * (float32 x, float32 y, float32 likelihood)
  Fixed-size records -> trivial to seek/parse. Convert to CSV offline with
  ``scripts/kp_to_csv.py``.
* ``csv``: a ``.csv`` with columns
    frame_index, capture_ts, roi_x1, roi_x2, <bp>_x, <bp>_y, <bp>_lik, ...
"""
from __future__ import annotations

import logging
import queue
import struct
import threading
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

KP_MAGIC = b"DLCKP1\n"
# header: magic, <i n_bodyparts, <i names_len, names
# record: <q frame_index, <d capture_ts, <i roi_x1, <i roi_x2, then n*<3f
_RECORD_HEAD = struct.Struct("<qdii")


class ParallelRecorder:
    def __init__(
        self,
        *,
        out_dir: Path,
        stem: str,
        bodyparts: Sequence[str],
        fps: float,
        frame_is_rgb: bool = True,
        record_video: bool = True,
        video_codec: str = "mp4v",
        record_keypoints: bool = True,
        kp_format: str = "binary",
        queue_size: int = 128,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log = logger or logging.getLogger("recorder")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stem = str(stem)
        self.bodyparts = list(bodyparts)
        self.n = len(self.bodyparts)
        self.fps = float(fps) if fps and fps > 0 else 100.0
        self.frame_is_rgb = bool(frame_is_rgb)
        self.record_video = bool(record_video)
        self.video_codec = str(video_codec)
        self.record_keypoints = bool(record_keypoints)
        self.kp_format = str(kp_format).lower()

        self._q: "queue.Queue" = queue.Queue(maxsize=int(queue_size))
        self._stop = threading.Event()
        self._dropped = 0
        self._written = 0
        self._writer: Optional[cv2.VideoWriter] = None
        self._kp = None
        self._csv = None
        self.video_path: Optional[Path] = None
        self.kp_path: Optional[Path] = None
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self.record_keypoints:
            self._open_keypoints()
        self._thread.start()
        self.log.info(
            "Recorder started: dir=%s stem=%s video=%s(%s) keypoints=%s(%s) fps=%.1f bodyparts=%d",
            self.out_dir, self.stem, self.record_video, self.video_codec,
            self.record_keypoints, self.kp_format, self.fps, self.n,
        )

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15.0)
        if self._writer is not None:
            self._writer.release()
        if self._kp is not None:
            self._kp.close()
        if self._csv is not None:
            self._csv.close()
        self.log.info(
            "Recorder closed: wrote=%d dropped=%d video=%s keypoints=%s",
            self._written, self._dropped, self.video_path, self.kp_path,
        )

    # -- producer side (hot loop) -------------------------------------------
    def submit(self, frame, frame_index: int, capture_ts: float, pose, roi) -> None:
        """Enqueue one frame + its full-frame pose. Non-blocking; drops on overflow."""
        if self._stop.is_set():
            return
        item = (
            frame.copy() if (self.record_video and frame is not None) else None,
            int(frame_index),
            float(capture_ts),
            np.asarray(pose, dtype=np.float32).copy() if self.record_keypoints else None,
            roi,
        )
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                self.log.warning(
                    "Recorder queue full -> dropped %d frame(s); writer cannot keep up "
                    "(try a faster codec, keypoints-only, or a larger queue).",
                    self._dropped,
                )

    # -- consumer side (worker thread) --------------------------------------
    def _run(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            frame, fidx, ts, pose, roi = item
            try:
                if frame is not None:
                    self._write_video(frame)
                if pose is not None:
                    self._write_keypoints(fidx, ts, pose, roi)
                self._written += 1
            except Exception as exc:  # never let a write error kill the thread
                self.log.warning("Recorder write failed at frame %d: %s", fidx, exc)
            finally:
                self._q.task_done()

    def _write_video(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            ext = "avi" if self.video_codec.upper() in ("FFV1", "MJPG", "XVID") else "mp4"
            self.video_path = self.out_dir / f"{self.stem}.{ext}"
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (w, h), True)
            if not self._writer.isOpened():
                self.log.warning(
                    "VideoWriter failed to open %s (codec=%s) -> video disabled", self.video_path, self.video_codec
                )
                self._writer = None
                self.record_video = False
                return
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if self.frame_is_rgb else frame
        self._writer.write(bgr)

    def _open_keypoints(self) -> None:
        if self.kp_format == "csv":
            self.kp_path = self.out_dir / f"{self.stem}.csv"
            self._csv = open(self.kp_path, "w", buffering=1 << 20, newline="")
            cols = ["frame_index", "capture_ts", "roi_x1", "roi_x2"]
            for b in self.bodyparts:
                cols += [f"{b}_x", f"{b}_y", f"{b}_lik"]
            self._csv.write(",".join(cols) + "\n")
        else:
            self.kp_path = self.out_dir / f"{self.stem}.dlckp"
            self._kp = open(self.kp_path, "wb", buffering=1 << 20)
            names = "\n".join(self.bodyparts).encode("utf-8")
            self._kp.write(KP_MAGIC)
            self._kp.write(struct.pack("<i", self.n))
            self._kp.write(struct.pack("<i", len(names)))
            self._kp.write(names)

    def _write_keypoints(self, fidx: int, ts: float, pose: np.ndarray, roi) -> None:
        x1 = int(roi[0]) if roi else -1
        x2 = int(roi[1]) if roi else -1
        if self._csv is not None:
            parts = [str(fidx), f"{ts:.6f}", str(x1), str(x2)]
            for i in range(self.n):
                parts += [f"{float(pose[i, 0]):.2f}", f"{float(pose[i, 1]):.2f}", f"{float(pose[i, 2]):.4f}"]
            self._csv.write(",".join(parts) + "\n")
        elif self._kp is not None:
            self._kp.write(_RECORD_HEAD.pack(int(fidx), float(ts), x1, x2))
            self._kp.write(np.ascontiguousarray(pose[:, :3], dtype="<f4").tobytes())
