"""Convert a .dlckp binary keypoint stream (from the single-camera bridge) to CSV.

Usage:
    python scripts/kp_to_csv.py recording.dlckp [out.csv]

Binary layout (little-endian), written by python/live_recorder.py:
    header : b"DLCKP1\\n", int32 n_bodyparts, int32 names_len, names (utf-8, "\\n"-joined)
    record : int64 frame_index, float64 capture_ts, int32 roi_x1, int32 roi_x2,
             then n_bodyparts * (float32 x, float32 y, float32 likelihood)

Coordinates are FULL-FRAME. For a crop-local coordinate (e.g. to fine-tune on the
ROI crop), use  x_local = x - roi_x1.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

KP_MAGIC = b"DLCKP1\n"
_HEAD = struct.Struct("<qdii")  # frame_index, capture_ts, roi_x1, roi_x2


def convert(src: Path, dst: Path) -> int:
    with open(src, "rb") as f:
        if f.read(len(KP_MAGIC)) != KP_MAGIC:
            raise ValueError(f"{src} is not a DLCKP1 file")
        (n,) = struct.unpack("<i", f.read(4))
        (names_len,) = struct.unpack("<i", f.read(4))
        names = f.read(names_len).decode("utf-8").split("\n")
        if len(names) != n:
            raise ValueError(f"header says {n} bodyparts but {len(names)} names")
        rec_size = _HEAD.size + n * 12  # 3 float32 per bodypart
        cols = ["frame_index", "capture_ts", "roi_x1", "roi_x2"]
        for b in names:
            cols += [f"{b}_x", f"{b}_y", f"{b}_lik"]

        rows = 0
        with open(dst, "w", newline="", buffering=1 << 20) as out:
            out.write(",".join(cols) + "\n")
            while True:
                chunk = f.read(rec_size)
                if len(chunk) < rec_size:
                    if len(chunk) != 0:  # truncated trailing record (e.g. crash mid-write)
                        print(f"warning: ignored a truncated trailing record of {len(chunk)} bytes", file=sys.stderr)
                    break
                fidx, ts, x1, x2 = _HEAD.unpack_from(chunk, 0)
                vals = struct.unpack_from(f"<{n * 3}f", chunk, _HEAD.size)
                parts = [str(fidx), f"{ts:.6f}", str(x1), str(x2)]
                for i in range(n):
                    parts += [f"{vals[3 * i]:.2f}", f"{vals[3 * i + 1]:.2f}", f"{vals[3 * i + 2]:.4f}"]
                out.write(",".join(parts) + "\n")
                rows += 1
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".csv")
    rows = convert(src, dst)
    print(f"wrote {rows} rows -> {dst}")


if __name__ == "__main__":
    main()
