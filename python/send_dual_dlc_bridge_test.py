"""
Send synthetic dual DLCLive bridge packets to the Open Ephys bridge plugin.

Default mode sends raw pose points (`dual_dlc_live.pose.v1`) so the plugin has
to compute validity, angle and TTL output itself. Use `--mode ttl` to exercise
the legacy ttl_lines protocol.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from typing import Any


SIDE_POINT_SETS = {
    "left": ("hl_hip_l", "hl_ankle_l", "hl_toes_l"),
    "right": ("hl_hip_r", "hl_ankle_r", "hl_toes_r"),
}
TRACKED_POINTS = sorted({name for triplet in SIDE_POINT_SETS.values() for name in triplet})
BINARY_POSE_MAGIC = b"DDLP"
BINARY_POSE_VERSION = 1
BINARY_FLAG_ACK = 1 << 0
BINARY_HEADER_STRUCT = struct.Struct("<4sHHqdffHH")
BINARY_SIDE_STRUCT = struct.Struct("<qqdfIHH")
BINARY_POINT_STRUCT = struct.Struct("<fff")


def point(x: float, y: float, likelihood: float) -> dict[str, float]:
    return {"x": x, "y": y, "likelihood": likelihood}


def low_point(x: float, y: float) -> dict[str, float]:
    return point(x, y, 0.01)


def side_points(index: int, camera_side: str, valid: bool, trigger_angle: bool) -> dict[str, dict[str, float]]:
    """Build six named points; only the camera_side triplet is high-confidence."""
    base_x = 100.0 if camera_side == "left" else 500.0
    base_y = 50.0 + float(index % 5)
    points = {
        "hl_hip_l": low_point(base_x, base_y),
        "hl_ankle_l": low_point(base_x + 40.0, base_y),
        "hl_toes_l": low_point(base_x + 20.0, base_y + 20.0),
        "hl_hip_r": low_point(base_x, base_y),
        "hl_ankle_r": low_point(base_x + 40.0, base_y),
        "hl_toes_r": low_point(base_x + 20.0, base_y + 20.0),
    }

    hip, ankle, toes = SIDE_POINT_SETS[camera_side]
    likelihood = 0.90 if valid else 0.05
    points[hip] = point(base_x, base_y, likelihood)
    points[ankle] = point(base_x + 40.0, base_y, likelihood)
    if trigger_angle:
        # Approximately 45 degrees at ankle.
        points[toes] = point(base_x + 20.0, base_y + 20.0, likelihood)
    else:
        # Approximately 135 degrees at ankle.
        points[toes] = point(base_x + 60.0, base_y + 20.0, likelihood)
    return points


def build_pose_packet(index: int, request_ack: bool = False) -> dict[str, Any]:
    left_valid = index % 2 == 0
    right_valid = index % 3 != 0
    left_trigger = index % 4 == 0
    right_trigger = index % 5 == 0
    now = time.time()
    packet: dict[str, Any] = {
        "schema": "dual_dlc_live.pose.v1",
        "pair_index": index,
        "host_time": now,
        "host_dt_ms": 0.0,
        "camera_dt_ms": None,
        "tracked_points": TRACKED_POINTS,
        "side_point_sets": {side: list(names) for side, names in SIDE_POINT_SETS.items()},
        "left": {
            "name": "left",
            "frame_id": index,
            "source_frame_id": index,
            "capture_ts": now,
            "infer_ms": 0.0,
            "drops": 0,
            "raw_visible": 3 if left_valid else 0,
            "raw_points": side_points(index, "left", left_valid, left_trigger),
        },
        "right": {
            "name": "right",
            "frame_id": index,
            "source_frame_id": index,
            "capture_ts": now,
            "infer_ms": 0.0,
            "drops": 0,
            "raw_visible": 3 if right_valid else 0,
            "raw_points": side_points(index, "right", right_valid, right_trigger),
        },
    }
    if request_ack:
        packet["ack"] = True
    return packet


def build_ttl_packet(index: int, request_ack: bool = False) -> dict[str, Any]:
    left_valid = index % 2 == 0
    right_valid = index % 3 != 0
    left_trigger = index % 4 == 0
    right_trigger = index % 5 == 0
    now = time.time()
    packet: dict[str, Any] = {
        "schema": "dual_dlc_live.v1",
        "pair_index": index,
        "host_time": now,
        "host_dt_ms": 0.0,
        "camera_dt_ms": None,
        "left": {
            "name": "left",
            "frame_id": index,
            "source_frame_id": index,
            "capture_ts": now,
            "valid": left_valid,
            "angle_deg": 45.0 if left_trigger else 135.0,
            "infer_ms": 0.0,
            "picked_side": "left",
            "drops": 0,
        },
        "right": {
            "name": "right",
            "frame_id": index,
            "source_frame_id": index,
            "capture_ts": now,
            "valid": right_valid,
            "angle_deg": 45.0 if right_trigger else 135.0,
            "infer_ms": 0.0,
            "picked_side": "right",
            "drops": 0,
        },
        "ttl_lines": [
            left_valid,
            right_valid,
            left_trigger,
            right_trigger,
            False,
            False,
            False,
            False,
        ],
    }
    if request_ack:
        packet["ack"] = True
    return packet


def build_binary_pose_packet(packet: dict[str, Any], request_ack: bool = False) -> bytes:
    flags = BINARY_FLAG_ACK if request_ack else 0
    camera_dt_ms = packet.get("camera_dt_ms")
    data = bytearray()
    data.extend(
        BINARY_HEADER_STRUCT.pack(
            BINARY_POSE_MAGIC,
            BINARY_POSE_VERSION,
            flags,
            int(packet["pair_index"]),
            float(packet["host_time"]),
            float(packet["host_dt_ms"]),
            float("nan") if camera_dt_ms is None else float(camera_dt_ms),
            len(TRACKED_POINTS),
            0,
        )
    )

    for side_name in ("left", "right"):
        side = packet[side_name]
        data.extend(
            BINARY_SIDE_STRUCT.pack(
                int(side["frame_id"]),
                -1 if side.get("source_frame_id") is None else int(side["source_frame_id"]),
                float(side["capture_ts"]),
                float(side["infer_ms"]),
                int(side["drops"]),
                int(side["raw_visible"]),
                0,
            )
        )
        raw_points = side["raw_points"]
        for point_name in TRACKED_POINTS:
            point_data = raw_points[point_name]
            data.extend(
                BINARY_POINT_STRUCT.pack(
                    float(point_data["x"]),
                    float(point_data["y"]),
                    float(point_data["likelihood"]),
                )
            )
    return bytes(data)


def expected_pose_ttl(index: int) -> str:
    # Plugin defaults keep angle triggers disabled, so synthetic pose mode only
    # guarantees validity bits unless the GUI toggle is enabled manually.
    left_valid = index % 2 == 0
    right_valid = index % 3 != 0
    ttl = (1 if left_valid else 0) | (2 if right_valid else 0)
    return f"0x{ttl:02X}"


def expected_ttl_lines(packet: dict[str, Any]) -> str:
    ttl = 0
    for line, enabled in enumerate(packet["ttl_lines"]):
        if enabled:
            ttl |= 1 << line
    return f"0x{ttl:02X}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47000)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--mode", choices=("pose", "ttl"), default="pose")
    parser.add_argument("--wire-format", choices=("binary", "json"), default="binary")
    parser.add_argument("--wait-ack", action="store_true")
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--check-ack-ttl", action="store_true")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.wait_ack:
        sock.settimeout(args.ack_timeout)

    try:
        acked = 0
        for index in range(1, args.count + 1):
            if args.mode == "pose":
                packet = build_pose_packet(index, request_ack=args.wait_ack)
                expected_ttl = expected_pose_ttl(index)
                data = (
                    build_binary_pose_packet(packet, request_ack=args.wait_ack)
                    if args.wire_format == "binary"
                    else (json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8")
                )
            else:
                packet = build_ttl_packet(index, request_ack=args.wait_ack)
                expected_ttl = expected_ttl_lines(packet)
                data = (json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8")
            sock.sendto(data, (args.host, args.port))
            wire = args.wire_format if args.mode == "pose" else "json"
            print(f"sent pair={index} mode={args.mode} wire={wire} expected_ttl={expected_ttl}")
            if args.wait_ack:
                try:
                    ack_data, ack_addr = sock.recvfrom(2048)
                except (socket.timeout, ConnectionResetError) as exc:
                    raise SystemExit(f"missing ack for pair={index}") from exc

                ack_text = ack_data.decode("utf-8", errors="replace").strip()
                if f"pair={index}" not in ack_text:
                    raise SystemExit(
                        f"unexpected ack for pair={index}: {ack_text!r} from {ack_addr}"
                    )
                if args.check_ack_ttl and expected_ttl not in ack_text:
                    raise SystemExit(
                        f"unexpected ttl for pair={index}: expected {expected_ttl}, got {ack_text!r}"
                    )

                acked += 1
                print(f"ack pair={index} from={ack_addr[0]}:{ack_addr[1]} {ack_text}")
            time.sleep(args.interval)

        if args.wait_ack:
            print(f"acked {acked}/{args.count}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
