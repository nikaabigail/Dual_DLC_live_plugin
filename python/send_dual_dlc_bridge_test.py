"""
Send synthetic dual DLCLive bridge packets to the Open Ephys bridge plugin.

Use this before starting cameras to verify that DualDLCLiveBridge receives UDP
packets and emits TTL events inside Open Ephys.
"""
from __future__ import annotations

import argparse
import json
import socket
import time


def build_packet(index: int, request_ack: bool = False) -> dict[str, object]:
    left_valid = index % 2 == 0
    right_valid = index % 3 != 0
    left_trigger = index % 4 == 0
    right_trigger = index % 5 == 0
    packet: dict[str, object] = {
        "schema": "dual_dlc_live.v1",
        "pair_index": index,
        "host_time": time.time(),
        "host_dt_ms": 0.0,
        "camera_dt_ms": None,
        "left": {
            "name": "left",
            "frame_id": index,
            "source_frame_id": index,
            "capture_ts": time.time(),
            "valid": left_valid,
            "angle_deg": 50.0 if left_trigger else 70.0,
            "infer_ms": 0.0,
            "picked_side": "left",
            "drops": 0,
        },
        "right": {
            "name": "right",
            "frame_id": index,
            "source_frame_id": index,
            "capture_ts": time.time(),
            "valid": right_valid,
            "angle_deg": 50.0 if right_trigger else 70.0,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47000)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--wait-ack", action="store_true")
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.wait_ack:
        sock.settimeout(args.ack_timeout)

    try:
        acked = 0
        for index in range(1, args.count + 1):
            packet = build_packet(index, request_ack=args.wait_ack)
            data = (json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8")
            sock.sendto(data, (args.host, args.port))
            print(f"sent pair={index} ttl={packet['ttl_lines']}")
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

                acked += 1
                print(f"ack pair={index} from={ack_addr[0]}:{ack_addr[1]} {ack_text}")
            time.sleep(args.interval)

        if args.wait_ack:
            print(f"acked {acked}/{args.count}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
