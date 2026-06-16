from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import live_profiles


LIVE_SCRIPT_NAMES = ("single_rt_dlc_live_bridge.py", "dual_rt_dlc_live.py")


def live_processes() -> list[dict[str, str]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and "
        "($_.CommandLine -like '*single_rt_dlc_live_bridge.py*' -or "
        "$_.CommandLine -like '*dual_rt_dlc_live.py*') } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=str(live_profiles.WORK_DIR),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return []
    if not output:
        return []

    import json

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]

    current_pid = os.getpid()
    processes: list[dict[str, str]] = []
    for item in parsed:
        pid = int(item.get("ProcessId", 0) or 0)
        command_line = str(item.get("CommandLine", ""))
        if pid <= 0 or pid == current_pid:
            continue
        processes.append({"pid": str(pid), "command": command_line})
    return processes


def print_live_processes(processes: list[dict[str, str]]) -> None:
    if not processes:
        print("No running DLC live Python process was found.")
        return
    print("Running DLC live Python process:")
    for proc in processes:
        print(f"  PID {proc['pid']}: {proc['command']}")


def stop_live_processes(processes: list[dict[str, str]]) -> None:
    if not processes:
        return
    ids = ",".join(proc["pid"] for proc in processes)
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {ids} -Force"],
        cwd=str(live_profiles.WORK_DIR),
        check=False,
    )


def resolve_profile(raw: str | None) -> live_profiles.LiveProfile:
    if not raw:
        return live_profiles.choose_profile_interactively()
    names = live_profiles.profile_names()
    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(names):
            return live_profiles.get_profile(names[index - 1])
    return live_profiles.get_profile(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Choose and run a DLC live profile.")
    parser.add_argument("profile", nargs="?", help="Profile name or menu number. Omit for interactive menu.")
    parser.add_argument("--list", action="store_true", help="Show all launch profiles and exit.")
    parser.add_argument("--status", action="store_true", help="Show currently running DLC live Python process.")
    parser.add_argument("--stop", action="store_true", help="Stop currently running DLC live Python process and exit.")
    parser.add_argument("--replace", action="store_true", help="Stop an existing live process before starting this profile.")
    parser.add_argument("--target", choices=["single", "dual"], default=None, help="Filter --list output.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting it.")
    parser.add_argument("--max-frames", type=int, default=0, help="Pass --max-frames to single-camera profiles.")
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", action="store_true", help="Force OpenCV preview window on.")
    display_group.add_argument("--no-display", action="store_true", help="Force headless mode with no preview window.")
    args = parser.parse_args()

    if args.list:
        live_profiles.print_profile_table(args.target)
        return 0

    running = live_processes()

    if args.status:
        print_live_processes(running)
        return 0

    if args.stop:
        if not running:
            print("No running DLC live Python process was found.")
            return 0
        print_live_processes(running)
        print("Stopping running DLC live process...")
        stop_live_processes(running)
        return 0

    profile = resolve_profile(args.profile)
    script = "single_rt_dlc_live_bridge.py" if profile.target == "single" else "dual_rt_dlc_live.py"
    script_path = live_profiles.WORK_DIR / script
    command = [str(live_profiles.PYTHON_EXE), str(script_path), "--profile", profile.name]
    if args.display:
        command.append("--display")
    elif args.no_display:
        command.append("--no-display")
    if profile.target == "single" and int(args.max_frames) > 0:
        command.extend(["--max-frames", str(int(args.max_frames))])

    if args.dry_run:
        display_value = live_profiles.display_value_for_profile(profile, args.display, args.no_display)
        print(live_profiles.profile_banner(profile, display_override=display_value), flush=True)
    print("Launch command:")
    print("  " + " ".join(command))
    print()
    if args.dry_run:
        return 0

    if running:
        if not args.replace:
            print_live_processes(running)
            print()
            print("Camera is already owned by that process. To restart with the selected profile, run:")
            restart_command = [str(live_profiles.PYTHON_EXE), str(live_profiles.WORK_DIR / "run_live_profile.py"), profile.name]
            if args.display:
                restart_command.append("--display")
            elif args.no_display:
                restart_command.append("--no-display")
            restart_command.append("--replace")
            print("  " + " ".join(restart_command))
            print()
            print("Or stop it first:")
            print("  " + " ".join([str(live_profiles.PYTHON_EXE), str(live_profiles.WORK_DIR / "run_live_profile.py"), "--stop"]))
            return 2
        print_live_processes(running)
        print("Stopping existing live process before launch...")
        stop_live_processes(running)

    return subprocess.call(command, cwd=str(live_profiles.WORK_DIR))


if __name__ == "__main__":
    raise SystemExit(main())
