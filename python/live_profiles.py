from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORK_DIR = Path(r"C:\dlc\DLC_OBS_Spinal_cord_stimulation")
PYTHON_EXE = Path(r"C:\dlc_live_env\Scripts\python.exe")


@dataclass(frozen=True)
class LiveProfile:
    name: str
    target: str
    label: str
    summary: str
    settings: dict[str, Any]
    recommended: bool = False
    notes: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        script = "single_rt_dlc_live_bridge.py" if self.target == "single" else "dual_rt_dlc_live.py"
        return f"{PYTHON_EXE} {WORK_DIR / script} --profile {self.name}"


COMMON_BRIDGE_SETTINGS: dict[str, Any] = {
    "DUAL_OE_BRIDGE_ENABLED": True,
    "DUAL_OE_BRIDGE_HOST": "127.0.0.1",
    "DUAL_OE_BRIDGE_PORT": 47000,
    "DUAL_OE_BRIDGE_SEND_EVERY_N_RESULTS": 1,
    "DUAL_OE_BRIDGE_PACKET_MODE": "pose",
    "DUAL_OE_BRIDGE_WIRE_FORMAT": "binary",
    "DUAL_OE_BRIDGE_REQUEST_ACK": False,
    "DUAL_FAST_POSE_ONLY": True,
    "DUAL_ENABLE_STAGE_PROFILER": True,
    "DUAL_PROFILE_EMA_ALPHA": 0.10,
    "ENABLE_BENCHMARK_CSV": False,
}

COMMON_SINGLE_SETTINGS: dict[str, Any] = {
    **COMMON_BRIDGE_SETTINGS,
    "SINGLE_CAMERA_NAME": "left",
    "SINGLE_PLUGIN_SIDE": "left",
    "SINGLE_DISPLAY_WINDOW": True,
    "SINGLE_WINDOW_NAME": "DLC Live single bridge",
    "SINGLE_LOG_EVERY_N_FRAMES": 120,
    "SINGLE_PROFILE_LOG_EVERY_N_FRAMES": 120,
    "LOG_PATH": WORK_DIR / "single_rt_dlc_live_bridge_debug.log",
    "BENCHMARK_CSV_PATH": WORK_DIR / "single_rt_dlc_live_bridge_benchmark.csv",
    "CROPPING": None,
    "RESIZE": 1.0,
    "DYNAMIC_CROPPING": (False, 0.5, 10),
}

COMMON_DUAL_SETTINGS: dict[str, Any] = {
    **COMMON_BRIDGE_SETTINGS,
    "DUAL_DISPLAY_WINDOW": True,
    "DUAL_LOG_EVERY_N_PAIRS": 120,
    "DUAL_PROFILE_LOG_EVERY_N_PAIRS": 120,
    "DUAL_PROCESS_EVERY_N_PAIRS": 1,
    "DUAL_ENABLE_BATCH_INFERENCE": True,
    "DUAL_BATCH_FALLBACK_TO_SEQUENTIAL": True,
    "LOG_PATH": WORK_DIR / "dual_rt_dlc_live_debug.log",
    "BENCHMARK_CSV_PATH": WORK_DIR / "dual_rt_dlc_live_benchmark.csv",
}


PROFILE_ORDER = (
    "single-best",
    "single-strict",
    "single-fp16",
    "single-cpu",
    "single-debug",
    "single-rgb-on",
    "dual-best",
    "dual-cpu",
    "dual-fp16",
)


PROFILES: dict[str, LiveProfile] = {
    "single-best": LiveProfile(
        name="single-best",
        target="single",
        label="1 camera, RGB no-convert, FP32+TF32",
        summary="Current best working mode for the left treadmill camera and Open Ephys plugin.",
        recommended=True,
        settings={
            **COMMON_SINGLE_SETTINGS,
            "PRECISION": "FP32",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": True,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_TORCH_COMPILE_BACKEND": "cudagraphs",
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 12,
            "DUAL_TORCH_INTEROP_THREADS": 12,
        },
        notes=(
            "Best balance seen so far: good FPS, GPU used, CPU lower than full auto.",
            "cudagraphs is on (accuracy-neutral, bit-identical kernels); first frame after init pays a one-time graph-capture cost, then inference drops ~21->~17ms. Auto-falls back to eager on any compile error.",
            "Use --profile single-strict for a guaranteed pure-eager run (no torch.compile).",
            "The plugin receives left-side pose; right side is sent empty.",
            "Preview window is enabled now; add --no-display for final stimulation runs.",
        ),
    ),
    "single-strict": LiveProfile(
        name="single-strict",
        target="single",
        label="1 camera, RGB no-convert, strict FP32",
        summary="Most conservative numeric mode: TF32 disabled.",
        settings={
            **COMMON_SINGLE_SETTINGS,
            "PRECISION": "FP32",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": False,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 0,
            "DUAL_TORCH_INTEROP_THREADS": 0,
        },
        notes=(
            "Use this when coordinate reproducibility matters more than FPS.",
            "Expected to be slower than single-best.",
            "Preview window is enabled now; add --no-display for final stimulation runs.",
        ),
    ),
    "single-fp16": LiveProfile(
        name="single-fp16",
        target="single",
        label="1 camera, RGB no-convert, FP16 experimental",
        summary="Fast experimental mode. Validate coordinates before stimulation.",
        settings={
            **COMMON_SINGLE_SETTINGS,
            "PRECISION": "FP16",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": False,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 0,
            "DUAL_TORCH_INTEROP_THREADS": 0,
        },
        notes=(
            "This was the fastest family in tests, but it can slightly change likelihoods/coordinates.",
            "Do not use for final stimulation until you compare TTL behavior with FP32.",
            "Preview window is enabled now; add --no-display for final stimulation runs.",
        ),
    ),
    "single-cpu": LiveProfile(
        name="single-cpu",
        target="single",
        label="1 camera, lower CPU, FP32+TF32",
        summary="Reduces Torch CPU threadpool. Some FPS loss is expected.",
        settings={
            **COMMON_SINGLE_SETTINGS,
            "PRECISION": "FP32",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": True,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 8,
            "DUAL_TORCH_INTEROP_THREADS": 8,
        },
        notes=(
            "Dual test showed much lower CPU with a visible FPS cost.",
            "Useful if Open Ephys or other apps stutter.",
            "Preview window is enabled now; add --no-display for final stimulation runs.",
        ),
    ),
    "single-debug": LiveProfile(
        name="single-debug",
        target="single",
        label="1 camera, visual debug window",
        summary="Same model path as single-best, but shows OpenCV overlay.",
        settings={
            **COMMON_SINGLE_SETTINGS,
            "PRECISION": "FP32",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": True,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 12,
            "DUAL_TORCH_INTEROP_THREADS": 12,
            "SINGLE_DISPLAY_WINDOW": True,
            "SINGLE_LOG_EVERY_N_FRAMES": 30,
            "SINGLE_PROFILE_LOG_EVERY_N_FRAMES": 30,
        },
        notes=(
            "Use for checking whether points visually sit on the animal.",
            "Display can lower FPS and should be off for stimulation.",
        ),
    ),
    "single-rgb-on": LiveProfile(
        name="single-rgb-on",
        target="single",
        label="1 camera, BGR input + DLCLive RGB conversion",
        summary="Color fallback if RGB no-convert looks wrong.",
        settings={
            **COMMON_SINGLE_SETTINGS,
            "PRECISION": "FP32",
            "GALAXY_OUTPUT_COLOR": "bgr",
            "CONVERT_TO_RGB": True,
            "DUAL_TORCH_ALLOW_TF32": True,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 12,
            "DUAL_TORCH_INTEROP_THREADS": 12,
        },
        notes=(
            "Use only as a color sanity check.",
            "The current best result is expected from RGB no-convert, not this mode.",
            "Preview window is enabled now; add --no-display for final stimulation runs.",
        ),
    ),
    "dual-best": LiveProfile(
        name="dual-best",
        target="dual",
        label="2 cameras, RGB no-convert, FP32+TF32, max throughput",
        summary="Dual mode for later, when both camera views are physically correct.",
        settings={
            **COMMON_DUAL_SETTINGS,
            "PRECISION": "FP32",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": True,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 0,
            "DUAL_TORCH_INTEROP_THREADS": 0,
        },
        notes=(
            "Do not use now if the right camera is not pointed at the treadmill.",
            "This keeps the higher-throughput dual setting.",
            "Preview windows are enabled now; add --no-display for final stimulation runs.",
        ),
    ),
    "dual-cpu": LiveProfile(
        name="dual-cpu",
        target="dual",
        label="2 cameras, lower CPU, FP32+TF32",
        summary="Dual CPU-balanced variant tested with Torch 12/12.",
        settings={
            **COMMON_DUAL_SETTINGS,
            "PRECISION": "FP32",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": True,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 12,
            "DUAL_TORCH_INTEROP_THREADS": 12,
        },
        notes=(
            "Dual test gave lower CPU than full auto, with a small FPS cost.",
            "Needs both camera views fixed before real use.",
            "Preview windows are enabled now; add --no-display for final stimulation runs.",
        ),
    ),
    "dual-fp16": LiveProfile(
        name="dual-fp16",
        target="dual",
        label="2 cameras, FP16 experimental",
        summary="Fast dual experiment; validate before stimulation.",
        settings={
            **COMMON_DUAL_SETTINGS,
            "PRECISION": "FP16",
            "GALAXY_OUTPUT_COLOR": "rgb",
            "CONVERT_TO_RGB": False,
            "DUAL_TORCH_ALLOW_TF32": False,
            "DUAL_TORCH_CUDNN_BENCHMARK": True,
            "DUAL_CV2_NUM_THREADS": 1,
            "DUAL_TORCH_NUM_THREADS": 0,
            "DUAL_TORCH_INTEROP_THREADS": 0,
        },
        notes=(
            "Fastest tested family, but not the safest numerically.",
            "Only for validation runs.",
            "Preview windows are enabled now; add --no-display for final stimulation runs.",
        ),
    ),
}


def profile_names(target: str | None = None) -> list[str]:
    return [name for name in PROFILE_ORDER if target is None or PROFILES[name].target == target]


def get_profile(name: str) -> LiveProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(PROFILE_ORDER)
        raise ValueError(f"Unknown live profile {name!r}. Valid profiles: {valid}") from exc


def apply_profile(config_module: Any, name: str) -> LiveProfile:
    profile = get_profile(name)
    for key, value in profile.settings.items():
        setattr(config_module, key, value)
    return profile


def display_key_for_target(target: str) -> str | None:
    if target == "single":
        return "SINGLE_DISPLAY_WINDOW"
    if target == "dual":
        return "DUAL_DISPLAY_WINDOW"
    return None


def display_value_for_profile(
    profile: LiveProfile,
    force_display: bool = False,
    force_no_display: bool = False,
) -> bool | None:
    if force_display:
        return True
    if force_no_display:
        return False
    key = display_key_for_target(profile.target)
    if key is None:
        return None
    value = profile.settings.get(key)
    return None if value is None else bool(value)


def apply_display_value(config_module: Any, target: str, display_value: bool | None) -> None:
    key = display_key_for_target(target)
    if key is not None and display_value is not None:
        setattr(config_module, key, bool(display_value))


def profile_banner(profile: LiveProfile, display_override: bool | None = None) -> str:
    marker = " [RECOMMENDED]" if profile.recommended else ""
    display_value = display_value_for_profile(profile) if display_override is None else display_override
    lines = [
        "=" * 78,
        f"LIVE PROFILE: {profile.name}{marker}",
        f"Mode: {profile.target}",
        f"Label: {profile.label}",
        f"Summary: {profile.summary}",
        "",
        "Important settings:",
        f"  PRECISION={profile.settings.get('PRECISION')}",
        f"  GALAXY_OUTPUT_COLOR={profile.settings.get('GALAXY_OUTPUT_COLOR')}",
        f"  CONVERT_TO_RGB={profile.settings.get('CONVERT_TO_RGB')}",
        f"  DUAL_TORCH_ALLOW_TF32={profile.settings.get('DUAL_TORCH_ALLOW_TF32')}",
        f"  DUAL_CV2_NUM_THREADS={profile.settings.get('DUAL_CV2_NUM_THREADS')}",
        f"  DUAL_TORCH_NUM_THREADS={profile.settings.get('DUAL_TORCH_NUM_THREADS')}",
        f"  DUAL_TORCH_INTEROP_THREADS={profile.settings.get('DUAL_TORCH_INTEROP_THREADS')}",
        f"  display={display_value}",
        f"  UDP={profile.settings.get('DUAL_OE_BRIDGE_HOST')}:{profile.settings.get('DUAL_OE_BRIDGE_PORT')}",
    ]
    if profile.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in profile.notes)
    lines.append("=" * 78)
    return "\n".join(lines)


def print_profile_table(target: str | None = None) -> None:
    names = profile_names(target)
    print("Available live profiles:")
    for index, name in enumerate(names, start=1):
        profile = PROFILES[name]
        marker = " (recommended)" if profile.recommended else ""
        print(f"  {index}. {profile.name:<16} {profile.target:<6} {profile.label}{marker}")
        print(f"     run: {profile.command}")
    print()
    print("To choose from a menu, run without --list:")
    print(f"  {PYTHON_EXE} {WORK_DIR / 'run_live_profile.py'}")
    print("To start a profile directly, pass its number or name:")
    print(f"  {PYTHON_EXE} {WORK_DIR / 'run_live_profile.py'} 1")
    print(f"  {PYTHON_EXE} {WORK_DIR / 'run_live_profile.py'} single-best")
    print("If another live process already owns the camera, add --replace:")
    print(f"  {PYTHON_EXE} {WORK_DIR / 'run_live_profile.py'} 1 --replace")
    print("Preview windows are enabled by default for checking points; add --no-display for headless stimulation:")
    print(f"  {PYTHON_EXE} {WORK_DIR / 'run_live_profile.py'} single-best --no-display")


def build_profile_parser(target: str | None, default_profile: str | None) -> argparse.ArgumentParser:
    names = profile_names(target)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--profile",
        choices=names,
        default=default_profile,
        help="Runtime profile to apply before opening cameras and loading DLCLive.",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", action="store_true", help="Force OpenCV preview window on.")
    display_group.add_argument("--no-display", action="store_true", help="Force headless mode with no preview window.")
    parser.add_argument("--list-profiles", action="store_true", help="Print available profiles and exit.")
    return parser


def apply_cli_profile(
    config_module: Any,
    target: str | None,
    default_profile: str | None = None,
    argv: list[str] | None = None,
) -> LiveProfile | None:
    parser = build_profile_parser(target, default_profile)
    args, _ = parser.parse_known_args(argv)
    if args.list_profiles:
        print_profile_table(target)
        raise SystemExit(0)
    if not args.profile:
        return None
    profile = apply_profile(config_module, args.profile)
    display_value = display_value_for_profile(profile, args.display, args.no_display)
    apply_display_value(config_module, profile.target, display_value)
    print(profile_banner(profile, display_override=display_value), flush=True)
    return profile


def choose_profile_interactively(target: str | None = None) -> LiveProfile:
    names = profile_names(target)
    print_profile_table(target)
    print()
    while True:
        raw = input("Select profile number or name: ").strip()
        if not raw:
            raw = "single-best"
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(names):
                return PROFILES[names[index - 1]]
        if raw in PROFILES and (target is None or PROFILES[raw].target == target):
            return PROFILES[raw]
        print("Unknown selection. Try again.")


def main() -> int:
    parser = argparse.ArgumentParser(description="List available DLC live runtime profiles.")
    parser.add_argument("--target", choices=["single", "dual"], default=None)
    args = parser.parse_args()
    print_profile_table(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
