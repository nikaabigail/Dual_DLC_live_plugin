"""
Smoke-test the DualDLCLiveBridge Windows DLL from the build tree.

This verifies that the DLL can be located and that the standard Open Ephys
plugin entrypoints are exported. Full plugin initialization still happens in
the Open Ephys GUI process.
"""
from __future__ import annotations

import os
import sys
from ctypes import wintypes
from pathlib import Path

import ctypes


DONT_RESOLVE_DLL_REFERENCES = 0x00000001


def main() -> None:
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    candidates = [
        cwd / "out" / "build" / "x64-Debug" / "plugins" / "DualDLCLiveBridge.dll",
        script_dir.parents[1] / "dist" / "windows-x64-debug" / "DualDLCLiveBridge.dll",
    ]

    plugin = next((path for path in candidates if path.exists()), None)
    if plugin is None:
        raise FileNotFoundError("DualDLCLiveBridge.dll not found in build tree or dist/windows-x64-debug")

    build_root = plugin.parent.parent if plugin.parent.name.lower() == "plugins" else None
    if build_root is not None:
        print(f"BUILD_ROOT {build_root}", flush=True)
        os.add_dll_directory(str(build_root))
        os.add_dll_directory(str(build_root / "plugins"))

    print(f"PLUGIN {plugin}", flush=True)
    os.add_dll_directory(str(plugin.parent))
    print("DLL_SEARCH_PATH_OK", flush=True)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    kernel32.LoadLibraryExW.restype = wintypes.HMODULE
    kernel32.GetProcAddress.argtypes = [wintypes.HMODULE, ctypes.c_char_p]
    kernel32.GetProcAddress.restype = ctypes.c_void_p
    kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
    kernel32.FreeLibrary.restype = wintypes.BOOL

    print("PLUGIN_MAP_START", flush=True)
    handle = kernel32.LoadLibraryExW(str(plugin), None, DONT_RESOLVE_DLL_REFERENCES)
    if not handle:
        error = ctypes.get_last_error()
        print(f"PLUGIN_MAP_FAILED winerror={error}", flush=True)
        raise SystemExit(2)
    print("PLUGIN_MAP_OK", flush=True)

    for name in ("getLibInfo", "getPluginInfo"):
        if not kernel32.GetProcAddress(handle, name.encode("ascii")):
            print(f"EXPORT_MISSING {name}", flush=True)
            kernel32.FreeLibrary(handle)
            raise SystemExit(3)
        print(f"EXPORT_OK {name}", flush=True)
    kernel32.FreeLibrary(handle)
    print("PLUGIN_EXPORTS_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CHECK_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
