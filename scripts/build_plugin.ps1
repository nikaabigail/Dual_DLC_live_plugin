<#
================================================================================
 build_plugin.ps1
 Repeatable build + deploy of the DualDLCLiveBridge Open Ephys plugin.

 Source of truth for the plugin code is THIS repo: open_ephys_plugin\DualDLCLiveBridge\.
 To build, the source is synced into a plugin-GUI tree's Plugins\ folder and built
 in the requested configuration; the resulting DLL is copied to dist\ and (optionally)
 straight into a target Open Ephys plugins folder.

 Update workflow:
   1. edit open_ephys_plugin\DualDLCLiveBridge\*.cpp / *.h in this repo
   2. .\scripts\build_plugin.ps1                      # builds Release, DLL -> dist\windows-x64-release\
   3. copy dist\windows-x64-release\DualDLCLiveBridge.dll to the target's plugins folder
      (or pass -DeployDir to copy automatically), then restart Open Ephys
   4. commit the changed source (+ dll) to the repo

 Params:
   -GuiRoot   : path to a plugin-GUI checkout that MATCHES the target Open Ephys
                version (GUI_VERSION must equal the target; here 1.0.1 / API 10).
   -Config    : Release (default) or Debug.
   -DeployDir : optional; if given, the built DLL is also copied here
                (e.g. "$env:LOCALAPPDATA\Open Ephys\plugins-api10" on the target,
                or a shared/USB folder to carry to the target).
   -VcVars    : optional explicit path to vcvars64.bat (auto-discovered otherwise).

 Example:
   .\scripts\build_plugin.ps1
   .\scripts\build_plugin.ps1 -Config Release -DeployDir "D:\transfer\plugins"
================================================================================
#>
[CmdletBinding()]
param(
  [string]$GuiRoot   = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'plugin-GUI-main\plugin-GUI-main'),
  [ValidateSet("Release","Debug")][string]$Config = "Release",
  [string]$DeployDir = "",
  [string]$VcVars    = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot   = Split-Path $PSScriptRoot -Parent
$PluginSrc  = Join-Path $RepoRoot "open_ephys_plugin\DualDLCLiveBridge"
$cfgLower   = $Config.ToLower()

function Find-VcVars {
  param([string]$Explicit)
  if ($Explicit -and (Test-Path $Explicit)) { return $Explicit }
  $vsw = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vsw) {
    $p = & $vsw -all -prerelease -latest -property installationPath 2>$null | Select-Object -First 1
    if ($p) { $c = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"; if (Test-Path $c) { return $c } }
  }
  foreach ($r in @(
      "C:\Program Files\Microsoft Visual Studio\18\Insiders",
      "C:\Program Files\Microsoft Visual Studio\2022\Community",
      "C:\Program Files\Microsoft Visual Studio\2022\Professional",
      "C:\Program Files\Microsoft Visual Studio\2022\BuildTools")) {
    $c = Join-Path $r "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $c) { return $c }
  }
  throw "vcvars64.bat not found. Install Visual Studio C++ (Desktop development with C++) or pass -VcVars."
}

if (-not (Test-Path $PluginSrc)) { throw "Plugin source not found: $PluginSrc" }
if (-not (Test-Path $GuiRoot))   { throw "plugin-GUI tree not found: $GuiRoot (clone the matching plugin-GUI and pass -GuiRoot)" }
$vc = Find-VcVars -Explicit $VcVars
Write-Host "==> plugin-GUI : $GuiRoot"
Write-Host "==> vcvars     : $vc"
Write-Host "==> config     : $Config"

# 1) sync the repo's plugin source into the GUI tree
$dstPlug = Join-Path $GuiRoot "Plugins\DualDLCLiveBridge"
Write-Host "==> syncing plugin source -> $dstPlug"
New-Item -ItemType Directory -Force $dstPlug | Out-Null
Copy-Item "$PluginSrc\*" $dstPlug -Recurse -Force
# ensure the GUI's Plugins\CMakeLists.txt registers our plugin
$pluginsCMake = Join-Path $GuiRoot "Plugins\CMakeLists.txt"
if ((Test-Path $pluginsCMake) -and -not (Select-String -Path $pluginsCMake -Pattern "add_subdirectory\(DualDLCLiveBridge\)" -Quiet)) {
  Add-Content $pluginsCMake "`nadd_subdirectory(DualDLCLiveBridge)"
  Write-Host "   added add_subdirectory(DualDLCLiveBridge) to Plugins\CMakeLists.txt"
}

# 2) configure + build (inside one cmd that first runs vcvars64 so cl/ninja are on PATH)
$buildDir = Join-Path $GuiRoot "out\build\x64-$Config"
$cmds = @(
  "call `"$vc`"",
  "cmake -S `"$GuiRoot`" -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=$Config -DOE_DONT_CHECK_BUILD_PATH=TRUE",
  "cmake --build `"$buildDir`" --target DualDLCLiveBridge"
) -join " && "
Write-Host "==> building (this can take a while on first Release configure)..."
& cmd /c $cmds
if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }

# 3) collect the DLL -> repo dist\ and optional deploy dir
$dll = Join-Path $buildDir "Plugins\DualDLCLiveBridge.dll"
if (-not (Test-Path $dll)) { throw "DLL not found after build: $dll" }
$distDir = Join-Path $RepoRoot "dist\windows-x64-$cfgLower"
New-Item -ItemType Directory -Force $distDir | Out-Null
Copy-Item $dll $distDir -Force
Write-Host "==> DLL -> $distDir\DualDLCLiveBridge.dll"
if ($DeployDir) {
  New-Item -ItemType Directory -Force $DeployDir | Out-Null
  Copy-Item $dll $DeployDir -Force
  Write-Host "==> deployed -> $DeployDir\DualDLCLiveBridge.dll"
}
Write-Host "`nDONE ($Config). On the target: copy the DLL into the Open Ephys plugins folder"
Write-Host "(%LOCALAPPDATA%\Open Ephys\plugins-api10), ensure VC++ Redistributable x64 is installed, restart Open Ephys."
