param(
  [string]$RuntimeRoot = "",
  [string]$GprMaxDir = "",
  [string]$GprMaxZip = "",
  [string]$CondaEnv = "uavgpr_gprmax_py310_gpu",
  [string]$MachineProfile = "auto",
  [string]$GpuIds = "0",
  [int]$OmpThreads = 8,
  [string]$RunScale = "auto",
  [string]$CondaEnvPrefix = "",
  [string]$MinicondaDir = "",
  [string]$CondaExe = "",
  [string]$DownloadDir = "",
  [switch]$UseExistingConda,
  [switch]$ForceRecreateEnv,
  [switch]$SkipCuda,
  [switch]$SkipVSBuildTools,
  [switch]$SkipPycuda,
  [switch]$NoGpuSmoke,
  [switch]$AllowCloneGprMax,
  [switch]$NoExternalCondaFallback
)

$ErrorActionPreference = "Continue"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

function Test-PathQuiet {
  param([string]$Path)
  if (-not $Path -or -not $Path.Trim()) { return $false }
  try { return [bool](Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue) }
  catch { return $false }
}

function Get-PathRootSafe {
  param([string]$Path)
  if (-not $Path -or -not $Path.Trim()) { return "" }
  try { return [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path)) }
  catch { return "" }
}

function Test-PathRootAvailable {
  param([string]$Path)
  $rootPath = Get-PathRootSafe $Path
  if (-not $rootPath) { return $false }
  try { return [bool](Test-Path -LiteralPath $rootPath -ErrorAction SilentlyContinue) }
  catch { return $false }
}

function Ensure-DirectoryChecked {
  param([string]$Path, [string]$Label)
  if (-not $Path -or -not $Path.Trim()) {
    Write-Host "$Label path is empty." -ForegroundColor Red
    return $false
  }
  if (-not (Test-PathRootAvailable $Path)) {
    Write-Host "$Label drive/root does not exist: $(Get-PathRootSafe $Path)" -ForegroundColor Red
    Write-Host "Path: $Path" -ForegroundColor Yellow
    return $false
  }
  try {
    New-Item -ItemType Directory -Force -Path $Path -ErrorAction Stop | Out-Null
    return $true
  } catch {
    Write-Host "Could not create $Label directory: $Path" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    return $false
  }
}

function Resolve-RequiredTool {
  param([string]$Name, [string[]]$Candidates)
  foreach ($candidate in $Candidates) {
    if (Test-PathQuiet $candidate) { return (Resolve-Path $candidate).Path }
  }
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  return ""
}

function Test-CoreWindowsTools {
  $systemRoot = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }
  $tools = @{
    "cmd.exe" = @((Join-PathSafe $systemRoot "System32\cmd.exe"))
    "where.exe" = @((Join-PathSafe $systemRoot "System32\where.exe"))
    "chcp.com" = @((Join-PathSafe $systemRoot "System32\chcp.com"))
    "powershell.exe" = @((Join-PathSafe $systemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"))
  }
  $missing = @()
  foreach ($name in $tools.Keys) {
    $resolved = Resolve-RequiredTool $name $tools[$name]
    if ($resolved) { Add-Log "Core Windows tool OK: $name -> $resolved" }
    else { $missing += $name }
  }
  if ($missing.Count -gt 0) {
    Write-Host "Core Windows command tools are missing from this process environment: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "This usually means PATH/SystemRoot is corrupted in the current shell." -ForegroundColor Yellow
    Add-Log "Missing core Windows tools: $($missing -join ', ')"
    return $false
  }
  return $true
}

function Join-PathSafe {
  param([string]$Base, [string]$Child)
  if (-not $Base -or -not $Base.Trim()) { return "" }
  try { return (Join-Path $Base $Child -ErrorAction Stop) }
  catch { return "" }
}

function Test-GprMaxSourceDir {
  param([string]$Candidate)
  $marker = Join-PathSafe $Candidate "gprMax\__main__.py"
  return (Test-PathQuiet $marker)
}

function Resolve-DefaultRuntimeRoot {
  if ($RuntimeRoot -and $RuntimeRoot.Trim()) { return $RuntimeRoot }
  if (Test-PathQuiet "D:\") { return "D:\UavGPR_Runtime" }
  if (Test-PathQuiet "E:\") { return "E:\UavGPR_Runtime" }
  return (Join-Path $Root "UavGPR_Runtime")
}

function Read-RuntimeEnvValue {
  param([string]$FilePath, [string]$Key)
  if (-not (Test-PathQuiet $FilePath)) { return "" }
  try {
    foreach ($line in Get-Content $FilePath -ErrorAction Stop) {
      $trimmed = $line.Trim()
      if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
      $idx = $trimmed.IndexOf("=")
      if ($idx -le 0) { continue }
      $k = $trimmed.Substring(0, $idx).Trim()
      $v = $trimmed.Substring($idx + 1).Trim()
      if ($k -eq $Key) { return $v }
    }
  } catch { return "" }
  return ""
}

function Resolve-DefaultGprMaxDir {
  if ($GprMaxDir -and $GprMaxDir.Trim()) { return $GprMaxDir }
  foreach ($envFile in @(
    (Join-Path $RuntimeRoot "uavgpr_runtime.env"),
    (Join-Path $Root ".simlab_env")
  )) {
    foreach ($key in @("UAVGPR_GPRMAX_ROOT", "GPRMAX_SOURCE_DIR")) {
      $value = Read-RuntimeEnvValue $envFile $key
      if ($value -and (Test-GprMaxSourceDir $value)) { return $value }
    }
  }
  foreach ($candidate in @(
    (Join-Path (Join-Path $RuntimeRoot "gprMax") "gprMax-v.3.1.7"),
    "D:\gprMax\gprMax-v.3.1.7",
    "E:\gprMax\gprMax-v.3.1.7",
    "D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7",
    "E:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7"
  )) {
    if (Test-GprMaxSourceDir $candidate) { return $candidate }
  }
  return (Join-Path (Join-Path $RuntimeRoot "gprMax") "gprMax-v.3.1.7")
}
$RuntimeRoot = [System.IO.Path]::GetFullPath((Resolve-DefaultRuntimeRoot))
if (-not $MinicondaDir) { $MinicondaDir = Join-Path $RuntimeRoot "miniconda3" }
if (-not $CondaEnvPrefix) { $CondaEnvPrefix = Join-Path (Join-Path $RuntimeRoot "conda_envs") $CondaEnv }
if (-not $GprMaxDir) { $GprMaxDir = Resolve-DefaultGprMaxDir }
if (-not $DownloadDir) { $DownloadDir = Join-Path $RuntimeRoot "downloads" }
$LogDir = Join-Path $RuntimeRoot "logs"

$requiredDirs = @(
  @{ Path = $RuntimeRoot; Label = "RuntimeRoot" },
  @{ Path = (Split-Path $MinicondaDir -Parent); Label = "Miniconda parent" },
  @{ Path = (Split-Path $CondaEnvPrefix -Parent); Label = "Conda env parent" },
  @{ Path = (Split-Path $GprMaxDir -Parent); Label = "gprMax parent" },
  @{ Path = $DownloadDir; Label = "DownloadDir" },
  @{ Path = $LogDir; Label = "LogDir" }
)
foreach ($d in $requiredDirs) {
  if (-not (Ensure-DirectoryChecked $d.Path $d.Label)) { exit 2 }
}

$Log = Join-Path $LogDir "setup_uavgpr_gpu_runtime_windows.log"
"UavGPR-SimLab unified GPU/gprMax setup started: $(Get-Date)" | Out-File -FilePath $Log -Encoding utf8
"Project root: $Root" | Add-Content -Path $Log
"RuntimeRoot: $RuntimeRoot" | Add-Content -Path $Log
"MinicondaDir: $MinicondaDir" | Add-Content -Path $Log
"CondaEnvPrefix: $CondaEnvPrefix" | Add-Content -Path $Log
"GprMaxDir: $GprMaxDir" | Add-Content -Path $Log
"UseExistingConda: $UseExistingConda" | Add-Content -Path $Log
"ForceRecreateEnv: $ForceRecreateEnv" | Add-Content -Path $Log
"AllowCloneGprMax: $AllowCloneGprMax" | Add-Content -Path $Log
"MachineProfile: $MachineProfile" | Add-Content -Path $Log
"GpuIds: $GpuIds" | Add-Content -Path $Log
"OmpThreads: $OmpThreads" | Add-Content -Path $Log

function Write-Step([string]$msg) {
  Write-Host "`n$msg" -ForegroundColor Cyan
  Add-Content -Path $Log -Value "`n$msg"
}

function Add-Log([string]$msg) { Add-Content -Path $Log -Value $msg }

function Show-LogTail([int]$lines = 80) {
  if (Test-Path $Log) {
    Write-Host "`n--- setup log tail ($Log) ---" -ForegroundColor Yellow
    Get-Content $Log -Tail $lines | ForEach-Object { Write-Host $_ }
    Write-Host "--- end log tail ---`n" -ForegroundColor Yellow
  }
}

function Refresh-Path {
  $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
  $user = [Environment]::GetEnvironmentVariable("Path", "User")
  $condaScripts = Join-Path $MinicondaDir "Scripts"
  $condaBin = Join-Path $MinicondaDir "Library\bin"
  $systemRoot = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }
  $mustHave = @(
    (Join-Path $systemRoot "System32"),
    $systemRoot,
    (Join-Path $systemRoot "System32\Wbem"),
    (Join-Path $systemRoot "System32\WindowsPowerShell\v1.0"),
    (Join-Path $systemRoot "System32\OpenSSH")
  ) -join ";"
  $env:Path = "$mustHave;$condaScripts;$condaBin;$machine;$user;$env:Path"
}

function ConvertTo-ProcessArgument([string]$arg) {
  if ($null -eq $arg) { return '""' }
  $s = [string]$arg
  if ($s.Length -eq 0) { return '""' }
  if ($s -notmatch '[\s"]') { return $s }
  return '"' + $s.Replace('"', '\"') + '"'
}

function Join-ProcessArguments([string[]]$argv) {
  return (($argv | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " ")
}

function Run-Logged([string]$exe, [string[]]$argv, [string]$cwd = $Root) {
  Refresh-Path
  $argLine = Join-ProcessArguments $argv
  Add-Log ("> " + $exe + " " + $argLine)
  Write-Host ("> " + $exe + " " + $argLine) -ForegroundColor DarkGray

  # Use PowerShell native invocation with an argument array instead of
  # Start-Process -ArgumentList <single string>. This prevents another round
  # of Windows command-line re-parsing from breaking commands such as:
  #   python -c "import sys; print(sys.version)"
  # It also keeps the refreshed PATH in the current process environment.
  Push-Location $cwd
  try {
    $global:LASTEXITCODE = 0
    $output = & $exe @argv 2>&1
    $exitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } elseif ($?) { 0 } else { 1 }
    foreach ($line in $output) { Add-Log ([string]$line) }
    Add-Log "[EXIT] $exitCode"
    return $exitCode
  } catch {
    Add-Log "[EXCEPTION] $($_.Exception.Message)"
    return 999
  } finally {
    Pop-Location
  }
}


function Get-SafeMachineProfileName {
  param([string]$Requested)
  $value = $Requested
  if (-not $value -or $value.Trim() -eq "" -or $value.Trim().ToLowerInvariant() -eq "auto") {
    $gpuText = "gpu"
    try {
      $gpuRaw = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
      if ($gpuRaw) { $gpuText = $gpuRaw }
    } catch {}
    $hostText = if ($env:COMPUTERNAME) { $env:COMPUTERNAME } else { "windows" }
    $value = ($hostText + "_" + $gpuText)
  }
  $value = ($value -replace "[^A-Za-z0-9_.-]", "_")
  $value = ($value -replace "_+", "_").Trim("_")
  if (-not $value) { $value = "windows_gpu" }
  return $value
}

$MachineProfile = Get-SafeMachineProfileName $MachineProfile
if ($RunScale -eq "auto") {
  if ($MachineProfile -match "4090") { $RunScale = "formal" }
  elseif ($MachineProfile -match "3060") { $RunScale = "small" }
  else { $RunScale = "smoke" }
}


function Find-VisualStudioInstallationPath {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-PathQuiet $vswhere) {
    try {
      $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1
      if ($path -and (Test-PathQuiet $path)) { return $path }
    } catch { Add-Log "vswhere lookup failed: $($_.Exception.Message)" }
  }
  foreach ($candidate in @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
    "D:\Visual Studio\2022\BuildTools",
    "E:\Visual Studio\2022\BuildTools",
    "D:\sisual stdio 2022",
    "E:\sisual stdio 2022"
  )) {
    if (Test-PathQuiet (Join-PathSafe $candidate "VC\Auxiliary\Build\vcvars64.bat")) { return $candidate }
    if (Test-PathQuiet (Join-PathSafe $candidate "Common7\Tools\VsDevCmd.bat")) { return $candidate }
  }
  return ""
}

function Import-VSDeveloperEnvironment {
  param([string]$VSPath)
  if (-not (Test-PathQuiet $VSPath)) { return $false }
  $vsDevCmd = Join-PathSafe $VSPath "Common7\Tools\VsDevCmd.bat"
  $vcVars64 = Join-PathSafe $VSPath "VC\Auxiliary\Build\vcvars64.bat"
  $cmdExe = Join-PathSafe $env:SystemRoot "System32\cmd.exe"
  if (-not (Test-PathQuiet $cmdExe)) { $cmdExe = "cmd.exe" }

  if (Test-PathQuiet $vsDevCmd) {
    $devCmd = '"' + $vsDevCmd + '" -no_logo -arch=x64 -host_arch=x64 && set'
    Add-Log "Loading VS developer environment with VsDevCmd: $vsDevCmd"
  } elseif (Test-PathQuiet $vcVars64) {
    $devCmd = '"' + $vcVars64 + '" && set'
    Add-Log "Loading VS developer environment with vcvars64: $vcVars64"
  } else {
    Add-Log "No VsDevCmd.bat or vcvars64.bat found under: $VSPath"
    return $false
  }

  try {
    $lines = & $cmdExe /d /s /c $devCmd 2>&1
    $code = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
    foreach ($line in $lines) { Add-Log ([string]$line) }
    if ($code -ne 0) {
      Add-Log "VS developer environment command failed with exit code $code"
      return $false
    }
    foreach ($line in $lines) {
      $s = [string]$line
      $idx = $s.IndexOf('=')
      if ($idx -le 0) { continue }
      $name = $s.Substring(0, $idx)
      $value = $s.Substring($idx + 1)
      if (-not $name) { continue }
      try { Set-Item -Path ("Env:" + $name) -Value $value -ErrorAction Stop } catch {}
    }
    Refresh-Path
    $env:DISTUTILS_USE_SDK = "1"
    $env:MSSdk = "1"
    return $true
  } catch {
    Add-Log "Failed to import VS developer environment: $($_.Exception.Message)"
    return $false
  }
}

function Test-HeaderInIncludePath {
  param([string]$HeaderName)
  if (-not $env:INCLUDE) { return $false }
  foreach ($dir in ($env:INCLUDE -split ";")) {
    if (-not $dir -or -not $dir.Trim()) { continue }
    $candidate = Join-PathSafe $dir $HeaderName
    if (Test-PathQuiet $candidate) { return $true }
  }
  return $false
}

function Find-WindowsSdkHeader {
  param([string]$RelativeHeader)
  foreach ($sdkRoot in @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\Include",
    "${env:ProgramFiles}\Windows Kits\10\Include",
    "${env:ProgramFiles(x86)}\Windows Kits\11\Include",
    "${env:ProgramFiles}\Windows Kits\11\Include"
  )) {
    if (-not (Test-PathQuiet $sdkRoot)) { continue }
    try {
      $matches = Get-ChildItem -Path $sdkRoot -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
      foreach ($versionDir in $matches) {
        $candidate = Join-PathSafe $versionDir.FullName $RelativeHeader
        if (Test-PathQuiet $candidate) { return $candidate }
      }
    } catch { Add-Log "Windows SDK header lookup failed in $sdkRoot: $($_.Exception.Message)" }
  }
  return ""
}

function Test-CudaBuildEnvironment {
  if ($SkipCuda -or $SkipPycuda) {
    Add-Log "CUDA/PyCUDA checks intentionally relaxed: SkipCuda=$SkipCuda SkipPycuda=$SkipPycuda"
    return $true
  }
  $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
  if (-not $nvcc) {
    Write-Host "nvcc was not found. Install CUDA Toolkit before enabling GPU/PyCUDA." -ForegroundColor Red
    Add-Log "nvcc missing before PyCUDA/gprMax GPU setup."
    return $false
  }
  Add-Log "nvcc resolved to: $($nvcc.Source)"
  $cudaRootsToCheck = @()
  if ($env:CUDA_PATH) { $cudaRootsToCheck += $env:CUDA_PATH }
  if ($env:CUDA_HOME) { $cudaRootsToCheck += $env:CUDA_HOME }
  try {
    $cudaRootsToCheck += (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | ForEach-Object { $_.FullName })
  } catch {}
  $cudaHeader = ""
  foreach ($rootCuda in $cudaRootsToCheck) {
    $candidate = Join-PathSafe $rootCuda "include\cuda.h"
    if (Test-PathQuiet $candidate) { $cudaHeader = $candidate; break }
  }
  if (-not $cudaHeader) {
    Write-Host "CUDA Toolkit is visible through nvcc, but cuda.h was not found." -ForegroundColor Red
    Write-Host "PyCUDA build requires CUDA Toolkit headers. Reinstall CUDA Toolkit or set CUDA_PATH." -ForegroundColor Yellow
    Add-Log "cuda.h missing. CUDA_PATH=$env:CUDA_PATH CUDA_HOME=$env:CUDA_HOME"
    return $false
  }
  Add-Log "CUDA header OK: $cudaHeader"
  return $true
}

function Initialize-MSVCBuildEnvironment {
  $vsPath = Find-VisualStudioInstallationPath
  if (-not $vsPath) {
    Write-Host "Visual Studio Build Tools with MSVC was not found." -ForegroundColor Red
    Add-Log "Visual Studio installation path not found."
    return $false
  }
  Write-Host "Using Visual Studio installation: $vsPath" -ForegroundColor Green
  Add-Log "Using Visual Studio installation: $vsPath"

  if (-not (Import-VSDeveloperEnvironment $vsPath)) {
    Write-Host "Could not load the Visual Studio developer environment." -ForegroundColor Red
    Add-Log "Import-VSDeveloperEnvironment failed."
    return $false
  }

  $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
  if (-not $cl) {
    Write-Host "cl.exe is still not available after loading the VS developer environment." -ForegroundColor Red
    Add-Log "cl.exe missing after VsDevCmd/vcvars64."
    return $false
  }
  Add-Log "cl.exe resolved to: $($cl.Source)"

  $hasIo = Test-HeaderInIncludePath "io.h"
  $hasWindows = Test-HeaderInIncludePath "windows.h"
  if (-not $hasIo) {
    $sdkIo = Find-WindowsSdkHeader "ucrt\io.h"
    if ($sdkIo) { Add-Log "io.h exists in Windows SDK but is not in INCLUDE: $sdkIo" }
  }
  if (-not $hasWindows) {
    $sdkWin = Find-WindowsSdkHeader "um\windows.h"
    if ($sdkWin) { Add-Log "windows.h exists in Windows SDK but is not in INCLUDE: $sdkWin" }
  }

  if (-not $hasIo -or -not $hasWindows) {
    Write-Host "MSVC is present, but Windows SDK/UCRT headers are not available in INCLUDE." -ForegroundColor Red
    Write-Host "Missing: io.h=$(-not $hasIo), windows.h=$(-not $hasWindows)" -ForegroundColor Red
    Write-Host "Fix: open Visual Studio Installer -> Modify Build Tools -> install 'Desktop development with C++' and a Windows 10/11 SDK, then rerun this script." -ForegroundColor Yellow
    Add-Log "INCLUDE=$env:INCLUDE"
    Add-Log "LIB=$env:LIB"
    Add-Log "WindowsSdkDir=$env:WindowsSdkDir"
    Add-Log "MSVC/Windows SDK header check failed: io.h=$hasIo windows.h=$hasWindows"
    return $false
  }

  Write-Host "MSVC + Windows SDK headers are available." -ForegroundColor Green
  Add-Log "MSVC + Windows SDK headers OK. INCLUDE=$env:INCLUDE"
  return $true
}

function Winget-Install([string[]]$ids, [string]$name, [string[]]$extra = @()) {
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host "winget not found; cannot install $name automatically." -ForegroundColor Yellow
    Add-Log "winget not found for $name"
    return $false
  }
  foreach ($id in $ids) {
    Write-Host "Trying winget package: $id" -ForegroundColor Yellow
    $args = @("install", "-e", "--id", $id, "--silent", "--accept-package-agreements", "--accept-source-agreements") + $extra
    $code = Run-Logged "winget" $args $Root
    Refresh-Path
    if ($code -eq 0) { return $true }
  }
  Write-Host "Could not install $name with known winget IDs. Check: $Log" -ForegroundColor Yellow
  return $false
}

function Download-File([string]$url, [string]$outPath) {
  Write-Host "Downloading: $url" -ForegroundColor Yellow
  Add-Log "Downloading: $url -> $outPath"
  try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
  try {
    Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing -ErrorAction Stop
    if (Test-PathQuiet $outPath) { return $true }
  } catch {
    Add-Log "Invoke-WebRequest failed: $($_.Exception.Message)"
  }
  $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
  if ($curl) {
    try {
      Add-Log "Trying curl.exe fallback."
      & $curl.Source -L --retry 3 --connect-timeout 30 -o $outPath $url 2>&1 | Add-Content -Path $Log
      if ($LASTEXITCODE -eq 0 -and (Test-PathQuiet $outPath)) { return $true }
      Add-Log "curl.exe exit code: $LASTEXITCODE"
    } catch { Add-Log "curl.exe fallback failed: $($_.Exception.Message)" }
  }
  if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
    try {
      Add-Log "Trying BITS fallback."
      Start-BitsTransfer -Source $url -Destination $outPath -ErrorAction Stop
      if (Test-PathQuiet $outPath) { return $true }
    } catch { Add-Log "BITS fallback failed: $($_.Exception.Message)" }
  }
  return $false
}
function Find-GprMaxSourceZip {
  param([string]$ExplicitZip)
  if ($ExplicitZip -and (Test-PathQuiet $ExplicitZip)) { return (Resolve-Path $ExplicitZip).Path }
  $candidates = @(
    (Join-Path $DownloadDir "gprMax-v.3.1.7.zip"),
    (Join-Path $RuntimeRoot "gprMax-v.3.1.7.zip"),
    (Join-Path $env:USERPROFILE "Downloads\gprMax-v.3.1.7.zip"),
    "D:\gprMax-v.3.1.7.zip",
    "E:\gprMax-v.3.1.7.zip"
  )
  foreach ($c in $candidates) { if (Test-PathQuiet $c) { return (Resolve-Path $c).Path } }
  return ""
}
function Resolve-GprMaxRootAfterExtract {
  param([string]$TargetDir)
  if (Test-GprMaxSourceDir $TargetDir) { return (Resolve-Path $TargetDir).Path }
  $children = Get-ChildItem $TargetDir -Directory -ErrorAction SilentlyContinue
  foreach ($d in $children) {
    if (Test-GprMaxSourceDir $d.FullName) { return $d.FullName }
  }
  return $TargetDir
}

function Find-CondaExe {
  if ($CondaExe -and (Test-PathQuiet $CondaExe)) { return (Resolve-Path $CondaExe).Path }
  $local = Join-Path $MinicondaDir "Scripts\conda.exe"
  if (Test-PathQuiet $local) { return (Resolve-Path $local).Path }
  if ($UseExistingConda) {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($c in @(
      "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
      "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
      "$env:ProgramData\miniconda3\Scripts\conda.exe",
      "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    )) { if (Test-PathQuiet $c) { return (Resolve-Path $c).Path } }
  }
  return ""
}

function Find-ExternalCondaExe {
  $cmd = Get-Command conda -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  foreach ($c in @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:ProgramData\miniconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
  )) { if (Test-PathQuiet $c) { return (Resolve-Path $c).Path } }
  return ""
}
function Test-CondaEnvPython {
  $py = Join-Path $CondaEnvPrefix "python.exe"
  if (-not (Test-PathQuiet $py)) { return $false }
  $code = Run-Logged $py @("-c", "import sys; print(sys.executable); print(sys.version)") $Root
  return ($code -eq 0)
}

function Remove-CondaEnvPrefix {
  if (Test-PathQuiet $CondaEnvPrefix) {
    Write-Host "Removing incomplete/stale conda prefix: $CondaEnvPrefix" -ForegroundColor Yellow
    Add-Log "Removing conda prefix: $CondaEnvPrefix"
    Run-Logged $CondaExe @("env", "remove", "-p", $CondaEnvPrefix, "-y") $Root | Out-Null
    Start-Sleep -Seconds 2
    if (Test-PathQuiet $CondaEnvPrefix) {
      try { Remove-Item -Recurse -Force $CondaEnvPrefix -ErrorAction Stop }
      catch { Add-Log "Remove-Item failed: $($_.Exception.Message)" }
    }
  }
}

function Create-CondaEnvFresh([string]$envFile) {
  New-Item -ItemType Directory -Force -Path (Split-Path $CondaEnvPrefix -Parent) | Out-Null
  Run-Logged $CondaExe @("config", "--set", "channel_priority", "flexible") $Root | Out-Null
  return (Run-Logged $CondaExe @("env", "create", "-p", $CondaEnvPrefix, "-f", $envFile) $Root)
}

function Ensure-CondaEnvironment([string]$envFile) {
  if ($ForceRecreateEnv) { Remove-CondaEnvPrefix }

  if (Test-CondaEnvPython) {
    Write-Host "Existing conda prefix detected; updating: $CondaEnvPrefix" -ForegroundColor Yellow
    $updateCode = Run-Logged $CondaExe @("env", "update", "-p", $CondaEnvPrefix, "-f", $envFile, "--prune") $Root
    if ($updateCode -eq 0 -and (Test-CondaEnvPython)) { return 0 }
    Write-Host "Conda env update failed; recreating prefix from scratch." -ForegroundColor Yellow
    Remove-CondaEnvPrefix
  } elseif (Test-PathQuiet $CondaEnvPrefix) {
    Write-Host "Conda prefix exists but is incomplete; recreating: $CondaEnvPrefix" -ForegroundColor Yellow
    Remove-CondaEnvPrefix
  }

  $createCode = Create-CondaEnvFresh $envFile
  if ($createCode -eq 0 -and (Test-CondaEnvPython)) { return 0 }

  Write-Host "First conda env create attempt failed; cleaning prefix and retrying once." -ForegroundColor Yellow
  Remove-CondaEnvPrefix
  $retryCode = Create-CondaEnvFresh $envFile
  if ($retryCode -eq 0 -and (Test-CondaEnvPython)) { return 0 }
  return $retryCode
}

Write-Step "[1/14] Check Windows / NVIDIA / core command tools"
if (-not $IsWindows -and $env:OS -notlike "Windows*") {
  Write-Host "This script is intended for Windows 10/11." -ForegroundColor Red
  exit 2
}
Refresh-Path
if (-not (Test-CoreWindowsTools)) { Show-LogTail; exit 2 }
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi | Tee-Object -FilePath $Log -Append } else { Write-Host "nvidia-smi not found. Install or update the NVIDIA driver first if this is a clean Windows machine." -ForegroundColor Yellow }

Write-Step "[2/14] Prepare centralized RuntimeRoot"
Write-Host "RuntimeRoot: $RuntimeRoot" -ForegroundColor Green
Write-Host "Miniconda:   $MinicondaDir" -ForegroundColor Green
Write-Host "Conda env:   $CondaEnvPrefix" -ForegroundColor Green
Write-Host "gprMax:      $GprMaxDir" -ForegroundColor Green
Write-Host "Downloads:   $DownloadDir" -ForegroundColor Green
Write-Host "Logs:        $LogDir" -ForegroundColor Green

Write-Step "[3/14] Ensure Miniconda / conda controller"
Refresh-Path
$CondaExe = Find-CondaExe
if (-not $CondaExe -or -not (Test-PathQuiet $CondaExe)) {
  $installer = Join-Path $DownloadDir "Miniconda3-latest-Windows-x86_64.exe"
  if (-not (Test-PathQuiet $installer)) {
    $ok = Download-File "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" $installer
    if (-not $ok) {
      $fallbackConda = if ($NoExternalCondaFallback) { "" } else { Find-ExternalCondaExe }
      if ($fallbackConda) {
        Write-Host "Could not download Miniconda installer. Falling back to existing conda controller: $fallbackConda" -ForegroundColor Yellow
        Write-Host "The actual gprMax/PyCUDA environment will still be created under: $CondaEnvPrefix" -ForegroundColor Yellow
        Add-Log "Miniconda download failed; using external conda controller only: $fallbackConda"
        $CondaExe = $fallbackConda
      } else {
        Write-Host "Could not download Miniconda installer." -ForegroundColor Red
        Write-Host "Manual option: put Miniconda3-latest-Windows-x86_64.exe in $installer and rerun." -ForegroundColor Yellow
        Write-Host "Alternative option: rerun with -UseExistingConda or -CondaExe if conda is already installed." -ForegroundColor Yellow
        Show-LogTail
        exit 2
      }
    }
  }
  if (-not $CondaExe -or -not (Test-PathQuiet $CondaExe)) {
    Write-Host "Installing Miniconda to $MinicondaDir" -ForegroundColor Yellow
    $installArgs = @("/S", "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/D=$MinicondaDir")
    $code = Run-Logged $installer $installArgs $Root
    if ($code -ne 0) { Write-Host "Miniconda installer failed with code $code." -ForegroundColor Red; Show-LogTail; exit 2 }
    Refresh-Path
    $CondaExe = Find-CondaExe
  }
}
if (-not $CondaExe -or -not (Test-PathQuiet $CondaExe)) { Write-Host "Conda still not found. Check $MinicondaDir." -ForegroundColor Red; Show-LogTail; exit 2 }
Write-Host "Using conda: $CondaExe" -ForegroundColor Green
Add-Log "Using conda: $CondaExe"
if ($CondaExe -notlike "$MinicondaDir*") {
  Write-Host "[WARN] Using external conda executable as controller only. Runtime env prefix remains centralized: $CondaEnvPrefix" -ForegroundColor Yellow
  Add-Log "[WARN] External conda controller in use: $CondaExe"
}

Write-Step "[4/14] Ensure Git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Winget-Install @("Git.Git") "Git" | Out-Null; Refresh-Path }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Run-Logged $CondaExe @("install", "git", "-y") $Root | Out-Null; Refresh-Path }

Write-Step "[5/14] Ensure Visual Studio 2022 Build Tools / MSVC / Windows SDK"
if (-not $SkipVSBuildTools) {
  $vsPathPre = Find-VisualStudioInstallationPath
  if (-not $vsPathPre) {
    Winget-Install @("Microsoft.VisualStudio.2022.BuildTools") "Visual Studio Build Tools" @("--override", "--wait --quiet --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100 --add Microsoft.VisualStudio.Component.Windows10SDK.19041 --includeRecommended") | Out-Null
  }
} else { Write-Host "Skipping Visual Studio Build Tools installation by user option; still checking existing MSVC/Windows SDK." -ForegroundColor Yellow }

if (-not (Initialize-MSVCBuildEnvironment)) {
  Write-Host "MSVC/Windows SDK environment check failed before gprMax build_ext." -ForegroundColor Red
  Show-LogTail
  exit 3
}

Write-Step "[6/14] Ensure CUDA Toolkit / nvcc"
Refresh-Path
if (-not $SkipCuda -and -not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
  Winget-Install @("Nvidia.CUDA", "NVIDIA.CUDA") "CUDA Toolkit" | Out-Null
  Refresh-Path
}
$cudaRoots = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue | Sort-Object Name -Descending
foreach ($rootCuda in $cudaRoots) {
  $bin = Join-Path $rootCuda.FullName "bin"
  if (Test-PathQuiet (Join-Path $bin "nvcc.exe")) {
    $env:Path = "$bin;$env:Path"
    if (-not $env:CUDA_PATH) { $env:CUDA_PATH = $rootCuda.FullName }
    if (-not $env:CUDA_HOME) { $env:CUDA_HOME = $rootCuda.FullName }
    break
  }
}
if (Get-Command nvcc -ErrorAction SilentlyContinue) { nvcc --version | Tee-Object -FilePath $Log -Append } else { Write-Host "nvcc not found. CPU gprMax can work; GPU/PyCUDA build requires CUDA Toolkit." -ForegroundColor Yellow }
if (-not (Test-CudaBuildEnvironment)) { Show-LogTail; exit 4 }

Write-Step "[7/14] Locate or prepare persistent gprMax source"
$zip = Find-GprMaxSourceZip $GprMaxZip
if (Test-GprMaxSourceDir $GprMaxDir) {
  Write-Host "Using existing persistent gprMax source: $GprMaxDir" -ForegroundColor Green
} elseif ($zip) {
  Write-Host "Using local gprMax zip: $zip" -ForegroundColor Green
  $cachedZip = Join-Path $DownloadDir "gprMax-v.3.1.7.zip"
  try {
    $zipResolved = (Resolve-Path $zip).Path
    $cachedResolved = if (Test-PathQuiet $cachedZip) { (Resolve-Path $cachedZip).Path } else { $cachedZip }
    if ($zipResolved -ne $cachedResolved) { Copy-Item -Force $zip $cachedZip }
  } catch { Add-Log "Could not cache gprMax zip: $($_.Exception.Message)" }
  $parent = Split-Path $GprMaxDir -Parent
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  Expand-Archive -Path $zip -DestinationPath $parent -Force
  $GprMaxDir = Resolve-GprMaxRootAfterExtract $parent
} elseif ($AllowCloneGprMax) {
  Write-Host "Local gprMax source/zip not found; cloning gprMax repository because -AllowCloneGprMax was provided." -ForegroundColor Yellow
  New-Item -ItemType Directory -Force -Path (Split-Path $GprMaxDir -Parent) | Out-Null
  if (Test-PathQuiet (Join-Path $GprMaxDir ".git")) { Run-Logged "git" @("pull") $GprMaxDir | Out-Null }
  else { Run-Logged "git" @("clone", "https://github.com/gprMax/gprMax.git", $GprMaxDir) $Root | Out-Null }
} else {
  Write-Host "gprMax source not found: $GprMaxDir" -ForegroundColor Red
  Write-Host "This release no longer bundles gprMax. Keep one persistent copy, then rerun for example:" -ForegroundColor Yellow
  Write-Host "  setup_gprmax_4090_windows.bat -RuntimeRoot `"$RuntimeRoot`" -GprMaxDir `"D:\UavGPR_Runtime\gprMax\gprMax-v.3.1.7`"" -ForegroundColor Yellow
  Write-Host "You may also pass -GprMaxZip if you still have a local gprMax-v.3.1.7.zip, or -AllowCloneGprMax to clone it." -ForegroundColor Yellow
  Show-LogTail
  exit 3
}
if (-not (Test-GprMaxSourceDir $GprMaxDir)) { Write-Host "gprMax source is invalid: $GprMaxDir" -ForegroundColor Red; Show-LogTail; exit 3 }
Write-Host "gprMax source root: $GprMaxDir" -ForegroundColor Green

Write-Step "[8/14] Create/update unified pinned conda environment under RuntimeRoot"
$envFile = Join-Path $Root "configs\environment_gprmax_4090_windows.yml"
if (-not (Test-PathQuiet $envFile)) { Write-Host "Missing environment file: $envFile" -ForegroundColor Red; exit 3 }
$envCode = Ensure-CondaEnvironment $envFile
if ($envCode -ne 0) {
  Write-Host "Conda environment create/update failed after clean retry. Check $Log." -ForegroundColor Red
  Show-LogTail
  exit 3
}
$EnvPython = Join-Path $CondaEnvPrefix "python.exe"
if (-not (Test-PathQuiet $EnvPython)) { Write-Host "Conda env python not found: $EnvPython" -ForegroundColor Red; Show-LogTail; exit 3 }
$envProbeCode = Run-Logged $EnvPython @("-c", "import sys, numpy, Cython, setuptools; print(sys.executable); print(sys.version); print('numpy', numpy.__version__); print('cython', Cython.__version__)") $Root
if ($envProbeCode -ne 0) { Write-Host "Conda env Python exists but required build modules are not importable." -ForegroundColor Red; Show-LogTail; exit 3 }

Write-Step "[9/14] Build gprMax Cython extensions"
if (-not (Initialize-MSVCBuildEnvironment)) { Write-Host "MSVC/Windows SDK environment was lost before build_ext." -ForegroundColor Red; Show-LogTail; exit 3 }
if (-not (Test-CudaBuildEnvironment)) { Show-LogTail; exit 4 }
$buildCode = Run-Logged $EnvPython @("setup.py", "build_ext", "--inplace") $GprMaxDir
if ($buildCode -ne 0) { Write-Host "gprMax build_ext failed. Check $Log." -ForegroundColor Red; Show-LogTail; exit 3 }
$compiledCount = 0
try { $compiledCount = @(Get-ChildItem -Path (Join-PathSafe $GprMaxDir "gprMax") -Filter "*.pyd" -Recurse -ErrorAction SilentlyContinue).Count } catch { $compiledCount = 0 }
Add-Log "Compiled gprMax .pyd count after build_ext: $compiledCount"
if ($compiledCount -lt 8) {
  Write-Host "gprMax build_ext returned success, but too few compiled .pyd extensions were found: $compiledCount" -ForegroundColor Red
  Show-LogTail
  exit 3
}

Write-Step "[10/14] Install gprMax editable into centralized conda environment"
$installGprCode = Run-Logged $EnvPython @("-m", "pip", "install", "-e", ".") $GprMaxDir
if ($installGprCode -ne 0) { Write-Host "gprMax editable install failed. Check $Log." -ForegroundColor Red; Show-LogTail; exit 3 }

Write-Step "[11/14] Install UavGPR-SimLab GUI package into same environment"
Run-Logged $EnvPython @("-m", "pip", "install", "--upgrade", "pip") $Root | Out-Null
$guiReqCode = Run-Logged $EnvPython @("-m", "pip", "install", "-r", "requirements_gui.txt") $Root
if ($guiReqCode -ne 0) { Write-Host "GUI dependency install failed. Check $Log." -ForegroundColor Red; Show-LogTail; exit 3 }
$simlabInstallCode = Run-Logged $EnvPython @("-m", "pip", "install", "-e", ".") $Root
if ($simlabInstallCode -ne 0) { Write-Host "UavGPR-SimLab editable install failed. Check $Log." -ForegroundColor Red; Show-LogTail; exit 3 }

Write-Step "[12/14] Install PyCUDA for GPU mode"
if (-not $SkipPycuda -and (Get-Command nvcc -ErrorAction SilentlyContinue)) {
  if (-not (Initialize-MSVCBuildEnvironment)) { Write-Host "MSVC/Windows SDK environment was lost before PyCUDA build." -ForegroundColor Red; Show-LogTail; exit 4 }
  if (-not (Test-CudaBuildEnvironment)) { Show-LogTail; exit 4 }
  $pycudaCode = Run-Logged $EnvPython @("-m", "pip", "install", "--no-cache-dir", "pycuda") $Root
  if ($pycudaCode -ne 0) {
    Write-Host "PyCUDA install failed. GPU mode cannot be accepted. Check CUDA/MSVC compatibility in $Log." -ForegroundColor Red
    Show-LogTail
    exit 4
  }
  $pycudaImportCode = Run-Logged $EnvPython @("-c", "import pycuda.driver as drv; drv.init(); print('pycuda_device_count', drv.Device.count())") $Root
  if ($pycudaImportCode -ne 0) {
    Write-Host "PyCUDA installed but CUDA driver check failed. GPU mode cannot be accepted. Check NVIDIA driver/CUDA Toolkit in $Log." -ForegroundColor Red
    Show-LogTail
    exit 4
  }
} elseif ($SkipPycuda) {
  Write-Host "Skipping PyCUDA by user option. GPU smoke will be disabled." -ForegroundColor Yellow
} else {
  Write-Host "nvcc not found. GPU mode cannot be configured unless -SkipPycuda is used intentionally." -ForegroundColor Red
  Show-LogTail
  exit 4
}

Write-Step "[13/14] Write persistent RuntimeRoot/profile settings"
$runtimeEnvPath = Join-Path $RuntimeRoot "uavgpr_runtime.env"
$projectEnvPath = Join-Path $Root ".simlab_env"
$envLines = @(
  "# UavGPR-SimLab persistent runtime settings. Safe to reuse across software versions and machines.",
  "UAVGPR_MACHINE_PROFILE=$MachineProfile",
  "UAVGPR_GPU_RUNTIME_ENV=unified_py310_gpu",
  "UAVGPR_RUN_SCALE=$RunScale",
  "UAVGPR_RUNTIME_ROOT=$RuntimeRoot",
  "UAVGPR_MINICONDA_DIR=$MinicondaDir",
  "UAVGPR_CONDA_EXE=$CondaExe",
  "UAVGPR_CONDA_ENV=$CondaEnv",
  "UAVGPR_CONDA_ENV_PREFIX=$CondaEnvPrefix",
  "UAVGPR_PYTHON_EXE=$EnvPython",
  "UAVGPR_GPRMAX_ROOT=$GprMaxDir",
  "GPRMAX_SOURCE_DIR=$GprMaxDir",
  "UAVGPR_USE_CONDA_RUN=0",
  "UAVGPR_GPU_IDS=$GpuIds",
  "UAVGPR_USE_GPU=1",
  "UAVGPR_GPU_ENABLED=1",
  "UAVGPR_OMP_THREADS=$OmpThreads"
)
$profileDir = Join-Path $RuntimeRoot "runtime_profiles"
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
$profileEnvPath = Join-Path $profileDir ("$MachineProfile.env")
$envLines | Set-Content -Path $runtimeEnvPath -Encoding utf8
$envLines | Set-Content -Path $projectEnvPath -Encoding utf8
$envLines | Set-Content -Path $profileEnvPath -Encoding utf8
Write-Host "Wrote persistent runtime env: $runtimeEnvPath" -ForegroundColor Green
Write-Host "Wrote machine profile env:     $profileEnvPath" -ForegroundColor Green
Write-Host "Wrote project launcher env:    $projectEnvPath" -ForegroundColor Green
Write-Host "Central runtime root: $RuntimeRoot" -ForegroundColor Green

Write-Step "[14/14] CPU/GPU smoke verification"
$verifyArgs = @("scripts\check_4090_gprmax_gpu.py", "--gprmax-root", $GprMaxDir, "--python-executable", $EnvPython, "--gpu-ids", $GpuIds, "--out", "logs\check_4090_gprmax_gpu_report.json")
if ($NoGpuSmoke) { $verifyArgs += "--no-gpu" }
$verifyCode = Run-Logged $EnvPython $verifyArgs $Root
if ($verifyCode -ne 0) {
  Write-Host "GPU/gprMax verification failed. Check logs\check_4090_gprmax_gpu_report.json and $Log." -ForegroundColor Red
  Show-LogTail
  exit 5
}

Write-Host "`nUnified GPU runtime setup finished and verified. Log: $Log" -ForegroundColor Green
Write-Host "Persistent runtime root: $RuntimeRoot" -ForegroundColor Green
Write-Host "Future UavGPR-SimLab versions will reuse $runtimeEnvPath and the persistent external gprMax source." -ForegroundColor Green
Write-Host "Launch GUI with run_gui.bat. Runtime uses the explicit conda-prefix python.exe by default; in Settings, confirm gprMax source, environment prefix, GPU=on, GPU IDs as configured." -ForegroundColor Green
exit 0
