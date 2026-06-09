#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Mamba-3 Windows Smoke Test Launcher
.DESCRIPTION
    Activates the local virtual environment, configures AMD ROCm or DirectML,
    and runs smoke_test.py on the selected device.
.PARAMETER Device
    Device backend: auto, cuda (ROCm), dml (DirectML/NPU), or cpu
.PARAMETER Amp
    Force bf16 AMP test even on CPU
#>

param(
    [string]$Device = "auto",
    [switch]$Amp = $false,
    [switch]$Help = $false
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvActivate = Join-Path $ScriptDir ".venv" "Scripts" "Activate.ps1"

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Definition -Full
    exit 0
}

# ────────────────────────────────────────────────────────────────
# 1. Check virtual environment
# ────────────────────────────────────────────────────────────────
if (-not (Test-Path $VenvActivate)) {
    Write-Host "ERROR: Virtual environment not found at $VenvActivate"
    Write-Host "Please run '.\setup_env.ps1' first to create the environment."
    exit 1
}

# ────────────────────────────────────────────────────────────────
# 2. Activate environment
# ────────────────────────────────────────────────────────────────
. $VenvActivate

# ────────────────────────────────────────────────────────────────
# 3. Auto-detect device if set to auto
# ────────────────────────────────────────────────────────────────
if ($Device -eq "auto") {
    Write-Host "--> Auto-detecting best available backend..."
    $pythonCheck = @"
import torch, sys
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    backend = torch.version.hip if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA'
    print(f'cuda|{gpu}|{backend}')
    sys.exit(0)
try:
    import torch_directml
    d = torch_directml.device()
    print(f'dml|DirectML device')
    sys.exit(0)
except:
    pass
print('cpu||')
"@
    $detect = python -c $pythonCheck 2>&1 | Select-Object -Last 1
    $parts = $detect.ToString().Split('|')
    $Device = $parts[0]
    $gpuName = $parts[1]
    $backend = $parts[2]

    Write-Host "  -> Detected backend: $Device"
    if ($Device -eq "cuda") {
        Write-Host "  -> GPU: $gpuName"
    }
}

# ────────────────────────────────────────────────────────────────
# 4. Configure environment variables
# ────────────────────────────────────────────────────────────────
if ($Device -eq "cuda") {
    Write-Host "--> Configuring ROCm environment..."

    $pythonCheckHsa = @"
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0).lower()
    if '860m' in name or '870m' in name or '880m' in name or 'radeon' in name:
        print('need_hsa_override')
    else:
        print('ok')
"@
    $hsaNeed = python -c $pythonCheckHsa 2>&1 | Select-Object -Last 1
    if ($hsaNeed -eq "need_hsa_override") {
        Write-Host "    AMD APU detected -- setting HSA_OVERRIDE_GFX_VERSION=11.0.0"
        $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
    }

    $env:HSA_ENABLE_SDMA = "0"
    $env:PYTORCH_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:256"

    Write-Host "    HSA_OVERRIDE_GFX_VERSION = $env:HSA_OVERRIDE_GFX_VERSION"
    Write-Host "    PYTORCH_ALLOC_CONF       = $env:PYTORCH_ALLOC_CONF"
}

if ($Device -eq "dml") {
    Write-Host "--> Using DirectML backend (NPU/GPU)"
}

# ────────────────────────────────────────────────────────────────
# 5. Execute smoke test
# ────────────────────────────────────────────────────────────────
$pythonPath = (Get-Command python).Source
$smokeScript = Join-Path $ScriptDir "smoke_test.py"

$pyArgs = @("--device", $Device)
if ($Amp) {
    $pyArgs += "--amp"
}

Write-Host "--> Launching Windows smoke tests..."
Write-Host "--> Python: $pythonPath"
Write-Host "--> Device: $Device"
Write-Host "-"x70

& $pythonPath $smokeScript @pyArgs
exit $LASTEXITCODE
