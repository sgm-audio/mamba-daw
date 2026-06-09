#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Mamba-3 Windows Training Loop Runner
.DESCRIPTION
    Automates ROCm/DirectML GPU configuration, memory tuning for Windows,
    automatic checkpoint detection, and logging.
.PARAMETER Data
    Path to MUSDB18-HQ train/ root directory
.PARAMETER Batch
    Batch size (default 2)
.PARAMeter Device
    Device backend: auto, cuda (ROCm), dml (DirectML/NPU), or cpu
.PARAMETER NoAmp
    Disable bf16 AMP
.PARAMETER Steps
    Total training steps
.PARAMETER Resume
    Path to specific checkpoint to resume from
#>

param(
    [string]$Data = "",
    [int]$Batch = 2,
    [string]$Device = "auto",
    [switch]$NoAmp = $false,
    [int]$Steps = 100000,
    [string]$Resume = "",
    [int]$CkptEvery = 500,
    [int]$ValEvery = 2000,
    [int]$NumWorkers = 0,
    [int]$GradAccum = 8,
    [switch]$Help = $false,
    [switch]$Laptop = $false
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvActivate = Join-Path $ScriptDir ".venv" "Scripts" "Activate.ps1"
$LogFile = Join-Path $ScriptDir "train.log"
$DefaultDataset = Join-Path $ScriptDir "datasets\musdb18hq\train"

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
# 3. Auto-detect best available device backend
# ────────────────────────────────────────────────────────────────
if ($Device -eq "auto") {
    Write-Host "--> Auto-detecting best available backend..."
    # Try ROCm/CUDA first
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

    Write-Host "  → Detected backend: $Device"

    if ($Device -eq "cuda") {
        Write-Host "  → GPU: $gpuName"
        Write-Host "  → PyTorch backend: $backend"
    } elseif ($Device -eq "dml") {
        Write-Host "  → Using DirectML (supports NPU / AMD / NVIDIA GPU via DirectX 12)"
    } else {
        Write-Host "  → No GPU detected, falling back to CPU"
    }
}

# ────────────────────────────────────────────────────────────────
# 4. Configure environment variables
# ────────────────────────────────────────────────────────────────
if ($Device -eq "cuda") {
    # AMD ROCm on Windows (HIP SDK)
    Write-Host "--> Configuring ROCm environment..."

    # Detect if this is an AMD GPU needing HSA override
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
        Write-Host "    AMD APU detected — setting HSA_OVERRIDE_GFX_VERSION=11.0.0"
        $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
    }

    # Disable SDMA to prevent memory pinning conflicts on APU shared RAM
    $env:HSA_ENABLE_SDMA = "0"

    # Optimize PyTorch memory allocator
    $env:PYTORCH_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:256"

    # Silence MIOpen kernel selection output (floods stderr, kills throughput)
    $env:MIOPEN_LOG_LEVEL = "3"
    # Fast find mode: hybrid heuristic, no exhaustive kernel search
    $env:MIOPEN_FIND_MODE = "2"

    Write-Host "    PYTORCH_ALLOC_CONF = $env:PYTORCH_ALLOC_CONF"
    Write-Host "    HSA_ENABLE_SDMA     = $env:HSA_ENABLE_SDMA"
    Write-Host "    MIOPEN_FIND_MODE    = $env:MIOPEN_FIND_MODE"

} elseif ($Device -eq "dml") {
    Write-Host "--> Using DirectML backend via torch-directml"
    Write-Host "    Device will be 'dml' — supports AMD/NVIDIA/Intel/NPU"
    $amp = $false  # DirectML bf16 support is limited
}

# ────────────────────────────────────────────────────────────────
# 5. Resolve dataset path
# ────────────────────────────────────────────────────────────────
if ([string]::IsNullOrEmpty($Data)) {
    if (Test-Path $DefaultDataset) {
        $Data = $DefaultDataset
        Write-Host "--> Using default dataset: $Data"
    } else {
        Write-Host "ERROR: No dataset path provided and default not found at $DefaultDataset"
        Write-Host "Pass --Data 'D:\path\to\musdb18hq\train'"
        exit 1
    }
}

if (-not (Test-Path $Data)) {
    Write-Host "ERROR: Dataset path does not exist: $Data"
    exit 1
}

# ────────────────────────────────────────────────────────────────
# 6. Check for checkpoints to resume automatically
# ────────────────────────────────────────────────────────────────
$CkptDir = if ($Laptop) { Join-Path $ScriptDir "checkpoints_laptop" } else { Join-Path $ScriptDir "checkpoints" }
$LatestCkpt = ""

if ([string]::IsNullOrEmpty($Resume) -and (Test-Path $CkptDir)) {
    $LatestCkpt = Get-ChildItem -Path $CkptDir -Filter "checkpoint_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $LatestCkpt) {
        $LatestCkpt = Get-ChildItem -Path $CkptDir -Filter "interrupt_*.pt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
} elseif (-not [string]::IsNullOrEmpty($Resume)) {
    $LatestCkpt = $Resume
}

# ────────────────────────────────────────────────────────────────
# 7. Build arguments
# ────────────────────────────────────────────────────────────────
$pyArgs = @(
    "--data", "$Data"
    "--device", $Device
    "--batch", $Batch
    "--steps", $Steps
    "--ckpt-every", $CkptEvery
    "--val-every", $ValEvery
    "--num-workers", $NumWorkers
    "--grad-accum", $GradAccum
)

if ($NoAmp) {
    $pyArgs += "--no-amp"
}

if ($LatestCkpt) {
    Write-Host "--> [Resume] Auto-detected latest checkpoint: $LatestCkpt"
    $pyArgs += "--resume", "$LatestCkpt"
} else {
    Write-Host "--> [Fresh Start] No existing checkpoints found."
}

# Append any splatted user arguments from -ArgumentList
# (captured via $args from splatting in parent scope if desired)

# ────────────────────────────────────────────────────────────────
# 8. Print banner and launch
# ────────────────────────────────────────────────────────────────
$trainScriptName = if ($Laptop) { "train_laptop.py" } else { "train.py" }
$logFileName    = if ($Laptop) { "train_laptop.log" } else { "train.log" }
$LogFile = Join-Path $ScriptDir $logFileName

Write-Host "="x70
if ($Laptop) {
    Write-Host " Mamba-3 LAPTOP Training (Rock GPU)"
} else {
    Write-Host " Mamba-3 Windows Training"
}
Write-Host " Device: $Device"
if ($Device -eq "cuda") {
    Write-Host " ROCm mode active"
}
if ($Device -eq "dml") {
    Write-Host " DirectML mode active (NPU/GPU)"
}
if ($Laptop) {
    Write-Host " Model: Lean (~785K params)  |  2-sec clips  |  1x STFT loss"
}
Write-Host " Dataset: $Data"
Write-Host " Batch: $Batch | Steps: $Steps | GradAccum: $GradAccum | Workers: $NumWorkers | AMP: $(-not $NoAmp)"
Write-Host " Logging to: $LogFile"
Write-Host "="x70

$pythonPath = (Get-Command python).Source
$trainScript = Join-Path $ScriptDir $trainScriptName

Write-Host "--> Python: $pythonPath"
Write-Host "--> Script: $trainScript"
Write-Host "--> Args: $($pyArgs -join ' ')"
Write-Host "-"x70

# Run training, tee to log file
$startTime = Get-Date
$stderrLog = Join-Path $ScriptDir "stderr.log"
try {
    & $pythonPath $trainScript @pyArgs 2>> $stderrLog | ForEach-Object {
        $_ | Out-File -FilePath $LogFile -Append
        $_
    }
} catch {
    Write-Host "`nTraining interrupted or errored: $_" -ForegroundColor Yellow
} finally {
    $elapsed = (Get-Date) - $startTime
    Write-Host "-"x70
    Write-Host " Session ended. Elapsed: $($elapsed.Hours)h $($elapsed.Minutes)m $($elapsed.Seconds)s"
    Write-Host " Log: $LogFile"
    Write-Host "-"x70
}
