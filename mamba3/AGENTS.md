# Mamba-3 Training Guide (Windows)

## Venv activation
- MUST use dot-sourcing: `. .\.venv\Scripts\Activate.ps1`
- NEVER use call operator (`& .\.venv\Scripts\Activate.ps1`) — it forks a child scope, deactivation doesn't work.

## Setup order
1. `env_314.ps1` — one-shot env setup (preferred). Installs Python 3.14→3.12 fallback, ROCm 7.13.0 gfx1152 wheels, all deps. Includes gates for Smart App Control, VC++ redist, GPU driver ≥31.x, username spaces.
2. `run_smoke_test.ps1` — runs `smoke_test.py` (793 lines, 9 sections) before training.
3. `extract_crepe.py` — offline CREPE f0 extraction, saves `.crepe.npy` + `.pitched.npy` per stem. Resumable.
4. `run_train.ps1` — auto-detects latest checkpoint for resume.

## Device quirks
- **Device string**: `cuda` for ROCm on Windows (`torch.cuda` API compat layer, not real CUDA).
- **HSA override**: NOT needed on Windows (native gfx1152 wheels). Linux Bazzite needs `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- **Required env vars** (set by `run_train.ps1` / `run_smoke_test.ps1`):
  - `HSA_ENABLE_SDMA=0` — prevents memory pinning conflicts on APU shared RAM
  - `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256` — defrag on 8 GB iGPU

## Training commands
- **Windows**: `run_train.ps1` (PowerShell wrapper). Sets env vars, auto-detects checkpoints, tee's to `train.log`.
- **Linux**: `run_train.sh` (Bash wrapper). Uses `--ckpt-every 1000`, sets `HSA_OVERRIDE_GFX_VERSION=11.0.0` + `MIOPEN_FIND_MODE=5`.
- `--ckpt-every 500` on Windows (gfx1152 GPU hang bug mitigation), `--ckpt-every 1000` on Linux.
- `--no-amp` disables bf16 — use if NaN/Inf every step.
- Resume via `--resume checkpoints/checkpoint_XXXXXX.pt` (loads model+optimizer+scheduler+step). Checkpoints written atomically (`.tmp` → rename).

## Dual smoke tests
- `smoke_test.py` (793 lines, 9 sections) — referenced and run by `run_smoke_test.ps1`
- `test_smoke.py` (764 lines, fewer sections) — referenced by `TRAINING_WINDOWS.md`
- Both exist. Scripts use `smoke_test.py`, docs reference `test_smoke.py`. Run `smoke_test.py`.

## Architecture
- **`model.py`**: `PerTrackProcessor` entrypoint. R=8 structured spectral features → Mamba-3 MIMO (complex poles, trapezoidal discretization) → FiLM conditioning → overlap-add decoder.
- **`train.py`**: `train()` entrypoint. 3-phase: Phase 1 (0-20%) NT-Xent contrastive, Phase 2 (20-50%) single-track MR-STFT+Mel-L1, Phase 3 (50%+) cross-track with SI-SDR ramp.
- **`extract_crepe.py`**: Offline CREPE. Pitched stems (vocals/bass/other/guitar/piano/keys) → real inference, unpitched (drums) → zeros, mixture → skipped.

## Dataset
- Expected at `$env:USERPROFILE\datasets\musdb18hq\train\`
- Layout: `{track_dir}/{bass,drums,vocals,other,mixture}.wav` (86 train tracks)
- CREPE files: `<stem>.crepe.npy` (T, 360 fp16) + `<stem>.pitched.npy` alongside each WAV
- Training works without CREPE files (zeros fill-in, f0 conditioning inert)

## ROCm on Windows caveats
- PyTorch-only ROCm: no rocprofiler, no HIP tools, no rocBLAS, no ROCm-aware Triton.
- Smart App Control MUST be OFF (one-way disable, requires Windows reinstall to re-enable).
- GPU hang bug `MES:0x82` on Ryzen AI 7 350 / Radeon 860M — aggressive checkpointing is mitigation.
- `torch.cuda.is_available()` returns True (API-compat layer, not real CUDA).

## Linux venv artifacts
- `.venv/`, `.venv_nightly/`, `venv/` are Linux-built venvs (pyvenv.cfg shows `home=/usr/bin`)
- Must delete before running `env_314.ps1` on Windows
- Detect by checking `pyvenv.cfg` for `home = /usr/bin` or `home = /home/`

## Testing
- No formal test framework — smoke tests are the gate
- `python smoke_test.py --device cuda` (venv activated)

## PowerShell quirks
- Backtick (`` ` ``) for line continuation
- `Tee-Object` for log tee (equivalent of `tee`)
- `$LASTEXITCODE` for checking exit codes (not `$?`)
