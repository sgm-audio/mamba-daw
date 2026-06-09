# Mamba-3 Training Workflow Commands

> **Quick reference for resuming/running training on Bazzite + ROCm + gfx1152 (spoofed as gfx1100).**
> Project lives at `~/mamba3/`. Venv at `~/venvs/mamba-rocm/`. Distrobox `rocm-mamba`.

---

## 1. Initial Session Setup

Run these **every time you open a fresh Bazzite terminal**.

### 1.1 On Bazzite host (one block):
```bash
# Prevent sleep / lid-close / idle suspend for the whole training session.
# Leave this terminal open. Sleep is blocked until you Ctrl+C or close it.
systemd-inhibit \
  --who="mamba3-training" \
  --why="Long ML training run" \
  --what="sleep:idle:handle-lid-switch" \
  sleep infinity
```

### 1.2 Enter the container (new terminal):
```bash
distrobox enter rocm-mamba
```

### 1.3 Inside the container — environment setup (one block):
```bash
source ~/venvs/mamba-rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export MIOPEN_FIND_MODE=5
cd ~/mamba3
```

### 1.4 Verify environment is sane:
```bash
echo "HSA=$HSA_OVERRIDE_GFX_VERSION  ARCH=$PYTORCH_ROCM_ARCH" && \
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('ROCm:', torch.version.hip)"
```
Expected output: `HSA=11.0.0 ARCH=gfx1100`, GPU name `AMD Radeon 860M`, ROCm version like `6.4.x`.

---

## 2. One-Time Setup (skip after first run)

### 2.1 Install monitoring tools:
```bash
sudo apt install -y htop rocm-smi-lib radeontop lm-sensors
```

### 2.2 Install torchcrepe for f0 extraction:
```bash
pip install torchcrepe
```

### 2.3 Pre-extract CREPE f0 for the entire dataset (~30-60 min):
```bash
# Train set
python extract_crepe.py --data ~/datasets/musdb18hq/train --device cuda --model large

# Test set (held-out — optional but recommended for final eval)
python extract_crepe.py --data ~/datasets/musdb18hq/test --device cuda --model large
```
**Note:** Resumable — re-run anytime to skip already-extracted files.

---

## 3. Training

### 3.1 Find the latest checkpoint:
```bash
ls -lt ~/mamba3/checkpoints/checkpoint_*.pt 2>/dev/null | head -5
```

### 3.2 Resume from the latest checkpoint:
Replace `XXXXXX` with the step number from 3.1 (e.g., `005000`):
```bash
python train.py \
  --data ~/datasets/musdb18hq/train \
  --resume ~/mamba3/checkpoints/checkpoint_XXXXXX.pt \
  --device cuda \
  --batch 4 \
  --ckpt-every 1000 \
  --val-every 2000 \
  2>&1 | tee -a ~/mamba3/train.log
```

### 3.3 Start fresh training (only if no checkpoint exists):
```bash
python train.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --batch 4 \
  --ckpt-every 1000 \
  --val-every 2000 \
  2>&1 | tee ~/mamba3/train.log
```

### 3.4 Resume with bigger model (24 GB UMA — only if restarting from scratch):
```bash
python train.py \
  --data ~/datasets/musdb18hq/train \
  --device cuda \
  --batch 4 \
  --d-model 384 \
  --d-state 48 \
  --n-layers 8 \
  --ckpt-every 1000 \
  --val-every 2000 \
  2>&1 | tee ~/mamba3/train_v2.log
```
**Warning:** You cannot resume a `d-model=256` checkpoint into a `d-model=384` model — weights won't fit. Larger model = fresh start.

---

## 4. Monitoring (separate terminals)

Each in its own distrobox terminal. None of these disturb the running training.

### 4.1 GPU temperature, power, VRAM, utilization (live):
```bash
distrobox enter rocm-mamba
watch -n 2 'rocm-smi --showtemp --showpower --showuse --showmemuse'
```

### 4.2 CPU + RAM + process tree:
```bash
distrobox enter rocm-mamba
htop
```

### 4.3 Tail the training log (no distrobox needed — log is in shared home):
```bash
tail -f ~/mamba3/train.log
```

### 4.4 All system temps (CPU package, NVMe, etc.):
```bash
distrobox enter rocm-mamba
watch -n 2 sensors
```

### 4.5 Graphical GPU TUI (alternative to rocm-smi):
```bash
distrobox enter rocm-mamba
radeontop
```

---

## 5. Recommended Layout: Single Tmux Session

Lets you walk away and reattach. Survives terminal close.

```bash
# Inside distrobox:
tmux new -s train
```

Inside tmux, set up panes:
- **Pane 1** (default): training
  ```bash
  source ~/venvs/mamba-rocm/bin/activate
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
  export PYTORCH_ROCM_ARCH=gfx1100
  cd ~/mamba3
  python train.py --data ~/datasets/musdb18hq/train --resume checkpoints/checkpoint_005000.pt --device cuda 2>&1 | tee -a train.log
  ```
- **Split horizontally**: `Ctrl+B` then `"` → run `watch -n 2 'rocm-smi --showtemp --showpower --showuse --showmemuse'`
- **Split vertically**: `Ctrl+B` then `%` → run `htop`

**Detach** (training keeps running): `Ctrl+B` then `D`
**Reattach** anytime: `tmux attach -t train`
**Kill** entirely: `tmux kill-session -t train`

---

## 6. Graceful Shutdown

The training loop now has a signal handler. If you `Ctrl+C` once, it saves an `interrupt_NNNNNN.pt` checkpoint **after the current step finishes**, then exits cleanly. Resume with:
```bash
python train.py --data ~/datasets/musdb18hq/train --resume ~/mamba3/checkpoints/interrupt_NNNNNN.pt --device cuda
```

`Ctrl+C` twice = hard exit (no checkpoint).

---

## 7. Expected Wall-Clock (860M iGPU, batch=4, d_model=256)

| Phase | Step range | Duration | What's training |
|---|---|---|---|
| 1 | 0 → 20,000 | ~6-12 hr | Contrastive semantic encoder (NT-Xent) |
| 2 | 20,000 → 50,000 | ~12-20 hr | Single-track adaptive filter (MR-STFT + Mel-L1) |
| 3 | 50,000 → 100,000 | ~24-48 hr | Cross-track-conditioned full model |
| **Total** | | **~2-4 days** | |

**Decision checkpoints:**
- **Step 5,000** — does loss trend down? (Phase 1 should be ~0.6 → ~0.2)
- **Step 20,000** — Phase 1 complete. Validation should show contrastive loss < 1.0.
- **Step 50,000** — Phase 2 complete. SI-SDR should be > 5 dB on validation.
- **Step 100,000** — Phase 3 complete. Final model.

Train stops automatically if validation loss plateaus for 10 val checks (~20,000 steps without improvement).

---

## 8. Thermal Thresholds

| Temp | Status | Action |
|---|---|---|
| <85°C | Healthy | Continue |
| 85-95°C | Warm but safe | Verify good airflow |
| >95°C | Throttling | Pause, improve cooling |
| >100°C | Emergency | Stop immediately |

If sustained >90°C, get a laptop cooler pad or lift the chassis for airflow.

---

## 9. Quick Troubleshooting

### "HIP error: invalid device function"
Environment vars dropped. Re-export from §1.3:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
```

### "ImportError: cannot import name 'ir' from 'triton._C.libtriton'"
Triton wheel went stale. Reinstall:
```bash
pip uninstall -y pytorch-triton-rocm
pip install pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm6.4 --force-reinstall --no-cache-dir
```

### "NaN/Inf loss — skipping batch" (occasional)
Normal. Single bad batch skipped automatically.

### "NaN/Inf loss" every step
Stop training. Run with `--no-amp` to disable bf16:
```bash
python train.py --data ~/datasets/musdb18hq/train --resume checkpoints/checkpoint_XXXXXX.pt --device cuda --no-amp
```

### OOM (out of memory)
Either reduce batch size or model size:
```bash
python train.py --data ~/datasets/musdb18hq/train --device cuda --batch 2 --d-model 192
```

### Step time > 30s
Acceptable for sequential SSM scan on iGPU. If it's > 60s consistently, reduce sequence length (edit `clip_seconds=4.0` → `2.0` in `train.py`'s `StemDataset` constructor).

---

## 10. File Locations

```
~/mamba3/
├── model.py                  # Mamba-3 MIMO architecture
├── train.py                  # Training loop with resume/val/early-stop
├── test_smoke.py             # Pre-training smoke test (9 sections)
├── extract_crepe.py          # Offline CREPE f0 extraction
├── train.log                 # Training output (appended on resume)
└── checkpoints/
    ├── checkpoint_001000.pt  # Regular checkpoints (every --ckpt-every steps)
    ├── checkpoint_002000.pt
    ├── ...
    ├── best.pt               # Best validation loss to date
    ├── final.pt              # Written at training completion
    └── interrupt_NNNNNN.pt   # Written on Ctrl+C

~/datasets/musdb18hq/
├── train/                    # 86 tracks
│   └── <Artist - Title>/
│       ├── bass.wav
│       ├── drums.wav
│       ├── vocals.wav
│       ├── other.wav
│       ├── mixture.wav
│       ├── bass.crepe.npy    # ← generated by extract_crepe.py
│       ├── bass.pitched.npy
│       └── ... (one .crepe.npy + .pitched.npy per stem)
└── test/                     # 50 held-out tracks
```

---

## 11. Sanity Checks Before You Walk Away

After kicking off training, **wait 5 minutes**, then verify:

1. Log is being written: `tail -3 ~/mamba3/train.log`
2. GPU is busy: `rocm-smi --showuse` — GPU% should be high (60-95%)
3. Temperature is OK: `rocm-smi --showtemp` — under 90°C
4. Loss is finite: scan log for `NaN/Inf` — should be rare or absent
5. Steps/sec is reasonable: log line includes `X.XX step/s` — should be > 0.1

If all five check out, detach tmux (`Ctrl+B D`) and walk away. Training will continue.
