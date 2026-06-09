# Mamba-3 Training on RunPod H200 NVL - Complete Setup Guide

**Target:** NVIDIA H200 NVL (143 GB VRAM, 283 GB RAM, 16 vCPU)  
**Cost:** $3.79 CAD/hour (~$2.75 USD)  
**Expected Training Time:** 5-7 hours for 100,000 steps  
**Total Cost:** ~$19-27 CAD

---

## 📋 Table of Contents

1. [Pod Selection & Launch](#1-pod-selection--launch)
2. [SSH Connection](#2-ssh-connection)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Download](#4-dataset-download)
5. [CREPE Extraction](#5-crepe-extraction)
6. [Training Execution](#6-training-execution)
7. [Monitoring & Troubleshooting](#7-monitoring--troubleshooting)
8. [Cost Tracking](#8-cost-tracking)
9. [Shutdown & Download](#9-shutdown--download)

---

## 1. Pod Selection & Launch

### Step-by-Step

1. **Navigate to RunPod Console:** https://runpod.io/console
2. **Click "Deploy" → "Select Template"**
3. **Choose Template:** `PyTorch 2.5.0 + CUDA 12.4` (or latest available)
4. **Select GPU:** `NVIDIA H200 NVL (143 GB VRAM)`
   - **DO NOT** select H100 SXM (80GB) - you need the extra RAM for caching
5. **Configure Storage:**
   - **Container Disk:** 40 GB minimum (required for dependencies + temp files)
   - **Network Volume:** Optional (for persistent storage across pods)
6. **Review & Launch:**
   - Expected cost: **$3.79 CAD/hour**
   - Click **"Deploy"**

### Wait For:
- Pod status changes from `Initializing` → `Running` (2-5 minutes)
- SSH command becomes available in the console

---

## 2. SSH Connection

### Get SSH Command

From the RunPod console, click **"Connect"** on your running pod. Copy the SSH command, which looks like:

```bash
ssh root@104.243.42.157 -p 41234 -i ~/.ssh/your_key.pem
```

### Connect

```bash
# Replace with your actual SSH command
ssh root@<your-pod-ip> -p <port> -i <your-key>
```

### First-Time Setup (if needed)

```bash
# If SSH key issues, RunPod also provides password login
# Use the password shown in the console under "Connect"
```

### Verify Connection

```bash
# You should see:
# root@<container-id>:/workspace#

# Check GPU availability
nvidia-smi
```

**Expected Output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 | Persistence-M | Bus-Id          | Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf          | Pwr:Usage/Cap | Memory-Usage    | GPU-Util |
|=========================================+========================+======================================|
|   0  NVIDIA H200 NVL        Off         | 00000000:01:00.0 Off |  Off                 |
| N/A   35C    P0           |  45W / 700W |  1024MB / 145920MB |   0%     |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Environment Setup

### Clone Repository & Create Virtual Environment

```bash
# Navigate to workspace
cd /workspace

# Clone your Mamba-3 repository
git clone https://github.com/yourusername/mamba3.git
cd mamba3

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install core tools
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install audio and ML dependencies
pip install numpy soundfile auraloss torchcrepe musdb tqdm

# Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA Version: {torch.version.cuda}')
"
```

**Expected Output:**
```
PyTorch: 2.5.0+cu124
CUDA Available: True
GPU: NVIDIA H200
CUDA Version: 12.4
```

### Create Checkpoint Directory

```bash
mkdir -p /workspace/mamba3/checkpoints
```

---

## 4. Dataset Download

### Overview

You need two datasets:
- **MUSDB18-HQ** (required): 11 GB download, ~25 GB uncompressed, 86 training tracks
- **Slakh2100** (optional): 35 GB download, ~80 GB uncompressed, 150+ training tracks

**Total download time on RunPod:** 15-25 minutes (1 Gbps+ connection)

### Download MUSDB18-HQ

```bash
cd /workspace/datasets

echo "=== Downloading MUSDB18-HQ ==="
wget --continue -O musdb18hq.zip "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"

echo "Extracting MUSDB18-HQ..."
unzip -q musdb18hq.zip -d musdb18hq

# Fix nested directory structure if needed
if [ -d "musdb18hq/musdb18hq" ]; then
    mv musdb18hq/musdb18hq/* musdb18hq/
    rmdir musdb18hq/musdb18hq
fi

# Remove zip file to save space
rm musdb18hq.zip

# Verify structure
echo "MUSDB18-HQ Training Tracks: $(ls musdb18hq/train | wc -l)"
echo "MUSDB18-HQ Test Tracks: $(ls musdb18hq/test | wc -l)"
```

### Download Slakh2100 (Optional)

```bash
cd /workspace/datasets

echo "=== Downloading Slakh2100 ==="
wget --continue -O slakh.tar.gz "https://zenodo.org/records/3691788/files/Slakh2100_flac_redux.tar.gz?download=1"

echo "Extracting Slakh2100..."
mkdir -p slakh2100
tar -xzf slakh.tar.gz -C slakh2100 --strip-components=1
rm slakh.tar.gz

# Verify structure
echo "Slakh2100 Training Tracks: $(ls slakh2100/train | wc -l)"
```

### Verify Dataset Structure

```bash
# MUSDB18-HQ should have:
ls /workspace/datasets/musdb18hq/train/

# Each track directory should contain:
# - bass.wav
# - drums.wav
# - vocals.wav
# - other.wav
# - mixture.wav

# Example check:
ls /workspace/datasets/musdb18hq/train/Track001/
```

---

## 5. CREPE Extraction

### Overview

CREPE (pitch estimation) must be pre-computed for all stems. This is a **one-time cost** (~30-45 minutes on H200).

### Extract for MUSDB18-HQ Training Set

```bash
cd /workspace/mamba3
source venv/bin/activate

echo "=== Extracting CREPE for MUSDB18-HQ Train ==="
python extract_crepe.py \
  --data /workspace/datasets/musdb18hq/train \
  --device cuda \
  --model full \
  --force \
  2>&1 | tee logs/crepe_musdb_train.log
```

### Extract for MUSDB18-HQ Test Set (Optional)

```bash
echo "=== Extracting CREPE for MUSDB18-HQ Test ==="
python extract_crepe.py \
  --data /workspace/datasets/musdb18hq/test \
  --device cuda \
  --model full \
  --force \
  2>&1 | tee logs/crepe_musdb_test.log
```

### Extract for Slakh2100 (Optional)

```bash
echo "=== Extracting CREPE for Slakh2100 Train ==="
python extract_crepe.py \
  --data /workspace/datasets/slakh2100/train \
  --device cuda \
  --model full \
  --force \
  2>&1 | tee logs/crepe_slakh_train.log
```

### Expected Output

```
Found 86 tracks under /workspace/datasets/musdb18hq/train
CREPE model: full
Device: cuda
GPU: NVIDIA H200
Warming up CREPE (first call triggers ~30s kernel compile)...
  warmup done in 35.2s

[  1/ 86] Track001
    ✓ bass.wav                  extracted (689 frames)           (0.4s)
    ✓ drums.wav                 unpitched (zeros)                (0.1s)
    ✓ vocals.wav                extracted (689 frames)           (0.4s)
    ✓ other.wav                 extracted (689 frames)           (0.3s)
...
────────────────────────────────────────────────────────────
Done in 28.3 min
  Extracted: 258
  Cached:    0
  Failed:    0
  Total:     344
```

### Verify CREPE Files

```bash
# Check that .crepe.npy and .pitched.npy files exist
ls /workspace/datasets/musdb18hq/train/Track001/*.crepe.npy
ls /workspace/datasets/musdb18hq/train/Track001/*.pitched.npy

# Should see:
# Track001/bass.crepe.npy
# Track001/bass.pitched.npy
# Track001/vocals.crepe.npy
# etc.
```

---

## 6. Training Execution

### Install tmux (Required for Detachable Sessions)

```bash
apt-get update
apt-get install -y tmux
```

### Create tmux Session

```bash
# Create a new tmux session named 'mamba3-train'
tmux new -s mamba3-train
```

**You are now inside the tmux session.** All subsequent commands run inside tmux.

### Activate Environment & Set Paths

```bash
# Activate virtual environment
source /workspace/mamba3/venv/bin/activate

# Navigate to project
cd /workspace/mamba3

# Create logs directory
mkdir -p logs
```

### Launch Training (OPTIMAL H200 NVL CONFIGURATION)

```bash
# Copy your latest checkpoint from your laptop if needed
# (Use scp from your laptop: scp checkpoints/*.pt root@<pod-ip>:/workspace/mamba3/checkpoints/)

# Start training with RAM caching
python train.py \
  --data /workspace/datasets/musdb18hq/train \
  --save ./checkpoints \
  --batch 128 \
  --grad-accum 2 \
  --num-workers 8 \
  --cache-in-ram \
  --device cuda \
  --steps 100000 \
  --ckpt-every 1000 \
  --val-every 2000 \
  --log-every 50 \
  --resume ./checkpoints/checkpoint_006000.pt \
  2>&1 | tee logs/train_h200_full.log
```

### Configuration Explanation

| Flag | Value | Reason |
|------|-------|--------|
| `--batch` | 128 | Fits in 143GB VRAM with room for larger effective batch |
| `--grad-accum` | 2 | Effective batch = 256 (stable for NT-Xent contrastive loss) |
| `--num-workers` | 8 | You have 16 vCPUs; use half for DataLoader, leave half for OS |
| `--cache-in-ram` | (flag) | Pre-load all stems in 283GB RAM (eliminates I/O bottleneck) |
| `--device` | cuda | Use CUDA backend (H200 is real CUDA, not ROCm) |
| `--resume` | checkpoint_006000.pt | Continue from your latest checkpoint |

### Verify Training Started Correctly

**Wait for and verify these log messages:**

```
Pre-loading 86 tracks into RAM cache...
  ✓ Cached 344 stems (23.45 GB)
Model parameters: 3,903,155
Tracks: 90 train, 10 val
Starting training: 100000 steps, batch=128, device=cuda
Gradient accumulation: 2 (effective batch = 256)
Phase 1 (contrastive):  step 0       → 20000
Phase 2 (single-track): step 20000 → 50000
Phase 3 (cross-track):  step 50000 → 100000
Resuming at step 6000

────────────────────────────────────────────────────────────
  ENTERING PHASE 1 at step 6000
────────────────────────────────────────────────────────────

[step   6000] phase=1 loss=0.2468 lr=2.99e-04 0.00 step/s
  Running validation at step 6000...
  val_loss = 0.4363  (best=0.4618)
  ✓ new best val loss — saved best.pt
[step   6050] phase=1 loss=0.1052 lr=2.99e-04 4.23 step/s
```

**CRITICAL:** If `step/s` is **< 2.0** after the first 100 steps, something is wrong. See [Troubleshooting](#7-monitoring--troubleshooting).

### Detach from tmux

Once you see stable step/s > 3.0:

1. Press `Ctrl+B` (release both keys)
2. Then press `D`

You'll see: `[detached]`

You can now safely close your laptop. Training continues in the background.

---

## 7. Monitoring & Troubleshooting

### Re-attach to tmux Session

```bash
# From any SSH session connected to the pod
tmux attach -t mamba3-train
```

### Monitor Without Attaching

```bash
# Tail the training log
tail -f /workspace/mamba3/logs/train_h200_full.log

# Show last 20 step entries
grep "\[step" /workspace/mamba3/logs/train_h200_full.log | tail -20

# Count total steps completed
grep -c "\[step" /workspace/mamba3/logs/train_h200_full.log
```

### GPU Monitoring Dashboard

**Option 1: Simple Watch**
```bash
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv'
```

**Option 2: Detailed dmon (in a separate SSH session)**
```bash
nvidia-smi dmon -s pucvmet -i 0
```

**Output Interpretation:**
```
# gpu   pwr  temp    sm   mem   enc   dec  mclk  pclk
# Idx     W     C     %     %     %     %   MHz   MHz
    0    250    65    85    72     0     0  9600  1980
```
- **pwr:** Power draw (W) - H200 TDP is 700W, expect 200-400W during training
- **temp:** Temperature (°C) - should stay < 80°C
- **sm:** GPU compute utilization - **target: 80-90%**
- **mem:** VRAM utilization - expect 60-80% @ batch=128

### Performance Diagnostics

```bash
# Check average steps/sec over last 100 steps
grep "step/s" /workspace/mamba3/logs/train_h200_full.log | tail -100 | awk '{print $NF}' | awk '{sum+=$1; count++} END {printf "Average: %.2f steps/sec\n", sum/count}'

# Check if validation loss is improving
grep "val_loss" /workspace/mamba3/logs/train_h200_full.log | tail -10

# List checkpoints created (with sizes and timestamps)
ls -lh --time-style=+%Y-%m-%d\ %H:%M /workspace/mamba3/checkpoints/checkpoint_*.pt | tail -10

# Check RAM usage (should be 45-65GB with cache)
free -h

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Troubleshooting Guide

#### Problem 1: Steps/sec < 2.0

**Symptoms:** Training is slow, GPU utilization < 50%

**Diagnosis:**
```bash
# Check if RAM cache actually loaded
grep "Pre-loading" /workspace/mamba3/logs/train_h200_full.log
grep "Cached" /workspace/mamba3/logs/train_h200_full.log

# If cache didn't load, the --cache-in-ram flag wasn't passed correctly

# Check disk I/O wait
iostat -x 2 | head -20
# If %util is > 80% on your main disk, I/O is the bottleneck
```

**Solutions:**
1. **Restart with cache:** Kill training (`Ctrl+C` in tmux), re-run with `--cache-in-ram`
2. **Copy dataset to RAM disk:**
   ```bash
   cp -r /workspace/datasets/musdb18hq /tmp/datasets
   # Then re-run training with --data /tmp/datasets/musdb18hq/train
   ```
3. **Increase workers:** Change `--num-workers 8` to `--num-workers 12`

#### Problem 2: GPU Utilization < 50%

**Symptoms:** `nvidia-smi` shows SM util at 30-50%

**Diagnosis:**
```bash
# Check if batch size is too small
# H200 needs large batches to saturate compute

# Check DataLoader starvation
top -b -n 1 | grep -E "python|DataLoader" | head -10
```

**Solutions:**
1. **Increase batch size:** `--batch 256 --grad-accum 1` (effective batch = 256)
2. **Increase workers:** `--num-workers 12` (you have 16 vCPUs available)
3. **Disable gradient checkpointing** (uses more VRAM but faster): `--no-grad-ckpt`

#### Problem 3: CUDA Out of Memory

**Symptoms:** Training crashes with `RuntimeError: CUDA out of memory`

**Diagnosis:**
```bash
# Check current VRAM usage
nvidia-smi --query-gpu=memory.used --format=csv
```

**Solutions:**
1. **Reduce batch size:** `--batch 64 --grad-accum 4` (effective batch still = 256)
2. **Enable gradient checkpointing** (already on by default, but verify you're not using `--no-grad-ckpt`)
3. **Reduce workers:** `--num-workers 4` (each worker uses some VRAM)

#### Problem 4: DataLoader Worker Killed

**Symptoms:** `RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Terminated`

**Diagnosis:**
```bash
# Check system RAM usage
free -h
# If "available" is < 10GB, you're running out of RAM
```

**Solutions:**
1. **Reduce workers:** `--num-workers 4`
2. **Disable RAM cache** (last resort, will be much slower): Remove `--cache-in-ram` flag
3. **Close other processes:** `htop` → kill unnecessary processes

#### Problem 5: Validation Loss Not Improving

**Symptoms:** `val_loss` stays flat or increases over multiple validation checks

**Diagnosis:**
```bash
# Check validation history
grep "val_loss" /workspace/mamba3/logs/train_h200_full.log

# Expected pattern (rough):
# step 6000:  val_loss = 0.4363
# step 8000:  val_loss = 0.3892
# step 10000: val_loss = 0.3654
```

**Solutions:**
1. **This is normal in Phase 1** (contrastive pretraining) - loss may oscillate
2. **Wait for Phase 2** (single-track reconstruction) - loss should stabilize
3. **If still not improving by step 20000:**
   - Reduce learning rate: `--lr 1e-4`
   - Increase batch size: `--batch 256`
   - Check data quality: Ensure CREPE files are correctly extracted

---

## 8. Cost Tracking

### Real-Time Cost Monitor

Run this in a **separate SSH session** (not inside tmux):

```bash
# Record start time
START_TIME=$(date +%s)
echo "Training started at: $(date)"

# Monitor loop (prints every 5 minutes)
while true; do
  NOW=$(date +%s)
  ELAPSED_SEC=$((NOW - START_TIME))
  HOURS=$(echo "scale=3; $ELAPSED_SEC / 3600" | bc)
  COST_CAD=$(echo "scale=2; $HOURS * 3.79" | bc)
  
  # Get step count from log
  STEPS=$(grep -c "\[step" /workspace/mamba3/logs/train_h200_full.log 2>/dev/null || echo "0")
  REMAINING=$((100000 - STEPS))
  
  # Estimate completion time
  if [ $STEPS -gt 100 ]; then
    FIRST_100_TIME=$(grep "\[step.*100\]" /workspace/mamba3/logs/train_h200_full.log | head -1 | awk '{print $NF}')
    if [ -n "$FIRST_100_TIME" ]; then
      STEPS_PER_SEC=$(echo "scale=2; 100 / $FIRST_100_TIME" | bc 2>/dev/null || echo "3.0")
      HOURS_REMAINING=$(echo "scale=1; $REMAINING / ($STEPS_PER_SEC * 3600)" | bc 2>/dev/null || echo "?")
    else
      HOURS_REMAINING="?"
    fi
  else
    HOURS_REMAINING="?"
  fi
  
  echo "⏱  $(date +%H:%M:%S) | 💰 \$${COST_CAD} CAD | 📊 ${STEPS}/100000 steps | 🎯 ${REMAINING} remaining | ⏳ ~${HOURS_REMAINING}h to finish"
  
  sleep 300  # Update every 5 minutes
done
```

### Quick Cost Check Commands

```bash
# One-liner to check current cost and progress
START=$(stat -c %Y /workspace/mamba3/logs/train_h200_full.log); NOW=$(date +%s); HRS=$(echo "scale=2; ($NOW-$START)/3600" | bc); COST=$(echo "scale=2; $HRS*3.79" | bc); STEPS=$(grep -c "\[step" /workspace/mamba3/logs/train_h200_full.log); echo "💰 \$${COST} CAD | 📊 ${STEPS} steps | ⏱ ${HRS}h elapsed"
```

### Cost Projection Table

| Duration | Steps (at 4/sec) | Cost (CAD) |
|----------|------------------|------------|
| 1 hour   | 14,400           | $3.79      |
| 3 hours  | 43,200           | $11.37     |
| 5 hours  | 72,000           | $18.95     |
| 7 hours  | 100,800          | $26.53     |
| 10 hours | 144,000          | $37.90     |

**Target:** Complete 100,000 steps in 5-7 hours = **$19-27 CAD**

---

## 9. Shutdown & Download

### Verify Training Completion

```bash
# Re-attach to tmux session
tmux attach -t mamba3-train

# Check for completion message
grep "Training complete" /workspace/mamba3/logs/train_h200_full.log

# Expected output:
# Training complete. Best val loss: 0.2847
```

### List Final Checkpoints

```bash
ls -lh /workspace/mamba3/checkpoints/

# You should see:
# - checkpoint_001000.pt through checkpoint_0100000.pt (or final step)
# - best.pt (checkpoint with lowest validation loss)
# - final.pt (last checkpoint before completion)
```

### Download Checkpoints to Your Laptop

**From your local machine (not the pod):**

```bash
# Create local checkpoints directory
mkdir -p ~/mamba3-checkpoints

# Download all checkpoints
scp -P <port> -r root@<pod-ip>:/workspace/mamba3/checkpoints/*.pt ~/mamba3-checkpoints/

# Or download only the best checkpoint
scp -P <port> root@<pod-ip>:/workspace/mamba3/checkpoints/best.pt ~/mamba3-checkpoints/

# Or download specific checkpoints
scp -P <port> root@<pod-ip>:/workspace/mamba3/checkpoints/checkpoint_0{50,60,70,80,90,100}000.pt ~/mamba3-checkpoints/
```

### Download Training Logs

```bash
# From your local machine
scp -P <port> -r root@<pod-ip>:/workspace/mamba3/logs/ ~/mamba3-logs/
```

### Terminate Pod

**IMPORTANT:** RunPod charges by the minute. Terminate immediately after downloading.

1. Go to https://runpod.io/console
2. Find your running pod
3. Click **"Stop"** (graceful shutdown) or **"Terminate"** (immediate deletion)
4. Confirm termination

**Verify pod is terminated:**
- Pod disappears from "Running" list
- No more charges appear on your account

---

## 📊 Expected Results Summary

### Performance Metrics (H200 NVL)

| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| **Steps/sec** | 4-6 | > 3.0 |
| **GPU SM Utilization** | 80-90% | > 70% |
| **VRAM Usage** | 55-70 GB | < 120 GB |
| **System RAM Usage** | 45-65 GB | < 200 GB |
| **Time to 100k steps** | 5-7 hours | < 10 hours |
| **Total Cost** | $19-27 CAD | < $40 CAD |

### Training Quality Metrics

| Phase | Steps | Expected Loss Range |
|-------|-------|---------------------|
| **Phase 1: Contrastive** | 0-20,000 | NT-Xent: 0.001-0.5 (oscillates) |
| **Phase 2: Single-Track** | 20,000-50,000 | Total: 0.5-1.5 (steady decline) |
| **Phase 3: Cross-Track** | 50,000-100,000 | Total: 0.2-0.8 (gradual improvement) |

**Final Validation Loss Target:** 0.25-0.35 (at step 100,000)

---

## 🆘 Emergency Commands

### Kill Training Immediately

```bash
# If something goes wrong and you need to stop NOW
tmux kill-session -t mamba3-train

# Or from inside tmux: Ctrl+C (may take 30s to checkpoint)
```

### Save Emergency Checkpoint

```bash
# If pod is about to be preempted, send SIGTERM to trigger checkpoint
# Find training PID
pgrep -f "python train.py"

# Send gentle terminate (triggers signal handler in train.py)
kill -15 <PID>

# Wait 1-2 minutes for checkpoint to save
# Checkpoint will be saved as: checkpoints/interrupt_XXXXXX.pt
```

### Resume from Interrupted Checkpoint

```bash
# If training was interrupted, resume from the last checkpoint
python train.py \
  --data /workspace/datasets/musdb18hq/train \
  --batch 128 \
  --grad-accum 2 \
  --num-workers 8 \
  --cache-in-ram \
  --device cuda \
  --steps 100000 \
  --resume /workspace/mamba3/checkpoints/interrupt_054321.pt \
  2>&1 | tee logs/train_resumed.log
```

---

## 📝 Pre-Departure Checklist

Before closing your laptop:

- [ ] tmux session is running (`tmux list-sessions` shows `mamba3-train`)
- [ ] Training log is growing (`tail -f logs/train_h200_full.log` shows new lines every few seconds)
- [ ] GPU utilization > 70% (`nvidia-smi` shows SM usage in 70-90% range)
- [ ] Steps/sec > 3.0 (check last 10 log entries)
- [ ] No OOM errors in log
- [ ] Checkpoints are being saved (`ls checkpoints/*.pt` shows recent files)
- [ ] Cost monitor is running (optional, in separate SSH session)
- [ ] You know how to re-attach: `tmux attach -t mamba3-train`

**If all boxes are checked, you can safely close your laptop for 5-7 hours.**

---

## Post-Training Next Steps

1. **Evaluate Checkpoints:**
   ```bash
   # Run inference on test set with best checkpoint
   python evaluate.py --checkpoint checkpoints/best.pt --data /workspace/datasets/musdb18hq/test
   ```

2. **Fine-Tune (Optional):**
   - Continue training to 200,000 steps if validation loss still improving
   - Adjust learning rate: `--lr 1e-4` for fine-tuning phase

3. **Export Model:**
   ```bash
   python export_model.py --checkpoint checkpoints/best.pt --output model.onnx
   ```

4. **Document Results:**
   - Save final training log
   - Record best validation loss
   - Note any observations or issues

---

**Good luck! This setup will get your Mamba-3 model trained efficiently on the H200 NVL.** 🚀

**Total estimated cost: $19-27 CAD for 100,000 steps.**