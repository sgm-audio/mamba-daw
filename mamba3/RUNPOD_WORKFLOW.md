# Mamba-3 RunPod H100 Workflow

> **Your pod:** H100 SXM 1x ($3.29/hr) • 26 vCPU • 251 GB RAM • 20 GB container disk • 50 GB network volume mounted at `/workspace` • Template: `runpod-torch-v240` (PyTorch + JupyterLab)
>
> **Estimated total cost:** $10-25 for full 100k-step training (3-7 hours wall-clock)
>
> **Key facts about your setup:**
> - `/workspace` is your **network volume** (persistent — survives pod stop/start)
> - Everything outside `/workspace` is the **container disk** (DESTROYED when pod is terminated)
> - **Put ALL your work in `/workspace`** — datasets, code, checkpoints, venv, everything

---

## Phase 0 — Critical Reminders Before You Start

### Cost discipline
- **You're paying $3.29/hr.** Every minute idle = ~5.5 cents. Every hour idle = $3.29.
- **Stop the pod when you walk away** (not just leave SSH). Use the "Stop" button in the RunPod web UI.
- Set a phone alarm if you walk away — even checking back in 30 min late = $1.65 lost.

### Storage layout
```
/workspace                          ← 50 GB network volume (PERSISTENT)
  ├── mamba3/                       ← code + venv + checkpoints
  └── datasets/musdb18hq/           ← dataset + CREPE files

/root/, /tmp/, everywhere else      ← container disk (LOST on pod termination)
```

### Data download strategy
**Do NOT upload from your laptop.** Your home upload would take hours. The pod has datacenter-grade internet — pull MUSDB18-HQ directly from Zenodo. Should take 5-15 minutes.

---

## Phase 1 — SSH In & Verify Environment

### 1.1 Connect to your pod
RunPod web UI shows your SSH command. Looks something like:
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

If you haven't set up SSH keys yet, use the "Connect" → "Web Terminal" button on RunPod's web UI for now. SSH setup can wait.

### 1.2 Verify PyTorch + H100 are alive
```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

**Expected output:**
- `torch: 2.4.x` or `2.5.x` (the runpod-torch-v240 template uses 2.4.0+cu121 or similar)
- `CUDA: True`
- `GPU: NVIDIA H100 80GB HBM3` (or similar)
- `VRAM: ~80 GB`

If `CUDA: False` → the template is broken. Stop the pod, pick a different PyTorch template.

### 1.3 Check disk + memory
```bash
df -h /workspace
free -h
nvidia-smi
```
- `/workspace` should show ~50 GB available
- RAM should show ~251 GB
- `nvidia-smi` should list the H100

---

## Phase 2 — Set Up Project Directory

### 2.1 Create project structure in the persistent volume
```bash
mkdir -p /workspace/mamba3
mkdir -p /workspace/datasets
cd /workspace/mamba3
```

### 2.2 Install system tools (tmux is CRITICAL)
```bash
apt-get update -qq
apt-get install -y -qq tmux htop git wget unzip vim
```

**tmux is non-negotiable.** Without it, if your SSH connection drops, your training dies.

---

## Phase 3 — Install Python Dependencies

The template ships with PyTorch already. We only need to add training-specific packages.

### 3.1 Install training dependencies
```bash
pip install --no-cache-dir soundfile numpy auraloss torchcrepe
```

### 3.2 Verify all imports work
```bash
python -c "import torch, torchaudio, torchcrepe, soundfile, numpy, auraloss; print('all imports OK')"
```

If `torchaudio` is missing (some templates skip it):
```bash
pip install torchaudio
```

---

## Phase 4 — Upload Your Python Files

You have four files to get onto the pod: `model.py`, `train.py`, `test_smoke.py`, `extract_crepe.py`.

### Option A — JupyterLab (easiest, no setup)
1. RunPod web UI → "Connect" → "Connect to HTTP Service [Port 8888]"
2. JupyterLab opens in your browser
3. Navigate to `/workspace/mamba3/` in the file browser (left panel)
4. Drag and drop your four `.py` files into the panel

### Option B — GitHub (cleanest for ongoing work)
If you have your files in a GitHub repo:
```bash
cd /workspace/mamba3
git clone https://github.com/<you>/<repo>.git .
# or for private repos with token auth:
git clone https://<your-token>@github.com/<you>/<private-repo>.git .
```

### Option C — runpodctl from your local machine
On your laptop (one-time install: `pip install runpodctl`):
```bash
# On laptop:
runpodctl send model.py train.py test_smoke.py extract_crepe.py
```
It gives you a one-time code. Then on the pod:
```bash
cd /workspace/mamba3
runpodctl receive <code>
```

### Option D — `scp` (if SSH is configured)
On your laptop:
```bash
scp -P <port> model.py train.py test_smoke.py extract_crepe.py root@<pod-ip>:/workspace/mamba3/
```

### 4.1 Verify files are there
```bash
ls -la /workspace/mamba3/
```
Should show all four `.py` files.

---

## Phase 5 — Download MUSDB18-HQ

### 5.1 Pull from Zenodo (5-15 min on datacenter internet)
```bash
cd /workspace/datasets
wget -O musdb18hq.zip https://zenodo.org/records/3338373/files/musdb18hq.zip
```

Watch the download speed. If it's < 10 MB/s for more than a minute, kill it with Ctrl+C and pick a different RunPod region (some have better Zenodo peering than others).

### 5.2 Extract
```bash
unzip -q musdb18hq.zip
rm musdb18hq.zip    # save 21 GB on the volume
ls musdb18hq/       # should show: train  test
ls musdb18hq/train | head -5    # should show track directories
```

### 5.3 Verify a sample
```bash
ls "/workspace/datasets/musdb18hq/train/$(ls /workspace/datasets/musdb18hq/train | head -1)/"
```
Should show `bass.wav`, `drums.wav`, `vocals.wav`, `other.wav`, `mixture.wav`.

---

## Phase 6 — Smoke Test

**Always run this first.** Confirms the model loads, runs forward+backward on the H100, and matches expected output shapes.

```bash
cd /workspace/mamba3
python test_smoke.py --device cuda
```

H100 will rip through this in 20-40 seconds. Expected: **9/9 sections passed**.

If any section fails, **DO NOT** proceed to training. Stop, debug, fix.

---

## Phase 7 — CREPE Pre-Extraction

H100 makes this fast — ~5-10 min for the train set.

```bash
cd /workspace/mamba3

# Train set
python extract_crepe.py --data /workspace/datasets/train --device cuda --model full

# Test set (optional but recommended for proper validation later)
python extract_crepe.py --data /workspace/datasets/test --device cuda --model full
```

**Resumable** — if your SSH drops mid-extraction, just re-run; already-done stems are skipped.

---

## Phase 8 — Training (the main event)

### 8.1 Start a tmux session FIRST
```bash
tmux new -s train
```
You're now inside tmux. If SSH drops, training keeps running.

### 8.2 Launch training inside tmux
```bash
cd /workspace/mamba3
python train.py \
  --data /home/ubuntu/SGM/train \
  --device cuda \
  --batch 4 \
  --ckpt-every 1000 \
  --val-every 2000 \
  --num-workers  \
  2>&1 | tee train.log
```

**What's different from the iGPU run:**
- `--batch 16` — H100 has 80 GB VRAM, can fit a bigger batch (faster training, better gradient estimates)
- `--num-workers 4` — H100 pod has 26 vCPUs, parallel data loading speeds things up significantly
- `--ckpt-every 1000` — checkpoints every 1000 steps (~5-10 min on H100). If the pod hiccups, max 1000 steps lost.

### 8.3 Detach tmux (training keeps running)
```
Ctrl+B  then  D
```

You're now back at the bare shell. Training is still running. You can:
- Disconnect SSH entirely
- Close your laptop
- Walk away

The pod and the tmux session don't care.

### 8.4 Reattach later
```bash
tmux attach -t train
```

---

## Phase 9 — Monitoring (separate SSH terminals or JupyterLab terminals)

Open new SSH sessions or JupyterLab terminals (don't disturb the training tmux). Each in its own.

### 9.1 GPU usage live
```bash
D
```
Or the modern version:
```bash
nvidia-smi dmon -s pucvmet -i 0
```

H100 should show:
- GPU-Util: 70-95%
- Memory-Used: 10-30 GB (depending on batch)
- Power: 400-700 W
- Temp: 50-70°C (datacenter cooling is excellent)

### 9.2 Training log live
```bash
tail -f /workspace/mamba3/train.log
```

### 9.3 System resources
```bash
htop
```

---

## Phase 10 — Expected Training Timeline

| Phase | Steps | Wall-clock on H100 | Cost |
|---|---|---|---|
| 1 (contrastive) | 0 → 20,000 | 40-80 min | $2.20-4.40 |
| 2 (single-track) | 20,000 → 50,000 | 60-120 min | $3.30-6.60 |
| 3 (cross-track) | 50,000 → 100,000 | 2-4 hours | $6.60-13.20 |
| **Total (100k)** | | **4-7 hours** | **$13-25** |

### Decision checkpoints:
- **Step 5,000** (~15 min): Phase 1 loss trending down (0.6 → 0.2)
- **Step 20,000** (~1 hour): Phase 1 complete, validation contrastive < 1.0
- **Step 50,000** (~3 hours): Phase 2 complete, SI-SDR > 5 dB on validation
- **Step 100,000** (~5 hours): Phase 3 complete, model ready

Early stopping is enabled — training auto-stops if validation plateaus for 10 val checks.

---

## Phase 11 — Download Checkpoints to Your Laptop

**Do this regularly throughout training, not just at the end.** Pods can die. Network volumes survive, but if your account expires or something weird happens, you want a local copy.

### Option A — runpodctl (best for big files)
On the pod:
```bash
cd /workspace/mamba3
runpodctl send checkpoints/best.pt
```
It prints a code. On your laptop:
```bash
runpodctl receive <code>
```

### Option B — scp (if SSH keys are set up)
On your laptop:
```bash
scp -P <port> root@<pod-ip>:/workspace/mamba3/checkpoints/best.pt ./
scp -P <port> root@<pod-ip>:/workspace/mamba3/train.log ./
```

### Option C — JupyterLab (drag and drop)
1. Open JupyterLab (port 8888)
2. Navigate to `/workspace/mamba3/checkpoints/`
3. Right-click `best.pt` → Download

---

## Phase 12 — When You're Done

### 12.1 Download EVERYTHING you care about
```bash
# On laptop:
scp -P <port> -r root@<pod-ip>:/workspace/mamba3/checkpoints/ ./
scp -P <port> root@<pod-ip>:/workspace/mamba3/train.log ./
```
Or use JupyterLab's "Download as Archive" on the `mamba3/` folder.

### 12.2 STOP THE POD
RunPod web UI → click your pod → click "Stop"

**This stops compute billing. ($3.29/hr → $0/hr)**

You will still pay for the network volume (~$0.07/GB/month for the 50 GB = $3.50/month). That's fine — it keeps your data and code intact for next time.

### 12.3 To resume training later
1. Start the pod again from the web UI (the network volume reattaches automatically)
2. SSH back in
3. Everything's where you left it under `/workspace/`
4. Re-attach tmux: `tmux attach -t train` (the session is gone if you terminated, but the data isn't)
5. Resume training:
   ```bash
   cd /workspace/mamba3
   tmux new -s train
   python train.py \
     --data /workspace/datasets/musdb18hq/train \
     --resume checkpoints/checkpoint_NNNNNN.pt \
     --device cuda \
     --batch 16 \
     --ckpt-every 1000 \
     2>&1 | tee -a train.log
   ```

---

## Troubleshooting

### "no module named torch" or similar
Template didn't pre-install PyTorch. Quick fix:
```bash
pip install torch torchvision torchaudio
```

### `nvidia-smi` works but `torch.cuda.is_available()` returns False
Driver/PyTorch CUDA mismatch — wrong template. Stop the pod, pick a different one (e.g., `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).

### Step time is way slower than expected
- Check `nvidia-smi` — if GPU-Util is < 50%, the DataLoader is the bottleneck. Increase `--num-workers 8` or `--num-workers 16`.
- Network volume I/O can be slow under heavy random reads. Move dataset to container disk (`/root/datasets/`) — faster but lost on pod termination.

### Out of disk space
Container disk is only 20 GB. If you accidentally wrote there:
```bash
df -h /
# If full, find the culprit:
du -sh /root/* | sort -h
```
Move everything to `/workspace/`.

### OOM (out of memory)
Unlikely on H100, but if it happens:
```bash
python train.py --batch 8 --d-model 256 ...
```

### Pod stops mid-training
- Restart the pod from web UI
- The 50 GB network volume reattaches automatically
- `tmux` session is gone, but your checkpoints are intact
- Resume: `python train.py --resume /workspace/mamba3/checkpoints/checkpoint_NNNNNN.pt ...`

---

## One-Shot Quick Reference Block

After SSH'ing into a fresh pod, paste this whole thing (assumes you've uploaded code via JupyterLab first):

```bash
# Setup
apt-get update -qq && apt-get install -y -qq tmux htop git wget unzip vim
pip install --no-cache-dir soundfile numpy auraloss torchcrepe torchaudio

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# Project directory (should already exist if volume is reattached)
mkdir -p /workspace/mamba3 /workspace/datasets
cd /workspace/datasets

# Download dataset if not already present
if [ ! -d musdb18hq ]; then
    wget -O musdb18hq.zip https://zenodo.org/records/3338373/files/musdb18hq.zip
    unzip -q musdb18hq.zip
    rm musdb18hq.zip
fi

# Smoke test
cd /workspace/mamba3
python test_smoke.py --device cuda

# CREPE extraction
python extract_crepe.py --data /workspace/datasets/musdb18hq/train --device cuda --model full
python extract_crepe.py --data /workspace/datasets/musdb18hq/test --device cuda --model full

# Start training in tmux
tmux new -s train
# Inside tmux:
python train.py \
  --data /workspace/datasets/musdb18hq/train \
  --device cuda \
  --batch 16 \
  --ckpt-every 1000 \
  --val-every 2000 \
  --num-workers 4 \
  2>&1 | tee train.log
# Detach: Ctrl+B then D
```

---

## Pre-Mortems (read these once)

1. **You forget to stop the pod overnight.** 8 hours unused = $26 wasted. **Set a phone alarm.** Or use RunPod's auto-stop feature if available.

2. **SSH drops during a long operation outside tmux.** That operation dies. **Always use tmux for anything longer than 30 seconds.**

3. **You terminate the pod (not stop) before downloading checkpoints.** Network volume survives termination, but if you also delete the volume to save $3.50/month, everything is gone. **Download checkpoints to your laptop before terminating anything.**

4. **You blow through your RunPod credit faster than expected.** $3.29/hr × 24 hours = $79/day. **Check your RunPod balance daily.**

5. **Step time is 10x slower than expected.** Almost always DataLoader bottleneck. Increase `--num-workers`, or check that the dataset isn't sitting on slow shared storage.

6. **You discover at step 95,000 that the loss has been NaN since step 60,000.** Tail your log periodically. The training will skip NaN batches but you won't be learning. If you see persistent NaNs, kill it, restart with `--no-amp`.

---

## Cost Sanity Check Calculator

Quick mental math:
- 1 hour H100 = $3.29
- 4 hours H100 = $13.16
- Full 100k training (~5 hours) = ~$16.45
- Add ~$1 for setup time
- **Realistic total: $15-20 for first complete run**

If your first run goes well, you can do another full run later (different hyperparameters, etc.) for the same ~$15-20. Compared to a $1500+ GPU, very reasonable.

Network volume monthly: $3.50 (whether or not pod is running)
