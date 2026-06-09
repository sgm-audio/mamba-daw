# RunPod Training Checklist — Mamba3 Per-Track Processor

> Date: 2026-05-31
> Goal: train Mamba3 model end-to-end TODAY
> Pod: H100/H200/A100 — anything with CUDA
> Expected step rate after fix: 5-30 step/s (vs <1 step/s before)
> Expected total wall-clock: 1-5 hours for 100k steps

---

## 0. Before you start the pod

You should already have local backups of:
- [ ] Dataset (MUSDB18-HQ stems on local disk)
- [ ] CREPE extractions (the `.crepe.npy` and `.pitched.npy` files)
- [ ] Last checkpoint (if you want to resume) — `best.pt` or `checkpoint_006000.pt`

If any of those are only on the dead Lambda pod or the still-running RunPod
right now, **download them first before the next pod spin-up.**

---

## 1. Pod selection

Pick **A100 80GB SXM4** or **H100 PCIe**. NOT H200 — you've confirmed it doesn't
buy you anything on this model, and it costs more.

- A100 80GB SXM4: ~$1.69-1.99/hr
- H100 PCIe: ~$2.69/hr

A100 is the right call. Pick a pod with:
- At least 200 GB persistent volume (dataset + checkpoints + dependencies)
- At least 16 vCPU (you need them for num_workers=8 dataloader)
- 64 GB+ RAM (so `--cache-in-ram` works for MUSDB18-HQ)

Pick the "RunPod PyTorch 2.4 CUDA 12.4" template or similar.

---

## 2. SSH in and bootstrap the environment

Once the pod is up, ssh in and run this whole block at once. Adapt paths to
where your dataset lives on the pod.

```bash
# ---- Install the basic tooling you actually use ----
apt-get update -qq
apt-get install -y -qq tmux vim git aria2 htop nvtop ffmpeg

# ---- Python deps for training ----
# Use the cu124 PyTorch wheels matched to the pod's CUDA version.
# If pod ships with PyTorch 2.4 already (RunPod default), skip torch reinstall.
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
# If above prints False for cuda, fix the pod template — do not continue.

# Dependencies your code actually imports
pip install --no-cache-dir \
    soundfile==0.12.1 \
    torchcrepe==0.0.23 \
    auraloss==0.4.0 \
    "numpy<2" \
    tqdm

# Verify torchaudio is present and matches torch CUDA version
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"

# If torchaudio is missing or version-mismatched, install matching wheel:
# pip install --index-url https://download.pytorch.org/whl/cu124 torchaudio
```

**Watch for:** if you see `NumPy 2.x incompatibility`, run `pip install "numpy<2"`
and restart Python. This is the same trap as the Lambda session.

---

## 3. tmux session

Always train inside tmux. If your SSH drops, training keeps running.

```bash
tmux new -s train
# If reconnecting later:  tmux attach -t train
# Detach without killing: Ctrl-b then d
```

---

## 4. Get code on the pod

You have three files to upload from your laptop:
- `model.py` (updated — parallel scan added)
- `train.py` (updated — perf-aware defaults)
- `test_parity.py` (new — verifies parallel scan is correct)

From your **laptop**, use scp:

```bash
# Adjust IP and key path to your actual pod
POD_IP=<paste-pod-ip>
POD_USER=root  # or whatever RunPod uses
SSH_KEY=~/.ssh/id_ed25519

scp -i $SSH_KEY model.py test_parity.py train.py ${POD_USER}@${POD_IP}:/workspace/mamba3/
```

Or if you keep the code in a git repo, clone it on the pod directly:

```bash
cd /workspace
git clone <your-repo-url> mamba3
cd mamba3
```

---

## 5. Get the dataset on the pod (aria2)

MUSDB18-HQ is ~30 GB. With aria2 doing multiple connections, expect 80-150 MB/s.

If you have a URL to your tarballed dataset:
```bash
cd /workspace
mkdir -p datasets && cd datasets
aria2c -x 16 -s 16 -j 1 --file-allocation=none <url-to-musdb18hq.tar>
tar -xf musdb18hq.tar
```

If you need MUSDB18-HQ from the official source, it lives behind a Zenodo URL
that may rate-limit. Better to host your own copy on a fast CDN and pull it.

**CREPE extractions:** if you precomputed `.crepe.npy` and `.pitched.npy` files
locally, tar them and aria2 them too. Do NOT re-extract on the pod — it costs
$15 of GPU time you don't need to spend.

```bash
# From laptop, before pod:
cd /path/to/musdb18hq/train
tar -czf crepe_files.tar.gz $(find . -name '*.crepe.npy' -o -name '*.pitched.npy')
# Upload to a host with aria2-able URL, OR scp directly:
scp -i $SSH_KEY crepe_files.tar.gz ${POD_USER}@${POD_IP}:/workspace/datasets/musdb18hq/train/
# On pod:
cd /workspace/datasets/musdb18hq/train
tar -xzf crepe_files.tar.gz
```

Verify CREPE files are there:
```bash
ls /workspace/datasets/musdb18hq/train/ | head -5
ls /workspace/datasets/musdb18hq/train/<one-track-name>/ | grep crepe
# Should see: bass.crepe.npy, bass.pitched.npy, drums.crepe.npy, etc.
```

---

## 6. Run parity test BEFORE training

This is the gate. **If parity test fails, do not start training.**

```bash
cd /workspace/mamba3
python test_parity.py
```

Expected output:
```
Device: cuda
Layer built: H=16, N=32, d_model=256
Output shapes:
  sequential: (2, 345, 16, 16)
  parallel:   (2, 345, 16, 16)
Difference stats:
  max abs:  X.XXe-04   <- should be small
  mean abs: X.XXe-05
  rel:      X.XXe-04   <- < 1e-3

PASS -- parallel scan matches sequential within tolerance
Speed comparison (10 iterations each):
  sequential: XXX ms per call
  parallel:   XX ms per call
  speedup:    XXx              <- should be 20-50x on A100/H100

Ready to train.
```

If parity FAILS: stop. Send me the actual output. There's a bug in the parallel scan that needs fixing before any training time gets spent.

If parity PASSES but speedup is < 5x on GPU: something's still wrong. Tell me and we look at it.

---

## 7. Smoke test training (100 steps, ~2 min)

Confirm end-to-end pipeline works before committing to a long run.

```bash
cd /workspace/mamba3
python train.py \
    --data /workspace/datasets/musdb18hq/train \
    --save /workspace/checkpoints \
    --steps 100 \
    --batch 16 \
    --num-workers 8 \
    --val-every 99999 \
    --log-every 10 \
    --ckpt-every 99999
```

Expected output:
- Model parameters: 3,903,155 (or similar)
- Tracks: N train, M val
- Loss starts ~2.5, drops within 50 steps
- **Step rate: should report 5-15+ step/s** (the whole point of the fix)

If step rate is still <1: stop, send me the log output, parallel scan didn't apply.

---

## 8. Resume from your existing checkpoint OR start fresh

If you have `best.pt` from the previous Lambda run at step 6000:

```bash
# Upload best.pt to the pod first (scp from laptop)
scp -i $SSH_KEY best.pt ${POD_USER}@${POD_IP}:/workspace/checkpoints/best.pt
```

If you do NOT have it, just start fresh — at 5-15 step/s, hitting step 6000
again takes 7-20 minutes, not 7 hours. **Starting fresh is fine.**

---

## 9. Real training command

```bash
cd /workspace/mamba3
python train.py \
    --data /workspace/datasets/musdb18hq/train \
    --save /workspace/checkpoints \
    --steps 100000 \
    --batch 16 \
    --num-workers 8 \
    --cache-in-ram \
    --val-every 2000 \
    --log-every 50 \
    --ckpt-every 1000 \
    --early-stop-patience 10 \
    2>&1 | tee training.log
```

Notes:
- `--cache-in-ram` only useful if pod has 64GB+ RAM. Check with `free -h` first.
- `tee training.log` saves output in case the session dies
- If you want to resume from a checkpoint: add `--resume /workspace/checkpoints/best.pt`

---

## 10. Monitor

In a second tmux window (Ctrl-b then c):

```bash
# GPU utilization
nvtop
# or
watch -n 1 nvidia-smi

# Disk
df -h /workspace

# RAM
htop
```

You want to see:
- GPU util consistently 70%+ (with parallel scan)
- Memory ~5-20 GB per process
- Step rate steady, not drifting down over time

---

## 11. Pull checkpoints to laptop periodically

Every hour or so, from laptop:

```bash
scp -i $SSH_KEY ${POD_USER}@${POD_IP}:/workspace/checkpoints/best.pt ~/mamba3_checkpoints/best_$(date +%Y%m%d_%H%M).pt
```

This protects against pod loss. RunPod has had reliability issues — don't
trust the pod to still exist tomorrow.

---

## 12. When training finishes

```bash
# On pod
ls -la /workspace/checkpoints/
# best.pt is the one you want
# final.pt is the last step (may not be the best)

# From laptop
scp -i $SSH_KEY ${POD_USER}@${POD_IP}:/workspace/checkpoints/best.pt ./best_final.pt
scp -i $SSH_KEY ${POD_USER}@${POD_IP}:/workspace/mamba3/training.log ./training_final.log
```

Then **terminate the pod immediately**. Storage costs accumulate even when
the pod is "stopped." Confirm $0/hr billing in the RunPod dashboard before
you walk away.

---

## 13. Cost expectations

At 10 step/s sustained on A100 at $1.99/hr:
- 100k steps = 10,000 seconds = 2.78 hours
- **Cost: ~$5.50**

At 5 step/s sustained on A100 at $1.99/hr:
- 100k steps = 20,000 seconds = 5.56 hours
- **Cost: ~$11**

At 15 step/s on H100 at $2.69/hr:
- 100k steps = 6,667 seconds = 1.85 hours
- **Cost: ~$5**

Either way, **the whole training run should cost under $15.** If it's costing
more than that, something's wrong — kill and diagnose.

---

## 14. What can go wrong (quick recovery table)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mamba_ssm` import error | Not needed anymore | Ignore, parallel scan is pure PyTorch |
| Step rate <1 step/s | Parity test wasn't run or didn't pass | Stop, run parity test, send output |
| NaN loss at step 0 | Initialization issue | Drop batch size, restart |
| NaN loss mid-training | Numerical instability in complex log | Increase epsilon in `_ssm_scan_parallel` from 1e-12 to 1e-8 |
| OOM | Batch too large | Drop --batch from 16 to 8 |
| `numpy.dtype size changed` | NumPy 2.x mismatch | `pip install "numpy<2"`, restart Python |
| Workers all CPU-pegged but GPU idle | Disk IO bottleneck | Add `--cache-in-ram` (needs RAM) |
| GPU at 100% but step rate slow | Validation taking forever | Increase `--val-every` |
| Pod disconnects | RunPod flakiness | tmux survives, just reconnect and `tmux attach -t train` |

---

## 15. Decision gate after smoke test

After step 7 (smoke test), you should know:
- [ ] Parity test passes
- [ ] Smoke training runs at 5+ step/s
- [ ] Loss decreases from 2.5 toward 1.0 within 100 steps
- [ ] No OOM, no NaN

**If all four:** start full training run.

**If any fail:** stop. Send the output. Don't burn money on a broken pipeline.

---

## 16. Quick reference of files in this kit

| File | What it does | Where it goes |
|------|--------------|---------------|
| `model.py` | Updated with parallel scan | `/workspace/mamba3/` |
| `train.py` | Updated defaults | `/workspace/mamba3/` |
| `test_parity.py` | Verifies parallel == sequential | `/workspace/mamba3/` |
| `CHECKLIST.md` | This file | Keep on laptop for reference |
