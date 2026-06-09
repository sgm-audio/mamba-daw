#!/usr/bin/env python3
"""Mamba-3 laptop training launcher — replaces run_train_laptop.ps1.

Usage:
  python launch_train.py [--fresh-best-val] [--batch N] [--grad-accum N] [--steps N] [--no-amp]
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# -- resolution order --------------------------------------------------
DATA_CANDIDATES = [
    os.path.join(os.environ.get("USERPROFILE", ""), "datasets", "musdb18hq", "train"),
    str(SCRIPT_DIR / "datasets" / "musdb18hq" / "train"),
]

CKPT_DIR = SCRIPT_DIR / "checkpoints_laptop"
SAVE_DIR = str(CKPT_DIR)

# -- auto-detect dataset -----------------------------------------------
dataset = None
for cand in DATA_CANDIDATES:
    if os.path.isdir(cand):
        dataset = cand
        break

if dataset is None:
    print("ERROR: No dataset found. Tried:")
    for c in DATA_CANDIDATES:
        print(f"  {c}")
    sys.exit(1)

# -- auto-detect latest checkpoint -------------------------------------
latest_ckpt = None
if CKPT_DIR.is_dir():
    ckpts = sorted(CKPT_DIR.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        latest_ckpt = str(ckpts[0])
    else:
        ckpts = sorted(CKPT_DIR.glob("interrupt_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            latest_ckpt = str(ckpts[0])

# -- ROCm env ----------------------------------------------------------
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# -- parse optional overrides ------------------------------------------
fresh_best_val = "--fresh-best-val" in sys.argv
no_amp = "--no-amp" in sys.argv
batch = 2
grad_accum = 8
steps = 100_000

for i, a in enumerate(sys.argv):
    if a == "--batch" and i + 1 < len(sys.argv):
        batch = int(sys.argv[i + 1])
    if a == "--grad-accum" and i + 1 < len(sys.argv):
        grad_accum = int(sys.argv[i + 1])
    if a == "--steps" and i + 1 < len(sys.argv):
        steps = int(sys.argv[i + 1])

# -- build args --------------------------------------------------------
cmd = [
    sys.executable,
    str(SCRIPT_DIR / "train_laptop.py"),
    "--data", dataset,
    "--save", SAVE_DIR,
    "--device", "cuda",
    "--batch", str(batch),
    "--steps", str(steps),
    "--grad-accum", str(grad_accum),
    "--ckpt-every", "500",
    "--val-every", "1000",
    "--num-workers", "2",
]

if no_amp:
    cmd.append("--no-amp")
if latest_ckpt:
    cmd.extend(["--resume", latest_ckpt])
    if fresh_best_val:
        cmd.append("--fresh-best-val")
        print(f"--fresh-best-val: resetting val_loss baseline")
    print(f"Resuming from: {latest_ckpt}")

print(f"Dataset: {dataset}")
print(f"Command: {' '.join(cmd)}")
print("=" * 70)

sys.exit(subprocess.run(cmd).returncode)
