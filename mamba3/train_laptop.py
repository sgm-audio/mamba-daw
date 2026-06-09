"""
Mamba-3 Per-Track Adaptive Filter Bank — LAPTOP TRAINING SCRIPT
================================================================
For CPU or 8GB shared-iGPU training. Optimized for low memory footprint
and minimal per-step compute.

Differences from train.py (cloud version):
  - Single-resolution STFT loss (1 FFT size, not 4)
  - Single mel-scale (not 3)
  - Shorter clips (2 sec, not 4)
  - Batch 2 default
  - num_workers 2 default
  - Smaller model defaults (from model_laptop.TrackProcessorConfig)
  - Auto-checkpoint every 200 steps so power loss doesn't kill progress

Usage:
    python train_laptop.py --data /path/to/musdb18hq/train

Resume:
    python train_laptop.py --data ... --resume ./checkpoints_laptop/best.pt

Recommended invocation for actual all-day training:
    nohup python train_laptop.py \
        --data ~/datasets/musdb18hq/train \
        --save ~/checkpoints_laptop \
        --steps 100000 \
        > train.log 2>&1 &
    tail -f train.log
"""

from __future__ import annotations

import random
import math
import sys
import time
import warnings
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import torchaudio

warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values"
)

if sys.platform != 'win32':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

from model_laptop import PerTrackProcessor, TrackProcessorConfig


# ---------------------------------------------------------------------------
# LEAN LOSS — single-resolution STFT, single mel
# ---------------------------------------------------------------------------

class LeanTrackLoss(nn.Module):
    """
    Single-resolution loss for laptop training.
    Drops multi-res STFT and 3-mel stack — those cost ~5x more compute
    for ~10% quality gain. Reintroduce on a cloud retrain when shipping.
    """

    def __init__(self, cfg: TrackProcessorConfig, sample_rate: int = 44100):
        super().__init__()
        self.cfg = cfg
        self.sr  = sample_rate

        # Single STFT for loss (was 4 resolutions)
        self.loss_n_fft = 1024
        self.loss_hop   = 256

        # Single mel (was 3 mel scales)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft       = 1024,
            hop_length  = 256,
            n_mels      = 64,
        )

        self.global_step = 0

    def _si_sdr(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-8
        target_energy = (target ** 2).sum(dim=-1, keepdim=True)
        dot = (pred * target).sum(dim=-1, keepdim=True)
        proj = dot / (target_energy + eps) * target
        noise = pred - proj
        si_sdr = 10 * torch.log10(
            (proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps
        )
        return -si_sdr.mean()

    def _mel_l1(self, pred_wave: Tensor, target_wave: Tensor) -> Tensor:
        mel = self.mel_transform.to(pred_wave.device)
        p = torch.log1p(mel(pred_wave))
        t = torch.log1p(mel(target_wave))
        return F.l1_loss(p, t)

    def _si_sdr_weight(self) -> float:
        if self.global_step < 5000:
            return 0.0
        elif self.global_step > 15000:
            return 0.1
        else:
            return 0.1 * (self.global_step - 5000) / 10000

    def forward(self, pred_stft: Tensor, target_wave: Tensor, model: nn.Module) -> dict[str, Tensor]:
        pred_wave = model.waveform_from_stft(pred_stft)

        L = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred_wave   = pred_wave[..., :L]
        target_wave = target_wave[..., :L]

        losses = {}

        # Single STFT magnitude L1
        window = torch.hann_window(self.loss_n_fft, device=pred_wave.device)
        p_stft = torch.stft(pred_wave, self.loss_n_fft, self.loss_hop,
                            window=window, return_complex=True)
        t_stft = torch.stft(target_wave, self.loss_n_fft, self.loss_hop,
                            window=window, return_complex=True)
        losses['stft'] = F.l1_loss(p_stft.abs(), t_stft.abs())

        losses['mel_l1'] = self._mel_l1(pred_wave, target_wave)

        si_sdr_w = self._si_sdr_weight()
        if si_sdr_w > 0:
            losses['si_sdr'] = self._si_sdr(pred_wave, target_wave) * si_sdr_w
        else:
            losses['si_sdr'] = torch.tensor(0.0, device=pred_wave.device)

        losses['total'] = (
            1.0 * losses['stft'] +
            0.5 * losses['mel_l1'] +
            losses['si_sdr']
        )

        self.global_step += 1
        return losses


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        B = z1.shape[0]
        z  = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.T

        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Dataset — same as cloud but with 2-sec default clip
# ---------------------------------------------------------------------------

class StemDataset(Dataset):
    STEM_NAMES = ['bass', 'drums', 'vocals', 'other']

    def __init__(
        self,
        root: str,
        cfg: TrackProcessorConfig,
        clip_seconds: float = 2.0,    # was 4.0
        augment: bool = True,
        target_stems: Optional[list[str]] = None,
        cache_in_ram: bool = True,
    ):
        self.root      = Path(root)
        self.cfg       = cfg
        self.clip_len  = int(clip_seconds * cfg.sample_rate)
        self.augment   = augment
        self.stems     = target_stems or ['bass', 'drums', 'vocals', 'other']
        self._cache_in_ram = cache_in_ram
        self._stem_cache: dict[tuple, Tensor] = {}

        self.tracks = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and any((d / f"{s}.wav").exists() for s in self.stems)
        ])
        assert len(self.tracks) > 0, f"No tracks found in {root}"

    def set_shared_cache(self, cache: dict):
        self._stem_cache = cache
        self._cache_in_ram = True

    def build_ram_cache(self):
        if self._cache_in_ram and len(self._stem_cache) == 0:
            print(f"Pre-loading {len(self.tracks)} tracks into RAM...", flush=True)
            for track_dir in self.tracks:
                for stem_name in self.stems:
                    wav = self._load_stem_disk(track_dir, stem_name)
                    if wav is not None:
                        self._stem_cache[(str(track_dir), stem_name)] = wav
            total_gb = sum(v.numel() * 4 for v in self._stem_cache.values()) / 1e9
            print(f"  RAM cache: {len(self._stem_cache)} stems, {total_gb:.2f} GB", flush=True)

    def _load_stem(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        if self._cache_in_ram:
            return self._stem_cache.get((str(track_dir), stem_name))
        return self._load_stem_disk(track_dir, stem_name)

    def _load_stem_disk(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        path = track_dir / f"{stem_name}.wav"
        if not path.exists():
            return None

        try:
            data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        except Exception as e:
            print(f"  Warning: Could not read {path.name}: {e}")
            return None

        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        wav = torch.from_numpy(data)

        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.cfg.sample_rate
            ).squeeze(0)
        return wav

    def __len__(self) -> int:
        return len(self.tracks) * len(self.stems) * 4

    def _load_crepe(self, track_dir: Path, stem_name: str, target_len: int):
        crepe_path   = track_dir / f"{stem_name}.crepe.npy"
        pitched_path = track_dir / f"{stem_name}.pitched.npy"

        T_expected = 1 + target_len // self.cfg.hop_length

        if not crepe_path.exists() or not pitched_path.exists():
            return torch.zeros(T_expected, 360, dtype=torch.float32), 0.0

        probs = np.load(str(crepe_path)).astype(np.float32)
        pitched = float(np.load(str(pitched_path)))

        return torch.from_numpy(probs), pitched

    def _crop_with_crepe(self, wav: Tensor, crepe: Tensor) -> tuple[Tensor, Tensor]:
        L = wav.shape[-1]
        T_clip = 1 + self.clip_len // self.cfg.hop_length

        if L <= self.clip_len:
            wav_padded = F.pad(wav, (0, self.clip_len - L))
            T_have = crepe.shape[0]
            if T_have >= T_clip:
                crepe_out = crepe[:T_clip]
            else:
                pad_amt = T_clip - T_have
                crepe_out = F.pad(crepe, (0, 0, 0, pad_amt))
            return wav_padded, crepe_out

        start = random.randint(0, L - self.clip_len)
        wav_crop = wav[start:start + self.clip_len]

        frame_start = start // self.cfg.hop_length
        frame_end   = frame_start + T_clip
        T_have = crepe.shape[0]

        if frame_end <= T_have:
            crepe_crop = crepe[frame_start:frame_end]
        else:
            crepe_crop = crepe[frame_start:]
            pad_amt = T_clip - crepe_crop.shape[0]
            crepe_crop = F.pad(crepe_crop, (0, 0, 0, pad_amt))

        return wav_crop, crepe_crop

    def _augment(self, wav: Tensor) -> Tensor:
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))
        if random.random() < 0.5:
            wav = -wav
        return wav.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        track_idx = (idx // (len(self.stems) * 4)) % len(self.tracks)
        stem_idx  = (idx // 4) % len(self.stems)
        track_dir = self.tracks[track_idx]
        target_name = self.stems[stem_idx]

        target = self._load_stem(track_dir, target_name)
        if target is None:
            T_clip = 1 + self.clip_len // self.cfg.hop_length
            return {
                'target':         torch.zeros(self.clip_len),
                'other_stems':    torch.zeros(len(self.stems) - 1, self.clip_len),
                'target_crepe':   torch.zeros(T_clip, 360),
                'other_crepe':    torch.zeros(len(self.stems) - 1, T_clip, 360),
                'target_pitched': 0.0,
                'other_pitched':  torch.zeros(len(self.stems) - 1),
                'stem_name':      target_name,
                'valid':          False,
            }

        target_crepe, target_pitched = self._load_crepe(
            track_dir, target_name, target.shape[-1]
        )
        target, target_crepe = self._crop_with_crepe(target, target_crepe)

        other_wavs       = []
        other_crepes     = []
        other_pitched_l  = []
        for name in self.stems:
            if name == target_name:
                continue
            wav = self._load_stem(track_dir, name)
            if wav is not None:
                cr, pit = self._load_crepe(track_dir, name, wav.shape[-1])
                w_crop, c_crop = self._crop_with_crepe(wav, cr)
                other_wavs.append(w_crop)
                other_crepes.append(c_crop)
                other_pitched_l.append(pit)
            else:
                T_clip = 1 + self.clip_len // self.cfg.hop_length
                other_wavs.append(torch.zeros(self.clip_len))
                other_crepes.append(torch.zeros(T_clip, 360))
                other_pitched_l.append(0.0)

        if self.augment:
            target = self._augment(target)
            other_wavs = [self._augment(w) for w in other_wavs]

        return {
            'target':         target,
            'other_stems':    torch.stack(other_wavs, dim=0),
            'target_crepe':   target_crepe,
            'other_crepe':    torch.stack(other_crepes, dim=0),
            'target_pitched': target_pitched,
            'other_pitched':  torch.tensor(other_pitched_l),
            'stem_name':      target_name,
            'valid':          True,
        }


def collate_valid(batch: list[dict]) -> dict:
    batch = [b for b in batch if b['valid']]
    if not batch:
        return None
    return {
        'target':         torch.stack([b['target'] for b in batch]),
        'other_stems':    torch.stack([b['other_stems'] for b in batch]),
        'target_crepe':   torch.stack([b['target_crepe'] for b in batch]),
        'other_crepe':    torch.stack([b['other_crepe'] for b in batch]),
        'target_pitched': torch.tensor([b['target_pitched'] for b in batch]),
        'other_pitched':  torch.stack([b['other_pitched'] for b in batch]),
        'stem_names':     [b['stem_name'] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training Loop (lean)
# ---------------------------------------------------------------------------

def train(
    data_root: str,
    save_dir: str,
    cfg: Optional[TrackProcessorConfig] = None,
    n_steps:     int   = 100_000,
    batch_size:  int   = 2,
    lr:          float = 3e-4,
    device:      str   = 'auto',
    resume_from: Optional[str] = None,
    val_split:   float = 0.1,
    ckpt_every:  int   = 200,
    log_every:   int   = 25,
    val_every:   int   = 1000,
    num_workers: int   = 2,
    grad_clip:   float = 1.0,
    grad_accum:  int   = 8,
    early_stop_patience: int = 20,
    amp:         bool  = True,
    compile_model: bool = False,
    profile_steps: int = 0,
    fresh_best_val: bool = False,
    cache_ram:    bool = True,
):
    if cfg is None:
        cfg = TrackProcessorConfig()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("No GPU detected, using CPU")
    elif device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    dev = torch.device(device)
    print(f"Device: {dev}")

    if device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
        print(f"CPU threads: {torch.get_num_threads()}")

    if amp and device == 'cpu':
        amp = False
        print("AMP disabled on CPU")

    autocast_dtype = torch.bfloat16 if amp else torch.float32
    print(f"AMP: {amp} ({autocast_dtype})")

    if device == 'cuda' and num_workers == 2:
        num_workers = 4
        print(f"num_workers auto-bumped to {num_workers} for GPU")

    model = PerTrackProcessor(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Model config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_state={cfg.d_state}")

    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile (mode='reduce-overhead')...")
        model = torch.compile(model, mode='reduce-overhead')
    elif compile_model:
        print("torch.compile not available — skipping")

    task_loss   = LeanTrackLoss(cfg, cfg.sample_rate).to(dev)
    contrastive = NTXentLoss(temperature=0.1).to(dev)

    decay_params  = [p for n, p in model.named_parameters()
                     if 'bias' not in n and 'norm' not in n and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters()
                      if ('bias' in n or 'norm' in n) and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': decay_params,   'weight_decay': 1e-4},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step: int) -> float:
        warmup = 2000
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    best_val_loss = float('inf')
    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            print(f"ERROR: --resume path does not exist: {ckpt_path}")
            sys.exit(1)
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step']
        if 'best_val_loss' in ckpt and not fresh_best_val:
            best_val_loss = ckpt['best_val_loss']
            print(f"  resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"  resumed at step {start_step}, best_val_loss reset for new device")

    full_dataset = StemDataset(data_root, cfg, clip_seconds=2.0, augment=True, cache_in_ram=cache_ram)
    if cache_ram:
        full_dataset.build_ram_cache()
    val_dataset  = StemDataset(data_root, cfg, clip_seconds=2.0, augment=False, cache_in_ram=False)
    if cache_ram:
        val_dataset.set_shared_cache(full_dataset._stem_cache)

    n_tracks = len(full_dataset.tracks)
    n_val    = max(1, int(n_tracks * val_split))
    random.Random(42).shuffle(full_dataset.tracks)
    val_tracks   = full_dataset.tracks[-n_val:]
    train_tracks = full_dataset.tracks[:-n_val]

    full_dataset.tracks = train_tracks
    val_dataset.tracks  = val_tracks

    print(f"Tracks: {len(train_tracks)} train, {len(val_tracks)} val")

    dataloader = DataLoader(
        full_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = (device == 'cuda'),
        collate_fn  = collate_valid,
        drop_last   = True,
        persistent_workers = (num_workers > 0),
        prefetch_factor = 4 if num_workers > 0 else None,
        multiprocessing_context = 'spawn' if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_valid,
        drop_last   = False,
    )

    PHASE_1_END = int(n_steps * 0.20)
    PHASE_2_END = int(n_steps * 0.50)

    import signal
    _state = {'step': start_step, 'interrupted': False}

    def signal_handler(signum, frame):
        if _state['interrupted']:
            print("\n! Second interrupt — exiting hard.")
            sys.exit(130)
        print(f"\n! Caught signal {signum} — will save after current step.")
        _state['interrupted'] = True

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM') and sys.platform != 'win32':
        signal.signal(signal.SIGTERM, signal_handler)
    if sys.platform == 'win32' and hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

    def save_checkpoint(step: int, loss_val: float, is_best: bool = False,
                        name: Optional[str] = None):
        ckpt = {
            'step':           step,
            'model':          model.state_dict(),
            'optimizer':      optimizer.state_dict(),
            'scheduler':      scheduler.state_dict(),
            'cfg':            cfg,
            'loss':           loss_val,
            'best_val_loss':  best_val_loss,
        }
        fname = name if name else f"checkpoint_{step:06d}.pt"
        tmp_path  = save_path / f".{fname}.tmp"
        final_path = save_path / fname

        torch.save(ckpt, tmp_path)
        tmp_path.replace(final_path)

        if is_best:
            best_tmp = save_path / ".best.pt.tmp"
            torch.save(ckpt, best_tmp)
            best_tmp.replace(save_path / "best.pt")

    @torch.no_grad()
    def validate(current_phase: int) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        max_val_batches = 20   # cap validation time on laptop
        for v_idx, v_batch in enumerate(val_loader):
            if v_idx >= max_val_batches:
                break
            if v_batch is None:
                continue
            v_target = v_batch['target'].to(dev)
            v_crepe  = v_batch['target_crepe'].to(dev)
            if current_phase == 1:
                z1 = _encode_for_contrastive(model, v_target, cfg, dev)
                z2 = _encode_for_contrastive(model, _augment_for_contrastive(v_target), cfg, dev)
                vl = contrastive(z1, z2)
            else:
                if current_phase == 3:
                    v_other = v_batch['other_stems'].to(dev)
                    K = v_other.shape[1]
                    embs = []
                    for k in range(K):
                        v_crepe_k = v_batch['other_crepe'][:, k, :, :].to(dev)
                        out_k = model(v_other[:, k, :], v_crepe_k, other_embs=None)
                        embs.append(out_k['track_emb'])
                    other_embs = torch.stack(embs, dim=1)
                    out = model(v_target, v_crepe, other_embs=other_embs)
                else:
                    out = model(v_target, v_crepe, other_embs=None)
                vl_dict = task_loss(out['output_stft'], v_target, model)
                vl = vl_dict['total']
            if not (torch.isnan(vl) or torch.isinf(vl)):
                total_loss += vl.item()
                n_batches += 1
        model.train()
        return total_loss / max(1, n_batches)

    step = start_step
    plateau_count = 0
    last_log_time = time.time()
    last_log_step = step
    accum_step = 0

    prof = None
    if profile_steps > 0:
        print(f"Profiler enabled — will capture {profile_steps} steps after warmup")
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if device == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=profile_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(save_path / "profile_trace")
            ),
        )
        prof.start()

    print(f"\nStarting LAPTOP training: {n_steps} steps")
    print(f"  batch={batch_size}, grad_accum={grad_accum} (effective {batch_size*grad_accum})")
    print(f"  num_workers={num_workers}")
    print(f"  ckpt every {ckpt_every} steps (protects against power loss)")
    print(f"  Phase 1 (contrastive):  step 0       -> {PHASE_1_END}")
    print(f"  Phase 2 (single-track): step {PHASE_1_END} -> {PHASE_2_END}")
    print(f"  Phase 3 (cross-track):  step {PHASE_2_END} -> {n_steps}")
    if start_step > 0:
        print(f"  Resuming at step {start_step}")
    print()

    if device == 'cuda':
        print("Warming up GPU kernels (one forward+backward pass)...", flush=True)
        try:
            warm_batch = next(iter(dataloader))
            if warm_batch is not None:
                wt = warm_batch['target'][:1].to(dev)
                wc = warm_batch['target_crepe'][:1].to(dev)
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                    _ = model(wt, wc, other_embs=None)
                torch.cuda.synchronize(dev)
        except StopIteration:
            pass
        print("GPU warmup complete.", flush=True)

    last_phase = None
    empty_batch_streak = 0

    while step < n_steps:
        for batch in dataloader:
            if batch is None:
                empty_batch_streak += 1
                if empty_batch_streak > len(dataloader) * 2:
                    print(f"\nERROR: all batches empty.")
                    print(f"Dataset root: {data_root}")
                    sys.exit(1)
                continue
            empty_batch_streak = 0

            target       = batch['target'].to(dev)
            other_stems  = batch['other_stems'].to(dev)
            target_crepe = batch['target_crepe'].to(dev)
            other_crepe  = batch['other_crepe'].to(dev)
            B = target.shape[0]

            current_phase = (1 if step < PHASE_1_END
                             else 2 if step < PHASE_2_END
                             else 3)

            if current_phase != last_phase:
                print(f"\n{'-'*60}")
                print(f"  ENTERING PHASE {current_phase} at step {step}")
                print(f"{'-'*60}\n")
                last_phase = current_phase

            loss_scale = 1.0 / grad_accum

            if current_phase == 1:
                aug1 = _augment_for_contrastive(target)
                aug2 = _augment_for_contrastive(target)
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                    z1 = _encode_for_contrastive(model, aug1, cfg, dev)
                    z2 = _encode_for_contrastive(model, aug2, cfg, dev)
                    loss = contrastive(z1, z2)
                loss_dict = {'total': loss, 'contrastive': loss}

            elif current_phase == 2:
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                    out = model(target, target_crepe, other_embs=None)
                    loss_dict = task_loss(out['output_stft'], target, model)

            else:
                K = other_stems.shape[1]
                other_embs_list = []
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                    for k in range(K):
                        stem_k  = other_stems[:, k, :]
                        crepe_k = other_crepe[:, k, :, :]
                        out_k = model(stem_k, crepe_k, other_embs=None)
                        other_embs_list.append(out_k['track_emb'])
                    other_embs = torch.stack(other_embs_list, dim=1)
                    out = model(target, target_crepe, other_embs=other_embs)
                    loss_dict = task_loss(out['output_stft'], target, model)

            loss = loss_dict['total']
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[step {step}] NaN/Inf loss — skipping")
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                continue

            (loss * loss_scale).backward()
            accum_step += 1

            if accum_step >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0

            if prof is not None:
                prof.step()

            if step % log_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                steps_per_sec = (step - last_log_step) / elapsed
                last_log_time = now
                last_log_step = step

                # ETA in hours
                if steps_per_sec > 0.001:
                    remaining = (n_steps - step) / steps_per_sec
                    eta_h = remaining / 3600
                    eta_str = f"ETA {eta_h:.1f}h"
                else:
                    eta_str = "ETA -"

                log_str = (
                    f"[step {step:6d}] phase={current_phase} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e} "
                    f"{steps_per_sec:.3f} step/s {eta_str}"
                )
                if device == 'cuda':
                    mem = torch.cuda.max_memory_allocated(dev) / 1024**3
                    log_str += f"  gpu_mem={mem:.2f}GB"
                print(log_str, flush=True)

            if step > 0 and step % val_every == 0:
                print(f"  Running validation at step {step}...")
                val_loss = validate(current_phase)
                print(f"  val_loss = {val_loss:.4f}  (best={best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(step, loss.item(), is_best=True)
                    print(f"  new best val loss -- saved best.pt")
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count >= early_stop_patience:
                        print(f"  Val plateaued for {early_stop_patience} checks -- early stopping")
                        save_checkpoint(step, loss.item(), name="final_earlystop.pt")
                        if prof is not None:
                            prof.stop()
                        return model

            if step > 0 and step % ckpt_every == 0:
                save_checkpoint(step, loss.item())

            if _state['interrupted']:
                print(f"\n! Saving interrupt checkpoint at step {step}...")
                save_checkpoint(step, loss.item(), name=f"interrupt_{step:06d}.pt")
                print(f"  saved. Resume with: --resume {save_path}/interrupt_{step:06d}.pt")
                if prof is not None:
                    prof.stop()
                return model

            step += 1
            _state['step'] = step
            if step >= n_steps:
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    save_checkpoint(step, loss.item(), name="final.pt")
    if prof is not None:
        prof.stop()
        print(f"Profile trace saved to {save_path / 'profile_trace'}.json")
    return model


def _augment_for_contrastive(wav: Tensor) -> Tensor:
    gain_db = random.uniform(-4, 4)
    wav = wav * (10 ** (gain_db / 20))
    if random.random() < 0.5:
        wav = -wav
    shift = random.randint(0, wav.shape[-1] // 4)
    wav = torch.roll(wav, shift, dims=-1)
    noise_level = random.uniform(0, 0.005)
    wav = wav + noise_level * torch.randn_like(wav)
    return wav.clamp(-1.0, 1.0)


def _encode_for_contrastive(model, wav, cfg, dev):
    mag_sq, _ = model._compute_stft(wav)
    mel  = model._compute_mel(mag_sq)
    r8   = model._extract_r8_raw(wav, mag_sq)
    return model.semantic_enc(mel, r8, return_sequence=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Mamba-3 laptop training")
    parser.add_argument('--data',       default='./datasets/musdb18hq/train')
    parser.add_argument('--save',       default='./checkpoints_laptop')
    parser.add_argument('--resume',     default=None)
    parser.add_argument('--steps',      type=int,   default=100_000)
    parser.add_argument('--batch',      type=int,   default=2)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--device',     default='auto')
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--val-split',  type=float, default=0.1)
    parser.add_argument('--val-every',  type=int,   default=1000)
    parser.add_argument('--ckpt-every', type=int,   default=200)
    parser.add_argument('--log-every',  type=int,   default=25)
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int,  default=2)
    parser.add_argument('--no-amp',     action='store_true')
    parser.add_argument('--compile',    action='store_true',
                        help='Enable torch.compile (may not work on all ROCm configs)')
    parser.add_argument('--profile',    type=int,  default=0, metavar='STEPS',
                        help='Run torch profiler for STEPS steps, save Chrome trace to save dir')
    parser.add_argument('--fresh-best-val', action='store_true',
                        help='Reset best_val_loss on resume (use when switching device type)')
    parser.add_argument('--no-cache',    action='store_true',
                        help='Disable RAM pre-caching of stems (saves RAM, increases disk I/O)')

    args = parser.parse_args()

    cfg = TrackProcessorConfig()

    train(
        data_root           = args.data,
        save_dir            = args.save,
        cfg                 = cfg,
        n_steps             = args.steps,
        batch_size          = args.batch,
        lr                  = args.lr,
        device              = args.device,
        resume_from         = args.resume,
        val_split           = args.val_split,
        ckpt_every          = args.ckpt_every,
        log_every           = args.log_every,
        val_every           = args.val_every,
        num_workers         = args.num_workers,
        grad_accum          = args.grad_accum,
        early_stop_patience = args.early_stop_patience,
        amp                 = not args.no_amp,
        compile_model       = args.compile,
        profile_steps       = args.profile,
        fresh_best_val      = args.fresh_best_val,
        cache_ram           = not args.no_cache,
    )
