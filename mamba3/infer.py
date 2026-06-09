#!/usr/bin/env python
"""
Mamba-3 Per-Track Processor — Inference Script
================================================
Load a trained checkpoint and process audio through the filter bank.

Usage:
    python infer.py --model checkpoints_laptop/final.pt --input song.wav --output out.wav
    python infer.py --model checkpoints_laptop/final.pt --input song.wav --output out.wav --device cuda
    python infer.py --model checkpoints_laptop/best.pt --input song.wav --output out.wav --chunk 8

Chunked mode (--chunk N) splits long audio into N-second windows with
overlap-add for GPU memory-constrained inference. Without --chunk, the
entire file is processed in one pass (best quality, needs RAM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model_laptop import PerTrackProcessor, TrackProcessorConfig


def main():
    parser = argparse.ArgumentParser(
        description="Mamba-3 Per-Track Processor inference"
    )
    parser.add_argument("--model", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--input", required=True, help="Input WAV file")
    parser.add_argument("--output", required=True, help="Output WAV file")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--chunk", type=float, default=0,
                        help="Chunk duration in seconds (0 = process entire file at once)")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Overlap ratio for chunked mode (default 0.25)")
    args = parser.parse_args()

    # ---------- device ----------
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    dev = torch.device(args.device)

    # ---------- load checkpoint ----------
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    step = ckpt.get("step", "unknown")

    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
    else:
        cfg = TrackProcessorConfig()

    print(f"Checkpoint: {ckpt_path.name}  (step {step})")
    print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"d_state={cfg.d_state}, n_heads={cfg.n_heads}")

    # ---------- build model ----------
    model = PerTrackProcessor(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # ---------- load audio ----------
    audio, sr = sf.read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    print(f"Input:  {args.input}  ({duration:.1f}s, {sr} Hz, mono)")

    # ---------- resample if needed ----------
    if sr != cfg.sample_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, cfg.sample_rate)
        audio_t = torch.from_numpy(audio.copy()).float().unsqueeze(0)
        audio = resampler(audio_t).squeeze(0).numpy()
        sr = cfg.sample_rate
        print(f"Resampled to {sr} Hz")

    # ---------- normalize ----------
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95  # -0.4 dBFS headroom
        print(f"Normalized (peak {peak:.3f} -> 0.95)")

    # ---------- inference ----------
    if args.chunk > 0:
        output = _process_chunked(model, audio, sr, cfg, dev,
                                  args.chunk, args.overlap)
    else:
        output = _process_full(model, audio, cfg, dev)

    # ---------- save ----------
    sf.write(args.output, output, cfg.sample_rate)
    out_dur = len(output) / cfg.sample_rate
    print(f"Output: {args.output}  ({out_dur:.1f}s)")


def _process_full(model, audio, cfg, dev):
    """Process entire file in one forward pass."""
    waveform = torch.from_numpy(audio.copy()).float().unsqueeze(0).to(dev)
    input_len = waveform.shape[-1]

    mag_sq, _ = model._compute_stft(waveform)
    T_frames = mag_sq.shape[1]
    crepe = torch.zeros(1, T_frames, 360, device=dev)

    use_amp = (dev.type == "cuda")
    with torch.no_grad():
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            out = model(waveform, crepe, other_embs=None)

    output = model.waveform_from_stft(out["output_stft"])
    output = output.squeeze(0).squeeze(0).cpu().numpy()
    if len(output) > input_len:
        output = output[:input_len]
    elif len(output) < input_len:
        output = np.pad(output, (0, input_len - len(output)))
    return output


def _process_chunked(model, audio, cfg, dev, chunk_s, overlap_ratio):
    """Process long audio in overlapping chunks with crossfade."""
    sr = cfg.sample_rate
    chunk_len = int(chunk_s * sr)
    hop_len = int(chunk_len * (1 - overlap_ratio))

    if hop_len < 1:
        hop_len = chunk_len // 2

    total = len(audio)
    n_chunks = max(1, (total - chunk_len) // hop_len + 1)

    # Hann window for crossfade
    window = np.hanning(chunk_len)
    output = np.zeros(total + chunk_len, dtype=np.float32)
    weight = np.zeros(total + chunk_len, dtype=np.float32)

    for i in range(n_chunks):
        start = i * hop_len
        end = min(start + chunk_len, total)
        chunk = np.zeros(chunk_len, dtype=np.float32)
        chunk[:end - start] = audio[start:end]

        waveform = torch.from_numpy(chunk).float().unsqueeze(0).to(dev)

        mag_sq, _ = model._compute_stft(waveform)
        T_frames = mag_sq.shape[1]
        crepe = torch.zeros(1, T_frames, 360, device=dev)

        use_amp = (dev.type == "cuda")
        with torch.no_grad():
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                out = model(waveform, crepe, other_embs=None)

        chunk_out = model.waveform_from_stft(out["output_stft"])
        chunk_out = chunk_out.squeeze(0).squeeze(0).cpu().numpy()

        w = window[:len(chunk_out)] if len(chunk_out) < len(window) else window
        output[start:start + len(chunk_out)] += chunk_out * w
        weight[start:start + len(chunk_out)] += w

        if (i + 1) % 10 == 0:
            print(f"  chunk {i + 1}/{n_chunks}")

    # Normalize by window overlap
    valid = weight > 1e-8
    output[valid] /= weight[valid]
    return output[:total]


if __name__ == "__main__":
    main()
