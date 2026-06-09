"""
Smoke test + memory profiler for Mamba-3 per-track processor.
Run this FIRST on your 860M before touching the training loop.

Usage (inside distrobox ROCm container):
  python test_smoke.py --device cuda
  python test_smoke.py --device cpu   # fallback if ROCm fails

Covers:
  [0] ROCm / HSA env validation (gfx1103 specific)
  [1] Model build + parameter count
  [2] Sub-module unit tests (R8, Mamba3MIMOLayer, CrossTrackAggregator, F0Conditioner)
  [3] ISTFT roundtrip (silent audio → STFT → ISTFT → SNR check)
  [4] Forward pass no-grad (NaN / Inf / shape checks)
  [5] AMP bf16 forward (matches train.py default — critical for SSM on ROCm)
  [6] Forward + backward with grad (NaN grad check, peak VRAM)
  [7] Gradient checkpointing (matches train.py enable path)
  [8] Memory scaling: single _compute_stft call, batch=4 × 4s clip

Expected on 860M (8 GB UMA, pure PyTorch, no Triton):
  Forward pass 1s clip batch=1 : 2–8 s
  Peak VRAM fwd+bwd d_model=128: < 3 GB
  ISTFT roundtrip SNR          : > 60 dB
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Callable

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import (
    PerTrackProcessor,
    TrackProcessorConfig,
    R8FeatureExtractor,
    Mamba3MIMOLayer,
    CrossTrackAggregator,
    F0Conditioner,
)


# ── helpers ────────────────────────────────────────────────────────────────

PASS  = "\033[92m✓\033[0m"
FAIL  = "\033[91m✗\033[0m"
WARN  = "\033[93m!\033[0m"
SKIP  = "\033[90m–\033[0m"


def vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def stft_frames(L: int, hop: int) -> int:
    """Frame count produced by torch.stft(center=True) — the model's default.
    Equivalent to: 1 + L // hop. NOT (L - n_fft) // hop + 1 (that's center=False).
    """
    return 1 + L // hop


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"    ⏱  {label}: {time.perf_counter()-t0:.3f}s")


def check(condition: bool, msg: str, fatal: bool = False) -> bool:
    status = PASS if condition else FAIL
    print(f"    {status} {msg}")
    if not condition and fatal:
        raise AssertionError(f"Fatal check failed: {msg}")
    return condition


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_section(label: str, fn: Callable) -> bool:
    """Run a test section, catch and report exceptions without aborting suite."""
    section(label)
    try:
        fn()
        return True
    except AssertionError as e:
        print(f"  {FAIL} FATAL: {e}")
        return False
    except Exception:
        print(f"  {FAIL} EXCEPTION:")
        traceback.print_exc()
        return False


# ── section 0: ROCm / HSA env ──────────────────────────────────────────────

def test_rocm_env():
    """
    Validate ROCm environment variables and gfx1103 visibility.
    Checks:
      - HSA_OVERRIDE_GFX_VERSION set (required for ROCm < 7.x on 860M)
      - PYTORCH_ROCM_ARCH set
      - torch.cuda.is_available() (ROCm exposes via CUDA API)
      - Device name contains expected RDNA3 identifier
      - rocminfo / hipconfig availability (optional, warn if absent)
    """
    cuda_ok = torch.cuda.is_available()
    check(cuda_ok, f"torch.cuda.is_available() → {cuda_ok}")

    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"    {PASS} GPU: {name}")
        print(f"    {PASS} VRAM: {total_gb:.1f} GB")

        # Warn if looks like NVIDIA — this stack is AMD-targeted
        if "NVIDIA" in name.upper():
            print(f"    {WARN} Detected NVIDIA GPU — ROCm-specific checks will be skipped")
            return

        # HSA override — mandatory for ROCm 5.x / 6.x on gfx1103
        hsa_override = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
        if hsa_override:
            check(True, f"HSA_OVERRIDE_GFX_VERSION={hsa_override}")
            if hsa_override not in ("11.0.0", "11.0.2"):
                print(f"    {WARN} Non-standard HSA_OVERRIDE value."
                      f" Recommended: 11.0.0 (ROCm 6.x) or unset (ROCm 7.x native)")
        else:
            print(f"    {WARN} HSA_OVERRIDE_GFX_VERSION not set."
                  f" Required for ROCm <7.x on gfx1103. Set: export HSA_OVERRIDE_GFX_VERSION=11.0.0")

        # PYTORCH_ROCM_ARCH
        rocm_arch = os.environ.get("PYTORCH_ROCM_ARCH", "")
        if rocm_arch:
            check("gfx1103" in rocm_arch or "gfx1100" in rocm_arch,
                  f"PYTORCH_ROCM_ARCH={rocm_arch} (expect gfx1103 or gfx1100)")
        else:
            print(f"    {WARN} PYTORCH_ROCM_ARCH not set."
                  f" Set: export PYTORCH_ROCM_ARCH=gfx1103")

        # rocminfo (optional — may not be in PATH inside distrobox)
        rocminfo_path = os.popen("which rocminfo 2>/dev/null").read().strip()
        if rocminfo_path:
            gfx = os.popen("rocminfo 2>/dev/null | grep 'gfx'").read().strip()
            print(f"    {PASS} rocminfo gfx targets: {gfx[:80]}")
        else:
            print(f"    {WARN} rocminfo not in PATH — run outside distrobox to verify gfx1103")

        # Tiny SGEMM on device (validates ROCm actually computes)
        a = torch.randn(64, 64, device='cuda')
        b = torch.randn(64, 64, device='cuda')
        c = a @ b
        check(not torch.isnan(c).any(), "64×64 SGEMM on device: no NaN")

    else:
        print(f"    {WARN} No CUDA/ROCm device — running CPU-only.")
        print(f"    {WARN} To enable: distrobox enter rocm && export HSA_OVERRIDE_GFX_VERSION=11.0.0")


# ── section 1: model build ─────────────────────────────────────────────────

_CFG = None   # populated in main, shared across sections

def test_build():
    global _CFG
    cfg = _CFG

    with timed("model instantiation"):
        model = PerTrackProcessor(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    {PASS} Parameters: {n_params:,}")

    # Verify all expected sub-modules present
    for attr in ['r8_extractor', 'semantic_enc', 'f0_conditioner',
                 'cross_track', 'cond_mlp', 'mamba_layers', 'decoder']:
        check(hasattr(model, attr), f"model.{attr} exists", fatal=True)

    check(len(model.mamba_layers) == cfg.n_layers,
          f"mamba_layers depth: {len(model.mamba_layers)} == {cfg.n_layers}")


# ── section 2: sub-module unit tests ──────────────────────────────────────

def test_submodules(dev: torch.device):
    cfg = _CFG
    sr  = cfg.sample_rate

    # shared dummy tensors
    B, L = 2, sr * 1
    T = stft_frames(L, cfg.hop_length)
    wav     = torch.randn(B, L,   device=dev)
    mag_sq  = torch.rand(B, T, cfg.n_fft // 2 + 1, device=dev)

    # ── R8FeatureExtractor ──
    print("  R8FeatureExtractor:")
    r8_ext = R8FeatureExtractor(cfg).to(dev)
    with torch.no_grad():
        r8_out = r8_ext(wav, mag_sq)
    check(r8_out.shape == (B, T, cfg.d_model),
          f"r8_ext output shape {r8_out.shape} == ({B},{T},{cfg.d_model})", fatal=True)
    check(not torch.isnan(r8_out).any(), "r8_ext: no NaN")
    check(not torch.isinf(r8_out).any(), "r8_ext: no Inf")

    # Raw R8 (8-dim before projection) — needed for MIMO input
    band   = r8_ext._soft_band_energy(mag_sq)    # (B,T,6)
    trans  = r8_ext._transient_density(wav, T)   # (B,T,1)
    cent   = r8_ext._spectral_centroid(mag_sq)    # (B,T,1)
    r8_raw = torch.cat([band, trans, cent], dim=-1)  # (B,T,8)
    check(r8_raw.shape[-1] == cfg.R,
          f"r8_raw last dim {r8_raw.shape[-1]} == R={cfg.R}", fatal=True)

    # ── Mamba3MIMOLayer ──
    print("  Mamba3MIMOLayer:")
    layer = Mamba3MIMOLayer(cfg).to(dev)
    x_emb = torch.randn(B, T, cfg.d_model, device=dev)
    cond  = torch.randn(B, T, cfg.d_cond,  device=dev)

    with torch.no_grad():
        # Without conditioning
        y_no_cond = layer(x_emb, r8_raw)
        check(y_no_cond.shape == x_emb.shape,
              f"MIMO no-cond shape {y_no_cond.shape} == {x_emb.shape}", fatal=True)
        check(not torch.isnan(y_no_cond).any(), "MIMO no-cond: no NaN")

        # With conditioning
        y_cond = layer(x_emb, r8_raw, cond=cond)
        check(y_cond.shape == x_emb.shape,
              f"MIMO with-cond shape {y_cond.shape} == {x_emb.shape}", fatal=True)
        check(not torch.isnan(y_cond).any(), "MIMO with-cond: no NaN")

        # Output should differ with vs without conditioning
        diff = (y_cond - y_no_cond).abs().mean().item()
        check(diff > 1e-6, f"MIMO cond changes output (mean Δ={diff:.4e})")

    # Verify A_d magnitude ≤ 1 (stability: poles inside unit disk)
    A_d, _ = layer._discretize_trapezoidal(
        layer.log_a_real,
        layer.a_imag,
        torch.full((B, T, cfg.n_heads), 0.01, device=dev)
    )
    pole_mag = A_d.abs().max().item()
    check(pole_mag <= 1.0 + 1e-5,
          f"Trapezoidal discretization: max pole magnitude {pole_mag:.6f} ≤ 1")

    # ── CrossTrackAggregator ──
    print("  CrossTrackAggregator:")
    agg = CrossTrackAggregator(cfg).to(dev)
    target_emb = torch.randn(B, T, cfg.d_semantic, device=dev)
    K = 3
    other_embs = torch.randn(B, K, T, cfg.d_semantic, device=dev)
    with torch.no_grad():
        ctx = agg(target_emb, other_embs)
    check(ctx.shape == (B, T, cfg.d_cross),
          f"CrossTrack output shape {ctx.shape} == ({B},{T},{cfg.d_cross})", fatal=True)
    check(not torch.isnan(ctx).any(), "CrossTrack: no NaN")

    # ── F0Conditioner ──
    print("  F0Conditioner:")
    f0c = F0Conditioner(cfg).to(dev)
    crepe = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    with torch.no_grad():
        f0_out = f0c(crepe)
    check(f0_out.shape == (B, T, cfg.d_f0),
          f"F0Conditioner output shape {f0_out.shape} == ({B},{T},{cfg.d_f0})", fatal=True)
    check(not torch.isnan(f0_out).any(), "F0Conditioner: no NaN")


# ── section 3: ISTFT roundtrip ─────────────────────────────────────────────

def test_istft_roundtrip(dev: torch.device):
    """
    Validate STFT → ISTFT reconstruction SNR.
    Uses a sine tone (known spectrum) so SNR is interpretable.
    Expectation: > 60 dB for clean roundtrip (no model involved).
    """
    cfg = _CFG
    sr  = cfg.sample_rate

    # 440 Hz sine, 2 seconds
    t   = torch.linspace(0, 2.0, sr * 2, device=dev)
    ref = torch.sin(2 * 3.14159265 * 440 * t).unsqueeze(0)  # (1, L)

    window = torch.hann_window(cfg.n_fft, device=dev)

    stft = torch.stft(
        ref,
        n_fft      = cfg.n_fft,
        hop_length  = cfg.hop_length,
        window     = window,
        return_complex = True,
        normalized = False,
    )  # (1, F, T)

    recon = torch.istft(
        stft,
        n_fft      = cfg.n_fft,
        hop_length  = cfg.hop_length,
        window     = window,
    )  # (1, L')

    # Align lengths
    L = min(ref.shape[-1], recon.shape[-1])
    signal_power = ref[..., :L].pow(2).mean()
    error_power  = (ref[..., :L] - recon[..., :L]).pow(2).mean()
    snr_db = 10 * torch.log10(signal_power / (error_power + 1e-12)).item()

    check(snr_db > 60.0, f"STFT→ISTFT roundtrip SNR: {snr_db:.1f} dB (threshold 60 dB)")
    check(recon.shape[-1] >= L - cfg.hop_length,
          f"ISTFT output length {recon.shape[-1]} ≥ {L - cfg.hop_length}")

    # Also test model.waveform_from_stft convenience wrapper
    model = PerTrackProcessor(cfg).to(dev)
    stft_T = stft.transpose(1, 2)          # (1, T, F) — model convention
    with torch.no_grad():
        recon2 = model.waveform_from_stft(stft_T)
    L2 = min(ref.shape[-1], recon2.shape[-1])
    err2 = (ref[..., :L2] - recon2[..., :L2]).pow(2).mean()
    snr2 = 10 * torch.log10(signal_power / (err2 + 1e-12)).item()
    check(snr2 > 60.0, f"model.waveform_from_stft roundtrip SNR: {snr2:.1f} dB")


# ── section 4: forward pass no-grad ───────────────────────────────────────

def test_forward_no_grad(dev: torch.device) -> dict:
    """Returns output dict for reuse in later sections."""
    cfg = _CFG
    sr  = cfg.sample_rate
    B   = 1
    L   = sr * 1    # 1s clip
    T   = stft_frames(L, cfg.hop_length)

    model = PerTrackProcessor(cfg).to(dev).eval()

    waveform    = torch.randn(B, L, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    K = 2
    other_embs  = torch.randn(B, K, T, cfg.d_semantic, device=dev)

    reset_peak()
    with timed("forward no-grad (1s, batch=1)"):
        with torch.no_grad():
            out = model(waveform, crepe_probs, other_embs=other_embs)

    print(f"    output_stft : {out['output_stft'].shape}")
    print(f"    track_emb   : {out['track_emb'].shape}")
    print(f"    r8          : {out['r8'].shape}")
    print(f"    VRAM after fwd: {vram_mb():.0f} MB  peak: {peak_vram_mb():.0f} MB")

    check(not torch.isnan(out['output_stft']).any(), "output_stft: no NaN", fatal=True)
    check(not torch.isinf(out['output_stft']).any(), "output_stft: no Inf", fatal=True)
    check(out['output_stft'].shape[0] == B,           "batch dim correct")
    check(out['output_stft'].dtype == torch.complex64, "output dtype complex64")

    return out


# ── section 5: AMP bf16 forward ───────────────────────────────────────────

def test_amp_bf16(dev: torch.device):
    """
    Validate bf16 autocast matches fp32 output to reasonable tolerance.
    Critical: SSM recurrence is numerically sensitive — if bf16 gives NaN
    or > 1.0 mean absolute difference vs fp32, flag it.

    train.py defaults to bf16 AMP. This test gates that decision.
    """
    if not torch.cuda.is_available():
        print(f"    {SKIP} No CUDA/ROCm — skipping AMP test")
        return

    cfg = _CFG
    sr  = cfg.sample_rate
    B, L = 1, sr * 1
    T = stft_frames(L, cfg.hop_length)

    model_fp32 = PerTrackProcessor(cfg).to(dev).eval()
    model_bf16 = PerTrackProcessor(cfg).to(dev).eval()
    # Copy weights so comparison is apples-to-apples
    model_bf16.load_state_dict(model_fp32.state_dict())

    wav         = torch.randn(B, L, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)

    with torch.no_grad():
        # fp32 baseline
        out_fp32 = model_fp32(wav, crepe_probs)

        # bf16 autocast
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out_bf16 = model_bf16(wav, crepe_probs)

    # Compare magnitudes (complex STFT)
    mag_fp32 = out_fp32['output_stft'].abs().float()
    mag_bf16 = out_bf16['output_stft'].abs().float()

    has_nan  = torch.isnan(mag_bf16).any().item()
    has_inf  = torch.isinf(mag_bf16).any().item()
    mean_err = (mag_fp32 - mag_bf16).abs().mean().item()
    max_err  = (mag_fp32 - mag_bf16).abs().max().item()

    check(not has_nan, f"bf16 output: no NaN")
    check(not has_inf, f"bf16 output: no Inf")
    check(mean_err < 0.5,
          f"bf16 vs fp32 mean |Δmag|: {mean_err:.4f} (threshold 0.5)")
    check(max_err  < 5.0,
          f"bf16 vs fp32 max  |Δmag|: {max_err:.4f} (threshold 5.0)")

    if has_nan or has_inf:
        print(f"    {WARN} bf16 NaN/Inf detected → train with --no-amp until resolved")
        print(f"    {WARN} Check: complex64 recurrence inside autocast context")
    elif mean_err > 0.1:
        print(f"    {WARN} Mean error {mean_err:.4f} > 0.1 — monitor for training divergence")
    else:
        print(f"    {PASS} bf16 AMP looks stable for training")


# ── section 6: forward + backward ─────────────────────────────────────────

def test_forward_backward(dev: torch.device):
    cfg = _CFG
    sr  = cfg.sample_rate
    B, L = 1, sr * 1
    T = stft_frames(L, cfg.hop_length)

    model = PerTrackProcessor(cfg).to(dev).train()

    wav         = torch.randn(B, L, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    target_wav  = torch.randn(B, L, device=dev)  # dummy target

    reset_peak()
    with timed("forward+backward (1s, batch=1)"):
        out  = model(wav, crepe_probs)
        recon = model.waveform_from_stft(out['output_stft'])
        Lm   = min(recon.shape[-1], target_wav.shape[-1])
        loss = F.l1_loss(recon[..., :Lm], target_wav[..., :Lm])
        loss.backward()

    print(f"    Loss: {loss.item():.4f}")
    print(f"    Peak VRAM fwd+bwd: {peak_vram_mb():.0f} MB")

    check(not torch.isnan(loss), "loss: no NaN", fatal=True)
    check(not torch.isinf(loss), "loss: no Inf", fatal=True)

    # Gradient checks
    grad_norms = {
        n: p.grad.norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    nan_grads = [n for n, v in grad_norms.items() if v != v]
    zero_grads = [n for n, v in grad_norms.items() if v == 0.0]

    check(len(nan_grads) == 0, f"NaN gradients: {len(nan_grads)} params affected"
          + (f" → {nan_grads[:3]}" if nan_grads else ""))
    check(len(zero_grads) < len(grad_norms) * 0.5,
          f"Dead grads: {len(zero_grads)}/{len(grad_norms)} params zero")

    # Top-5 grad norms (useful for debugging exploding/vanishing)
    top5 = sorted(grad_norms.items(), key=lambda x: -x[1])[:5]
    print("    Top-5 grad norms (explode check):")
    for n, v in top5:
        flag = WARN if v > 100 else PASS
        print(f"      {flag} {v:8.4f}  {n}")

    # SSM-specific: check log_a_real and a_imag got gradients
    # (these are the pole parameters — must be trained)
    for i, layer in enumerate(model.mamba_layers):
        for pname in ['log_a_real', 'a_imag']:
            p = getattr(layer, pname, None)
            if p is not None and p.grad is not None:
                check(not torch.isnan(p.grad).any(),
                      f"mamba_layers[{i}].{pname} grad: no NaN")


# ── section 7: gradient checkpointing ─────────────────────────────────────

def test_grad_checkpointing(dev: torch.device):
    """
    Validate that gradient checkpointing reduces peak VRAM without
    breaking gradients. Compares peak VRAM with/without checkpointing
    for the same forward+backward pass.

    train.py sets layer.gradient_checkpointing = True on all Mamba layers.
    We replicate that here and verify:
      1. Output is numerically close (same weights, same input)
      2. Peak VRAM with checkpointing ≤ peak VRAM without
      3. Gradients are non-NaN post-recompute

    IMPLEMENTATION NOTE: torch.utils.checkpoint internally calls
    torch._compile.disable(), which forces an import of torch._dynamo even
    when use_reentrant=True. Under pytorch-triton-rocm 3.5.1 the libtriton.so
    is broken (missing 'ir' symbol), so the import explodes regardless of
    reentrant mode. We bypass this entirely with a manual autograd.Function
    that recomputes forward during backward. Zero dependency on
    torch._compile / torch._dynamo / triton.
    """
    if not torch.cuda.is_available():
        print(f"    {SKIP} No CUDA/ROCm — skipping grad-checkpointing test")
        return

    # ── Manual gradient checkpoint via autograd.Function ──
    # This is the "old reentrant" semantics, hand-implemented to bypass
    # torch.utils.checkpoint's transitive torch._dynamo dependency.
    class ManualCheckpoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fn, has_cond, x, x_r8, *cond_arg):
            """fn: layer.forward bound. has_cond: bool. cond_arg: optional cond tensor."""
            ctx.fn = fn
            ctx.has_cond = has_cond
            ctx.save_for_backward(x, x_r8, *cond_arg)
            with torch.no_grad():
                if has_cond:
                    out = fn(x, x_r8, cond=cond_arg[0])
                else:
                    out = fn(x, x_r8, cond=None)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            saved = ctx.saved_tensors
            if ctx.has_cond:
                x, x_r8, cond = saved
                inputs = (x.detach().requires_grad_(True),
                          x_r8.detach().requires_grad_(True),
                          cond.detach().requires_grad_(True))
            else:
                x, x_r8 = saved
                inputs = (x.detach().requires_grad_(True),
                          x_r8.detach().requires_grad_(True))

            with torch.enable_grad():
                if ctx.has_cond:
                    out = ctx.fn(inputs[0], inputs[1], cond=inputs[2])
                else:
                    out = ctx.fn(inputs[0], inputs[1], cond=None)

            grads = torch.autograd.grad(out, inputs, grad_out,
                                        allow_unused=True)
            # (fn, has_cond, x_grad, r8_grad, [cond_grad])
            if ctx.has_cond:
                return (None, None, grads[0], grads[1], grads[2])
            else:
                return (None, None, grads[0], grads[1])

    def manual_checkpoint(fn, x, x_r8, cond):
        if cond is None:
            return ManualCheckpoint.apply(fn, False, x, x_r8)
        else:
            return ManualCheckpoint.apply(fn, True, x, x_r8, cond)

    cfg = _CFG
    sr  = cfg.sample_rate
    B, L = 1, sr * 1
    T   = stft_frames(L, cfg.hop_length)

    wav         = torch.randn(B, L, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    target      = torch.randn(B, L, device=dev)

    # ── Without checkpointing ──
    model_plain = PerTrackProcessor(cfg).to(dev).train()
    reset_peak()
    torch.manual_seed(1234)  # deterministic dropout for plain vs ckpt comparison
    out_plain = model_plain(wav, crepe_probs)
    recon     = model_plain.waveform_from_stft(out_plain['output_stft'])
    Lm        = min(recon.shape[-1], target.shape[-1])
    F.l1_loss(recon[..., :Lm], target[..., :Lm]).backward()
    peak_plain = peak_vram_mb()
    print(f"    Peak VRAM without checkpointing: {peak_plain:.0f} MB")

    # ── With manual checkpointing ──
    model_ckpt = PerTrackProcessor(cfg).to(dev).train()
    model_ckpt.load_state_dict(model_plain.state_dict())

    # Wrap each Mamba layer's forward in ManualCheckpoint
    original_forwards = {}
    for i, layer in enumerate(model_ckpt.mamba_layers):
        original_forwards[i] = layer.forward

        def make_ckpt_forward(orig_fwd):
            def ckpt_forward(x, x_r8, cond=None):
                return manual_checkpoint(orig_fwd, x, x_r8, cond)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_forwards[i])

    reset_peak()
    torch.manual_seed(1234)  # SAME seed as plain → identical dropout masks
    out_ckpt  = model_ckpt(wav, crepe_probs)
    recon_ck  = model_ckpt.waveform_from_stft(out_ckpt['output_stft'])
    Lck       = min(recon_ck.shape[-1], target.shape[-1])
    loss_ck   = F.l1_loss(recon_ck[..., :Lck], target[..., :Lck])
    loss_ck.backward()
    peak_ckpt = peak_vram_mb()
    print(f"    Peak VRAM with    checkpointing: {peak_ckpt:.0f} MB")

    saving_mb = peak_plain - peak_ckpt
    print(f"    VRAM saved by checkpointing: {saving_mb:.0f} MB")

    check(peak_ckpt <= peak_plain + 10,   # 10 MB tolerance
          f"Checkpointing VRAM ≤ plain VRAM (+10 MB tolerance)")

    # Numerical check: outputs should be identical (same weights, same input)
    mag_plain = out_plain['output_stft'].abs().float().detach()
    mag_ckpt  = out_ckpt['output_stft'].abs().float().detach()
    mean_diff = (mag_plain - mag_ckpt).abs().mean().item()
    check(mean_diff < 1e-4,
          f"Checkpointed output matches plain (mean Δ={mean_diff:.2e})")

    # Gradient validity after recompute
    nan_grads = [
        n for n, p in model_ckpt.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    check(len(nan_grads) == 0,
          f"No NaN gradients after checkpointed backward"
          + (f" (affected: {nan_grads[:3]})" if nan_grads else ""))

    # Restore original forwards
    for i, layer in enumerate(model_ckpt.mamba_layers):
        layer.forward = original_forwards[i]


# ── section 8: memory scaling (single _compute_stft call) ─────────────────

def test_memory_scaling(dev: torch.device):
    """
    Estimate VRAM for batch=4 × 4s clip.
    Fix from original: _compute_stft called ONCE, result reused.
    Full model forward not run (too slow on sequential scan at this size);
    encoder-only pass gives the scaling signal.
    """
    cfg = _CFG
    sr  = cfg.sample_rate
    clip_4s = sr * 4
    B4 = 4
    T4 = stft_frames(clip_4s, cfg.hop_length)

    model = PerTrackProcessor(cfg).to(dev).eval()

    try:
        reset_peak()
        with torch.no_grad():
            w4 = torch.randn(B4, clip_4s, device=dev)

            # Single _compute_stft call — result reused for both mel and r8
            mag_sq4, _ = model._compute_stft(w4)          # ONCE
            mel4   = model._compute_mel(mag_sq4)           # reuse
            r8_4   = model._extract_r8_raw(w4, mag_sq4)   # reuse
            emb4   = model.semantic_enc(mel4, r8_4, return_sequence=False)

        peak_mb = peak_vram_mb()
        print(f"    batch=4, clip=4s, encoder only: peak VRAM = {peak_mb:.0f} MB")
        check(emb4.shape == (B4, cfg.d_semantic),
              f"encoder output shape {emb4.shape} == ({B4},{cfg.d_semantic})")

        # Budget guidance
        if peak_mb < 3000:
            print(f"    {PASS} VRAM OK for training batch=4 — proceed")
        elif peak_mb < 6000:
            print(f"    {WARN} VRAM tight ({peak_mb:.0f} MB) — enable grad checkpointing")
        else:
            print(f"    {WARN} VRAM high ({peak_mb:.0f} MB) — reduce to batch=2 or d_model=64")

    except torch.cuda.OutOfMemoryError:
        print(f"    {FAIL} OOM at batch=4 × 4s encoder pass")
        print(f"    → reduce: --d-model 64 --d-state 8 --batch 2")


# ── summary ────────────────────────────────────────────────────────────────

def print_summary(results: dict[str, bool]):
    section("SUMMARY")
    passed = sum(v for v in results.values())
    total  = len(results)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{total} sections passed")
    if passed == total:
        print(f"\n  {PASS} All checks passed — proceed to train.py")
        print(f"     python train.py --data /path/to/musdb18hq --device cuda")
    else:
        failed = [n for n, v in results.items() if not v]
        print(f"\n  {FAIL} Failed sections: {failed}")
        print(f"  Fix these before running train.py")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for Mamba-3 per-track processor"
    )
    parser.add_argument('--device',  default='cuda',
                        help='cuda (ROCm) or cpu')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Reduced d_model for smoke test (default 128)')
    parser.add_argument('--d-state', type=int, default=16)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--skip',    nargs='*', default=[],
                        help='Section numbers to skip, e.g. --skip 5 7')
    args = parser.parse_args()

    global _CFG
    _CFG = TrackProcessorConfig(
        d_model   = args.d_model,
        d_state   = args.d_state,
        n_heads   = 8,
        n_layers  = args.n_layers,
        chunk_size = 64,
    )

    dev_str = args.device if torch.cuda.is_available() else 'cpu'
    if dev_str == 'cuda' and not torch.cuda.is_available():
        print(f"{WARN} CUDA/ROCm not available — falling back to CPU")
    dev = torch.device(dev_str)

    skip = set(args.skip)

    results = {}

    def maybe(idx: str, label: str, fn):
        if idx in skip:
            print(f"\n{SKIP} [{idx}] {label} — skipped")
            return
        results[f"[{idx}] {label}"] = run_section(f"[{idx}] {label}", fn)

    maybe("0", "ROCm / HSA env",             test_rocm_env)
    maybe("1", "Model build",                 lambda: test_build())
    maybe("2", "Sub-module unit tests",       lambda: test_submodules(dev))
    maybe("3", "ISTFT roundtrip",             lambda: test_istft_roundtrip(dev))
    maybe("4", "Forward no-grad",             lambda: test_forward_no_grad(dev))
    maybe("5", "AMP bf16 forward",            lambda: test_amp_bf16(dev))
    maybe("6", "Forward + backward",          lambda: test_forward_backward(dev))
    maybe("7", "Gradient checkpointing",      lambda: test_grad_checkpointing(dev))
    maybe("8", "Memory scaling (batch=4×4s)", lambda: test_memory_scaling(dev))

    print_summary(results)


if __name__ == '__main__':
    main()
