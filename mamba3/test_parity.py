"""
Parity test: verifies _ssm_scan_parallel produces equivalent output to
_ssm_scan_sequential for the same input.

Run this BEFORE training. If it fails, do NOT proceed.

Usage:
    python test_parity.py
"""

import sys
import torch

from model import Mamba3MIMOLayer, TrackProcessorConfig


def main():
    # ---- Reproducibility ----
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Build a layer ----
    cfg = TrackProcessorConfig()
    layer = Mamba3MIMOLayer(cfg).to(device).eval()
    print(f"Layer built: H={cfg.n_heads}, N={cfg.d_state}, d_model={cfg.d_model}")

    # ---- Build a realistic-size batch ----
    B, T = 2, 345          # batch 2, 4-sec clip at hop 512
    x_r8  = torch.randn(B, T, cfg.R, device=device)
    dt    = torch.full((B, T, cfg.n_heads), 0.01, device=device)

    # Get discretized A, B exactly as the layer would
    with torch.no_grad():
        A_d, B_d = layer._discretize_trapezoidal(
            layer.log_a_real.float(),
            layer.a_imag.float(),
            dt,
        )

    # ---- Run both scans ----
    with torch.no_grad():
        out_seq = layer._ssm_scan_sequential(x_r8, A_d, B_d, dt)
        out_par = layer._ssm_scan_parallel(x_r8, A_d, B_d, dt)

    # ---- Compare ----
    print(f"\nOutput shapes:")
    print(f"  sequential: {tuple(out_seq.shape)}")
    print(f"  parallel:   {tuple(out_par.shape)}")

    if out_seq.shape != out_par.shape:
        print("FAIL: shape mismatch")
        sys.exit(1)

    abs_diff  = (out_seq - out_par).abs()
    max_diff  = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    seq_norm  = out_seq.abs().mean().item()
    rel_diff  = mean_diff / (seq_norm + 1e-12)

    print(f"\nDifference stats:")
    print(f"  max abs:  {max_diff:.6e}")
    print(f"  mean abs: {mean_diff:.6e}")
    print(f"  ref norm: {seq_norm:.6e}")
    print(f"  rel:      {rel_diff:.6e}")

    # ---- Pass / fail ----
    # 1e-3 relative tolerance is acceptable for fp32 complex arithmetic
    # with 345 sequential operations of mixed precision.
    TOL_REL = 1e-3
    TOL_ABS = 1e-2

    if rel_diff < TOL_REL and max_diff < TOL_ABS:
        print(f"\nPASS — parallel scan matches sequential within tolerance")
        print(f"   (rel < {TOL_REL}, max_abs < {TOL_ABS})")
    else:
        print(f"\nFAIL — parallel scan diverges from sequential")
        print(f"   (rel < {TOL_REL} required, max_abs < {TOL_ABS} required)")
        sys.exit(1)

    # ---- Speed check ----
    print("\nSpeed comparison (10 iterations each):")
    import time

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = layer._ssm_scan_sequential(x_r8, A_d, B_d, dt)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_seq = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = layer._ssm_scan_parallel(x_r8, A_d, B_d, dt)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_par = time.time() - t0

    print(f"  sequential: {t_seq*100:.1f} ms per call")
    print(f"  parallel:   {t_par*100:.1f} ms per call")
    print(f"  speedup:    {t_seq/t_par:.1f}x")

    if t_seq / t_par < 5:
        print(f"\nWarning: speedup is less than 5x. Expected 20-50x on GPU.")
        print(f"   If on CPU, this is normal — Python overhead dominates either way.")
        print(f"   If on GPU, something is wrong; investigate before training.")

    print("\nReady to train.")


if __name__ == '__main__':
    main()
