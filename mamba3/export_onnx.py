#!/usr/bin/env python
"""
ONNX export for Mamba-3 Per-Track Filter Bank.

Decomposes all complex arithmetic (SSM parallel scan + trapezoidal
discretization) into real-valued ops since ONNX has no complex dtype.

Usage:
    python export_onnx.py --model checkpoints_laptop/final.pt --output mamba_filter.onnx

Verification (after export):
    python export_onnx.py --model checkpoints_laptop/final.pt --output mamba_filter.onnx --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model_laptop import TrackProcessorConfig

# ═══════════════════════════════════════════════════════════════════
# Real-arithmetic complex helpers
# (a+bi) represented as (real, imag) pair of tensors
# ═══════════════════════════════════════════════════════════════════

def _cmul(ar, ai, br, bi):
    """(a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
    return ar * br - ai * bi, ar * bi + ai * br


def _cdiv(ar, ai, br, bi):
    """(a+bi)/(c+di) = (ac+bd + (bc-ad)i) / (c²+d²)"""
    denom = br * br + bi * bi + 1e-12
    return (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom


def _clog(r, i):
    """log(r + i·i) = 0.5·log(r²+i²) + i·atan2(i, r)"""
    mag_sq = r * r + i * i + 1e-12
    return 0.5 * torch.log(mag_sq), torch.atan2(i, r)


def _cexp(r, i):
    """exp(r + i·i) = exp(r)·cos(i) + i·exp(r)·sin(i)"""
    e = torch.exp(r)
    return e * torch.cos(i), e * torch.sin(i)


# ═══════════════════════════════════════════════════════════════════
# Real-arithmetic SSM discretization + parallel scan
# ═══════════════════════════════════════════════════════════════════

def _discretize_trapezoidal_real(
    log_a_real: Tensor,
    a_imag: Tensor,
    dt: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Exponential-trapezoidal (bilinear) discretization, pure real arithmetic.

    Returns A_d_real, A_d_imag, B_d_real, B_d_imag — all (B,T,H).
    """
    a_r = -torch.exp(log_a_real)   # (H,)  — alpha < 0 ensures stability
    hd_r = dt * a_r / 2             # (B,T,H)
    hd_i = dt * a_imag / 2

    num_r = 1 + hd_r
    num_i = hd_i
    den_r = 1 - hd_r
    den_i = -hd_i

    A_d_r, A_d_i = _cdiv(num_r, num_i, den_r, den_i)
    B_d_r, B_d_i = _cdiv(dt, torch.zeros_like(dt), den_r, den_i)
    return A_d_r, A_d_i, B_d_r, B_d_i


def _ssm_scan_parallel_real(
    x_r8: Tensor,           # (B, T, R)
    A_d_r: Tensor,           # (B, T, H)
    A_d_i: Tensor,           # (B, T, H)
    B_d_r: Tensor,           # (B, T, H)
    B_d_i: Tensor,           # (B, T, H)
    dt: Tensor,              # (B, T, H)  (unused in scan, used only in discretization)
    B_weight: Tensor,        # (N*2, R)
    B_bias: Tensor,          # (N*2,)
    C_weight: Tensor,        # (head_dim, N*2)
    H: int,
    N: int,
    head_dim: int,
) -> Tensor:
    """
    Parallel scan (Heinsen log-cumsum), pure real arithmetic.
    Returns (B, T, H, head_dim).
    """
    B_batch, T, R = x_r8.shape

    # B projection — mimic self.B_proj: Linear(R, N*2)
    bc = F.linear(x_r8, B_weight, B_bias)        # (B, T, N*2)
    Bv_r, Bv_i = bc.chunk(2, dim=-1)              # each (B, T, N)

    # b_input = B_d * B_vec  [complex multiply, broadcast H×N]
    Bdr = B_d_r.unsqueeze(-1); Bdi = B_d_i.unsqueeze(-1)  # (B,T,H,1)
    Bvr = Bv_r.unsqueeze(2);  Bvi = Bv_i.unsqueeze(2)    # (B,T,1,N)
    bi_r, bi_i = _cmul(Bdr, Bdi, Bvr, Bvi)               # (B,T,H,N)

    # log_A = log(A_d)
    lAr, lAi = _clog(A_d_r, A_d_i)               # (B,T,H)

    # Prefix products via cumsum(log)
    lA_cum_r = torch.cumsum(lAr, dim=1)
    lA_cum_i = torch.cumsum(lAi, dim=1)

    # inv_prefix = exp(-cumsum), fwd_prefix = exp(+cumsum)
    inv_r, inv_i = _cexp(-lA_cum_r, -lA_cum_i)
    fwd_r, fwd_i = _cexp( lA_cum_r,  lA_cum_i)

    inv_r = inv_r.unsqueeze(-1); inv_i = inv_i.unsqueeze(-1)  # (B,T,H,1)
    fwd_r = fwd_r.unsqueeze(-1); fwd_i = fwd_i.unsqueeze(-1)

    # b_normalized = b_input * inv_prefix
    bn_r, bn_i = _cmul(bi_r, bi_i, inv_r, inv_i)   # (B,T,H,N)

    # h_normalized = cumsum(b_normalized)
    hn_r = torch.cumsum(bn_r, dim=1)
    hn_i = torch.cumsum(bn_i, dim=1)

    # h = fwd_prefix * h_normalized
    h_r, h_i = _cmul(fwd_r, fwd_i, hn_r, hn_i)     # (B,T,H,N)

    # Readout: stack real+imag → linear
    h_cat = torch.cat([h_r, h_i], dim=-1)           # (B,T,H,N*2)
    out = F.linear(h_cat, C_weight)                  # (B,T,H,head_dim)
    return out


# ═══════════════════════════════════════════════════════════════════
# Per-layer real-arithmetic Mamba forward
# ═══════════════════════════════════════════════════════════════════

def _mamba_layer_forward_real(
    x: Tensor,
    x_r8: Tensor,
    cond: Tensor | None,
    # Parameters (extracted from trained layer)
    log_a_real: Tensor,          # (H,)
    a_imag: Tensor,              # (H,)
    B_weight: Tensor,            # (N*2, R)
    B_bias: Tensor | None,       # (N*2,)
    C_weight: Tensor,            # (head_dim, N*2)
    dt_proj_weight: Tensor,      # (H, D)
    dt_proj_bias: Tensor | None, # (H,)
    dt_bias: Tensor,             # (H,)
    out_proj_weight: Tensor,     # (D, D)
    out_proj_bias: Tensor | None, # (D,)
    norm_weight: Tensor,         # (D,)
    norm_bias: Tensor,           # (D,)
    film_proj_weight: Tensor,    # (D*2, d_cond)
    film_proj_bias: Tensor | None,  # (D*2,)
    dt_film_proj_weight: Tensor, # (H*2, d_cond)
    dt_film_proj_bias: Tensor | None,  # (H*2,)
    H: int,
    N: int,
    D: int,
    head_dim: int,
    dt_min: float,
    dt_max: float,
) -> Tensor:
    residual = x
    B_batch, T, _ = x.shape

    # FiLM conditioning
    if cond is not None:
        gb = F.linear(cond, film_proj_weight, film_proj_bias)
        gamma, beta = gb.chunk(2, dim=-1)
        x = x * (1 + gamma) + beta

    # dt projection
    dt_raw = F.linear(x, dt_proj_weight, dt_proj_bias)

    if cond is not None:
        dt_gb = F.linear(cond, dt_film_proj_weight, dt_film_proj_bias)
        dt_gamma, dt_beta = dt_gb.chunk(2, dim=-1)
        dt_raw = dt_raw * (1 + dt_gamma) + dt_beta

    dt = F.softplus(dt_raw + dt_bias)
    dt = dt.clamp(min=dt_min, max=dt_max)

    # Discretize (force fp32 for precision)
    A_d_r, A_d_i, B_d_r, B_d_i = _discretize_trapezoidal_real(
        log_a_real.float(), a_imag.float(), dt.float()
    )

    # SSM scan (force fp32 inside)
    out = _ssm_scan_parallel_real(
        x_r8.float(), A_d_r, A_d_i, B_d_r, B_d_i, dt.float(),
        B_weight.float(), B_bias.float() if B_bias is not None else torch.zeros(1, device=x_r8.device),
        C_weight.float(), H, N, head_dim,
    )

    out = out.to(x.dtype).reshape(B_batch, T, D)
    out = F.linear(out, out_proj_weight, out_proj_bias)
    out = F.dropout(out, p=0.0, training=False)  # eval mode

    return F.layer_norm(out + residual, (D,), norm_weight, norm_bias, 1e-5)


# ═══════════════════════════════════════════════════════════════════
# ONNX Export Wrapper
# ═══════════════════════════════════════════════════════════════════

class MambaFilterONNX(nn.Module):
    """
    Self-contained ONNX-traceable wrapper.

    Copies all weights from a trained PerTrackProcessor and reimplements
    the forward pass with pure real arithmetic for the Mamba SSM layers.
    """

    def __init__(self, trained_model: nn.Module, cfg: TrackProcessorConfig):
        super().__init__()
        self.cfg = cfg
        m = trained_model

        # ── R8 extractor ──
        self.r8_log_centers    = nn.Parameter(m.r8_extractor.log_centers.clone())
        self.r8_log_bandwidths = nn.Parameter(m.r8_extractor.log_bandwidths.clone())
        self.r8_input_proj_0_weight = nn.Parameter(m.r8_extractor.input_proj[0].weight.clone())
        self.r8_input_proj_0_bias   = nn.Parameter(m.r8_extractor.input_proj[0].bias.clone())
        self.r8_input_proj_1_weight = nn.Parameter(m.r8_extractor.input_proj[1].weight.clone())
        self.r8_input_proj_1_bias   = nn.Parameter(m.r8_extractor.input_proj[1].bias.clone())

        self.register_buffer("r8_transient_conv_w",
                             m.r8_extractor.transient_conv.weight.clone().detach())

        # ── Mel projection ──
        self.register_buffer("mel_weight", m.mel_proj.weight.clone().detach())

        # ── Semantic encoder conv stem (conv[0], gelu[1], conv[2], layernorm[3]) ──
        self.sem_conv0_weight = nn.Parameter(m.semantic_enc.conv_stem[0].weight.clone())
        self.sem_conv0_bias   = nn.Parameter(m.semantic_enc.conv_stem[0].bias.clone())
        self.sem_conv2_weight = nn.Parameter(m.semantic_enc.conv_stem[2].weight.clone())
        self.sem_conv2_bias   = nn.Parameter(m.semantic_enc.conv_stem[2].bias.clone())
        self.sem_norm_weight  = nn.Parameter(m.semantic_enc.conv_stem[3].weight.clone())
        self.sem_norm_bias    = nn.Parameter(m.semantic_enc.conv_stem[3].bias.clone())

        self.sem_proj_up_weight   = nn.Parameter(m.semantic_enc.proj_up.weight.clone())
        self.sem_proj_up_bias     = nn.Parameter(m.semantic_enc.proj_up.bias.clone())
        self.sem_proj_down_weight = nn.Parameter(m.semantic_enc.proj_down.weight.clone())
        self.sem_proj_down_bias   = nn.Parameter(m.semantic_enc.proj_down.bias.clone())

        self.sem_proj_head_0_weight = nn.Parameter(
            m.semantic_enc.proj_head[0].weight.clone())
        self.sem_proj_head_0_bias   = nn.Parameter(
            m.semantic_enc.proj_head[0].bias.clone())
        self.sem_proj_head_2_weight = nn.Parameter(
            m.semantic_enc.proj_head[2].weight.clone())
        self.sem_proj_head_2_bias   = nn.Parameter(
            m.semantic_enc.proj_head[2].bias.clone())

        # ── Semantic encoder Mamba layer (1 in laptop, 2 in cloud) ──
        assert len(m.semantic_enc.mamba_layers) <= 2, \
            f"Expected ≤2 semantic mamba layers, got {len(m.semantic_enc.mamba_layers)}"
        self.num_sem_mamba = len(m.semantic_enc.mamba_layers)
        for i in range(self.num_sem_mamba):
            self._register_mamba_layer(f"sem_mamba_{i}",
                                       m.semantic_enc.mamba_layers[i])

        # ── F0 conditioner (proj[0]=Linear, proj[1]=GELU, proj[2]=Linear, proj[3]=LayerNorm) ──
        self.f0_lin0_weight = nn.Parameter(m.f0_conditioner.proj[0].weight.clone())
        self.f0_lin0_bias   = nn.Parameter(m.f0_conditioner.proj[0].bias.clone())
        self.f0_lin1_weight = nn.Parameter(m.f0_conditioner.proj[2].weight.clone())
        self.f0_lin1_bias   = nn.Parameter(m.f0_conditioner.proj[2].bias.clone())
        self.f0_norm_weight = nn.Parameter(m.f0_conditioner.proj[3].weight.clone())
        self.f0_norm_bias   = nn.Parameter(m.f0_conditioner.proj[3].bias.clone())

        # ── Conditioning MLP (cond_mlp[0]=Linear, [1]=GELU, [2]=Linear, [3]=LayerNorm) ──
        self.cmlp_lin0_weight = nn.Parameter(m.cond_mlp[0].weight.clone())
        self.cmlp_lin0_bias   = nn.Parameter(m.cond_mlp[0].bias.clone())
        self.cmlp_lin1_weight = nn.Parameter(m.cond_mlp[2].weight.clone())
        self.cmlp_lin1_bias   = nn.Parameter(m.cond_mlp[2].bias.clone())
        self.cmlp_norm_weight = nn.Parameter(m.cond_mlp[3].weight.clone())
        self.cmlp_norm_bias   = nn.Parameter(m.cond_mlp[3].bias.clone())

        # ── Main Mamba layers (4) ──
        self._register_mamba_layer("mamba_0", m.mamba_layers[0])
        self._register_mamba_layer("mamba_1", m.mamba_layers[1])
        self._register_mamba_layer("mamba_2", m.mamba_layers[2])
        self._register_mamba_layer("mamba_3", m.mamba_layers[3])

        # ── Decoder ──
        self.dec_0_weight = nn.Parameter(m.decoder[0].weight.clone())
        self.dec_0_bias   = nn.Parameter(m.decoder[0].bias.clone())
        self.dec_2_weight = nn.Parameter(m.decoder[2].weight.clone())
        self.dec_2_bias   = nn.Parameter(m.decoder[2].bias.clone())

        # ── Buffers ──
        self.register_buffer("freqs",
                             torch.linspace(0, cfg.sample_rate / 2, cfg.n_fft // 2 + 1))
        self.log_max_freq = math.log(cfg.sample_rate / 2)
        self.log_dt_min   = math.log(cfg.dt_min)
        self.log_dt_max   = math.log(cfg.dt_max)

    def _register_mamba_layer(self, prefix: str, layer: nn.Module):
        """Register all parameters from a Mamba3MIMOLayer."""
        # Direct parameters
        for pname in ["log_a_real", "a_imag", "dt_bias"]:
            val = getattr(layer, pname)
            safe = pname  # no dots
            self.register_parameter(f"{prefix}_{safe}",
                                    nn.Parameter(val.clone()))

        # Sub-module parameters: B_proj, C_proj, dt_proj, out_proj, norm, film, dt_film
        for mod_name, suffix in [("B_proj", "B_proj"),
                                  ("C_proj", "C_proj"),
                                  ("dt_proj", "dt_proj"),
                                  ("out_proj", "out_proj"),
                                  ("norm", "norm"),
                                  ("film_proj", "film_proj"),
                                  ("dt_film_proj", "dt_film_proj")]:
            mod = getattr(layer, mod_name)
            self.register_parameter(f"{prefix}_{suffix}_weight",
                                    nn.Parameter(mod.weight.clone()))
            if mod.bias is not None:
                self.register_parameter(f"{prefix}_{suffix}_bias",
                                        nn.Parameter(mod.bias.clone()))

    # ── Forward components ──

    def _compute_stft(self, wav: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        window = torch.hann_window(cfg.n_fft, device=wav.device)
        w = wav.squeeze(1) if wav.dim() == 3 else wav
        stft = torch.stft(
            w, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            window=window, return_complex=True, normalized=False,
        )
        stft = stft.transpose(1, 2)
        return stft.abs().pow(2), stft

    def _compute_mel(self, mag_sq: Tensor) -> Tensor:
        projected = F.linear(mag_sq, self.mel_weight).clamp(min=0.0)
        return torch.log1p(projected)

    def _soft_band_energy(self, mag_sq: Tensor) -> Tensor:
        freqs = self.freqs.view(1, 1, -1, 1)
        centers = torch.exp(self.r8_log_centers).view(1, 1, 1, -1)
        bws = torch.exp(self.r8_log_bandwidths).view(1, 1, 1, -1)
        log_ratio = torch.log(freqs.clamp(min=1.0) / centers)
        weights = F.relu(1.0 - log_ratio.abs() / bws)
        weights = weights / (weights.sum(dim=2, keepdim=True).clamp(min=1e-8))
        energies = (mag_sq.unsqueeze(-1) * weights).sum(dim=2)
        return torch.log1p(energies)

    def _transient_density(self, waveform: Tensor, T: int) -> Tensor:
        envelope = waveform.abs().unsqueeze(1)
        diff = F.conv1d(envelope, self.r8_transient_conv_w, padding=2)
        diff = F.relu(diff)
        pooled = F.adaptive_avg_pool1d(diff, T)
        return pooled.transpose(1, 2)

    def _spectral_centroid(self, mag_sq: Tensor) -> Tensor:
        freqs = self.freqs.clamp(min=1.0)
        weights = mag_sq / (mag_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        centroid_hz = (weights * freqs).sum(dim=-1, keepdim=True)
        return torch.log(centroid_hz.clamp(min=1.0)) / self.log_max_freq

    def _extract_r8_raw(self, waveform: Tensor, mag_sq: Tensor) -> Tensor:
        band = self._soft_band_energy(mag_sq)
        T_frames = mag_sq.shape[1]
        trans = self._transient_density(waveform, T_frames)
        cent = self._spectral_centroid(mag_sq)
        return torch.cat([band, trans, cent], dim=-1)

    def _r8_input_proj(self, r8: Tensor) -> Tensor:
        x = F.linear(r8, self.r8_input_proj_0_weight, self.r8_input_proj_0_bias)
        x = F.layer_norm(x, (self.cfg.d_model,),
                         self.r8_input_proj_1_weight, self.r8_input_proj_1_bias, 1e-5)
        return F.gelu(x)

    def _loudness_features(self, waveform: Tensor, T: int) -> Tensor:
        B = waveform.shape[0]
        L = waveform.shape[-1]
        frame_size = (L + T - 1) // T
        pad_amount = T * frame_size - L
        if pad_amount > 0:
            waveform = F.pad(waveform, (0, pad_amount))
        frames = waveform.reshape(B, T, frame_size)
        rms = frames.pow(2).mean(-1, keepdim=True).sqrt()
        peak = frames.abs().max(-1, keepdim=True).values
        crest = (peak / (rms + 1e-8)).clamp(min=1e-8).log()
        lufs = 10 * torch.log10(rms.pow(2) + 1e-8)
        return torch.cat([rms, peak, crest, lufs], dim=-1)

    def _semantic_enc(self, mel: Tensor, r8_raw: Tensor,
                      return_sequence: bool = False) -> Tensor:
        cfg = self.cfg

        # Conv stem: [0] Conv1d, [1] GELU, [2] Conv1d, [3] LayerNorm
        x = mel.transpose(1, 2)
        x = F.conv1d(x, self.sem_conv0_weight, self.sem_conv0_bias, padding=1)
        x = F.gelu(x)
        x = F.conv1d(x, self.sem_conv2_weight, self.sem_conv2_bias, padding=1)
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (cfg.d_semantic,),
                         self.sem_norm_weight, self.sem_norm_bias, 1e-5)

        x = F.linear(x, self.sem_proj_up_weight, self.sem_proj_up_bias)

        # Semantic mamba layer(s)
        for i in range(self.num_sem_mamba):
            pfx = f"sem_mamba_{i}"
            x = _mamba_layer_forward_real(
                x, r8_raw, None,
                getattr(self, f"{pfx}_log_a_real"),
                getattr(self, f"{pfx}_a_imag"),
                getattr(self, f"{pfx}_B_proj_weight"),
                getattr(self, f"{pfx}_B_proj_bias", None),
                getattr(self, f"{pfx}_C_proj_weight"),
                getattr(self, f"{pfx}_dt_proj_weight"),
                getattr(self, f"{pfx}_dt_proj_bias", None),
                getattr(self, f"{pfx}_dt_bias"),
                getattr(self, f"{pfx}_out_proj_weight"),
                getattr(self, f"{pfx}_out_proj_bias", None),
                getattr(self, f"{pfx}_norm_weight"),
                getattr(self, f"{pfx}_norm_bias"),
                getattr(self, f"{pfx}_film_proj_weight"),
                getattr(self, f"{pfx}_film_proj_bias", None),
                getattr(self, f"{pfx}_dt_film_proj_weight"),
                getattr(self, f"{pfx}_dt_film_proj_bias", None),
                cfg.n_heads, cfg.d_state, cfg.d_model,
                cfg.d_model // cfg.n_heads,
                cfg.dt_min, cfg.dt_max,
            )

        x = F.linear(x, self.sem_proj_down_weight, self.sem_proj_down_bias)

        if return_sequence:
            return x
        pooled = x.mean(dim=1)
        h = F.linear(pooled, self.sem_proj_head_0_weight, self.sem_proj_head_0_bias)
        h = F.gelu(h)
        h = F.linear(h, self.sem_proj_head_2_weight, self.sem_proj_head_2_bias)
        return F.normalize(h, dim=-1)

    def _f0_condition(self, crepe: Tensor) -> Tensor:
        x = F.linear(crepe, self.f0_lin0_weight, self.f0_lin0_bias)
        x = F.gelu(x)
        x = F.linear(x, self.f0_lin1_weight, self.f0_lin1_bias)
        x = F.layer_norm(x, (self.cfg.d_f0,),
                         self.f0_norm_weight, self.f0_norm_bias, 1e-5)
        return x

    def _cond_mlp(self, cond_in: Tensor) -> Tensor:
        x = F.linear(cond_in, self.cmlp_lin0_weight, self.cmlp_lin0_bias)
        x = F.gelu(x)
        x = F.linear(x, self.cmlp_lin1_weight, self.cmlp_lin1_bias)
        x = F.layer_norm(x, (self.cfg.d_cond,),
                         self.cmlp_norm_weight, self.cmlp_norm_bias, 1e-5)
        return x

    def _main_mamba(self, x: Tensor, r8_raw: Tensor, cond: Tensor) -> Tensor:
        cfg = self.cfg
        prefixes = ["mamba_0", "mamba_1", "mamba_2", "mamba_3"]
        for pfx in prefixes:
            x = _mamba_layer_forward_real(
                x, r8_raw, cond,
                getattr(self, f"{pfx}_log_a_real"),
                getattr(self, f"{pfx}_a_imag"),
                getattr(self, f"{pfx}_B_proj_weight"),
                getattr(self, f"{pfx}_B_proj_bias", None),
                getattr(self, f"{pfx}_C_proj_weight"),
                getattr(self, f"{pfx}_dt_proj_weight"),
                getattr(self, f"{pfx}_dt_proj_bias", None),
                getattr(self, f"{pfx}_dt_bias"),
                getattr(self, f"{pfx}_out_proj_weight"),
                getattr(self, f"{pfx}_out_proj_bias", None),
                getattr(self, f"{pfx}_norm_weight"),
                getattr(self, f"{pfx}_norm_bias"),
                getattr(self, f"{pfx}_film_proj_weight"),
                getattr(self, f"{pfx}_film_proj_bias", None),
                getattr(self, f"{pfx}_dt_film_proj_weight"),
                getattr(self, f"{pfx}_dt_film_proj_bias", None),
                cfg.n_heads, cfg.d_state, cfg.d_model,
                cfg.d_model // cfg.n_heads,
                cfg.dt_min, cfg.dt_max,
            )
        return x

    def _decode(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.dec_0_weight, self.dec_0_bias)
        x = F.gelu(x)
        x = F.linear(x, self.dec_2_weight, self.dec_2_bias)
        return x

    def forward(self, waveform: Tensor, crepe_probs: Tensor) -> Tensor:
        """
        Args:
            waveform: (B, T_samples) mono audio
            crepe_probs: (B, T_frames, 360) f0 probabilities (zeros OK)
        Returns:
            processed_stft: (B, T_frames, F_bins*2) predicted complex STFT
                First half = real, second half = imag.
                F_bins = n_fft // 2 + 1 = 513.
                iSTFT must be performed by the caller.
        """
        cfg = self.cfg

        # STFT
        mag_sq, stft_in = self._compute_stft(waveform)
        T = mag_sq.shape[1]

        # Features
        r8 = self._r8_input_proj(self._extract_r8_raw(waveform, mag_sq))
        r8_raw = self._extract_r8_raw(waveform, mag_sq)
        mel = self._compute_mel(mag_sq)

        # Semantic embedding (sequence mode)
        sem_emb = self._semantic_enc(mel, r8_raw, return_sequence=True)

        # F0 conditioning
        f0_emb = self._f0_condition(crepe_probs)

        # Cross-track context (zeros)
        cross_ctx = torch.zeros(waveform.shape[0], T, cfg.d_cross,
                                device=waveform.device)

        # Loudness
        loudness = self._loudness_features(waveform, T)

        # Conditioning vector
        cond_in = torch.cat([sem_emb, f0_emb, cross_ctx, loudness], dim=-1)
        cond = self._cond_mlp(cond_in)

        # Main Mamba layers
        x = r8
        x = self._main_mamba(x, r8_raw, cond)

        # Decoder → predicted STFT (real+imag stacked along last dim)
        stft_pred = self._decode(x)  # (B, T_frames, F_bins*2)

        return stft_pred


import math  # noqa: E402 (used in __init__)


def _get_ort_session_options():
    """ORT session options with graph optimization for audio workloads."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_profiling = False
    return opts


# ═══════════════════════════════════════════════════════════════════
# Post-export patch: fix int32/int64 type mismatches
# ═══════════════════════════════════════════════════════════════════

def _patch_scatternd_int32(onnx_path: str):
    """
    Fix int32/int64 type mismatches from PyTorch dynamo ONNX export.
    1. Downgrade IR version to 9 (ORT 1.17.1 doesn't support IR v10)
    2. Convert all int32 initializers → int64
    3. Change Cast(to=6/int32) → Cast(to=7/int64)
    4. Convert metadata types int32 → int64
    5. Shape inference to propagate changes
    """
    import onnx
    import numpy as np

    model = onnx.load(onnx_path)

    # 0. Downgrade IR version to 9 (ORT 1.17.1 only supports IR ≤ v9)
    if model.ir_version > 9:
        print(f"  downgrading IR version from {model.ir_version} → 9")
        model.ir_version = 9

    # 1. Convert int32 initializers → int64
    for init in list(model.graph.initializer):
        if init.data_type == 6:
            arr = onnx.numpy_helper.to_array(init)
            new_init = onnx.numpy_helper.from_array(
                arr.astype(np.int64), name=init.name)
            model.graph.initializer.remove(init)
            model.graph.initializer.append(new_init)

    # 2. Change Cast(to=6) → Cast(to=7) in all nodes
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 6:
                    attr.i = 7  # int32 → int64

    # 3. Convert metadata int32 → int64
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == 6:
            vi.type.tensor_type.elem_type = 7
    for io_list in [model.graph.input, model.graph.output]:
        for io in io_list:
            if io.type.tensor_type.elem_type == 6:
                io.type.tensor_type.elem_type = 7

    # 4. Shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    onnx.save(model, onnx_path)
    print("  patched int32→int64 (initializers + Cast nodes + metadata)")


# ═══════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════

def export(checkpoint_path: str, output_path: str, verify: bool = False):
    dev = torch.device("cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        cfg = TrackProcessorConfig()
    print(f"  step={ckpt.get('step')}, d_model={cfg.d_model}, "
          f"n_layers={cfg.n_layers}")

    # Build original model + load weights
    from model_laptop import PerTrackProcessor
    original = PerTrackProcessor(cfg).to(dev)
    original.load_state_dict(ckpt["model"])
    original.eval()
    print(f"  params: {sum(p.numel() for p in original.parameters()):,}")

    # Build ONNX wrapper + copy weights
    wrapper = MambaFilterONNX(original, cfg).to(dev)
    wrapper.eval()

    # Test input — keep short (max safe T_frames ≈ 500 before log-cumsum overflow)
    sr = cfg.sample_rate
    duration = 2  # seconds  (T_frames ≈ 345, well within stable range)
    n_samples = duration * sr
    # Use a real test input that exercises all frequencies
    t = torch.arange(n_samples, dtype=torch.float32) / sr
    test_wav = (
        0.3 * torch.sin(2 * math.pi * 80 * t) +
        0.3 * torch.sin(2 * math.pi * 440 * t) +
        0.2 * torch.sin(2 * math.pi * 2000 * t) +
        0.1 * torch.sin(2 * math.pi * 6000 * t)
    ).unsqueeze(0).to(dev)  # (1, T)

    T_frames = 1 + test_wav.shape[-1] // cfg.hop_length
    test_crepe = torch.zeros(1, T_frames, 360, device=dev)

    # Verify PyTorch↔ONNX parity before export
    print("\nVerifying ONNX wrapper vs PyTorch model (STFT output)...")
    with torch.no_grad():
        # PyTorch original
        mag_sq, stft_in = original._compute_stft(test_wav)
        T_frames_actual = mag_sq.shape[1]
        test_crepe2 = torch.zeros(1, T_frames_actual, 360, device=dev)
        orig_out = original(test_wav, test_crepe2)

        # PyTorch model outputs dict with 'output_stft' (complex) + decoder output
        # Get the raw decoder output by running through the model steps
        r8 = original.r8_extractor(test_wav, mag_sq)
        r8_raw = original._extract_r8_raw(test_wav, mag_sq)
        mel = original._compute_mel(mag_sq)
        sem_emb = original.semantic_enc(mel, r8_raw, return_sequence=True)
        f0_emb = original.f0_conditioner(test_crepe2)
        cross_ctx = torch.zeros(1, T_frames_actual, cfg.d_cross, device=dev)
        loudness = original._loudness_features(test_wav, T_frames_actual)
        cond_in = torch.cat([sem_emb, f0_emb, cross_ctx, loudness], dim=-1)
        cond = original.cond_mlp(cond_in)
        x = r8
        for layer in original.mamba_layers:
            x = layer(x, r8_raw, cond=cond)
        orig_stft = original.decoder(x)  # Raw decoder output (B, T, F_bins*2)

        # ONNX wrapper
        wrap_stft = wrapper(test_wav, test_crepe)

    diff = (orig_stft.float() - wrap_stft.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (orig_stft.float().abs() + 1e-8)).mean().item()

    has_nan = (max_diff != max_diff)  # NaN check
    print(f"  max  abs diff: {max_diff:.2e}")
    print(f"  mean abs diff: {mean_diff:.2e}")
    print(f"  mean rel diff: {rel_diff:.2e}")

    if has_nan or max_diff > 1e-3:
        if has_nan:
            print(f"\n  ERROR: NaN detected in ONNX wrapper output!")
        else:
            print(f"\n  WARNING: max diff {max_diff:.2e} > 1e-3 — "
                  f"ONNX output may differ audibly from PyTorch.")
        if not verify:
            print("  Continuing with export anyway (run with --verify to block).")
    elif max_diff > 1e-6:
        print(f"  OK: diff within FP32 tolerance (max {max_diff:.2e})")
    else:
        print(f"  Excellent: near bit-identical (max {max_diff:.2e})")

    if verify and (has_nan or max_diff > 1e-3):
        if has_nan:
            print("\nERROR: NaN detected. Aborting export.")
        else:
            print("\nERROR: Parity check failed. Aborting export.")
        sys.exit(1)

    # Export
    print(f"\nExporting to {output_path} ...")

    # Dynamic axes for variable-length input
    dynamic_axes = {
        "waveform": {1: "seq_len"},
        "crepe": {1: "n_frames"},
        "processed_stft": {1: "n_frames"},
    }

    torch.onnx.export(
        wrapper,
        (test_wav, test_crepe),
        output_path,
        input_names=["waveform", "crepe"],
        output_names=["processed_stft"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
    )

    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  exported: {output_path} ({size_mb:.1f} MB)")

    # Post-export patch: fix ScatterND int32→int64 bug
    _patch_scatternd_int32(output_path)

    # Quick ONNX Runtime verification
    print("\nVerifying ONNX Runtime inference...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping ORT verification")
        print(f"\nDone. Model exported to {output_path}")
        return

    sess = ort.InferenceSession(
        output_path,
        providers=['CPUExecutionProvider'],
        sess_options=_get_ort_session_options(),
    )
    onnx_out = sess.run(["processed_stft"], {
        "waveform": test_wav.cpu().numpy().astype(np.float32),
        "crepe": test_crepe.cpu().numpy().astype(np.float32),
    })[0]

    onnx_out_t = torch.from_numpy(onnx_out).float()
    ort_diff = (orig_stft.float() - onnx_out_t).abs()
    print(f"  ORT vs PyTorch max diff: {ort_diff.max().item():.2e}")
    print(f"  ORT vs PyTorch mean diff: {ort_diff.mean().item():.2e}")

    print(f"\nDone. Model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Mamba-3 filter bank to ONNX"
    )
    parser.add_argument("--model", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--output", default="mamba_filter.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--verify", action="store_true",
                        help="Abort if PyTorch/ONNX parity > 1e-3")
    args = parser.parse_args()

    export(args.model, args.output, args.verify)
