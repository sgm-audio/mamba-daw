"""
Mamba-3 MIMO R=8 Per-Track Adaptive Filter Bank — LAPTOP EDITION
=================================================================
Lean version for CPU / 8GB shared-iGPU training. Same architecture
as model.py but with:
  - Smaller defaults (d_model=128, n_layers=4, n_heads=8, d_state=16)
  - Single-resolution STFT path (no multi-res)
  - Parallel scan (Heinsen log-cumsum)
  - Checkpoint format compatible with full-size model.py (can swap configs)

Use with train_laptop.py.

Same checkpoints can be loaded by the full model.py if you ever return
to GPU training, as long as you pass the same TrackProcessorConfig.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


@dataclass
class TrackProcessorConfig:
    sample_rate: int = 44100
    n_fft: int = 1024            # smaller FFT (was 2048) — 4x less compute in feature extraction
    hop_length: int = 256        # smaller hop (was 512) — keeps T similar for shorter clips
    n_mels: int = 64             # smaller (was 128)

    band_boundaries_hz: tuple = (
        (20,   80),
        (80,   250),
        (250,  800),
        (800,  2500),
        (2500, 8000),
        (8000, 20000),
    )
    R: int = 8

    # LAPTOP DEFAULTS — much smaller than cloud config
    d_model: int = 128           # was 256 — 4x less parameters in linears
    d_state: int = 16            # was 32 — 4x less complex-state compute
    n_heads: int = 8             # was 16
    n_layers: int = 4            # was 6 — fewer mamba layers
    chunk_size: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1

    d_semantic: int = 64         # was 128
    d_f0: int = 16               # was 32
    d_cross: int = 32            # was 64
    d_cond: int = 64             # was 128

    max_tracks: int = 8
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# R=8 Structured Feature Extractor
# ---------------------------------------------------------------------------

class R8FeatureExtractor(nn.Module):
    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.cfg = cfg
        sr = cfg.sample_rate
        n_fft = cfg.n_fft

        self.register_buffer(
            "freqs",
            torch.linspace(0, sr / 2, n_fft // 2 + 1)
        )

        centers_hz = [math.sqrt(lo * hi) for (lo, hi) in cfg.band_boundaries_hz]
        self.log_centers = nn.Parameter(
            torch.log(torch.tensor(centers_hz, dtype=torch.float32))
        )
        self.log_bandwidths = nn.Parameter(torch.full((6,), math.log(0.5)))

        self.transient_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.log_max_freq = math.log(sr / 2)

        self.input_proj = nn.Sequential(
            nn.Linear(cfg.R, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

    def _soft_band_energy(self, mag_sq: Tensor) -> Tensor:
        freqs = self.freqs.view(1, 1, -1, 1)
        centers = torch.exp(self.log_centers).view(1, 1, 1, -1)
        bws     = torch.exp(self.log_bandwidths).view(1, 1, 1, -1)

        log_ratio = torch.log(freqs.clamp(min=1.0) / centers)
        weights = F.relu(1.0 - log_ratio.abs() / bws)
        weights = weights / (weights.sum(dim=2, keepdim=True).clamp(min=1e-8))

        energies = (mag_sq.unsqueeze(-1) * weights).sum(dim=2)
        return torch.log1p(energies)

    def _transient_density(self, waveform: Tensor, T: int) -> Tensor:
        envelope = waveform.abs().unsqueeze(1)
        diff = self.transient_conv(envelope)
        diff = F.relu(diff)
        pooled = F.adaptive_avg_pool1d(diff, T)
        return pooled.transpose(1, 2)

    def _spectral_centroid(self, mag_sq: Tensor) -> Tensor:
        freqs = self.freqs.clamp(min=1.0)
        weights = mag_sq / (mag_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        centroid_hz = (weights * freqs).sum(dim=-1, keepdim=True)
        return torch.log(centroid_hz.clamp(min=1.0)) / self.log_max_freq

    def forward(self, waveform: Tensor, stft_mag_sq: Tensor) -> Tensor:
        band_energy  = self._soft_band_energy(stft_mag_sq)
        T_frames = stft_mag_sq.shape[1]
        transient    = self._transient_density(waveform, T_frames)
        centroid     = self._spectral_centroid(stft_mag_sq)

        r8 = torch.cat([band_energy, transient, centroid], dim=-1)
        return self.input_proj(r8)


# ---------------------------------------------------------------------------
# Mamba3 MIMO Layer (PARALLEL SCAN)
# ---------------------------------------------------------------------------

class Mamba3MIMOLayer(nn.Module):
    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.H = cfg.n_heads
        self.N = cfg.d_state
        self.d = cfg.d_model
        self.R = cfg.R

        head_dim = cfg.d_model // cfg.n_heads
        assert head_dim * cfg.n_heads == cfg.d_model
        self.head_dim = head_dim

        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model * 2 + cfg.n_heads, bias=False)

        log_a_real = torch.zeros(cfg.n_heads)
        a_imag = torch.linspace(0, math.pi, cfg.n_heads + 1)[:-1]
        self.log_a_real = nn.Parameter(log_a_real)
        self.a_imag     = nn.Parameter(a_imag)

        self.B_proj = nn.Linear(cfg.R, cfg.d_state * 2, bias=True)
        self.C_proj = nn.Linear(cfg.d_state * 2, self.head_dim, bias=False)

        self.dt_proj  = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        self.dt_bias  = nn.Parameter(torch.full((cfg.n_heads,), -4.0))
        self.log_dt_min = math.log(cfg.dt_min)
        self.log_dt_max = math.log(cfg.dt_max)

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm     = nn.LayerNorm(cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)

        self.film_proj    = nn.Linear(cfg.d_cond, cfg.d_model * 2, bias=True)
        self.dt_film_proj = nn.Linear(cfg.d_cond, cfg.n_heads * 2, bias=True)
        with torch.no_grad():
            self.film_proj.weight.zero_()
            self.film_proj.bias.zero_()
            self.dt_film_proj.weight.zero_()
            self.dt_film_proj.bias.zero_()

        self.gradient_checkpointing = False

    def _discretize_trapezoidal(self, a_real, a_imag, dt):
        half_dt_a = torch.complex(
            dt * (-torch.exp(a_real)) / 2,
            dt * a_imag / 2
        )
        numerator   = 1 + half_dt_a
        denominator = 1 - half_dt_a
        A_d = numerator / denominator
        B_d_scale = torch.complex(dt, torch.zeros_like(dt)) / denominator
        return A_d, B_d_scale

    def _ssm_scan_parallel(self, x_r8, A_d, B_d, dt):
        """Parallel scan via Heinsen log-cumsum. Pure PyTorch."""
        B_batch, T, R = x_r8.shape
        H, N = self.H, self.N

        bc = self.B_proj(x_r8)
        B_real, B_imag = bc.chunk(2, dim=-1)
        B_vec = torch.complex(B_real, B_imag)

        b_input = B_d.unsqueeze(-1) * B_vec.unsqueeze(2)

        log_A = torch.log(A_d + 1e-12)
        log_A_cumsum = torch.cumsum(log_A, dim=1)

        inv_prefix = torch.exp(-log_A_cumsum).unsqueeze(-1)
        fwd_prefix = torch.exp( log_A_cumsum).unsqueeze(-1)

        b_normalized = b_input * inv_prefix
        h_normalized = torch.cumsum(b_normalized, dim=1)

        h = fwd_prefix * h_normalized

        h_cat = torch.cat([h.real, h.imag], dim=-1)
        out = self.C_proj(h_cat)
        return out

    def forward(self, x, x_r8, cond=None):
        B, T, D = x.shape
        residual = x

        if cond is not None:
            gamma_beta = self.film_proj(cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        dt_raw = self.dt_proj(x)

        if cond is not None:
            dt_gamma_beta = self.dt_film_proj(cond)
            dt_gamma, dt_beta = dt_gamma_beta.chunk(2, dim=-1)
            dt_raw = dt_raw * (1 + dt_gamma) + dt_beta

        dt = F.softplus(dt_raw + self.dt_bias)
        dt = dt.clamp(
            min=math.exp(self.log_dt_min),
            max=math.exp(self.log_dt_max)
        )

        # Always fp32 for complex math
        with torch.autocast(device_type=x.device.type, enabled=False):
            dt_fp32 = dt.float()
            x_r8_fp32 = x_r8.float()
            A_d, B_d_scale = self._discretize_trapezoidal(
                self.log_a_real.float(), self.a_imag.float(), dt_fp32
            )
            out = self._ssm_scan_parallel(x_r8_fp32, A_d, B_d_scale, dt_fp32)

        out = out.to(x.dtype)
        out = out.reshape(B, T, D)

        out = self.out_proj(out)
        out = self.drop(out)
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Cross-Track Attention Aggregator (kept, used in phase 3 only)
# ---------------------------------------------------------------------------

class CrossTrackAggregator(nn.Module):
    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_semantic,
            num_heads=2,             # was 4 — smaller for laptop
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(cfg.d_semantic, cfg.d_cross)
        self.norm = nn.LayerNorm(cfg.d_cross)

    def forward(self, target_emb, other_embs):
        B, T, D = target_emb.shape
        K = other_embs.shape[1]

        q  = target_emb.reshape(B * T, 1, D)
        kv = other_embs.permute(0, 2, 1, 3).reshape(B * T, K, D)

        ctx, _ = self.attn(q, kv, kv)
        ctx = ctx.squeeze(1).reshape(B, T, D)
        return self.norm(self.proj(ctx))


# ---------------------------------------------------------------------------
# Semantic Encoder
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.d_semantic, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.d_semantic, cfg.d_semantic, kernel_size=3, padding=1),
            nn.LayerNorm(cfg.d_semantic),
        )
        self.proj_up = nn.Linear(cfg.d_semantic, cfg.d_model)
        self.proj_down = nn.Linear(cfg.d_model, cfg.d_semantic)

        # Only 1 Mamba layer in encoder (was 2)
        self.mamba_layers = nn.ModuleList([
            Mamba3MIMOLayer(cfg) for _ in range(1)
        ])
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.d_semantic, cfg.d_semantic),
            nn.GELU(),
            nn.Linear(cfg.d_semantic, cfg.d_semantic),
        )

    def forward(self, mel, r8, return_sequence=False):
        x = mel.transpose(1, 2)
        x = self.conv_stem[0](x)
        x = self.conv_stem[1](x)
        x = self.conv_stem[2](x)
        x = x.transpose(1, 2)
        x = self.conv_stem[3](x)

        x = self.proj_up(x)

        for layer in self.mamba_layers:
            x = layer(x, r8)

        x = self.proj_down(x)

        if return_sequence:
            return x

        pooled = x.mean(dim=1)
        return F.normalize(self.proj_head(pooled), dim=-1)


# ---------------------------------------------------------------------------
# F0 Conditioning Module
# ---------------------------------------------------------------------------

class F0Conditioner(nn.Module):
    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(360, 64),
            nn.GELU(),
            nn.Linear(64, cfg.d_f0),
            nn.LayerNorm(cfg.d_f0),
        )

    def forward(self, crepe_probs):
        return self.proj(crepe_probs)


# ---------------------------------------------------------------------------
# Full Per-Track Processor (laptop edition)
# ---------------------------------------------------------------------------

class PerTrackProcessor(nn.Module):
    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.cfg = cfg

        self.r8_extractor   = R8FeatureExtractor(cfg)
        self.semantic_enc   = SemanticEncoder(cfg)
        self.f0_conditioner = F0Conditioner(cfg)
        self.cross_track    = CrossTrackAggregator(cfg)

        cond_in = cfg.d_semantic + cfg.d_f0 + cfg.d_cross + 4
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cfg.d_cond),
            nn.GELU(),
            nn.Linear(cfg.d_cond, cfg.d_cond),
            nn.LayerNorm(cfg.d_cond),
        )

        self.mamba_layers = nn.ModuleList([
            Mamba3MIMOLayer(cfg) for _ in range(cfg.n_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, 2 * (cfg.n_fft // 2 + 1)),
        )

        F_bins = cfg.n_fft // 2 + 1
        self.mel_proj = nn.Linear(F_bins, cfg.n_mels, bias=False)
        with torch.no_grad():
            self.mel_proj.weight.abs_()

    def _compute_stft(self, waveform):
        cfg = self.cfg
        window = torch.hann_window(cfg.n_fft, device=waveform.device)
        w = waveform.squeeze(1) if waveform.dim() == 3 else waveform
        B = w.shape[0]
        MAX_FFT_BATCH = 8
        if B <= MAX_FFT_BATCH:
            stft = torch.stft(
                w, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                window=window, return_complex=True, normalized=False,
            )
        else:
            chunks = []
            for i in range(0, B, MAX_FFT_BATCH):
                c = torch.stft(
                    w[i:i+MAX_FFT_BATCH], n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length, window=window,
                    return_complex=True, normalized=False,
                )
                chunks.append(c)
            stft = torch.cat(chunks, dim=0)
        stft = stft.transpose(1, 2)
        return stft.abs().pow(2), stft

    def _compute_mel(self, mag_sq):
        projected = self.mel_proj(mag_sq).clamp(min=0.0)
        return torch.log1p(projected)

    def _loudness_features(self, waveform, T):
        B = waveform.shape[0]
        L = waveform.shape[-1]
        frame_size = (L + T - 1) // T
        pad_amount = T * frame_size - L
        if pad_amount > 0:
            waveform = F.pad(waveform, (0, pad_amount))
        frames = waveform.reshape(B, T, frame_size)

        rms    = frames.pow(2).mean(-1, keepdim=True).sqrt()
        peak   = frames.abs().max(-1, keepdim=True).values
        crest  = (peak / (rms + 1e-8)).clamp(min=1e-8).log()
        lufs   = 10 * torch.log10(rms.pow(2) + 1e-8)

        return torch.cat([rms, peak, crest, lufs], dim=-1)

    def forward(self, waveform, crepe_probs, other_embs=None):
        cfg = self.cfg

        mag_sq, stft_in = self._compute_stft(waveform)
        T = mag_sq.shape[1]

        r8  = self.r8_extractor(waveform, mag_sq)
        r8_raw = self._extract_r8_raw(waveform, mag_sq)
        mel = self._compute_mel(mag_sq)

        sem_emb = self.semantic_enc(mel, r8_raw, return_sequence=True)

        f0_emb = self.f0_conditioner(crepe_probs)

        if other_embs is not None:
            cross_ctx = self.cross_track(sem_emb, other_embs)
        else:
            cross_ctx = torch.zeros(
                waveform.shape[0], T, cfg.d_cross, device=waveform.device
            )

        loudness = self._loudness_features(waveform, T)
        cond_in  = torch.cat([sem_emb, f0_emb, cross_ctx, loudness], dim=-1)
        cond     = self.cond_mlp(cond_in)

        x = r8
        for layer in self.mamba_layers:
            x = layer(x, r8_raw, cond=cond)

        stft_pred_real = self.decoder(x)
        F_bins = cfg.n_fft // 2 + 1
        stft_real = stft_pred_real[..., :F_bins].float()
        stft_imag = stft_pred_real[..., F_bins:2*F_bins].float()
        stft_out  = torch.complex(stft_real, stft_imag)

        return {
            'output_stft': stft_out,
            'track_emb':   sem_emb,
            'r8':          r8_raw,
        }

    def _extract_r8_raw(self, waveform, mag_sq):
        ext = self.r8_extractor
        band_energy = ext._soft_band_energy(mag_sq)
        T_frames = mag_sq.shape[1]
        transient   = ext._transient_density(waveform, T_frames)
        centroid    = ext._spectral_centroid(mag_sq)
        return torch.cat([band_energy, transient, centroid], dim=-1)

    def waveform_from_stft(self, stft):
        cfg = self.cfg
        window = torch.hann_window(cfg.n_fft, device=stft.device)
        return torch.istft(
            stft.transpose(1, 2),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=window,
        )
