#!/usr/bin/env python3
"""Spectral + temporal parity analysis between two f32 waveforms (native vs ort).

Finer than whisper coverage: quantifies WHERE two audio signals diverge in time
and frequency, to localize a native-vs-reference parity gap.

  python3 spectral_temporal.py native.f32 ort.f32 [sample_rate]

Temporal: length ratio, raw cos, best cross-correlation lag + aligned cos, RMSE,
SNR(dB), amplitude-envelope correlation, per-window aligned cos (time profile).
Spectral: STFT log-magnitude cos (overall + per-frame profile), spectral-centroid
MAE, low/mid/high band energy ratios.
"""
import sys
import numpy as np


def stft_mag(x, n_fft=1024, hop=256):
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + max(0, (len(x) - n_fft) // hop)
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n)]) if n else np.zeros((0, n_fft))
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec).astype(np.float32)  # [frames, freq]


def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    m = min(len(a), len(b)); a, b = a[:m], b[:m]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / (d + 1e-12))


def best_lag(a, b, max_lag=512):
    a = a.astype(np.float64); b = b.astype(np.float64)
    m = min(len(a), len(b)); a, b = a[:m], b[:m]
    best, bl = -2.0, 0
    for lag in range(-max_lag, max_lag + 1, 4):
        if lag >= 0:
            c = cos(a[lag:], b[:len(b) - lag])
        else:
            c = cos(a[:len(a) + lag], b[-lag:])
        if c > best:
            best, bl = c, lag
    return bl, best


def main():
    na, nb = sys.argv[1], sys.argv[2]
    sr = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
    a = np.fromfile(na, dtype=np.float32)
    b = np.fromfile(nb, dtype=np.float32)
    m = min(len(a), len(b)); A, B = a[:m], b[:m]
    print(f"=== TEMPORAL ({na.split('/')[-1]} vs {nb.split('/')[-1]}, sr={sr}) ===")
    print(f"  len native={len(a)} ort={len(b)} ratio={len(a)/max(1,len(b)):.4f}")
    print(f"  raw cos            = {cos(A, B):.5f}")
    lag, alc = best_lag(A, B)
    print(f"  best-lag           = {lag} samples ({1000*lag/sr:+.2f} ms), aligned cos = {alc:.5f}")
    rmse = float(np.sqrt(np.mean((A - B) ** 2)))
    sig = float(np.mean(B ** 2)); noise = float(np.mean((A - B) ** 2))
    snr = 10 * np.log10(sig / (noise + 1e-12))
    print(f"  RMSE={rmse:.5f}  peak native={np.abs(a).max():.4f} ort={np.abs(b).max():.4f}  SNR={snr:.1f} dB")
    env_a = np.abs(A); env_b = np.abs(B)
    k = max(1, sr // 200)
    ea = np.convolve(env_a, np.ones(k) / k, 'same'); eb = np.convolve(env_b, np.ones(k) / k, 'same')
    print(f"  amplitude-envelope corr = {cos(ea - ea.mean(), eb - eb.mean()):.5f}")
    # time profile: aligned cos per 8 chunks
    nseg = 8; seg = m // nseg
    prof = [round(cos(A[i*seg:(i+1)*seg], B[i*seg:(i+1)*seg]), 3) for i in range(nseg)] if seg else []
    print(f"  per-eighth cos     = {prof}")

    print("=== SPECTRAL ===")
    Sa, Sb = stft_mag(A), stft_mag(B)
    f = min(len(Sa), len(Sb))
    Sa, Sb = Sa[:f], Sb[:f]
    la, lb = np.log1p(Sa), np.log1p(Sb)
    print(f"  STFT log-mag cos   = {cos(la, lb):.5f}  ({f} frames, {Sa.shape[1]} bins)")
    fps = [round(cos(la[i], lb[i]), 3) for i in range(0, f, max(1, f // 8))]
    print(f"  per-frame cos      = {fps}")
    freqs = np.fft.rfftfreq(1024, 1 / sr)
    ca = (Sa * freqs).sum(1) / (Sa.sum(1) + 1e-9)
    cb = (Sb * freqs).sum(1) / (Sb.sum(1) + 1e-9)
    print(f"  spectral-centroid MAE = {np.mean(np.abs(ca - cb)):.1f} Hz  (native {ca.mean():.0f} vs ort {cb.mean():.0f})")
    bins = Sa.shape[1]
    lo, hi = bins // 3, 2 * bins // 3
    for name, sl in [("low", slice(0, lo)), ("mid", slice(lo, hi)), ("high", slice(hi, bins))]:
        ea = Sa[:, sl].mean(); eb = Sb[:, sl].mean()
        print(f"  band {name:4} energy  native={ea:.4f} ort={eb:.4f} ratio={ea/(eb+1e-9):.3f}")


if __name__ == "__main__":
    main()
