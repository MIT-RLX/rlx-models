#!/usr/bin/env python3
# RLX — GPLv3.
# Wav / mel / units helpers for folded native E2E (`model.gguf`).
from __future__ import annotations

import math
import os
import wave
from pathlib import Path

import numpy as np

FRAMES = 389
BLANK = 0
N_MEL = 80
SAMPLE_RATE = 16_000
FRAME_LENGTH = 400  # 25 ms
FRAME_SHIFT = 160  # 10 ms
N_FFT = 512


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _looks_populated(p: Path) -> bool:
    return (p / "model.gguf").is_file()


def asr_dir() -> Path:
    preferred = repo_root() / "weights" / "asr"
    env = os.environ.get("RLX_ASR_DIR")
    if env:
        p = Path(env)
        if _looks_populated(p) or p.is_dir():
            return p
    return preferred


def model_gguf(root: Path | None = None) -> Path | None:
    root = root or asr_dir()
    env = os.environ.get("RLX_ASR_GGUF")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for name in ("model.gguf", "asr.gguf", "rlx-asr.gguf"):
        p = root / name
        if p.is_file():
            return p
    return None


def load_silence_fbank(asr: Path) -> np.ndarray:
    try:
        from gguf_io import GgufFile, resolve_gguf

        gguf = resolve_gguf(Path(asr))
        if gguf is not None:
            g = GgufFile(gguf)
            if "silence_fbank" in g.tensors:
                vals = g.tensor_f32("silence_fbank").astype(np.float32).reshape(-1)
                if vals.size >= N_MEL:
                    return vals[:N_MEL]
    except Exception:
        pass
    return np.full(N_MEL, 7.0, dtype=np.float32)


def _mel_filterbank(low_freq: float = 20.0, high_freq: float = 8000.0) -> np.ndarray:
    m_lo, m_hi = 2595 * np.log10(1 + np.array([low_freq, high_freq]) / 700.0)
    m_pts = np.linspace(m_lo, m_hi, N_MEL + 2)
    hz = 700 * (10 ** (m_pts / 2595) - 1)
    bins = np.clip(((N_FFT + 1) * hz / SAMPLE_RATE).astype(int), 0, N_FFT // 2)
    fb = np.zeros((N_MEL, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(N_MEL):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center == left:
            center += 1
        if right == center:
            right += 1
        fb[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        fb[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    return fb


_FB = None
_POVEY = None


def _filters():
    global _FB, _POVEY
    if _FB is None:
        _FB = _mel_filterbank()
        _POVEY = np.power(np.hanning(FRAME_LENGTH), 0.85).astype(np.float32)
    return _FB, _POVEY


def log_mel_fbank(
    pcm: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    dither: float = 1.0,
    preemph: float = 0.97,
    remove_dc: bool = True,
    seed: int | None = 0,
) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if sample_rate != SAMPLE_RATE:
        n_out = int(round(len(pcm) * SAMPLE_RATE / sample_rate))
        if n_out <= 0:
            return np.zeros((0, N_MEL), dtype=np.float32)
        x_old = np.linspace(0, 1, len(pcm), endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)

    x = pcm * 32768.0
    if dither and dither > 0:
        rng = np.random.default_rng(seed)
        x = x + dither * rng.standard_normal(len(x)).astype(np.float32)
    if remove_dc and x.size:
        x = x - float(x.mean())
    if preemph and x.size > 1:
        x = np.concatenate([[x[0]], x[1:] - preemph * x[:-1]]).astype(np.float32)

    fb, window = _filters()
    if len(x) < FRAME_LENGTH:
        x = np.pad(x, (0, FRAME_LENGTH - len(x)))
    n_frames = 1 + (len(x) - FRAME_LENGTH) // FRAME_SHIFT
    out = np.empty((n_frames, N_MEL), dtype=np.float32)
    for i in range(n_frames):
        frame = x[i * FRAME_SHIFT : i * FRAME_SHIFT + FRAME_LENGTH] * window
        spec = np.abs(np.fft.rfft(frame, n=N_FFT)) ** 2
        mel = fb @ spec.astype(np.float32)
        out[i] = np.log(np.maximum(mel, 1e-10))
    return out


def fit_silence_calibration(silence: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    pcm = np.zeros(SAMPLE_RATE, dtype=np.float32)
    raw = log_mel_fbank(pcm, seed=seed)
    m = raw.mean(axis=0)
    A = np.vstack([m, np.ones_like(m)]).T
    a_g, _b_g = np.linalg.lstsq(A, silence.astype(np.float64), rcond=None)[0]
    a = np.full(N_MEL, float(a_g), dtype=np.float32)
    b = (silence.astype(np.float32) - a * m).astype(np.float32)
    return a, b


class Fbank:
    def __init__(self, asr: Path):
        self.asr_dir = Path(asr)
        self.silence = load_silence_fbank(self.asr_dir)
        self.a, self.b = fit_silence_calibration(self.silence)

    def __call__(
        self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE, *, seed: int | None = 0
    ) -> np.ndarray:
        raw = log_mel_fbank(pcm, sample_rate=sample_rate, seed=seed)
        if raw.size == 0:
            return raw
        return (raw * self.a[None, :] + self.b[None, :]).astype(np.float32)


def silence_fbank(asr: Path | None = None) -> np.ndarray:
    return Fbank(asr or asr_dir()).silence


def load_wav_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise SystemExit(f"unsupported sample width {sw} in {path}")
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return pcm, sr


def mel_from_wav(path: Path, asr: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    asr = asr or asr_dir()
    frontend = Fbank(asr)
    pcm, sr = load_wav_pcm(path)
    feat = frontend(pcm, sample_rate=sr, seed=0)
    return feat, frontend.silence


def mel_windows(
    feat: np.ndarray,
    silence: np.ndarray,
    *,
    hop_frames: int = FRAMES,
    max_chunks: int = 32,
) -> list[np.ndarray]:
    if feat.shape[0] < FRAMES:
        pad = np.tile(silence[None, :], (FRAMES - feat.shape[0], 1))
        return [np.concatenate([feat, pad], axis=0)]
    step = max(1, int(hop_frames))
    starts = list(range(0, feat.shape[0] - FRAMES + 1, step))
    last = feat.shape[0] - FRAMES
    if last > 0 and (not starts or starts[-1] != last):
        starts.append(last)
    return [feat[s : s + FRAMES] for s in starts[:max_chunks]]


def mel_chunks_from_wav(
    path: Path,
    *,
    asr: Path | None = None,
    hop_frames: int = FRAMES,
    max_chunks: int = 32,
) -> list[np.ndarray]:
    feat, silence = mel_from_wav(path, asr=asr)
    return mel_windows(feat, silence, hop_frames=hop_frames, max_chunks=max_chunks)


def resolve_units(asr: Path | None = None) -> list[str]:
    asr = asr or asr_dir()
    from gguf_io import GgufFile, resolve_gguf

    gguf = resolve_gguf(asr)
    if gguf is None:
        raise SystemExit(f"model.gguf not found under {asr} (run: just asr-pack-gguf)")
    g = GgufFile(gguf)
    units = g.metadata.get("rlx-asr.units")
    if isinstance(units, list) and units:
        return [str(u) for u in units]
    raise SystemExit(f"rlx-asr.units missing in {gguf}")


def decode_pieces(pieces: list[str], ids: list[int]) -> str:
    out: list[str] = []
    for i in ids:
        if i < 0 or i >= len(pieces):
            continue
        p = pieces[i]
        if p in ("<blank>", "<pad>", "<s>", "</s>", "<unk>") or p.startswith("<"):
            continue
        if p.startswith("\u2581"):
            out.append(" ")
            out.append(p[1:])
        else:
            out.append(p)
    return "".join(out).strip()


def ctc_beam_decode(
    logp: np.ndarray, blank: int = 0, beam: int = 5
) -> tuple[list[int], float]:
    t_len, v = logp.shape

    def lse(a: float, b: float) -> float:
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    prune = 9.0
    for t in range(t_len):
        row = logp[t]
        maxlp = float(row.max())
        thresh = maxlp - prune
        cands = [c for c in range(v) if c != blank and row[c] > thresh]
        nxt: dict[tuple[int, ...], tuple[float, float]] = {}

        def bump(key: tuple[int, ...], db: float, dnb: float) -> None:
            pb, pnb = nxt.get(key, (-math.inf, -math.inf))
            nxt[key] = (lse(pb, db), lse(pnb, dnb))

        for prefix, (pb, pnb) in beams.items():
            bump(prefix, lse(pb, pnb) + float(row[blank]), -math.inf)
            last = prefix[-1] if prefix else None
            for c in cands:
                lp = float(row[c])
                if last == c:
                    bump(prefix, -math.inf, pnb + lp)
                    bump(prefix + (c,), -math.inf, pb + lp)
                else:
                    bump(prefix + (c,), -math.inf, lse(pb, pnb) + lp)
        ranked = sorted(
            nxt.items(),
            key=lambda kv: -lse(kv[1][0], kv[1][1]),
        )[: max(beam, 1)]
        beams = dict(ranked)
    best = max(beams.items(), key=lambda kv: lse(kv[1][0], kv[1][1]))
    return list(best[0]), float(lse(best[1][0], best[1][1]))
