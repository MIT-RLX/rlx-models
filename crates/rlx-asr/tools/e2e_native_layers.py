#!/usr/bin/env python3
# RLX — GPLv3.
# Apple Voice / Espresso `streaming_encoder_64_16` native forward probe.
#
# Matches E5 capture I/O:
#   mel 389 → ×6 → 64 enc frames
#   cnn_cache [28,512,1,7], att_k [4,16,8,64], att_v [28,64,8,16], mask 80
#   Macaron ½FFN → streaming MHSA → causal fused conv → ½FFN → γ·LN
#   TEXT-K {0,7,14,21}: Q≡K codebook, identity-attn out, K history 16
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_io import (  # noqa: E402
    BLANK,
    FRAMES,
    ctc_beam_decode,
    decode_pieces,
    mel_from_wav,
    resolve_units,
)
from e2e_native_whole import ctc_logp, load_native_pack, teacher_encoder_chunks  # noqa: E402
from gguf_io import open_pack  # noqa: E402

DIM = 512
N_HEADS = 8
HEAD_DIM = 64
V_HEAD_DIM = 16
N_LAYERS = 28
SUB = 6
OUT_T = 64
LOOKAHEAD = 16
CNN_K = 7
MASK_LEN = OUT_T + LOOKAHEAD
LN_EPS = 1e-5
TEXT_K_LAYERS = {0, 7, 14, 21}
TEXT_K_SLOT = {0: 0, 7: 1, 14: 2, 21: 3}


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, eps: float = LN_EPS) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    if w.shape[1] == x.shape[1]:
        y = x @ w.T
    elif w.shape[0] == x.shape[1]:
        y = x @ w
    else:
        raise ValueError(f"linear mismatch x={x.shape} w={w.shape}")
    if b is not None:
        y = y + b
    return y


def expand_v128(v: np.ndarray) -> np.ndarray:
    t = v.shape[0]
    out = np.zeros((t, DIM), dtype=np.float32)
    for h in range(N_HEADS):
        out[:, h * HEAD_DIM : h * HEAD_DIM + V_HEAD_DIM] = v[
            :, h * V_HEAD_DIM : (h + 1) * V_HEAD_DIM
        ]
    return out


def pack_v128(v512: np.ndarray) -> np.ndarray:
    t = v512.shape[0]
    out = np.zeros((t, N_HEADS * V_HEAD_DIM), dtype=np.float32)
    for h in range(N_HEADS):
        out[:, h * V_HEAD_DIM : (h + 1) * V_HEAD_DIM] = v512[
            :, h * HEAD_DIM : h * HEAD_DIM + V_HEAD_DIM
        ]
    return out


def dequant_k(int8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (int8.astype(np.float32) * scale[:, None]).astype(np.float32)


def tensor_i8(pack, name: str) -> np.ndarray:
    t = pack._tensors[name]
    off = int(t["offset"])
    ln = int(t["length"])
    raw = pack._data[pack._data_base + off : pack._data_base + off + ln]
    shape = [int(x) for x in t.get("shape") or []]
    return np.frombuffer(raw, dtype=np.int8).reshape(shape).copy()


class StreamingState:
    def __init__(self) -> None:
        self.cnn = np.zeros((N_LAYERS, DIM, CNN_K), np.float32)
        self.att_k = np.zeros((4, LOOKAHEAD, N_HEADS, HEAD_DIM), np.float32)
        self.att_v = np.zeros((N_LAYERS, OUT_T, N_HEADS, V_HEAD_DIM), np.float32)
        self.mask = np.ones(MASK_LEN, np.float32)


def ffn_macaron(x: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
    h = silu(np.clip(linear(x, wa), -20.0, 20.0))
    return 0.5 * linear(h, wb)


def conv_causal(x: np.ndarray, w: np.ndarray, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal fused conv with k=7 left taps. hist: [DIM, 7]."""
    t = x.shape[0]
    # pad: hist taps 1..6 + x  (drop oldest)
    pad = np.concatenate([hist[:, 1:].T, x], axis=0)  # [6+T, DIM]
    out = np.zeros_like(x)
    for i in range(t):
        win = pad[i : i + CNN_K].mean(axis=0, keepdims=True)
        h = linear(win, w)
        out[i] = h[0, :DIM]
    new_hist = pad[-CNN_K:].T.copy()  # [DIM, 7]
    return out, new_hist


def self_attn_streaming(
    x: np.ndarray,
    layer: dict,
    k_codebook: np.ndarray | None,
    v_pad: np.ndarray | None,
    text_k: bool,
    state: StreamingState,
    layer_i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_w = k_codebook if k_codebook is not None else layer["q"]
    q = linear(x, q_w)
    k = q.copy() if (text_k or k_codebook is not None) else linear(x, layer["k"])
    if v_pad is not None:
        v512 = linear(x, v_pad)
        v128 = pack_v128(v512)
    else:
        v128 = linear(x, layer["v"])
        v512 = expand_v128(v128)

    # Left context from caches
    v_hist = state.att_v[layer_i]  # [64,8,16]
    v_left = expand_v128(v_hist.reshape(OUT_T, N_HEADS * V_HEAD_DIM))
    use_v = not np.allclose(v_hist, 0)

    if text_k:
        slot = TEXT_K_SLOT[layer_i]
        k_hist = state.att_k[slot].reshape(LOOKAHEAD, DIM)
        use_k = not np.allclose(k_hist, 0)
        if use_k and use_v:
            k_all = np.concatenate([k_hist, k], 0)
            v_all = np.concatenate([v_left[-LOOKAHEAD:], v512], 0)
        elif use_v:
            k_all = np.concatenate([np.zeros_like(v_left), k], 0)
            v_all = np.concatenate([v_left, v512], 0)
        else:
            k_all, v_all = k, v512
    else:
        if use_v:
            k_all = np.concatenate([np.zeros_like(v_left), k], 0)
            v_all = np.concatenate([v_left, v512], 0)
        else:
            k_all, v_all = k, v512

    t = x.shape[0]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    qh = q.reshape(t, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    kh = k_all.reshape(-1, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    vh = v_all.reshape(-1, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    scores = np.matmul(qh, kh.transpose(0, 2, 1)) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    ctx = np.matmul(attn, vh).transpose(1, 0, 2).reshape(t, DIM)
    out = ctx if text_k else linear(ctx, layer["out"])
    return out, k, v128


def load_native_layers(pack_path: Path):
    enc = load_native_pack()
    layers: dict[int, dict[str, np.ndarray]] = {}
    for i in range(N_LAYERS):
        pfx = f"layers.{i}."
        layers[i] = {
            "conv": enc[f"{pfx}conv.weight"].astype(np.float32),
            "ffn_a": enc[f"{pfx}ffn_a.weight"].astype(np.float32),
            "ffn_b": enc[f"{pfx}ffn_b.weight"].astype(np.float32),
            "q": enc[f"{pfx}self_attn.linear_q.weight"].astype(np.float32),
            "k": enc[f"{pfx}self_attn.linear_k.weight"].astype(np.float32),
            "v": enc[f"{pfx}self_attn.linear_v.weight"].astype(np.float32),
            "out": enc[f"{pfx}self_attn.linear_out.weight"].astype(np.float32),
            "gamma": enc[f"{pfx}bias.weight"].astype(np.float32),
        }

    p = open_pack(pack_path)
    k_codebook: dict[int, np.ndarray] = {}
    for layer in TEXT_K_LAYERS:
        i8_key = f"codebook.layer{layer}.linear_k.int8"
        sc_key = f"codebook.layer{layer}.linear_k.scale"
        if i8_key in p._tensors and sc_key in p._tensors:
            k_codebook[layer] = dequant_k(tensor_i8(p, i8_key), p.tensor_f32(sc_key))

    v_pad: dict[int, np.ndarray] = {}
    for layer in TEXT_K_LAYERS:
        pad_key = f"ls.layer{layer}.linear_v.weight_pad"
        if pad_key in p._tensors:
            v_pad[layer] = p.tensor_f32(pad_key).astype(np.float32)

    frontend = {
        "W": enc["frontend.input_proj_eff.W"].astype(np.float32),
        "b": enc.get("frontend.input_proj_eff.b"),
    }
    if frontend["W"].shape == (DIM, 80):
        frontend["W"] = frontend["W"].T
    elif frontend["W"].shape != (80, DIM):
        raise ValueError(f"unexpected input_proj shape {frontend['W'].shape}")
    return layers, k_codebook, frontend, v_pad


def subsample_mel(feat389: np.ndarray) -> np.ndarray:
    return feat389[np.arange(OUT_T) * SUB].astype(np.float32)


def forward_native(
    feat389: np.ndarray,
    layers: dict[int, dict[str, np.ndarray]],
    k_codebook: dict[int, np.ndarray],
    frontend: dict,
    v_pad: dict[int, np.ndarray],
    state: StreamingState | None = None,
) -> tuple[np.ndarray, StreamingState]:
    if state is None:
        state = StreamingState()
    x = linear(subsample_mel(feat389), frontend["W"], frontend.get("b"))
    ones = np.ones(DIM, dtype=np.float32)
    for i in range(N_LAYERS):
        layer = layers[i]
        text_k = i in TEXT_K_LAYERS
        h = x
        h = h + ffn_macaron(layer_norm(h, ones), layer["ffn_a"], layer["ffn_b"])
        delta, k_cur, v128 = self_attn_streaming(
            layer_norm(h, ones),
            layer,
            k_codebook.get(i),
            v_pad.get(i),
            text_k,
            state,
            i,
        )
        h = h + delta
        # update att caches
        state.att_v[i] = v128.reshape(OUT_T, N_HEADS, V_HEAD_DIM)
        if text_k:
            slot = TEXT_K_SLOT[i]
            state.att_k[slot] = k_cur[-LOOKAHEAD:].reshape(LOOKAHEAD, N_HEADS, HEAD_DIM)

        c_delta, new_cnn = conv_causal(layer_norm(h, ones), layer["conv"], state.cnn[i])
        h = h + c_delta
        state.cnn[i] = new_cnn

        h = h + ffn_macaron(layer_norm(h, ones), layer["ffn_a"], layer["ffn_b"])
        x = layer["gamma"] * layer_norm(h, ones)
    return x.astype(np.float32), state


def enc_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", nargs="+")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument(
        "--probe",
        type=Path,
        default=None,
        help="teacher_work dir for enc_corr vs e5_encoder_cache.bin",
    )
    args = ap.parse_args()

    from gguf_io import resolve_pack

    pack_path = resolve_pack()
    if pack_path is None:
        raise SystemExit("model.rlxp not found")
    layers, k_codebook, frontend, v_pad = load_native_layers(pack_path)
    pack = load_native_pack()
    pieces = resolve_units()

    if args.probe is not None and args.wav:
        for wav in args.wav:
            chunks = teacher_encoder_chunks(args.probe, Path(wav))
            if chunks is None:
                print(f"{wav}: no teacher enc under {args.probe}/{Path(wav).stem}")
                continue
            feat, _ = mel_from_wav(Path(wav))
            if feat.shape[0] < FRAMES:
                pad = np.tile(feat[-1:], (FRAMES - feat.shape[0], 1))
                feat = np.concatenate([feat, pad], axis=0)
            enc, _ = forward_native(feat[:FRAMES], layers, k_codebook, frontend, v_pad)
            corr = enc_corr(enc, chunks[0])
            print(f"{wav}: streaming native enc_corr={corr:.4f} (chunk 0 vs teacher)")

    if not args.wav:
        raise SystemExit("pass --wav")

    for wav in args.wav:
        feat, _ = mel_from_wav(Path(wav))
        state = StreamingState()
        hop = OUT_T * SUB
        logps = []
        for start in range(0, max(1, feat.shape[0] - FRAMES + 1), hop):
            ch = feat[start : start + FRAMES]
            if ch.shape[0] < FRAMES:
                ch = np.concatenate([ch, np.tile(ch[-1:], (FRAMES - ch.shape[0], 1))], 0)
            enc, state = forward_native(ch, layers, k_codebook, frontend, v_pad, state)
            logps.append(ctc_logp(enc, pack))
        logp = np.concatenate(logps, 0)
        ids, score = ctc_beam_decode(logp, beam=args.beam)
        text = decode_pieces(pieces, ids)
        nb = sum(int(row.argmax() != BLANK) for row in logp)
        print(
            f"{wav}: streaming native CTC {text!r} score={score:.2f} "
            f"non-blank-argmax={nb}/{len(logp)} chunks={len(logps)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
