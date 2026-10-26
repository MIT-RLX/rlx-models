#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: GPL-3.0
"""Voxtral-4B-TTS reference in plain PyTorch (CPU, f32), transcribed from
vLLM-Omni's `voxtral_tts_audio_generation.py` / `voxtral_tts_audio_tokenizer.py`
and vLLM's `MistralForCausalLM` (Mistral-native q/k permuted to NeoX halves).

  python voxtral_reference.py --model-dir DIR --prompt-tokens P.txt \
      --voice neutral_female --frames 200 --codes-out C.txt --wav-out W.wav
  python voxtral_reference.py --model-dir DIR --decode-codes C.txt --wav-out W.wav
  python voxtral_reference.py --model-dir DIR --prompt-tokens P.txt \
      --voice neutral_female --force-codes C.txt --semantic-out S.txt

`--force-codes` feeds the given frames back (teacher forcing) and writes the
semantic code the model picks after each, so an LM can be checked frame by
frame against these codes.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_CODEBOOKS = 37
N_SPECIAL = 2  # empty_audio, end_audio
END_AUDIO = 1
EMPTY_AUDIO = 0


def load_safetensors(path, prefixes):
    """The tensors of `path` whose names start with one of `prefixes`, as f32."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__" or not name.startswith(prefixes):
                continue
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if meta["dtype"] == "BF16":
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif meta["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                raise ValueError(f"{name}: {meta['dtype']}")
            out[name] = torch.from_numpy(arr.reshape(meta["shape"]).copy())
    return out


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


# ── LM (vLLM MistralForCausalLM over permuted q/k) ──────────────────────────


def build_lm(params, w):
    from transformers import MistralConfig, MistralModel

    cfg = MistralConfig(
        vocab_size=params["vocab_size"],
        hidden_size=params["dim"],
        intermediate_size=params["hidden_dim"],
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params["n_kv_heads"],
        head_dim=params["head_dim"],
        rms_norm_eps=params["norm_eps"],
        rope_theta=params["rope_theta"],
        max_position_embeddings=4096,
        sliding_window=None,
        attn_implementation="eager",
    )
    model = MistralModel(cfg)
    hd = params["head_dim"]
    dim = params["dim"]

    def permute(t, n_heads):
        attn_in = hd * n_heads
        return t.view(n_heads, attn_in // n_heads // 2, 2, dim).transpose(1, 2).reshape(attn_in, dim)

    sd = {"embed_tokens.weight": w["mm_audio_embeddings.tok_embeddings.weight"], "norm.weight": w["norm.weight"]}
    for i in range(params["n_layers"]):
        p = f"layers.{i}."
        q = f"layers.{i}."
        sd[q + "self_attn.q_proj.weight"] = permute(w[p + "attention.wq.weight"], params["n_heads"])
        sd[q + "self_attn.k_proj.weight"] = permute(w[p + "attention.wk.weight"], params["n_kv_heads"])
        sd[q + "self_attn.v_proj.weight"] = w[p + "attention.wv.weight"]
        sd[q + "self_attn.o_proj.weight"] = w[p + "attention.wo.weight"]
        sd[q + "mlp.gate_proj.weight"] = w[p + "feed_forward.w1.weight"]
        sd[q + "mlp.down_proj.weight"] = w[p + "feed_forward.w2.weight"]
        sd[q + "mlp.up_proj.weight"] = w[p + "feed_forward.w3.weight"]
        sd[q + "input_layernorm.weight"] = w[p + "attention_norm.weight"]
        sd[q + "post_attention_layernorm.weight"] = w[p + "ffn_norm.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval()


# ── acoustic transformer (FlowMatchingAudioTransformer) ─────────────────────


class Acoustic:
    def __init__(self, w, args, model_args):
        self.w = {k[len("acoustic_transformer.") :]: v for k, v in w.items() if k.startswith("acoustic_transformer.")}
        self.a = dict(args)
        self.a.setdefault("norm_eps", 1e-5)  # AcousticTransformerArgs' default
        self.levels = model_args["acoustic_codebook_size"]
        self.semantic_size = model_args["semantic_codebook_size"]
        self.n_steps = args.get("n_decoding_steps") or 7
        dim = args["dim"]
        half = dim // 2
        self.inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)

    def lin(self, x, name):
        return x @ self.w[name].T

    def layer(self, x, i):
        a = self.a
        p = f"layers.{i}."
        h = rms_norm(x, self.w[p + "attention_norm.weight"], a["norm_eps"])
        B, S, _ = h.shape
        q = self.lin(h, p + "attention.wq.weight").view(B, S, a["n_heads"], a["head_dim"])
        k = self.lin(h, p + "attention.wk.weight").view(B, S, a["n_kv_heads"], a["head_dim"])
        v = self.lin(h, p + "attention.wv.weight").view(B, S, a["n_kv_heads"], a["head_dim"])
        rep = a["n_heads"] // a["n_kv_heads"]
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = ((q / math.sqrt(a["head_dim"])) @ k.transpose(-2, -1)).softmax(-1) @ v
        att = att.transpose(1, 2).reshape(B, S, -1)
        x = x + self.lin(att, p + "attention.wo.weight")
        h = rms_norm(x, self.w[p + "ffn_norm.weight"], a["norm_eps"])
        ff = self.lin(F.silu(self.lin(h, p + "feed_forward.w1.weight")) * self.lin(h, p + "feed_forward.w3.weight"),
                      p + "feed_forward.w2.weight")
        return x + ff

    def velocity(self, x_t, llm_proj, t_proj):
        h = torch.stack([self.lin(x_t, "input_projection.weight"), t_proj, llm_proj], dim=1)
        for i in range(self.a["n_layers"]):
            h = self.layer(h, i)
        h = rms_norm(h, self.w["norm.weight"], self.a["norm_eps"])
        return self.lin(h[:, 0, :], "acoustic_codebook_output.weight")

    def semantic(self, hidden):
        logit = self.lin(hidden, "semantic_codebook_output.weight").float()
        logit[:, EMPTY_AUDIO] = -float("inf")
        logit[:, N_SPECIAL + self.semantic_size :] = -float("inf")
        return logit.argmax(-1)

    def frame(self, hidden, cfg_alpha, x0):
        sem = self.semantic(hidden)
        ts = torch.linspace(0, 1, self.n_steps + 1)
        emb = ts.view(-1, 1) * self.inv_freq.view(1, -1)
        t_proj_table = self.lin(torch.cat([emb.cos(), emb.sin()], -1), "time_projection.weight")
        llm_proj = self.lin(torch.cat([hidden, torch.zeros_like(hidden)], 0), "llm_projection.weight")
        sampled = x0
        for i in range(self.n_steps):
            dt = ts[i + 1] - ts[i]
            t_proj = t_proj_table[i].unsqueeze(0).expand(2, -1)
            v = self.velocity(torch.cat([sampled, sampled], 0), llm_proj, t_proj)
            v = cfg_alpha * v[:1] + (1 - cfg_alpha) * v[1:]
            sampled = sampled + v * dt
        sampled = sampled.clamp(-1, 1)
        codes = (((sampled + 1) / 2) * (self.levels - 1)).round().long()
        if int(sem) == END_AUDIO:
            codes[:] = EMPTY_AUDIO
        return [int(sem)] + [int(c) + N_SPECIAL for c in codes[0]]


# ── codec decoder (VoxtralTTSAudioTokenizer.decode) ─────────────────────────


def weight_norm(w, prefix):
    g = w.get(prefix + ".conv.parametrizations.weight.original0")
    v = w.get(prefix + ".conv.parametrizations.weight.original1")
    if g is None:
        return w[prefix + ".conv.weight"]
    norm = v.reshape(v.shape[0], -1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    return g * v / norm


def causal_conv(x, weight, stride, pad_mode):
    k = weight.shape[-1]
    pad = k - stride
    n_frames = (x.shape[-1] - k + pad) / stride + 1
    target = (math.ceil(n_frames) - 1) * stride + (k - pad)
    extra = target - x.shape[-1]
    if pad_mode == "reflect":
        x = F.pad(x, (pad, extra), mode="reflect")
    else:
        x = F.pad(x, (pad, extra), mode=pad_mode)
    return F.conv1d(x, weight, stride=stride)


def causal_conv_t(x, weight, stride):
    k = weight.shape[-1]
    out = F.conv_transpose1d(x, weight, stride=stride)
    total = k - stride
    right = math.ceil(total * 1.0)
    left = total - right
    return out[..., left : out.shape[-1] - right]


def alibi_slopes(n):
    def p2(m):
        r = 2.0 ** (-8.0 / m)
        return [r**i for i in range(m)]

    if math.log2(n).is_integer():
        s = p2(n)
    else:
        m = 2 ** math.floor(math.log2(n))
        s = p2(m) + p2(2 * m)[::2][: n - m]
    return torch.tensor(s)


def codec_layer(x, w, p, a, window):
    h = rms_norm(x, w[p + "attention_norm.weight"], a["norm_eps"])
    q = rms_norm(h @ w[p + "attention.wq.weight"].T, w[p + "attention.q_norm.weight"], a["qk_norm_eps"])
    k = rms_norm(h @ w[p + "attention.wk.weight"].T, w[p + "attention.k_norm.weight"], a["qk_norm_eps"])
    v = h @ w[p + "attention.wv.weight"].T
    S = x.shape[0]
    H, D = a["n_heads"], a["head_dim"]
    q, k, v = (t.view(S, -1, D).transpose(0, 1) for t in (q, k, v))
    pos = torch.arange(S)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)
    bias = alibi_slopes(H).view(H, 1, 1) * rel.unsqueeze(0).float()
    bias = bias.masked_fill(rel.unsqueeze(0) > 0, float("-inf"))
    bias = bias.masked_fill((rel < -window).unsqueeze(0), float("-inf"))
    att = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=bias.unsqueeze(0))
    att = att[0].transpose(0, 1).reshape(S, -1)
    r = att @ w[p + "attention.wo.weight"].T
    h = x + w[p + "attention_scale"] * r
    n = rms_norm(h, w[p + "ffn_norm.weight"], a["norm_eps"])
    ff = (F.silu(n @ w[p + "feed_forward.w1.weight"].T) * (n @ w[p + "feed_forward.w3.weight"].T)) @ w[
        p + "feed_forward.w2.weight"
    ].T
    return h + w[p + "ffn_scale"] * ff


def codec_decode(w, a, codes):
    """`codes` `[T, 37]` (vLLM layout, offsets included, no END frame)."""
    w = {k[len("audio_tokenizer.") :]: v for k, v in w.items() if k.startswith("audio_tokenizer.")}
    a = dict(a)
    a.setdefault("qk_norm_eps", 1e-6)
    c = torch.tensor(codes) - N_SPECIAL
    usage = w["quantizer.semantic_codebook.cluster_usage"].clamp(min=1e-5)
    table = w["quantizer.semantic_codebook.embedding_sum"] / usage[:, None]
    sem = table[c[:, 0]]  # [T, 256]
    ac = c[:, 1:].float() * 2 / (a["acoustic_codebook_size"] - 1) - 1  # [T, 36]
    emb = torch.cat([sem, ac], 1).T.unsqueeze(0)  # [1, 292, T]
    kernels = [int(x) for x in a["decoder_convs_kernels_str"].split(",")]
    strides = [int(x) for x in a["decoder_convs_strides_str"].split(",")]
    lengths = [int(x) for x in a["decoder_transformer_lengths_str"].split(",")]
    enc_strides = [int(x) for x in a.get("encoder_convs_strides_str", "2,2,2,1").split(",")]
    window = a["attn_sliding_window_size"]
    for s in enc_strides:
        if s > 1:
            window //= 2
    bi = 0
    x = causal_conv(emb, weight_norm(w, f"decoder_blocks.{bi}"), strides[0], "replicate")
    bi += 1
    if strides[0] > 1:
        window *= 2
    for idx, n in enumerate(lengths):
        h = x[0].T
        for li in range(n):
            h = codec_layer(h, w, f"decoder_blocks.{bi}.layers.{li}.", a, window)
        x = h.T.unsqueeze(0)
        bi += 1
        if idx + 1 != len(lengths) and (kernels[idx + 1] != 1 or strides[idx + 1] != 1):
            x = causal_conv_t(x, weight_norm(w, f"decoder_blocks.{bi}"), strides[idx + 1])
            bi += 1
            if strides[idx + 1] > 1:
                window *= 2
    out = causal_conv(x, weight_norm(w, "output_proj"), 1, "reflect")  # [1, 240, T']
    return out[0].T.reshape(-1).numpy()


def write_wav(path, pcm, sr):
    import wave

    pcm = np.clip(pcm, -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((pcm * 32767).astype("<i2").tobytes())


def read_codes(path):
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    flat = [int(x) for x in lines[1].split()]
    return [flat[i * N_CODEBOOKS : (i + 1) * N_CODEBOOKS] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-tokens")
    ap.add_argument("--voice", default="neutral_female")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--cfg-alpha", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--codes-out")
    ap.add_argument("--wav-out")
    ap.add_argument("--decode-codes")
    ap.add_argument("--force-codes")
    ap.add_argument("--semantic-out")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    d = Path(args.model_dir)
    params = json.loads((d / "params.json").read_text())
    mm = params["multimodal"]
    ckpt = d / "consolidated.safetensors"

    if args.decode_codes:
        w = load_safetensors(ckpt, ("audio_tokenizer.",))
        codes = read_codes(args.decode_codes)
        pcm = codec_decode(w, mm["audio_tokenizer_args"], codes)
        write_wav(args.wav_out, pcm, mm["audio_tokenizer_args"]["sampling_rate"])
        print(f"decoded {len(codes)} frames -> {len(pcm)} samples")
        return

    w = load_safetensors(ckpt, ("layers.", "norm.", "mm_audio_embeddings.", "acoustic_transformer."))
    lm = build_lm(params, w)
    am = mm["audio_model_args"]
    ac = Acoustic(w, am["acoustic_transformer_args"], am)
    tok_emb = w["mm_audio_embeddings.tok_embeddings.weight"]
    cb = w["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]
    sizes = [am["semantic_codebook_size"] + N_SPECIAL] + [am["acoustic_codebook_size"] + N_SPECIAL] * am[
        "n_acoustic_codebook"
    ]
    offsets = torch.tensor(np.cumsum([0] + sizes[:-1]))

    tokens = [int(x) for x in Path(args.prompt_tokens).read_text().split()]
    voice = np.fromfile(d / "voice_embedding" / f"{args.voice}.f32", dtype="<f4").reshape(-1, params["dim"])
    embeds = tok_emb[torch.tensor(tokens)].clone()
    slots = [i for i, t in enumerate(tokens) if t == mm["audio_model_args"]["audio_token_id"]]
    assert len(slots) == voice.shape[0], (len(slots), voice.shape)
    embeds[torch.tensor(slots)] = torch.from_numpy(voice)

    out = lm(inputs_embeds=embeds.unsqueeze(0), use_cache=True)
    cache = out.past_key_values
    hidden = out.last_hidden_state[:, -1, :]
    forced = read_codes(args.force_codes) if args.force_codes else None
    gen = torch.Generator().manual_seed(args.seed)
    frames, semantic = [], []
    n = len(forced) if forced else args.frames
    for fi in range(n):
        if forced:
            semantic.append(int(ac.semantic(hidden)))
            frame = forced[fi]
        else:
            x0 = torch.randn(1, am["n_acoustic_codebook"], generator=gen)
            frame = ac.frame(hidden, args.cfg_alpha, x0)
            if frame[0] == END_AUDIO:
                break
        frames.append(frame)
        e = cb[torch.tensor(frame) + offsets].sum(0)
        out = lm(inputs_embeds=e.view(1, 1, -1), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]
        print(f"frame {fi}: semantic {frame[0]}", file=sys.stderr)
    if args.semantic_out:
        Path(args.semantic_out).write_text(" ".join(map(str, semantic)) + "\n")
    if args.codes_out:
        flat = [c for f in frames for c in f]
        Path(args.codes_out).write_text(f"{len(frames)}\n" + " ".join(map(str, flat)) + "\n")
    if args.wav_out and frames:
        wc = load_safetensors(ckpt, ("audio_tokenizer.",))
        pcm = codec_decode(wc, mm["audio_tokenizer_args"], frames)
        write_wav(args.wav_out, pcm, mm["audio_tokenizer_args"]["sampling_rate"])
    print(f"{len(frames)} frames")


if __name__ == "__main__":
    main()
