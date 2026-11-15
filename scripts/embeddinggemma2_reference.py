#!/usr/bin/env python3
"""Reference dumps of `google/embeddinggemma-2` and the null-space audio head
(`BrunoHays/embeddinggemma-2-nullspace-audio`) for
`crates/rlx-gemma/tests/embedding_gemma2_parity.rs`.

    python scripts/embeddinggemma2_reference.py \
        --model-dir weights/embed/embeddinggemma-2 \
        --head weights/embed/embeddinggemma-2-nullspace-audio/nullspace_audio_head.safetensors \
        --author-code weights/embed/embeddinggemma-2-nullspace-audio/embeddinggemma2_nullspace.py \
        --out fixtures/embeddinggemma2 [--dtype float32|bfloat16] WAV...

Each audio case writes `OUT/<dtype>/audio_<name>/`:

    samples.f32       16 kHz mono input, exactly what the model saw
    input_ids.i64     token ids (prompt + <boa> + audio placeholders + <eoa>)
    features.f32      [frames, 128] log-mel, valid frames only
    tower_l{2,6,12}.f32  audio-tower hidden states after Conformer layer 2/6/12 [seq, 1024]
    tower_out.f32     tower output after `output_proj` [seq, 1536]
    soft_tokens.f32   `embed_audio` output spliced into the trunk [seq, 512]
    token_states.f32  trunk output after norm + embedding_projection [tokens, 768]
    base.f32          sentence-transformers embedding [768]
    pool_l2_ms.f32 / pool_l6_mean.f32   the head's pooled inputs [2048] / [1024]
    z.f32             null-space coordinates [256]
    embedding.f32     base + head, normalized [768] (this script's reading)
    author.f32        the author's wrapper on the same clip [768]
    meta.json         shapes, counts

and each text case `OUT/<dtype>/text_<i>/` with `input_ids.i64`,
`token_states.f32`, `base.f32`, `meta.json`. Cases include inputs past the
512-token sliding window and audio past the extractor's 30 s cut.

The prompt is the author's: `task: search result | query: ` then the audio.
Needs the versions the author pins: transformers 5.20.2, torch 2.14.1,
sentence-transformers 6.1.0, torchaudio 2.11.0, soundfile.
"""

import argparse
import importlib.util
import json
import math
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

PROMPT = "task: search result | query: "
TEXTS = [
    "a man giving a speech about his country",
    "Le chat est sur le tapis.",
    "dog barking in the distance",
    # Past the 512-token sliding window: sliding layers then mask by distance.
    " ".join(f"Sentence {i}: the quick brown fox jumps over the lazy dog near the river bank." for i in range(40)),
]
EPS = 1e-5


def resample_16k(path):
    import torchaudio

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        x = torchaudio.transforms.Resample(sr, 16000)(torch.from_numpy(x)).numpy()
    return x.astype(np.float32)


def save(d, name, a):
    a = np.ascontiguousarray(a)
    if a.dtype == np.int64:
        a.tofile(os.path.join(d, name))
    else:
        a.astype(np.float32).tofile(os.path.join(d, name))


def masked_mean(x, m):
    x = x.float()
    n = m.sum(dim=1, keepdim=True).clamp(min=1).float()
    x = torch.where(m.unsqueeze(-1), x, torch.zeros_like(x))
    return F.normalize(x.sum(dim=1) / n, dim=-1)


def masked_mean_std(x, m, eps=1e-6):
    x = x.float()
    n = m.sum(dim=1, keepdim=True).clamp(min=1).float()
    m3 = m.unsqueeze(-1)
    z = torch.zeros_like(x)
    mu = torch.where(m3, x, z).sum(dim=1, keepdim=True) / n.unsqueeze(-1)
    d = torch.where(m3, x - mu, z)
    sigma = torch.sqrt((d**2).sum(dim=1) / n + eps)
    return F.normalize(torch.cat([mu.squeeze(1), sigma], dim=-1), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--author-code")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("wavs", nargs="*")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    dtype = getattr(torch, a.dtype)
    st = SentenceTransformer(a.model_dir, device="cpu", model_kwargs={"torch_dtype": a.dtype},
                             config_kwargs={"vision_config": None})
    st.max_seq_length = 1024
    st.prompts = {}
    st.default_prompt_name = None
    model = st[0].auto_model
    head = {k: v.float() for k, v in load_file(a.head).items()}

    # Capture what sentence-transformers feeds the model, and the tower's states.
    seen = {}
    orig_fwd = model.forward

    def fwd(*args, **kw):
        seen["kw"] = {k: v for k, v in kw.items() if torch.is_tensor(v)}
        out = orig_fwd(*args, **kw)
        seen["last"] = out.last_hidden_state
        return out

    model.forward = fwd
    tower = model.audio_tower
    tower.config.tie_last_hidden_states = False
    orig_tower = tower.forward

    def tower_fwd(*args, **kw):
        kw["output_hidden_states"] = True
        out = orig_tower(*args, **kw)
        seen["tower"] = out
        return out

    tower.forward = tower_fwd
    orig_embed_audio = model.embed_audio.forward

    def embed_audio_fwd(*args, **kw):
        out = orig_embed_audio(*args, **kw)
        seen["soft"] = out
        return out

    model.embed_audio.forward = embed_audio_fwd

    root = os.path.join(a.out, a.dtype)
    os.makedirs(root, exist_ok=True)

    # Audio cases: the given files, a tone with noise, a short clip, and two
    # clips back to back.
    cases = []
    for p in a.wavs:
        cases.append((os.path.splitext(os.path.basename(p))[0], resample_16k(p)))
    t = np.arange(32000) / 16000.0
    rng = np.random.default_rng(0)
    tone = (0.3 * np.sin(2 * math.pi * 440 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
    cases.append(("tone_noise", tone))
    if cases:
        cases.append(("short", cases[0][1][: 11_213].copy()))
    if len(a.wavs) >= 2:
        both = np.concatenate([cases[0][1], cases[1][1]]).astype(np.float32)
        cases.append(("concat", both))
        # 25 s: past the trunk's 512-token sliding window. 35 s: past the
        # extractor's 30 s cut (the file keeps all 35 s; the model sees 30).
        cases.append(("long25", np.tile(both, 3)[: 25 * 16000].copy()))
        cases.append(("long35", np.tile(both, 4)[: 35 * 16000].copy()))

    author = None
    if a.author_code:
        spec = importlib.util.spec_from_file_location("eg2_nullspace", a.author_code)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        author = mod.EmbeddingGemma2NullSpaceWrapper(a.model_dir, revision=None, head_path=a.head,
                                                     device="cpu", model_kwargs={"torch_dtype": a.dtype})

    for name, x in cases:
        d = os.path.join(root, f"audio_{name}")
        os.makedirs(d, exist_ok=True)
        seen.clear()
        with torch.no_grad():
            base = st.encode([{"text": PROMPT, "audio": {"array": x, "sampling_rate": 16000}}],
                             convert_to_numpy=True, normalize_embeddings=False)[0]
        kw = seen["kw"]
        tow = seen["tower"]
        hs = tow.hidden_states
        mask = tow.attention_mask.bool()
        valid = int(mask.sum())
        fmask = kw["input_features_mask"][0].bool()
        feats = kw["input_features"][0][fmask].float().numpy()
        l2 = masked_mean_std(hs[2], mask)
        l6 = masked_mean(hs[6], mask)
        z1 = F.normalize(((l2 - head["l2_ms_mean"]) @ head["l2_ms_proj"]) / (head["l2_ms_scale"] + EPS), dim=-1)
        z2 = F.normalize(((l6 - head["l6_m_mean"]) @ head["l6_m_proj"]) / (head["l6_m_scale"] + EPS), dim=-1)
        w = math.sqrt(128 / 256)
        z = F.normalize(torch.cat([w * z1, w * z2], dim=-1), dim=-1)
        v = F.normalize(torch.from_numpy(base).float().unsqueeze(0), dim=-1)
        emb = F.normalize(v + z @ head["u_perp"].T, dim=-1)[0]

        ids = kw["input_ids"][0]
        save(d, "samples.f32", x)
        save(d, "input_ids.i64", ids.numpy().astype(np.int64))
        save(d, "features.f32", feats)
        for li in (2, 6, 12):
            save(d, f"tower_l{li}.f32", hs[li][0][mask[0]].float().numpy())
        save(d, "tower_out.f32", tow.last_hidden_state[0][mask[0]].float().numpy())
        save(d, "soft_tokens.f32", seen["soft"][0][mask[0]].float().numpy())
        save(d, "token_states.f32", seen["last"][0].float().numpy())
        save(d, "base.f32", base)
        save(d, "pool_l2_ms.f32", l2[0].numpy())
        save(d, "pool_l6_mean.f32", l6[0].numpy())
        save(d, "z.f32", z[0].numpy())
        save(d, "embedding.f32", emb.numpy())
        meta = {"samples": int(x.size), "frames": int(feats.shape[0]), "audio_tokens": valid,
                "tokens": int(ids.numel()), "hidden_states": len(hs), "dtype": a.dtype}
        if author is not None:
            ae = author.encode_audio([(x, 16000)], query_text="")[0]
            save(d, "author.f32", ae)
            meta["author_cos"] = float(np.dot(ae, emb.numpy()) / (np.linalg.norm(ae) * np.linalg.norm(emb.numpy())))
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(name, meta)

    for i, text in enumerate(TEXTS):
        d = os.path.join(root, f"text_{i}")
        os.makedirs(d, exist_ok=True)
        seen.clear()
        with torch.no_grad():
            base = st.encode([PROMPT + text], convert_to_numpy=True, normalize_embeddings=False)[0]
        ids = seen["kw"]["input_ids"][0]
        save(d, "input_ids.i64", ids.numpy().astype(np.int64))
        save(d, "token_states.f32", seen["last"][0].float().numpy())
        save(d, "base.f32", base)
        json.dump({"text": PROMPT + text, "tokens": int(ids.numel()), "dtype": a.dtype},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
        print("text", i, int(ids.numel()))


if __name__ == "__main__":
    main()
