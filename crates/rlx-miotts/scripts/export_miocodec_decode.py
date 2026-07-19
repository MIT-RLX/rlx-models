#!/usr/bin/env python3
"""Export MioCodec decode body ONNX (tokens+emb → mag,phase) for rlx-miotts.

- RoPE patched to float cos/sin (classic ONNX cannot export complex RoPE).
- FSQ replaced by a materialised Gather codebook (12800×768) so ORT/RLX avoid
  the Mod/Div FSQ chain.
- ISTFT stays in Rust via `rlx_xcodec::istft_same`.

Usage (repo root):
  .venv-miotts/bin/python crates/rlx-miotts/scripts/export_miocodec_decode.py
  .venv-miotts/bin/python crates/rlx-miotts/scripts/export_miocodec_decode.py --regen-lm
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
CODEC_DIR = REPO / "weights" / "tts" / "miocodec"
LM_DIR = REPO / "weights" / "tts" / "miotts"
FIX = CODEC_DIR / "fixtures"
TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")
FOX = "The quick brown fox jumps over the lazy dog."
SPEECH_LEN = 100


def parse_speech_tokens(text: str) -> list[int]:
    toks = [int(v) for v in TOKEN_RE.findall(text)]
    if not toks:
        raise ValueError(f"no speech tokens in LLM output: {text[:200]!r}")
    return toks


def _patch_float_rope():
    import miocodec.module.transformer as T

    def precompute_freqs_cis_float(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)

    def apply_rotary_emb_float(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_pair = x.float().reshape(*x.shape[:-1], -1, 2)
        seq = x_pair.shape[1]
        cos = freqs_cis[:seq, :, 0].view(1, seq, 1, -1)
        sin = freqs_cis[:seq, :, 1].view(1, seq, 1, -1)
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return torch.stack([y0, y1], dim=-1).flatten(-2).type_as(x)

    T.precompute_freqs_cis = precompute_freqs_cis_float
    T.apply_rotary_emb = apply_rotary_emb_float
    return precompute_freqs_cis_float


class DecodeBodyGather(nn.Module):
    """tokens [1,T] + global [1,128] → mag [1,F,S], phase [1,F,S] (no ISTFT)."""

    def __init__(self, model, table: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("codebook", table)

    def forward(self, content_token_indices: torch.Tensor, global_embedding: torch.Tensor):
        model = self.model
        content = self.codebook[content_token_indices.long()]
        seq_len = content.size(1)
        target_audio_length = model._calculate_original_audio_length(seq_len)
        stft_length = model._calculate_target_stft_length(target_audio_length)
        local_latent = model.wave_prenet(content)
        if model.wave_conv_upsample is not None:
            local_latent = model.wave_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
        local_latent = F.interpolate(
            local_latent.transpose(1, 2),
            size=stft_length,
            mode=model.config.wave_interpolation_mode,
        ).transpose(1, 2)
        local_latent = model.wave_prior_net(local_latent.transpose(1, 2)).transpose(1, 2)
        local_latent = model.wave_decoder(local_latent, condition=global_embedding.unsqueeze(1))
        local_latent = model.wave_post_net(local_latent.transpose(1, 2)).transpose(1, 2)
        if model.wave_upsampler is not None:
            local_latent = model.wave_upsampler(local_latent.transpose(1, 2))
        x = model.istft_head.out(local_latent).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        return torch.exp(mag).clamp(max=1e2), phase


def main() -> int:
    FIX.mkdir(parents=True, exist_ok=True)
    precompute = _patch_float_rope()
    from miocodec import MioCodecModel
    from miocodec.module.istft_head import ISTFT

    print("loading MioCodec…")
    codec = MioCodecModel.from_pretrained(
        config_path=str(CODEC_DIR / "config.yaml"),
        weights_path=str(CODEC_DIR / "model.safetensors"),
    )
    codec.eval()
    for name, mod in codec.named_modules():
        if type(mod).__name__ == "Transformer" and hasattr(mod, "freqs_cis"):
            head_dim = mod.layers[0].attention.head_dim
            end = mod.freqs_cis.shape[0]
            theta = getattr(mod, "rope_theta", 10000.0)
            mod._buffers["freqs_cis"] = precompute(head_dim, end, theta)
            print("patched RoPE", name)

    print("building codebook table…")
    with torch.inference_mode():
        table = codec.decode_token_indices(torch.arange(12800)).float()

    emb = torch.load(LM_DIR / "presets" / "en_female.pt", map_location="cpu", weights_only=True)
    if isinstance(emb, dict):
        emb = emb.get("global_embedding", emb.get("embedding"))
    emb = emb.detach().float().cpu().reshape(-1)
    np.save(FIX / "en_female.npy", emb.numpy())
    (FIX / "en_female.f32").write_bytes(emb.numpy().astype(np.float32).tobytes())

    for pt in (LM_DIR / "presets").glob("*.pt"):
        e = torch.load(pt, map_location="cpu", weights_only=True)
        if isinstance(e, dict):
            e = e.get("global_embedding", e.get("embedding"))
        e = e.detach().float().cpu().reshape(-1).numpy().astype(np.float32)
        pt.with_suffix(".f32").write_bytes(e.tobytes())

    tokens_path = FIX / "fox_tokens.json"
    if tokens_path.is_file() and "--regen-lm" not in sys.argv:
        tokens = json.loads(tokens_path.read_text())["tokens"]
        print(f"reusing tokens: {len(tokens)}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(LM_DIR), trust_remote_code=True)
        lm = AutoModelForCausalLM.from_pretrained(
            str(LM_DIR), torch_dtype=torch.float32, trust_remote_code=True
        )
        lm.eval()
        messages = [{"role": "user", "content": FOX}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        with torch.inference_mode():
            out = lm.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.8,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        tokens = parse_speech_tokens(text)
        tokens_path.write_text(json.dumps({"text": FOX, "tokens": tokens, "raw": text}, indent=2))
        print(f"speech tokens: {len(tokens)}")

    t = (tokens + [0] * SPEECH_LEN)[:SPEECH_LEN]
    tokens_t = torch.tensor([t], dtype=torch.long)
    emb_t = emb.unsqueeze(0)
    body = DecodeBodyGather(codec, table).eval()
    with torch.inference_mode():
        mag, phase = body(tokens_t, emb_t)
        istft = ISTFT(n_fft=1920, hop_length=480, win_length=1920, padding="same")
        wav = istft(torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))).squeeze(0)

    np.save(FIX / "fox_mag.npy", mag.cpu().numpy().astype(np.float32))
    np.save(FIX / "fox_phase.npy", phase.cpu().numpy().astype(np.float32))
    np.save(FIX / "fox_fixed_ref.npy", wav.cpu().numpy().astype(np.float32))
    (FIX / "fox_fixed_ref.f32").write_bytes(wav.cpu().numpy().astype(np.float32).tobytes())
    np.save(FIX / "hann_window.npy", istft.window.cpu().numpy().astype(np.float32))
    (FIX / "hann_window.f32").write_bytes(istft.window.cpu().numpy().astype(np.float32).tobytes())
    sf.write(FIX / "fox_ref.wav", wav.cpu().numpy(), 24000)

    meta = {
        "sample_rate": 24000,
        "n_fft": 1920,
        "hop_length": 480,
        "speech_len": SPEECH_LEN,
        "n_freq": 961,
        "stft_frames": int(mag.shape[-1]),
        "global_dim": 128,
        "vocab_size": 12800,
        "istft_padding": "same",
        "onnx": "decoder_body.onnx",
        "codebook": "gather",
    }
    (FIX / "meta.json").write_text(json.dumps(meta, indent=2))

    onnx_path = CODEC_DIR / "decoder_body.onnx"
    print(f"exporting {onnx_path}")
    torch.onnx.export(
        body,
        (tokens_t, emb_t),
        str(onnx_path),
        input_names=["content_token_indices", "global_embedding"],
        output_names=["mag", "phase"],
        opset_version=17,
        dynamo=False,
    )
    print("size", onnx_path.stat().st_size)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    om, op = sess.run(
        None,
        {"content_token_indices": tokens_t.numpy(), "global_embedding": emb_t.numpy()},
    )
    dm = float(np.max(np.abs(om - mag.numpy())))
    dp = float(np.max(np.abs(op - phase.numpy())))
    print(f"ORT vs torch max|Δ| mag={dm:.3e} phase={dp:.3e}")
    return 0 if dm < 5e-3 and dp < 5e-3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
