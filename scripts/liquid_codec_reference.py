#!/usr/bin/env python3
"""Reference outputs for `rlx-liquid-codec`, from LiquidAI's own codec code.

Runs the released `modeling_liquid_codec.py` / `modeling_lfm3.py` from the
checkpoint directory (the `trust_remote_code` files) in float32 over the bf16
weights. Those files target a transformers newer than the one installed here;
the three symbols they import and this path never exercises are filled in
below (`create_recurrent_attention_mask`: no padding mask; the accelerate hook
decorator: identity; `slice_logits_to_keep`: LM head only), and its configs,
dataclasses in newer releases, get an equivalent `__init__`; the attention
masks, whose builders take other arguments here, are built directly (causal,
plus the sliding window on the decoder).

    python3 -I scripts/liquid_codec_reference.py \
        --model weights/audio/liquid-codec \
        --wav assets/harness/jfk_16k.wav \
        --out-dir crates/rlx-liquid-codec/tests/fixtures

Writes `clip_24k.wav` (the first `--seconds` of `--wav` resampled to 24 kHz,
16-bit, so both sides read identical samples) and `reference.safetensors`:
the front-end output, the latents handed to the quantizer, the codes and the
audio they decode to (checked equal to the model's `forward`).
"""

import argparse
import importlib.util
import inspect
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from safetensors.torch import load_file, save_file

torch.set_grad_enabled(False)


def install_shims():
    import transformers.integrations.accelerate as acc
    import transformers.masking_utils as mu
    import transformers.modeling_utils as mod

    if not hasattr(mu, "create_recurrent_attention_mask"):
        mu.create_recurrent_attention_mask = lambda **kw: kw.get("attention_mask")
    if not hasattr(acc, "force_accelerate_hooks"):
        acc.force_accelerate_hooks = lambda *a, **k: (lambda f: f)
    if not hasattr(mod, "slice_logits_to_keep"):
        mod.slice_logits_to_keep = lambda h, keep: h

    # Configs became dataclasses after the installed transformers: give the
    # `@strict` classes a field-setting `__init__` that ends in
    # `__post_init__`, and the base class a `__post_init__` that runs its old
    # `__init__` without clobbering the fields already set.
    import huggingface_hub.dataclasses as hd
    from transformers.configuration_utils import PreTrainedConfig

    if not getattr(PreTrainedConfig, "__dataclass_fields__", None):
        base_init = PreTrainedConfig.__init__

        def base_post_init(self, **kwargs):
            mine = dict(self.__dict__)
            base_init(self, **kwargs)
            self.__dict__.update(mine)

        PreTrainedConfig.__post_init__ = base_post_init

        def strict(cls=None, **_):
            def wrap(cls):
                fields = {k: getattr(cls, k, None) for k in inspect.get_annotations(cls)}
                cls.__dataclass_fields__ = fields

                def __init__(self, **kwargs):
                    for k, default in fields.items():
                        v = kwargs.pop(k, default)
                        object.__setattr__(self, k, list(v) if isinstance(v, list) else v)
                    self.__post_init__(**kwargs)

                cls.__init__ = __init__
                return cls

            return wrap(cls) if cls is not None else wrap

        hd.strict = strict

        # Its docstring generator rejects config classes it has not registered.
        import transformers.utils as tu

        tu.auto_docstring = lambda obj=None, **_: obj if obj is not None else (lambda o: o)


def patch_masks(m):
    """Additive causal / sliding-window masks for the no-cache path (the
    installed transformers' mask builders take different arguments)."""

    def build(window, inputs_embeds, position_ids, **_):
        pos = position_ids[0]
        q, k = pos[:, None], pos[None, :]
        ok = k <= q
        if window is not None:
            ok &= k > q - window
        bias = torch.zeros(ok.shape, dtype=torch.float32)
        bias[~ok] = torch.finfo(torch.float32).min
        return bias[None, None].to(inputs_embeds.dtype)

    m.create_causal_mask = lambda config, **kw: build(None, **kw)
    m.create_sliding_window_causal_mask = lambda config, **kw: build(config.sliding_window, **kw)
    m.create_recurrent_attention_mask = lambda config, attention_mask=None, **kw: attention_mask


def load_package(model_dir: Path):
    """Import the checkpoint's modeling files as package `liquid_codec_ref`."""
    name = "liquid_codec_ref"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(model_dir)]
    sys.modules[name] = pkg
    for mod in ("configuration_lfm3", "configuration_liquid_codec", "modeling_lfm3", "modeling_liquid_codec"):
        spec = importlib.util.spec_from_file_location(f"{name}.{mod}", model_dir / f"{mod}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.{mod}"] = m
        spec.loader.exec_module(m)
        if mod == "modeling_lfm3":
            patch_masks(m)
    return sys.modules[f"{name}.configuration_liquid_codec"], sys.modules[f"{name}.modeling_liquid_codec"]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, "16-bit PCM only"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
        return x, w.getframerate()


def write_wav(path: Path, x: np.ndarray, rate: int):
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/audio/liquid-codec")
    ap.add_argument("--wav", default="assets/harness/jfk_16k.wav")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out-dir", default="crates/rlx-liquid-codec/tests/fixtures")
    args = ap.parse_args()
    model_dir = Path(args.model).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    install_shims()
    cfg_mod, codec_mod = load_package(model_dir)
    raw = json.loads((model_dir / "config.json").read_text())
    keep = {k: v for k, v in raw.items() if k in cfg_mod.LiquidCodecConfig.__dataclass_fields__}
    config = cfg_mod.LiquidCodecConfig(**keep)
    config._attn_implementation = "eager"
    model = codec_mod.LiquidCodecModel(config).float().eval()

    sd = load_file(str(model_dir / "model.safetensors"))
    renames = [("_frontend.", "frontend."), ("quantizer.fsq.", "quantizer.")]
    fixed = {}
    for k, v in sd.items():
        for a, b in renames:
            if k.startswith(a):
                k = b + k[len(a):]
        fixed[k.replace(".self_attn.gate_proj.", ".self_attn.g_proj.")] = v.float()
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    missing = [k for k in missing if not k.endswith("window")]
    assert not missing and not unexpected, (missing, unexpected)

    # The clip: resampled once here and stored as 16-bit, then re-read.
    x, rate = read_wav(Path(args.wav))
    x = torch.from_numpy(x[: int(args.seconds * rate)])
    if rate != config.sampling_rate:
        x = AF.resample(x, rate, config.sampling_rate)
    clip = out / "clip_24k.wav"
    write_wav(clip, x.numpy(), config.sampling_rate)
    pcm, _ = read_wav(clip)
    audio = torch.from_numpy(pcm)[None]

    # Stage by stage, the same calls `encode` makes.
    frame = config.frame_size
    n = -(audio.shape[-1] // -frame) * frame
    padded = torch.nn.functional.pad(audio, (0, n - audio.shape[-1]))
    frontend = model.frontend(model.stft(padded))  # [1, 512, F]
    hidden = model.encoder(frontend.mT)
    latents = model.quantizer.project_in(model.to_quantizer(hidden))  # [1, F, 48]
    codes = model.encode(audio).audio_codes  # [1, 12, F]
    decoded = model.decode(codes).audio_values[0]
    recon, codes2 = model(audio)
    assert torch.equal(codes, codes2) and torch.equal(recon[0], decoded[: recon.shape[-1]])

    tensors = {
        "frontend": frontend[0].mT.contiguous(),  # [F, 512]
        "latents": latents[0].contiguous(),  # [F, 48]
        "codes": codes[0].T.contiguous().to(torch.int32),  # [F, 12]
        "decoded": decoded.contiguous(),  # [F * 1920]
    }
    save_file(tensors, str(out / "reference.safetensors"))
    print(f"frames {codes.shape[-1]}, samples {decoded.shape[0]}; wrote {out}")


if __name__ == "__main__":
    main()
