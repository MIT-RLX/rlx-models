#!/usr/bin/env python3
"""Reference fixtures of the Gemma 4 audio tower for
`crates/rlx-gemma/tests/gemma4_audio_parity.rs`.

    python scripts/gemma4_audio_reference.py MODEL_DIR WAV OUT_DIR [--seconds S]

MODEL_DIR is `google/gemma-4-E2B-it-qat-mobile-transformers` (the test reads it
from the Hugging Face cache). Writes little-endian f32 `.bin` files, valid
frames / positions only:

    feats.bin        log-mel features [frames, 128]
    subsample.bin    sub-sampler output, hidden_states[0] [seq, 1024]
    layer0.bin       after the first conformer layer [seq, 1024]
    l0_attn_in.bin   layer 0 attention input (after norm_pre_attn) [seq, 1024]
    l0_attn_out.bin  layer 0 attention output [seq, 1024]
    l0_dwin.bin      layer 0 depthwise conv input [1024, seq]
    l0_dwout.bin     layer 0 depthwise conv output [1024, seq]
    out.bin          tower output after output_proj [seq, 1536]
    feat_out.bin     embed_audio(tower output): the LM soft tokens [seq, hidden]

The reference is HF with two adjustments (see `fix_reference`): the f32 linears
transformers mis-loads as int2 are restored, and activation rounding (SRQ) is
off, as in the Rust tower. Use a clip longer than one attention block (12 positions, about 0.5 s): with a
single block every query already sees the whole block, so the attention window
cannot be told apart. Needs transformers >= 5.18, torch, soundfile, torchaudio.
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


def fix_reference(model, model_dir):
    """Two adjustments, so the reference computes what the Rust tower implements.

    1. transformers (5.20.2) converts every audio-tower linear to an int2
       `QuantizedLinear`, including the three this checkpoint stores as f32
       (`output_proj`, the sub-sampler's `input_proj_linear`, `embed_audio`'s
       projection). Their forward then decodes the f32 weights as packed int2:
       garbage. They go back to plain linears with the checkpoint's weights.
    2. The quantized linears round activations to 8 bits (SRQ) wherever the
       checkpoint has a calibrated scale; the Rust tower dequantizes the weights
       but does not model SRQ. A zero scale switches SRQ off.
    """
    import torch.nn as nn
    from safetensors import safe_open

    ckpt = safe_open(os.path.join(model_dir, "model.safetensors"), "pt")
    for parent, attr, key in [
        (model.audio_tower, "output_proj", "model.audio_tower.output_proj"),
        (model.audio_tower.subsample_conv_projection, "input_proj_linear",
         "model.audio_tower.subsample_conv_projection.input_proj_linear"),
        (model.embed_audio, "embedding_projection", "model.embed_audio.embedding_projection"),
    ]:
        w = ckpt.get_tensor(key + ".weight").float()
        has_bias = key + ".bias" in ckpt.keys()
        lin = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
        with torch.no_grad():
            lin.weight.copy_(w)
            if has_bias:
                lin.bias.copy_(ckpt.get_tensor(key + ".bias").float())
        setattr(parent, attr, lin)
    for mod in model.audio_tower.modules():
        for name in ("input_activation_scale", "output_activation_scale"):
            if hasattr(mod, name):
                with torch.no_grad():
                    getattr(mod, name).zero_()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()

    x, sr = sf.read(a.wav, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        import torchaudio

        x = torchaudio.transforms.Resample(sr, 16000)(torch.from_numpy(x)).numpy()
    x = x[: int(a.seconds * 16000)].astype(np.float32)

    model = AutoModel.from_pretrained(a.model_dir, dtype=torch.float32).eval()
    fix_reference(model, a.model_dir)
    tower = model.audio_tower
    tower.config.tie_last_hidden_states = False
    fe = AutoProcessor.from_pretrained(a.model_dir).feature_extractor([x], sampling_rate=16000,
                                                                       return_tensors="pt")
    cap = {}
    layer0 = tower.layers[0]

    def keep(name):
        def hook(mod, inp, out):
            if inp:  # attention gets its input as a keyword: none here
                cap[name + "_in"] = inp[0].detach()
            cap[name + "_out"] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    layer0.norm_pre_attn.register_forward_hook(keep("pre_attn"))
    layer0.self_attn.register_forward_hook(keep("attn"))
    layer0.lconv1d.depthwise_conv1d.register_forward_hook(keep("dw"))

    with torch.no_grad():
        out = tower(fe["input_features"], fe["input_features_mask"], output_hidden_states=True)
        soft = model.embed_audio(inputs_embeds=out.last_hidden_state)
    mask = out.attention_mask[0].bool()
    fmask = fe["input_features_mask"][0].bool()

    os.makedirs(a.out, exist_ok=True)

    def save(name, t):
        np.ascontiguousarray(t.float().numpy()).astype("<f4").tofile(os.path.join(a.out, name))

    save("feats.bin", fe["input_features"][0][fmask])
    save("subsample.bin", out.hidden_states[0][0][mask])
    save("layer0.bin", out.hidden_states[1][0][mask])
    save("l0_attn_in.bin", cap["pre_attn_out"][0][mask])
    save("l0_attn_out.bin", cap["attn_out"][0][mask])
    # The causal depthwise conv pads inside its forward: its input is unpadded.
    save("l0_dwin.bin", cap["dw_in"][0][:, mask])
    save("l0_dwout.bin", cap["dw_out"][0][:, mask])
    save("out.bin", out.last_hidden_state[0][mask])
    save("feat_out.bin", soft[0][mask])
    print(f"{a.out}: {int(fmask.sum())} frames, {int(mask.sum())} positions, "
          f"soft tokens {tuple(soft[0][mask].shape)}")


if __name__ == "__main__":
    main()
