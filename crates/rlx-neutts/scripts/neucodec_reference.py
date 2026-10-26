"""NeuCodec's decoder as transformers runs it, over NeuTTS's decoder weights:
the reference the exported package is checked against.

    python crates/rlx-neutts/scripts/neucodec_reference.py \
        <neucodec_decoder.safetensors> <out.json>

Writes the codes it decodes and the audio (`{"codes": [...], "audio": [...]}`).
Needs torch, safetensors and transformers (>= 5, with `neucodec`).
"""

import json
import sys

import torch
from safetensors.torch import load_file
from transformers.models.neucodec.configuration_neucodec import NeuCodecConfig
from transformers.models.neucodec.modeling_neucodec import NeuCodecDecoder, NeuCodecQuantizer


def main(weights: str, out: str) -> None:
    st = load_file(weights)
    config = NeuCodecConfig()
    quantizer = NeuCodecQuantizer(config).eval()
    decoder = NeuCodecDecoder(config).eval()

    g = "generator."
    q = {
        "project_out.weight": st[g + "quantizer.project_out.weight"],
        "project_out.bias": st[g + "quantizer.project_out.bias"],
        "project_in.weight": st[g + "quantizer.project_in.weight"],
        "project_in.bias": st[g + "quantizer.project_in.bias"],
    }
    quantizer.load_state_dict(q, strict=False)

    d = {
        "fc.weight": st["fc_post_a.weight"],
        "fc.bias": st["fc_post_a.bias"],
        "embed.weight": st[g + "backbone.embed.weight"],
        "embed.bias": st[g + "backbone.embed.bias"],
        "norm.weight": st[g + "backbone.final_layer_norm.weight"],
        "norm.bias": st[g + "backbone.final_layer_norm.bias"],
        "head.linear.weight": st[g + "head.out.weight"],
        "head.linear.bias": st[g + "head.out.bias"],
    }
    for net in ("prior_net", "post_net"):
        for i in range(2):
            for part in ("norm1", "conv1", "norm2", "conv2"):
                for p in ("weight", "bias"):
                    d[f"{net}.{i}.{part}.{p}"] = st[f"{g}backbone.{net}.{i}.{part}.{p}"]
    dim = config.hidden_size
    for i in range(config.num_hidden_layers):
        s = f"{g}backbone.transformers.{i}."
        qkv = st[s + "att.c_attn.weight"]
        layer = f"layers.{i}."
        d[layer + "self_attn.q_proj.weight"] = qkv[:dim]
        d[layer + "self_attn.k_proj.weight"] = qkv[dim : 2 * dim]
        d[layer + "self_attn.v_proj.weight"] = qkv[2 * dim :]
        d[layer + "self_attn.o_proj.weight"] = st[s + "att.c_proj.weight"]
        d[layer + "input_layernorm.weight"] = st[s + "att_norm.weight"]
        d[layer + "post_attention_layernorm.weight"] = st[s + "ffn_norm.weight"]
        d[layer + "mlp.fc1.weight"] = st[s + "mlp.fc1.weight"]
        d[layer + "mlp.fc2.weight"] = st[s + "mlp.fc2.weight"]
    missing, unexpected = decoder.load_state_dict(d, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    # The checkpoint's window (the Hann window within 6e-8).
    decoder.head.window = st[g + "head.istft.window"]

    # Fifty frames of codes over the whole range, two seconds of audio.
    gen = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 65536, (50,), generator=gen)
    with torch.no_grad():
        latents = quantizer.from_codes(codes[None, :, None])
        audio = decoder(latents).reshape(-1)
    with open(out, "w") as f:
        json.dump({"codes": codes.tolist(), "audio": audio.tolist()}, f)
    print(f"{len(codes)} codes, {audio.numel()} samples -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
