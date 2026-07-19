#!/usr/bin/env python3
# RLX — DINOv3 reference dump for cross-backend parity testing.
#
# Builds a tiny (or real) HuggingFace `DINOv3ViTModel`, runs a forward pass,
# and writes a fixtures directory the Rust parity test consumes:
#
#   model.safetensors   HF state_dict (verbatim keys)
#   meta.json           config the Rust `DinoV3Config` mirrors
#   px.bin              pixel_values [C, H, W] little-endian f32 (batch 1)
#   last_hidden.bin     reference last_hidden_state [seq, hidden] f32
#   pooled.bin          reference pooler_output [hidden] f32
#
# Usage:
#   # Tiny random-weight fixture (default, hermetic — no gated download):
#   python dump_reference.py --out fixtures
#
#   # Real checkpoint (requires HF auth for the gated weights):
#   python dump_reference.py --model facebook/dinov3-vitb16-pretrain-lvd1689m \
#       --image some.jpg --out fixtures_vitb16
#
# Then, from the workspace root:
#   DINOV3_FIXTURES=$PWD/crates/rlx-dinov3/scripts/fixtures \
#   DINOV3_DEVICES=cpu,metal,mlx,wgpu \
#     cargo test -p rlx-dinov3 --test dinov3_parity --features metal,mlx,gpu -- --nocapture

from __future__ import annotations
import argparse, json, os
import numpy as np
import torch


def build_tiny():
    from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(
        patch_size=16, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, hidden_act="gelu",
        layer_norm_eps=1e-5, rope_theta=100.0, image_size=32, num_channels=3,
        query_bias=True, key_bias=False, value_bias=True, proj_bias=True,
        mlp_bias=True, layerscale_value=1.0, use_gated_mlp=False,
        num_register_tokens=4,
    )
    m = DINOv3ViTModel(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    px = torch.randn(1, 3, cfg.image_size, cfg.image_size) * 0.5
    return cfg, m, px


def build_real(model_id: str, image: str | None):
    from transformers import AutoImageProcessor, AutoModel
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    m = AutoModel.from_pretrained(model_id, token=tok).eval()
    cfg = m.config
    if image:
        from transformers.image_utils import load_image
        proc = AutoImageProcessor.from_pretrained(model_id, token=tok)
        px = proc(images=[load_image(image)], return_tensors="pt")["pixel_values"]
    else:
        torch.manual_seed(0)
        px = torch.randn(1, 3, cfg.image_size, cfg.image_size) * 0.5
    return cfg, m, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id (omit for tiny random fixture)")
    ap.add_argument("--image", default=None, help="image path (real model only)")
    ap.add_argument("--out", default="fixtures")
    args = ap.parse_args()

    if args.model:
        cfg, m, px = build_real(args.model, args.image)
    else:
        cfg, m, px = build_tiny()

    with torch.inference_mode():
        out = m(pixel_values=px)

    os.makedirs(args.out, exist_ok=True)
    from safetensors.numpy import save_file
    sd = {k: v.detach().float().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    save_file(sd, f"{args.out}/model.safetensors")

    def dump(name, arr):
        np.asarray(arr, dtype="<f4").tofile(f"{args.out}/{name}.bin")

    dump("px", px[0].float().numpy())
    dump("last_hidden", out.last_hidden_state[0].float().numpy())
    dump("pooled", out.pooler_output[0].float().numpy())

    g = lambda k, d: getattr(cfg, k, d)
    meta = dict(
        hidden_size=int(g("hidden_size", 384)),
        intermediate_size=int(g("intermediate_size", 1536)),
        num_hidden_layers=int(g("num_hidden_layers", 12)),
        num_attention_heads=int(g("num_attention_heads", 6)),
        image_size=int(px.shape[-1]),
        patch_size=int(g("patch_size", 16)),
        num_channels=int(g("num_channels", 3)),
        hidden_act=str(g("hidden_act", "gelu")),
        layer_norm_eps=float(g("layer_norm_eps", 1e-5)),
        rope_theta=float(g("rope_theta", 100.0)),
        query_bias=bool(g("query_bias", True)),
        key_bias=bool(g("key_bias", False)),
        value_bias=bool(g("value_bias", True)),
        proj_bias=bool(g("proj_bias", True)),
        mlp_bias=bool(g("mlp_bias", True)),
        layerscale_value=float(g("layerscale_value", 1.0)),
        use_gated_mlp=bool(g("use_gated_mlp", False)),
        num_register_tokens=int(g("num_register_tokens", 0)),
        seq=int(out.last_hidden_state.shape[1]),
    )
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    print(f"wrote {args.out}: seq={meta['seq']} hidden={meta['hidden_size']} "
          f"gated={meta['use_gated_mlp']} reg={meta['num_register_tokens']}")


if __name__ == "__main__":
    main()
