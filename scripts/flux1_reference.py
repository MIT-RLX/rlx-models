#!/usr/bin/env python3
"""FLUX.1 reference dumps for rlx-flux1 parity (diffusers / transformers).

Reads the same files rlx-flux1 does (BFL single-file transformer, HF T5 and
CLIP directories, BFL ae.safetensors) and writes raw little-endian f32 / i32
`.bin` files plus a `meta.json` into --out:

  text   prompt -> T5 ids + hidden states, CLIP ids + pooled vector (f32)
  dit    a truncated denoiser (first --doubles / --singles blocks + final
         layer) in f32 on CPU, LoRA merged in *diffusers* naming
  vae    random latents -> decoded RGB (f32)
  e2e    the full FluxPipeline (bf16 on MPS by default) with explicit noise

Needs: torch, diffusers, transformers, sentencepiece, safetensors, numpy, pillow.
"""

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open


def save(out, name, arr, dtype=np.float32):
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    a.tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


def tokenize(t5_dir, clip_dir, prompt, t5_len):
    from transformers import CLIPTokenizerFast, T5TokenizerFast

    t5_tok = T5TokenizerFast(tokenizer_file=os.path.join(t5_dir, "tokenizer.json"))
    clip_tok = CLIPTokenizerFast.from_pretrained(clip_dir)
    t5_ids = t5_tok(prompt, padding="max_length", max_length=t5_len, truncation=True,
                    return_tensors="pt").input_ids
    clip_ids = clip_tok(prompt, padding="max_length", max_length=77, truncation=True,
                        return_tensors="pt").input_ids
    return t5_ids, clip_ids


@torch.no_grad()
def cmd_text(a):
    from transformers import CLIPTextModel, T5EncoderModel

    os.makedirs(a.out, exist_ok=True)
    t5_ids, clip_ids = tokenize(a.t5, a.clip, a.prompt, a.t5_len)
    clip = CLIPTextModel.from_pretrained(a.clip, torch_dtype=torch.float32).eval()
    pooled = clip(clip_ids, output_hidden_states=False).pooler_output[0]
    del clip
    t5 = T5EncoderModel.from_pretrained(a.t5, torch_dtype=torch.float32).eval()
    hidden = t5(t5_ids, output_hidden_states=False)[0][0]
    meta = {
        "prompt": a.prompt,
        "t5_ids": save(a.out, "t5_ids", t5_ids[0], np.int32),
        "t5_hidden": save(a.out, "t5_hidden", hidden),
        "clip_ids": save(a.out, "clip_ids", clip_ids[0], np.int32),
        "pooled": save(a.out, "pooled", pooled),
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


def merge_lora_diffusers(state, lora_path, scale, keep):
    """W += scale * B @ A, keyed by diffusers module names."""
    n = 0
    with safe_open(lora_path, "pt") as f:
        keys = list(f.keys())
        for k in keys:
            if not k.endswith(".lora_A.weight"):
                continue
            mod = k[: -len(".lora_A.weight")]
            if mod.startswith("transformer."):
                mod = mod[len("transformer."):]
            if not keep(mod):
                continue
            A = f.get_tensor(k).float()
            B = f.get_tensor(k.replace(".lora_A.", ".lora_B.")).float()
            alpha_key = k.replace(".lora_A.weight", ".alpha")
            s = scale * (f.get_tensor(alpha_key).item() / A.shape[0] if alpha_key in keys else 1.0)
            state[mod + ".weight"] += s * (B @ A)
            n += 1
    return n


@torch.no_grad()
def cmd_dit(a):
    from diffusers import FluxTransformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers

    os.makedirs(a.out, exist_ok=True)

    def keep_bfl(k):
        if k.startswith("double_blocks."):
            return int(k.split(".")[1]) < a.doubles
        if k.startswith("single_blocks."):
            return int(k.split(".")[1]) < a.singles
        return True

    with safe_open(a.transformer, "pt") as f:
        sd = {k: f.get_tensor(k).float() for k in f.keys() if keep_bfl(k)}
    guidance = "guidance_in.in_layer.weight" in sd
    conv = convert_flux_transformer_checkpoint_to_diffusers(sd)
    model = FluxTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=a.doubles, num_single_layers=a.singles,
        attention_head_dim=128, num_attention_heads=24, joint_attention_dim=4096,
        pooled_projection_dim=768, guidance_embeds=guidance, axes_dims_rope=(16, 56, 56),
    ).eval()
    if a.lora:

        def keep_diff(m):
            for p, lim in (("transformer_blocks.", a.doubles), ("single_transformer_blocks.", a.singles)):
                if m.startswith(p):
                    return int(m.split(".")[1]) < lim
            return True

        print("lora modules merged:", merge_lora_diffusers(conv, a.lora, a.lora_scale, keep_diff))
    model.load_state_dict(conv, strict=True)

    g = torch.Generator().manual_seed(a.seed)
    gh, gw = a.grid_h, a.grid_w
    si = gh * gw
    if a.text_dir:
        txt = torch.from_numpy(np.fromfile(os.path.join(a.text_dir, "t5_hidden.bin"), np.float32)).reshape(-1, 4096)
        pooled = torch.from_numpy(np.fromfile(os.path.join(a.text_dir, "pooled.bin"), np.float32))
    else:
        txt = torch.randn(a.txt_len, 4096, generator=g) * 0.2
        pooled = torch.randn(768, generator=g)
    st = txt.shape[0]
    img = torch.randn(si, 64, generator=g)
    img_ids = torch.zeros(gh, gw, 3)
    img_ids[..., 1] = torch.arange(gh)[:, None]
    img_ids[..., 2] = torch.arange(gw)[None, :]
    img_ids = img_ids.reshape(si, 3)
    txt_ids = torch.zeros(st, 3)
    t = torch.tensor([a.t])
    gd = torch.tensor([a.guidance]) if guidance else None
    out = model(hidden_states=img[None], encoder_hidden_states=txt[None], pooled_projections=pooled[None],
                timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=gd, return_dict=False)[0][0]
    meta = {
        "doubles": a.doubles, "singles": a.singles, "grid_h": gh, "grid_w": gw, "txt_len": st,
        "t": a.t, "guidance": a.guidance, "lora": a.lora, "lora_scale": a.lora_scale,
        "img": save(a.out, "img", img), "txt": save(a.out, "txt", txt),
        "pooled": save(a.out, "pooled", pooled), "out": save(a.out, "out", out),
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta), "out std", out.std().item())


def flux_vae(ae_path, dtype=torch.float32):
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    cfg = dict(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
               up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(128, 256, 512, 512),
               layers_per_block=2, latent_channels=16, norm_num_groups=32, use_quant_conv=False,
               use_post_quant_conv=False, scaling_factor=0.3611, shift_factor=0.1159)
    vae = AutoencoderKL(**cfg).eval()
    with safe_open(ae_path, "pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    vae.load_state_dict(convert_ldm_vae_checkpoint(sd, vae.config), strict=True)
    return vae.to(dtype)


@torch.no_grad()
def cmd_vae(a):
    os.makedirs(a.out, exist_ok=True)
    vae = flux_vae(a.vae)
    g = torch.Generator().manual_seed(a.seed)
    z = torch.randn(16, a.latent_h, a.latent_w, generator=g)
    rgb = vae.decode(z[None] / 0.3611 + 0.1159).sample[0]
    meta = {"latents": save(a.out, "latents", z), "rgb": save(a.out, "rgb", rgb)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


@torch.no_grad()
def cmd_e2e(a):
    from accelerate import init_empty_weights
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    from transformers import CLIPTextModel, CLIPTokenizerFast, T5EncoderModel, T5TokenizerFast

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    dt = {"bf16": torch.bfloat16, "f32": torch.float32}[a.dtype]
    # 1. prompt embeddings in f32 on CPU, then free the encoders.
    t5_ids, clip_ids = tokenize(a.t5, a.clip, a.prompt, 512)
    clip = CLIPTextModel.from_pretrained(a.clip, torch_dtype=torch.float32).eval()
    pooled = clip(clip_ids).pooler_output
    del clip
    t5 = T5EncoderModel.from_pretrained(a.t5, torch_dtype=torch.float32).eval()
    prompt_embeds = t5(t5_ids)[0]
    del t5
    # 2. transformer, streamed into an empty model to keep one copy resident.
    with safe_open(a.transformer, "pt") as f:
        sd = {k: f.get_tensor(k).to(dt) for k in f.keys()}
    conv = convert_flux_transformer_checkpoint_to_diffusers(sd)
    del sd
    if a.lora:
        conv32 = {}
        with safe_open(a.lora, "pt") as f:
            mods = {k[len("transformer."):-len(".lora_A.weight")] for k in f.keys() if k.endswith(".lora_A.weight")}
        for m in mods:
            conv32[m + ".weight"] = conv[m + ".weight"].float()
        merge_lora_diffusers(conv32, a.lora, a.lora_scale, lambda m: True)
        for k, v in conv32.items():
            conv[k] = v.to(dt)
    with init_empty_weights():
        transformer = FluxTransformer2DModel(
            patch_size=1, in_channels=64, num_layers=19, num_single_layers=38, attention_head_dim=128,
            num_attention_heads=24, joint_attention_dim=4096, pooled_projection_dim=768,
            guidance_embeds=True, axes_dims_rope=(16, 56, 56))
    transformer.load_state_dict(conv, strict=True, assign=True)
    del conv
    transformer = transformer.to(dev).eval()
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
                                            base_shift=0.5, max_shift=1.15, base_image_seq_len=256,
                                            max_image_seq_len=4096)
    vae = flux_vae(a.vae, dt).to(dev)
    pipe = FluxPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None,
                        tokenizer_2=None, transformer=transformer)
    lh, lw = 2 * (a.height // 16), 2 * (a.width // 16)
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn(1, 16, lh, lw, generator=g)
    packed = FluxPipeline._pack_latents(noise, 1, 16, lh, lw)
    img = pipe(prompt_embeds=prompt_embeds.to(dev, dt), pooled_prompt_embeds=pooled.to(dev, dt),
               latents=packed.to(dev, dt), height=a.height, width=a.width, num_inference_steps=a.steps,
               guidance_scale=a.guidance, output_type="np").images[0]
    from PIL import Image

    Image.fromarray((img * 255).round().astype(np.uint8)).save(os.path.join(a.out, "image.png"))
    meta = {"prompt": a.prompt, "steps": a.steps, "guidance": a.guidance, "width": a.width, "height": a.height,
            "dtype": a.dtype, "device": a.device, "lora": a.lora,
            "noise": save(a.out, "noise", noise[0]), "image": save(a.out, "image", img)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="weights/image/flux1-dev")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text")
    t.add_argument("--prompt", required=True)
    t.add_argument("--t5-len", type=int, default=512)
    d = sub.add_parser("dit")
    d.add_argument("--doubles", type=int, default=2)
    d.add_argument("--singles", type=int, default=2)
    d.add_argument("--grid-h", type=int, default=4)
    d.add_argument("--grid-w", type=int, default=6)
    d.add_argument("--txt-len", type=int, default=16)
    d.add_argument("--text-dir", help="use a `text` dump's T5 states / pooled vector")
    d.add_argument("--t", type=float, default=0.7)
    d.add_argument("--guidance", type=float, default=3.5)
    d.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("vae")
    v.add_argument("--latent-h", type=int, default=8)
    v.add_argument("--latent-w", type=int, default=12)
    v.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("e2e")
    e.add_argument("--prompt", required=True)
    e.add_argument("--width", type=int, default=160)
    e.add_argument("--height", type=int, default=160)
    e.add_argument("--steps", type=int, default=20)
    e.add_argument("--guidance", type=float, default=5.0)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    e.add_argument("--dtype", default="bf16", choices=["bf16", "f32"])
    for s in (d, e):
        s.add_argument("--lora")
        s.add_argument("--lora-scale", type=float, default=1.0)
    for s in (t, d, v, e):
        s.add_argument("--out", required=True)
    a = p.parse_args()
    m = a.model_dir
    a.transformer = os.path.join(m, "flux1-dev.safetensors")
    a.t5, a.clip, a.vae = os.path.join(m, "t5"), os.path.join(m, "clip"), os.path.join(m, "vae", "ae.safetensors")
    {"text": cmd_text, "dit": cmd_dit, "vae": cmd_vae, "e2e": cmd_e2e}[a.cmd](a)


if __name__ == "__main__":
    main()
