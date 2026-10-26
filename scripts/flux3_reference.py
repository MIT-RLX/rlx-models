#!/usr/bin/env python3
"""FLUX 3 Image Klein reference dumps for rlx-flux3 parity.

Runs Black Forest Labs' own inference code (github.com/black-forest-labs/
flux-3-image-klein, passed as --bfl-repo) on CPU and writes raw little-endian
f32 / i32 `.bin` files plus a `meta.json` into --out/<case>:

  text   caption and "" -> Qwen3-VL token ids + stacked text tokens [L, 20480]
  t2i    --steps CFG steps at --size from torch.randn(seed) noise: first-step
         velocity of both samples, final latent tokens, decoded RGB
  edit   one reference image (the t2i output) -> its latent tokens, and the
         first-step velocity of an edit
  flow_map  (--flow-map-dir: the distilled checkpoint) the first-step velocity
         and a --fm-steps run without CFG, from the same caption and noise;
         --skip-base reuses an earlier `text` dump and skips t2i / edit

Weights stay bf16 (as shipped) and every nn.Linear runs in f32 on an
upcast copy, so peak memory is the bf16 checkpoint (~11 GB for the DiT)
rather than its f32 copy. Activations are f32 throughout; the casts the
reference makes explicitly (bf16 timesteps, bf16 text / reference tokens,
bf16-rounded noise) are kept.

Needs: torch, transformers (Qwen3-VL), safetensors, numpy, pillow.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def save(out, name, arr, dtype=np.float32):
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    a.tofile(os.path.join(out, name + ".bin"))
    return list(a.shape)


def f32_linears():
    """Every nn.Linear computes in f32 from its stored (bf16) weight."""

    def forward(self, x):
        b = None if self.bias is None else self.bias.float()
        return F.linear(x.float(), self.weight.float(), b)

    nn.Linear.forward = forward


@torch.no_grad()
def encode_texts(model_dir, prompts, max_tokens):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from flux3 import text_encoder as te

    path = os.path.join(model_dir, "text_encoder")
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side = "right"
    tok = processor.tokenizer
    out = []
    for prompt in prompts:
        text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        real = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"].shape[1]
        length = min(-(-real // te.PAD_MULTIPLE) * te.PAD_MULTIPLE, max_tokens)
        inputs = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"]
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        embeds = model.get_input_embeddings()(inputs["input_ids"]).float()
        output = model(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([output.hidden_states[k].float() for k in te.OUTPUT_LAYERS], dim=1)
        tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
        ids = torch.stack([torch.zeros(length)] * 3 + [torch.arange(length).float()], dim=-1)
        out.append((inputs["input_ids"][0, :real], tokens, ids))
    del model
    return out


def load_dit(model_dir):
    from flux3.checkpoint import is_flow_map, load_checkpoint
    from flux3.dit import DiT

    path = os.path.join(model_dir, "model.safetensors")
    with torch.device("meta"):
        dit = DiT(is_flow_map(path))
    return load_checkpoint(dit, path, "cpu", torch.bfloat16)


def as_f32(packed):
    packed.data = {m: v.float() for m, v in packed.data.items()}
    return packed


def patch_timestep_embedding():
    """The reference returns the embedding as bf16 (`.to(t)`); feed it to the
    f32 linears as f32 (same values)."""
    from flux3 import dit as d

    orig = d.timestep_embedding
    d.timestep_embedding = lambda t, *a, **k: orig(t, *a, **k).float()


@torch.no_grad()
def flow_map_case(a, text):
    from flux3.autoencoder import latent_ids, latent_shape
    from flux3.pipeline import pack, scaled_ids
    from flux3.sampling import denoise_flow_map, timesteps

    dit = load_dit(a.flow_map_dir)
    assert dit.flow_map, f"{a.flow_map_dir} is not a flow-map checkpoint"
    h, w = latent_shape(a.size, a.size)
    noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(a.seed))
    noise = noise.to(torch.bfloat16).float()
    image = (noise, scaled_ids(latent_ids(h, w)))
    schedule = timesteps(a.fm_steps, a.alpha)
    packed = as_f32(pack([[text, None, image]]))
    packed.timesteps["image"].fill_(schedule[0].item())
    v0 = dit(packed, timesteps_ref=schedule[1].reshape(1))
    packed = as_f32(pack([[text, None, image]]))
    tokens = denoise_flow_map(dit, packed, schedule)
    d = os.path.join(a.out, "flow_map")
    os.makedirs(d, exist_ok=True)
    meta = {
        "size": a.size,
        "steps": a.fm_steps,
        "alpha": a.alpha,
        "seed": a.seed,
        "schedule": [float(x) for x in schedule],
        "velocity0": save(d, "velocity0", v0.float()),
        "latents": save(d, "latents", tokens.float()),
    }
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"flow_map: velocity0 {tuple(v0.shape)}, latents {tuple(tokens.shape)}", flush=True)
    del dit


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bfl-repo", required=True, help="checkout of black-forest-labs/flux-3-image-klein")
    p.add_argument("--model-dir", required=True, help="flux-3-image-klein-base download")
    p.add_argument("--out", required=True)
    p.add_argument("--caption", required=True, help="caption text, or @FILE")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--alpha", type=float, default=6.93)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--flow-map-dir", help="flux-3-image-klein (distilled) download, for the flow_map case")
    p.add_argument("--fm-steps", type=int, default=4)
    p.add_argument("--skip-base", action="store_true", help="reuse --out/text, run only the flow_map case")
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.bfl_repo, "src"))
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    f32_linears()
    patch_timestep_embedding()

    from flux3.autoencoder import AutoEncoder, latent_ids, latent_shape
    from flux3.checkpoint import load_checkpoint
    from flux3.pipeline import pack, scaled_ids
    from flux3.sampling import denoise, timesteps

    caption = open(a.caption[1:]).read().strip() if a.caption.startswith("@") else a.caption

    # text
    d_text = os.path.join(a.out, "text")
    if a.skip_base:
        shape = json.load(open(os.path.join(d_text, "meta.json")))["cond_tokens"]
        c_tok = np.fromfile(os.path.join(d_text, "cond_tokens.bin"), dtype=np.float32).reshape(shape)
        c_tok = torch.from_numpy(c_tok).to(torch.bfloat16)
        c_pos = torch.stack([torch.zeros(c_tok.shape[0])] * 3 + [torch.arange(c_tok.shape[0]).float()], dim=-1)
    else:
        os.makedirs(d_text, exist_ok=True)
        (u_ids, u_tok, u_pos), (c_ids, c_tok, c_pos) = encode_texts(a.model_dir, ["", caption], a.max_tokens)
        meta = {
            "caption": caption,
            "uncond_ids": save(d_text, "uncond_ids", u_ids, np.int32),
            "uncond_tokens": save(d_text, "uncond_tokens", u_tok.float()),
            "cond_ids": save(d_text, "cond_ids", c_ids, np.int32),
            "cond_tokens": save(d_text, "cond_tokens", c_tok.float()),
        }
        json.dump(meta, open(os.path.join(d_text, "meta.json"), "w"), indent=1)
        print(f"text: uncond {tuple(u_tok.shape)}, cond {tuple(c_tok.shape)}", flush=True)

    if not a.skip_base:
        # t2i
        dit = load_dit(a.model_dir)
        ae = load_checkpoint(AutoEncoder(), os.path.join(a.model_dir, "ae.safetensors"), "cpu", torch.float32)
        h, w = latent_shape(a.size, a.size)
        noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(a.seed))
        noise = noise.to(torch.bfloat16).float()
        ids = latent_ids(h, w)
        image = (noise, scaled_ids(ids))
        text = [(u_tok, u_pos), (c_tok, c_pos)]
        schedule = timesteps(a.steps, a.alpha)

        def first_velocity(cond):
            packed = as_f32(pack([[text[0], cond, image], [text[1], cond, image]]))
            packed.timesteps["image"].fill_(schedule[0].item())
            return dit(packed)

        d_t2i = os.path.join(a.out, "t2i")
        os.makedirs(d_t2i, exist_ok=True)
        v0 = first_velocity(None)
        packed = as_f32(pack([[text[0], None, image], [text[1], None, image]]))
        tokens = denoise(dit, packed, schedule, a.guidance)
        rgb = ae.decode(tokens.float(), ids)
        meta = {
            "size": a.size,
            "steps": a.steps,
            "guidance": a.guidance,
            "alpha": a.alpha,
            "seed": a.seed,
            "noise": save(d_t2i, "noise", noise),
            "schedule": [float(x) for x in schedule],
            "velocity0": save(d_t2i, "velocity0", v0.float()),
            "latents": save(d_t2i, "latents", tokens.float()),
            "rgb": save(d_t2i, "rgb", rgb.float()),
        }
        json.dump(meta, open(os.path.join(d_t2i, "meta.json"), "w"), indent=1)
        print(f"t2i: velocity0 {tuple(v0.shape)}, latents {tuple(tokens.shape)}", flush=True)

        # edit: the t2i output as the single reference (already on the 16-px grid)
        pixels = ((rgb.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        ref = pixels.float() / 127.5 - 1.0
        r_tok, r_ids = ae.encode(ref)
        r_ids = scaled_ids(r_ids)
        r_ids[:, 3] = -1
        cond = (r_tok.to(torch.bfloat16), r_ids)
        d_edit = os.path.join(a.out, "edit")
        os.makedirs(d_edit, exist_ok=True)
        v_edit = first_velocity(cond)
        meta = {
            "size": a.size,
            "ref_rgb8": save(d_edit, "ref_rgb8", pixels.permute(1, 2, 0).numpy(), np.uint8),
            "ref_tokens": save(d_edit, "ref_tokens", r_tok.float()),
            "velocity0": save(d_edit, "velocity0", v_edit.float()),
        }
        json.dump(meta, open(os.path.join(d_edit, "meta.json"), "w"), indent=1)
        print(f"edit: ref tokens {tuple(r_tok.shape)}, velocity0 {tuple(v_edit.shape)}", flush=True)


    if a.flow_map_dir:
        flow_map_case(a, (c_tok, c_pos))


if __name__ == "__main__":
    main()
