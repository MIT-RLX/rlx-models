#!/usr/bin/env python
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# GPL-3.0-only (the rlx-vlash crate); VLASH itself is Apache-2.0.
"""Dump staged VLASH π₀ / π₀.₅ reference tensors for rlx-vlash CPU parity.

Runs the upstream VLASH policy (https://github.com/mit-han-lab/vlash) on CPU in
float32 with a fixed seed, a synthetic image / state / prompt, and an injected
Gaussian noise, then writes each intermediate as a little-endian f32 `.bin`
alongside a `manifest.json` (name → {shape, file}). The rlx-vlash `tests/parity.rs`
reads these and asserts cosine > 0.999 stage-by-stage.

Prereqs (see vlash/pyproject.toml — pins transformers @ dcddb97 + lerobot 0.4.1):

    conda create -n vlash python=3.10 && conda activate vlash
    pip install -e /path/to/vlash          # pulls the pinned transformers + lerobot
    pip install numpy

Usage:

    python vlash_ref_dump.py --variant pi05 \
        --checkpoint lerobot/pi05_base \
        --out ~/.cache/rlx-vlash/fixtures/pi05 \
        --num-images 1 --prompt "pick up the cube" --prompt-len 0 --seed 0

`--prompt-len 0` uses natural (unpadded) tokenization; a positive value pads to
that many tokens (must match the runner's `prompt_tokens`).

Stages dumped: image_chw01, pixel_values, token_ids, image_features_raw
(projector output, pre-scaling), image_features_scaled, prefix_embeds, noise,
state_padded, velocity_step0, actions_padded. Also writes checkpoint_keys.txt
(the raw `state_dict()` keys) so the crate's key remap can be validated.
"""

import argparse
import json
import os

import numpy as np
import torch


def w(out_dir, manifest, name, tensor):
    """Write one tensor as f32 .bin + record its shape in the manifest."""
    arr = tensor.detach().to(torch.float32).cpu().numpy().astype("<f4", copy=False)
    path = os.path.join(out_dir, f"{name}.bin")
    arr.tofile(path)
    manifest[name] = {"shape": list(arr.shape), "file": f"{name}.bin", "dtype": "f32"}
    print(f"  dumped {name:22s} shape={list(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pi0", "pi05"], required=True)
    ap.add_argument("--checkpoint", required=True, help="HF repo id or local dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--prompt", default="do the task")
    ap.add_argument("--prompt-len", type=int, default=0, help="0 = natural length")
    ap.add_argument("--img-h", type=int, default=224)
    ap.add_argument("--img-w", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument(
        "--tokens",
        default="",
        help="comma-separated fixed token ids; skips the (gated) PaliGemma tokenizer",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    fixed_tokens = [int(x) for x in args.tokens.split(",") if x.strip() != ""]
    if fixed_tokens:
        # Bypass the gated `google/paligemma-3b-pt-224` tokenizer: the policy
        # constructor calls AutoTokenizer.from_pretrained unconditionally, but
        # for numeric parity we only need identical token ids on both sides.
        import transformers

        class _DummyTok:
            def __call__(self, *a, **k):
                raise RuntimeError("tokenizer bypassed (--tokens given)")

        transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: _DummyTok())

    # --- load policy on CPU, float32, unfused (matches the crate) ---
    if args.variant == "pi0":
        from vlash.policies.pi0.configuration_pi0 import (
            PI0ActionExpertConfig as ExpCfg,
            PI0Config as Cfg,
            PI0VLMConfig as VlmCfg,
        )
        from vlash.policies.pi0.modeling_pi0 import PI0Policy as Policy
    else:
        from vlash.policies.pi05.configuration_pi05 import (
            PI05ActionExpertConfig as ExpCfg,
            PI05Config as Cfg,
            PI05VLMConfig as VlmCfg,
        )
        from vlash.policies.pi05.modeling_pi05 import PI05Policy as Policy

    # lerobot's `from_pretrained` decode does not run the dataclass __init__, so
    # it drops fields absent from config.json (state_cond, vlm_config, …). The
    # `lerobot/pi05_base` config.json matches the dataclass defaults (verified:
    # chunk 50, dims 32, 10 steps, gemma_2b/gemma_300m, state_cond absent→False),
    # so a fresh `Cfg()` reproduces it exactly. (Repair nested cfgs defensively.)
    cfg = Cfg()
    if getattr(cfg, "vlm_config", None) is None:
        cfg.vlm_config = VlmCfg()
    if getattr(cfg, "action_expert_config", None) is None:
        cfg.action_expert_config = ExpCfg()
    cfg.dtype = "float32"
    cfg.device = "cpu"
    # Enable Q/K/V + gate/up fusion (VLASH's default). It is numerically
    # identical to the unfused path — a fused projection is just the stacked
    # per-head matmul — and pi0's `denoise_step` reads `self_attn.qkv_proj`, so
    # it *requires* fusion to run. The rlx crate loads the unfused checkpoint
    # weights and matches either way.
    cfg.fuse_qkv = True
    cfg.fuse_gate_up = True
    policy = Policy.from_pretrained(args.checkpoint, config=cfg)
    policy.eval()
    policy.model.float()
    m = policy.model

    manifest = {"variant": args.variant, "seed": args.seed}

    with open(os.path.join(args.out, "checkpoint_keys.txt"), "w") as f:
        for k in policy.state_dict().keys():
            f.write(k + "\n")

    # --- synthetic inputs (deterministic) ---
    C, H, W = 3, args.img_h, args.img_w
    imgs01 = [
        torch.rand(1, C, H, W, generator=torch.Generator().manual_seed(args.seed + i))
        for i in range(args.num_images)
    ]
    w(args.out, manifest, "image_chw01", imgs01[0][0])  # first image, [C,H,W] in [0,1]

    # Preprocess exactly like PI0Policy.prepare_images.
    from vlash.policies.pi0.utils import resize_with_pad

    images, img_masks = [], []
    for im in imgs01:
        pv = resize_with_pad(im, *policy.config.image_resolution, pad_value=0)
        pv = pv * 2.0 - 1.0
        images.append(pv.float())
        img_masks.append(torch.ones(1, dtype=torch.bool))
    w(args.out, manifest, "pixel_values", images[0][0])  # [C,224,224] in [-1,1]

    # Tokenize prompt, or use the fixed ids (when the tokenizer is bypassed).
    if fixed_tokens:
        tokens = torch.tensor([fixed_tokens], dtype=torch.long)
        masks = torch.ones_like(tokens, dtype=torch.bool)
    else:
        tok = policy.language_tokenizer(
            [args.prompt if args.prompt.endswith("\n") else args.prompt + "\n"],
            padding="max_length" if args.prompt_len > 0 else "longest",
            max_length=args.prompt_len if args.prompt_len > 0 else policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        tokens = tok["input_ids"]
        masks = tok["attention_mask"].bool()
    w(args.out, manifest, "token_ids", tokens.float())
    w(args.out, manifest, "token_mask", masks.float())

    # State (raw dim = max_state_dim here for simplicity) + injected noise.
    state = torch.randn(1, policy.config.max_state_dim)
    w(args.out, manifest, "state_padded", state)
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim)
    w(args.out, manifest, "noise", noise)

    # --- vision: raw projector output (pre /√hidden) via a hook ---
    proj_out = {}
    proj_mod = m.vlm.model.multi_modal_projector
    h = proj_mod.register_forward_hook(lambda _mod, _in, out: proj_out.__setitem__("v", out))
    image_features_scaled = m.vlm.model.get_image_features(images[0])
    h.remove()
    w(args.out, manifest, "image_features_raw", proj_out["v"][0])  # [256, 2048]
    w(args.out, manifest, "image_features_scaled", image_features_scaled[0])

    # --- prefix embeddings ---
    prefix_embs, prefix_pad, prefix_att = m.prefix_embedder(images, img_masks, tokens, masks)
    w(args.out, manifest, "prefix_embeds", prefix_embs[0])  # [P, 2048]
    w(args.out, manifest, "prefix_pad", prefix_pad[0].float())

    # --- sample_actions with injected noise; capture velocity_step0 ---
    caps = {"n": 0}
    orig = m.denoise_step

    def wrapped(*a, **k):
        v = orig(*a, **k)
        if caps["n"] == 0:
            caps["v0"] = v.clone()
        caps["n"] += 1
        return v

    m.denoise_step = wrapped
    actions = m.sample_actions(
        images, img_masks, tokens, masks, state, noise=noise, num_steps=args.num_steps
    )
    m.denoise_step = orig
    if "v0" in caps:
        w(args.out, manifest, "velocity_step0", caps["v0"][0])  # [chunk, 32]
    w(args.out, manifest, "actions_padded", actions[0])  # [chunk, 32], normalized

    manifest["num_images"] = args.num_images
    manifest["prompt"] = args.prompt
    manifest["prompt_len"] = int(tokens.shape[1])
    manifest["num_steps"] = args.num_steps
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} entries → {args.out}/manifest.json")


if __name__ == "__main__":
    main()
