"""Dump the DeepSeek-V4.1 vision tower (ViT + aligner) reference outputs."""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RLX_REF_NOQUANT", "1")
import prng, dump as D
import model as M

N_H, N_W = 3, 4  # n_h is odd on purpose: the aligner must pad it


def main():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    args = D.small_args(vision=True)
    args.vision_n_layers = 2
    args.vision_dim = 32
    args.vision_n_heads = 2
    args.vision_inter_dim = 48
    args.vision_patch_size = 2
    args.vision_downsample_ratio = 2
    args.vision_rope_theta = 10000.0
    m = M.Transformer(args, tokenizer=object())
    shapes = {}
    for name, p in m.named_parameters():
        if not (name.startswith("vision.") or name.startswith("aligner.")):
            continue
        shapes[name] = list(p.shape)
        p.data = torch.from_numpy(prng.param_values(name, tuple(p.shape)).copy())
    m.float().eval()

    patches = torch.from_numpy(
        prng.param_values("patches", (N_H * N_W, 3, args.vision_patch_size, args.vision_patch_size)).copy()
    )
    with torch.inference_mode():
        vit = m.vision(patches, N_H, N_W)
        out = m.aligner(vit, N_H, N_W)
    json.dump(
        {
            "shapes": shapes,
            "n_h": N_H,
            "n_w": N_W,
            "vision_config": {
                "vision_n_layers": args.vision_n_layers,
                "vision_dim": args.vision_dim,
                "vision_n_heads": args.vision_n_heads,
                "vision_inter_dim": args.vision_inter_dim,
                "vision_patch_size": args.vision_patch_size,
                "vision_downsample_ratio": args.vision_downsample_ratio,
                "vision_rope_theta": args.vision_rope_theta,
                "vision_max_n_token": args.vision_max_n_token,
                "vision_min_pixels": args.vision_min_pixels,
                "dim": args.dim,
            },
            "patches": patches.reshape(-1).tolist(),
            "vit": vit.detach().reshape(-1).tolist(),
            "embeds": out.detach().reshape(-1).tolist(),
        },
        open("dsv41_vision_ref.json", "w"),
        separators=(",", ":"),
    )
    print("vit", tuple(vit.shape), "embeds", tuple(out.shape))


if __name__ == "__main__":
    main()
