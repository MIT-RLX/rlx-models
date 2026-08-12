#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Reference forward for the *real* DiffusionGemma vision tower.

Pairs with `diffusiongemma_fetch_subset.py --subset vision`, which pulls the
~1.15 GB of real tower weights by tensor:

    python3 scripts/diffusiongemma_fetch_subset.py .weights/dg-vision --subset vision
    python3 scripts/diffusiongemma_real_vision.py .weights/dg-vision
    RLX_DG_REAL_VISION_DIR=.weights/dg-vision \\
        cargo test -p rlx-diffusiongemma --test real_vision -- --nocapture

Everything the tower needs is emitted alongside the weights (image, patch
tensor, positions, pooling matrix, reference soft tokens) so the Rust side
replays the identical input and the comparison isolates the tower itself.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from diffusiongemma_reference import Cfg, pool_matrix, vision_tower  # noqa: E402

# A small budget keeps the run quick; the tower is resolution-agnostic and 70 is
# one of the processor's supported soft-token budgets.
MAX_SOFT_TOKENS = 70
POOLING_KERNEL = 3
PATCH = 16


def target_size(h, w, patch, max_patches, pool):
    """`get_aspect_ratio_preserving_size`, mirrored by the Rust preprocessor."""
    factor = ((max_patches * patch**2) / (h * w)) ** 0.5
    side = pool * patch
    return int(factor * h // side) * side, int(factor * w // side) * side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=pathlib.Path, help="subset dir from the fetcher")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = args.dir

    cfg = Cfg(json.loads((d / "config.json").read_text()))
    w = load_file(str(d / "model.safetensors"))
    w = {k: v.float() for k, v in w.items()}
    print(f"loaded {len(w)} real tensors", file=sys.stderr)

    # Deterministic RGB image, then the real processor's resize path.
    rng = np.random.RandomState(args.seed)
    src_h, src_w = 300, 200
    img_u8 = rng.randint(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
    max_patches = MAX_SOFT_TOKENS * POOLING_KERNEL**2
    th, tw = target_size(src_h, src_w, PATCH, max_patches, POOLING_KERNEL)
    resized = np.asarray(Image.fromarray(img_u8, "RGB").resize((tw, th), Image.BICUBIC))
    print(f"image {src_h}x{src_w} -> {th}x{tw}", file=sys.stderr)

    # Patchify: (nph, npw, patch, patch, C) with channel innermost.
    nph, npw = th // PATCH, tw // PATCH
    pix = resized.astype(np.float32) / 255.0
    patches = (
        pix.reshape(nph, PATCH, npw, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(nph * npw, -1)
    )
    positions = [(x, y) for y in range(nph) for x in range(npw)]
    n_patches = len(positions)
    n_soft = n_patches // POOLING_KERNEL**2
    pool = pool_matrix(positions, POOLING_KERNEL, n_soft)
    print(f"{n_patches} patches -> {n_soft} soft tokens", file=sys.stderr)

    with torch.no_grad():
        soft, taps = vision_tower(
            w, cfg, torch.from_numpy(patches)[None], positions, pool, n_soft
        )

    def dump(name, t):
        arr = t.detach().float().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
        (d / f"{name}.bin").write_bytes(arr.astype(np.float32).tobytes())

    dump("real_pixels", torch.from_numpy(patches))
    dump("real_pool", pool)
    dump("real_soft_tokens", soft)
    dump("real_patch_embed", taps["patch_embed"])
    dump("real_encoder_out", taps["encoder_out"])
    dump("real_pooled", taps["pooled"])
    (d / "real_image.bin").write_bytes(img_u8.tobytes())

    meta = {
        "src_h": src_h,
        "src_w": src_w,
        "target_h": th,
        "target_w": tw,
        "patches": n_patches,
        "soft_tokens": n_soft,
        "grid_cols": npw,
        "grid_rows": nph,
        "max_soft_tokens": MAX_SOFT_TOKENS,
        "pooling_kernel_size": POOLING_KERNEL,
        "patch_size": PATCH,
    }
    (d / "real_meta.json").write_text(json.dumps(meta, indent=2))
    s = soft.flatten()
    print(
        f"soft tokens {tuple(soft.shape)}  "
        f"absmax {s.abs().max():.4f}  rms {s.pow(2).mean().sqrt():.4f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
