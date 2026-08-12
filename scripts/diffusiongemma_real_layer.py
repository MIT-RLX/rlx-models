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
"""Reference forward for one *real* DiffusionGemma text layer.

Pairs with `diffusiongemma_fetch_subset.py --subset layer0`, which pulls one
layer (~1.63 GB) by tensor — the routed experts are 761 M of its 800 M
parameters, so this is the cheapest way to exercise the real 128-expert MoE:

    python3 scripts/diffusiongemma_fetch_subset.py /w/dg-layer0 --subset layer0
    python3 scripts/diffusiongemma_real_layer.py /w/dg-layer0
    RLX_DG_REAL_LAYER_DIR=/w/dg-layer0 \\
        cargo test -p rlx-diffusiongemma --test real_layer -- --nocapture

Layer 0 is a *sliding* layer, so it carries a real `v_proj` and the 16×256 / 8
KV-head geometry. Router statistics are printed because a router that collapses
onto a few experts would still look numerically fine in a cosine check.
"""

import argparse
import json
import pathlib
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from diffusiongemma_reference import (  # noqa: E402
    Cfg,
    causal_sliding_mask,
    layer_forward,
    rope_tables,
    router,
)

LAYER = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = args.dir

    cfg = Cfg(json.loads((d / "config.json").read_text()))
    w = {k: v.float() for k, v in load_file(str(d / "model.safetensors")).items()}
    print(f"loaded {len(w)} real tensors", file=sys.stderr)

    gen = torch.Generator().manual_seed(args.seed)
    seq = args.seq
    # Post-embedding hidden states are ~sqrt(hidden) times the embedding rows,
    # so use a magnitude in that ballpark rather than unit normal.
    x = torch.randn(1, seq, cfg.hidden_size, generator=gen) * 20.0

    cos, sin = rope_tables(cfg, LAYER, 0, seq)
    mask = causal_sliding_mask(seq, cfg.sliding_window)
    scalar_key = f"model.encoder.language_model.layers.{LAYER}.layer_scalar"

    with torch.no_grad():
        out, (k_tap, v_tap) = layer_forward(
            w, cfg, LAYER, x, cos, sin, mask, scalar_key
        )
        # Router diagnostics on the same input the layer routed with.
        flat = None
        # Recompute the post-attention residual the router actually sees is
        # internal to layer_forward; instead report routing of the layer input,
        # which is enough to show the router is not degenerate.
        top_i, top_w = router(w, f"model.decoder.layers.{LAYER}", cfg, x.reshape(-1, cfg.hidden_size))
        _ = flat

    hist = torch.bincount(top_i.flatten(), minlength=cfg.num_experts)
    print(
        f"router: {int((hist > 0).sum())}/{cfg.num_experts} experts hit over "
        f"{seq} tokens x top-{cfg.top_k_experts}; "
        f"weight range [{top_w.min():.4f}, {top_w.max():.4f}]",
        file=sys.stderr,
    )

    def dump(name, t):
        (d / f"{name}.bin").write_bytes(
            t.detach().float().contiguous().numpy().tobytes()
        )

    dump("layer_in", x)
    dump("layer_out", out)
    dump("layer_k", k_tap)
    dump("layer_v", v_tap)
    (d / "layer_meta.json").write_text(
        json.dumps(
            {
                "layer": LAYER,
                "seq": seq,
                "hidden": cfg.hidden_size,
                "head_dim": cfg.head_dim_at(LAYER),
                "kv_heads": cfg.kv_heads_at(LAYER),
                "is_full": cfg.is_full(LAYER),
                "experts_hit": int((hist > 0).sum()),
            },
            indent=2,
        )
    )
    print(
        f"layer {LAYER} out {tuple(out.shape)}  absmax {out.abs().max():.3f}  "
        f"rms {out.pow(2).mean().sqrt():.3f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
