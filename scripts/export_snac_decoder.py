#!/usr/bin/env python3
"""Export SNAC 24 kHz decoder + quantizer weights to safetensors for rlx-orpheus.

Usage:
  python3 scripts/export_snac_decoder.py [--repo hubertsiuzdak/snac_24khz] [--out DIR]

Requires: `pip install snac torch safetensors`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def flatten_weight_norm(state: dict, prefix: str, export: dict, torch) -> None:
    g_key = f"{prefix}.parametrizations.weight.original0"
    v_key = f"{prefix}.parametrizations.weight.original1"
    b_key = f"{prefix}.bias"
    if g_key not in state or v_key not in state:
        return
    g = state[g_key]
    v = state[v_key]
    # Match PyTorch weight_norm: normalize `v` over all dims except out_channels (dim 0).
    v_norm = torch.linalg.vector_norm(v, dim=tuple(range(1, v.ndim)), keepdim=True)
    w = g * (v / v_norm)
    export[f"{prefix}.weight"] = w.contiguous()
    if b_key in state:
        export[f"{prefix}.bias"] = state[b_key].contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="hubertsiuzdak/snac_24khz")
    parser.add_argument("--out", type=Path, default=Path("crates/rlx-orpheus/weights"))
    args = parser.parse_args()

    try:
        import torch
        from safetensors.torch import save_file
        from snac import SNAC
    except ImportError as e:
        print(f"error: {e}\ninstall: pip install snac torch safetensors", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    model = SNAC.from_pretrained(args.repo).eval()
    state = model.state_dict()
    export: dict = {}

    for k, v in state.items():
        if k.endswith(".alpha"):
            export[k] = v.contiguous()

    for i in range(model.n_codebooks):
        export[f"quantizer.quantizers.{i}.codebook.weight"] = state[
            f"quantizer.quantizers.{i}.codebook.weight"
        ].contiguous()

    prefixes = {
        k.rsplit(".parametrizations.weight.original0", 1)[0]
        for k in state
        if "parametrizations.weight.original0" in k
    }
    for p in sorted(prefixes):
        flatten_weight_norm(state, p, export, torch)

    cfg = {
        "sampling_rate": model.sampling_rate,
        "encoder_dim": model.encoder_dim,
        "encoder_rates": model.encoder_rates,
        "decoder_dim": model.decoder_dim,
        "decoder_rates": model.decoder_rates,
        "attn_window_size": model.attn_window_size,
        "codebook_size": model.codebook_size,
        "codebook_dim": model.codebook_dim,
        "vq_strides": model.vq_strides,
        "noise": True,
        "depthwise": True,
        "latent_dim": model.latent_dim,
    }

    st_path = args.out / "snac_24khz_decoder.safetensors"
    cfg_path = args.out / "snac_24khz_decoder_config.json"
    save_file(export, st_path)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {len(export)} tensors -> {st_path}")
    print(f"wrote config -> {cfg_path}")

    # Optional parity fixtures (torch.manual_seed(42), same codes as tests).
    try:
        import numpy as np

        torch.manual_seed(42)
        noises: list = []
        orig_randn = torch.randn

        def capture_randn(*shape, **kwargs):
            t = orig_randn(*shape, **kwargs)
            noises.append(t.detach().cpu().numpy())
            return t

        torch.randn = capture_randn  # type: ignore[assignment]
        codes_path = args.out / "ref_codes.json"
        if codes_path.is_file():
            codes = json.loads(codes_path.read_text())
            c0 = torch.tensor(codes["codes_0"])
            c1 = torch.tensor(codes["codes_1"])
            c2 = torch.tensor(codes["codes_2"])
            with torch.inference_mode():
                z_q = model.quantizer.from_codes([c0, c1, c2])
                audio = model.decoder(z_q).squeeze().numpy()
            np.save(args.out / "ref_decode.npy", audio)
            for i, n in enumerate(noises):
                np.save(args.out / f"ref_noise_{i}.npy", n)
            print(f"wrote parity fixtures: ref_decode.npy + {len(noises)} noise planes")
    except Exception as e:
        print(f"note: parity fixtures skipped ({e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
