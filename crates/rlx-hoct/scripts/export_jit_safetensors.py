#!/usr/bin/env python3
"""Export HOCT TorchScript general_v0.pt weights to safetensors."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pt", type=Path, help="Path to general_v0.pt")
    p.add_argument("-o", "--out", type=Path, required=True)
    args = p.parse_args()
    m = torch.jit.load(str(args.pt), map_location="cpu")
    # Clone so shared RoPE buffers (block.pos_enc ≡ attn.pos_enc) are unique on disk.
    sd = {k: v.detach().contiguous().cpu().float().clone() for k, v in m.state_dict().items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(args.out))
    print(f"wrote {len(sd)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
