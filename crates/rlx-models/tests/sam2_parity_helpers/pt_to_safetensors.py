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

#
#
# Convert a sam2 .pt checkpoint to .safetensors with key names preserved.
# The sam2 package's `build_sam2()` wants the .pt format; the Rust
# WeightMap reads safetensors — so for parity we keep both, with
# identical key sets, so neither side has to remap.
#
# Usage:
#   python pt_to_safetensors.py /path/in.pt /path/out.safetensors

import sys
import torch
from safetensors.torch import save_file


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: pt_to_safetensors.py <in.pt> <out.safetensors>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    # sam2 ckpts wrap weights under "model" key.
    state = ckpt.get("model", ckpt)
    # safetensors requires contiguous tensors with no shared storage; clone defensively.
    cleaned = {k: v.detach().contiguous().clone() for k, v in state.items() if torch.is_tensor(v)}
    print(f"[+] {len(cleaned)} tensors → {dst}")
    save_file(cleaned, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
