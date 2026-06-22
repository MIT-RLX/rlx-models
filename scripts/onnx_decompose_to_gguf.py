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

"""Convert decomposed safetensors (f32) to a simple GGUF for RLX WeightLoader experiments.

Usage:
  python3 scripts/onnx_decompose_to_gguf.py weights/model.safetensors weights/model.gguf

Requires: pip install gguf safetensors numpy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: onnx_decompose_to_gguf.py in.safetensors out.gguf", file=sys.stderr)
        sys.exit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        from safetensors.numpy import load_file
        import gguf
    except ImportError as e:
        print(f"install: pip install gguf safetensors numpy ({e})", file=sys.stderr)
        sys.exit(1)

    tensors = load_file(str(src))
    writer = gguf.GGUFWriter(str(dst), "kitten-tts-mini")
    for name, arr in tensors.items():
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {dst} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
