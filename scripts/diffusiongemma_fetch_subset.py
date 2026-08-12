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
"""Fetch a *subset* of DiffusionGemma's real weights, by tensor.

The full checkpoint is 51 GB and its routed experts alone are ~91 GB as f32, so
a whole-model run needs the paged expert path. Individual subsystems are much
smaller and can be tested for real right now: the vision tower is ~1.2 GB, one
text layer ~1.6 GB.

Safetensors headers carry per-tensor byte offsets, so this pulls exactly the
tensors asked for with HTTP Range requests instead of downloading whole shards:

    python3 scripts/diffusiongemma_fetch_subset.py .weights/dg-vision --subset vision
    python3 scripts/diffusiongemma_fetch_subset.py .weights/dg-layer0 --subset layer0

Tensors are written out in their original bf16; `WeightMap` converts on load.
"""

import argparse
import json
import pathlib
import struct
import subprocess
import sys

REPO = "google/diffusiongemma-26B-A4B-it"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"

# bytes-per-element for the dtypes this checkpoint uses
DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1}


def curl(url: str, rng: str | None = None) -> bytes:
    cmd = ["curl", "-sL", "--fail", "--retry", "3"]
    if rng:
        cmd += ["-H", f"Range: bytes={rng}"]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url} [{rng}]: {r.stderr[:200]!r}")
    return r.stdout


def shard_header(shard: str) -> tuple[dict, int]:
    """Return (header, data_start_offset)."""
    n = struct.unpack("<Q", curl(BASE + shard, "0-7")[:8])[0]
    raw = curl(BASE + shard, f"8-{8 + n - 1}")
    return json.loads(raw.decode()), 8 + n


def select(index: dict, subset: str) -> list[str]:
    keys = list(index["weight_map"])
    if subset == "vision":
        return [
            k
            for k in keys
            if k.startswith("model.encoder.vision_tower.")
            or k.startswith("model.encoder.embed_vision.")
        ]
    if subset.startswith("layer"):
        n = subset[len("layer") :]
        pre = f"model.decoder.layers.{n}."
        enc = f"model.encoder.language_model.layers.{n}.layer_scalar"
        return [k for k in keys if k.startswith(pre) or k == enc]
    if subset == "embed":
        return [
            k
            for k in keys
            if k in ("model.decoder.embed_tokens.weight", "model.decoder.norm.weight")
            or k.startswith("model.decoder.self_conditioning.")
        ]
    raise SystemExit(f"unknown subset {subset!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--subset", required=True, help="vision | layer<N> | embed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    index = json.loads(curl(BASE + "model.safetensors.index.json").decode())
    wanted = select(index, args.subset)
    if not wanted:
        raise SystemExit(f"subset {args.subset!r} matched no tensors")

    by_shard: dict[str, list[str]] = {}
    for k in wanted:
        by_shard.setdefault(index["weight_map"][k], []).append(k)

    tensors: dict[str, dict] = {}
    blobs: dict[str, bytes] = {}
    total = 0
    for shard, keys in sorted(by_shard.items()):
        header, data_start = shard_header(shard)
        # One Range request per tensor. Contiguous runs could be coalesced, but
        # tensors here are MBs each so per-tensor requests are already efficient.
        for k in sorted(keys):
            meta = header[k]
            s, e = meta["data_offsets"]
            blob = curl(BASE + shard, f"{data_start + s}-{data_start + e - 1}")
            want = e - s
            if len(blob) != want:
                raise RuntimeError(f"{k}: got {len(blob)} bytes, expected {want}")
            blobs[k] = blob
            tensors[k] = {"dtype": meta["dtype"], "shape": meta["shape"]}
            total += want
            print(f"  {k}  {meta['shape']} {meta['dtype']} ({want / 1e6:.1f} MB)", flush=True)

    # Re-emit as a single safetensors file.
    out_header: dict = {}
    off = 0
    for k in sorted(blobs):
        n = len(blobs[k])
        out_header[k] = {
            "dtype": tensors[k]["dtype"],
            "shape": tensors[k]["shape"],
            "data_offsets": [off, off + n],
        }
        off += n
    hdr = json.dumps(out_header, separators=(",", ":")).encode()
    pad = (8 - (len(hdr) % 8)) % 8
    hdr += b" " * pad

    path = args.out / "model.safetensors"
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr)))
        fh.write(hdr)
        for k in sorted(blobs):
            fh.write(blobs[k])

    (args.out / "config.json").write_bytes(curl(BASE + "config.json"))
    print(
        f"\nwrote {len(blobs)} tensors ({total / 1e9:.2f} GB) to {path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
