#!/usr/bin/env python3
"""Rename an mlx-community gemma-3n checkpoint to the naming mlx-lm 0.31.x expects.

The published `mlx-community/gemma-3n-E2B-it-4bit` safetensors names its text LM
tensors `language_model.model.<inner>` (+ `vision_tower.*` / `audio_tower.*`),
but mlx-lm 0.31.x's gemma3n `Model` tree wants `model.language_model.<inner>`
and its `sanitize` is text-only. This copies the RAW tensor bytes (preserving the
uint32 packed codes + bf16 scales/biases exactly), renaming
`language_model.model.` → `model.language_model.` and dropping vision/audio, so
mlx-lm can load it as an oracle.

The weights are numerically identical to the original — only the keys change — so
logits from mlx-lm on this copy match rlx running on the original checkpoint.

Usage:  python3 scripts/gemma3n_mlxlm_rename.py <src_dir> <dst_dir>
"""
import json
import os
import struct
import sys


def newname(k: str):
    if k.startswith("language_model.model."):
        return "model.language_model." + k[len("language_model.model."):]
    # drop vision_tower / audio_tower / embed_vision / embed_audio (text-only oracle)
    return None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    srcf = os.path.join(src, "model.safetensors")
    with open(srcf, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_base = 8 + n

        meta = hdr.get("__metadata__", {})
        kept = []
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            nk = newname(k)
            if nk is not None:
                kept.append((nk, k, v))
        kept.sort()

        # Build new header with contiguous offsets over kept tensors only.
        new_hdr = {"__metadata__": meta}
        offset = 0
        plan = []  # (src_begin, src_end, length)
        for nk, ok, v in kept:
            b0, b1 = v["data_offsets"]
            length = b1 - b0
            new_hdr[nk] = {
                "dtype": v["dtype"],
                "shape": v["shape"],
                "data_offsets": [offset, offset + length],
            }
            plan.append((data_base + b0, length))
            offset += length

        hdr_bytes = json.dumps(new_hdr, separators=(",", ":")).encode("utf-8")
        # safetensors requires 8-byte alignment of the header length.
        pad = (-len(hdr_bytes)) % 8
        hdr_bytes += b" " * pad

        dstf = os.path.join(dst, "model.safetensors")
        total = 0
        with open(dstf, "wb") as out:
            out.write(struct.pack("<Q", len(hdr_bytes)))
            out.write(hdr_bytes)
            for (begin, length) in plan:
                f.seek(begin)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(remaining, 1 << 24))
                    out.write(chunk)
                    remaining -= len(chunk)
                total += length
    print(f"wrote {len(kept)} tensors, {total/1e9:.2f} GB payload → {dstf}")

    # Copy config + tokenizer verbatim (config.text_config drives mlx-lm gemma3n).
    for extra in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]:
        s = os.path.join(src, extra)
        if os.path.exists(s):
            with open(s, "rb") as a, open(os.path.join(dst, extra), "wb") as b:
                b.write(a.read())
    print(f"copied config + tokenizer to {dst}")


if __name__ == "__main__":
    main()
