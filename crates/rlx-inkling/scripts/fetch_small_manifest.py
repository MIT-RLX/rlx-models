#!/usr/bin/env python3
"""Pin the real thinkingmachines/Inkling-Small contract as test fixtures.

Fetches `config.json` verbatim and the tensor manifest (name -> dtype, shape)
from the safetensors *headers* only (HTTP Range: first 8 bytes give the header
length, then the JSON). No weight bytes are downloaded.

    python3 -I crates/rlx-inkling/scripts/fetch_small_manifest.py OUT_DIR

Writes OUT_DIR/config.json and OUT_DIR/manifest.json.
"""
import json
import struct
import sys
import urllib.request

REPO = "thinkingmachines/Inkling-Small"
BASE = f"https://huggingface.co/{REPO}/resolve/main"


def get(url, rng=None):
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        return r.read()


def header(shard):
    url = f"{BASE}/{shard}"
    n = struct.unpack("<Q", get(url, (0, 7)))[0]
    return json.loads(get(url, (8, 8 + n - 1)))


def main(out):
    cfg = get(f"{BASE}/config.json")
    index = json.loads(get(f"{BASE}/model.safetensors.index.json"))
    shards = sorted(set(index["weight_map"].values())) + ["mtp.safetensors"]
    manifest = {}
    for shard in shards:
        for name, meta in header(shard).items():
            if name == "__metadata__":
                continue
            manifest[name] = {"dtype": meta["dtype"], "shape": meta["shape"]}
        print(f"{shard}: {len(manifest)} tensors so far", file=sys.stderr)
    with open(f"{out}/config.json", "wb") as f:
        f.write(cfg)
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=0, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1])
