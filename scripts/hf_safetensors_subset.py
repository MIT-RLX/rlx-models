#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Pull the first N decoder layers of a sharded HF safetensors LM by HTTP range.

A safetensors file is an 8-byte header length, a JSON header naming every
tensor's dtype, shape and byte range, then the data. So the tensors worth
testing can be fetched without the rest of the checkpoint. This writes a small,
self-consistent HF directory: one `model.safetensors` with the globals (embedding,
final norm, LM head) and decoder layers `0..N`, a `config.json` with the depth
(and `layer_types`) cut to N and the MTP block switched off, plus the tokenizer
files — a model that transformers and rlx both load as-is.

    python3 scripts/hf_safetensors_subset.py ORG/REPO OUT_DIR --layers 4
    python3 scripts/hf_safetensors_subset.py ORG/REPO --manifest manifest.json

`--manifest` writes only `{name: [dtype, shape]}` for every tensor in the repo
(a few hundred KB of header reads) — the fixture for a no-weights contract test.
Vision / audio towers and `mtp.*` are skipped unless `--keep PREFIX` names them.
"""
import argparse
import json
import os
import re
import struct
import urllib.request

HF = "https://huggingface.co"
TEXT_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "generation_config.json",
    "chat_template.jinja",
]


def get(url, a=None, b=None):
    headers = {"Range": f"bytes={a}-{b}"} if a is not None else {}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        return r.read()


def header(url):
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    hdr = json.loads(get(url, 8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    return 8 + n, hdr


def shard_files(repo, rev):
    base = f"{HF}/{repo}/resolve/{rev}"
    try:
        idx = json.loads(get(f"{base}/model.safetensors.index.json"))
        return base, sorted(set(idx["weight_map"].values()))
    except urllib.error.HTTPError:
        return base, ["model.safetensors"]


LAYER = re.compile(r"\.layers\.(\d+)\.")
TOWERS = ("visual", "vision", "audio", "merger", "mtp")


def wanted(name, n_layers, keep):
    if any(name.startswith(k) for k in keep):
        return True
    if name.startswith("mtp.") or any(f".{t}." in f".{name}" for t in TOWERS):
        return False
    m = LAYER.search(name)
    return m is None or int(m.group(1)) < n_layers


def cut_config(cfg, n):
    text = cfg.get("text_config", cfg)
    text["num_hidden_layers"] = n
    if isinstance(text.get("layer_types"), list):
        text["layer_types"] = text["layer_types"][:n]
    for c in (cfg, text):
        if "mtp_num_hidden_layers" in c:
            c["mtp_num_hidden_layers"] = 0
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rev", default="main")
    ap.add_argument("--keep", action="append", default=[], help="extra name prefix to keep")
    ap.add_argument("--manifest", help="write {name: [dtype, shape]} and exit")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base, files = shard_files(a.repo, a.rev)
    plan, manifest = [], {}
    for f in files:
        start, hdr = header(f"{base}/{f}")
        for name, t in hdr.items():
            manifest[name] = [t["dtype"], t["shape"]]
            if wanted(name, a.layers, a.keep):
                plan.append((f, start, name, t))

    if a.manifest:
        with open(a.manifest, "w") as fh:
            json.dump(dict(sorted(manifest.items())), fh, separators=(",", ":"))
        print(f"wrote {a.manifest}: {len(manifest)} tensors from {len(files)} files")
        return
    if not a.out:
        ap.error("OUT_DIR is required unless --manifest is given")

    total = sum(t["data_offsets"][1] - t["data_offsets"][0] for *_, t in plan)
    print(f"{len(plan)} of {len(manifest)} tensors, {total / 1e9:.2f} GB")
    if a.dry_run:
        return

    os.makedirs(a.out, exist_ok=True)
    out_hdr, off = {}, 0
    for _, _, name, t in sorted(plan, key=lambda p: p[2]):
        n = t["data_offsets"][1] - t["data_offsets"][0]
        out_hdr[name] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + n]}
        off += n
    hj = json.dumps(out_hdr, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    with open(os.path.join(a.out, "model.safetensors"), "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)) + hj)
        for f, start, name, t in sorted(plan, key=lambda p: p[2]):
            lo, hi = t["data_offsets"]
            fh.write(get(f"{base}/{f}", start + lo, start + hi - 1))
            print(f"  {name} {t['shape']}")

    cfg = cut_config(json.loads(get(f"{base}/config.json")), a.layers)
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    for f in TEXT_FILES:
        try:
            data = get(f"{base}/{f}")
        except urllib.error.HTTPError:
            continue
        with open(os.path.join(a.out, f), "wb") as fh:
            fh.write(data)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
