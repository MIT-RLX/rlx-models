#!/usr/bin/env python3
"""Shard Kimi-K3's routed experts to a worker node's local NVMe.

The full checkpoint's routed experts (MXFP4-packed, ~1.45 TB) don't fit on any one
worker, so each node holds a contiguous expert-id range across all MoE layers. This
extracts `[lo, hi)` for every MoE layer into per-layer safetensors on the remote node
(streamed one layer at a time so the Mac staging stays small), plus a merged index the
worker's CheckpointLoader reads. Resumable: skips layers already present on the node.

  ./shard_experts.py --node msi --lo 0   --hi 430 --dest /data/kimi-experts
  ./shard_experts.py --node amd --lo 466 --hi 896 --dest /data/kimi-experts
  # (experts 430..466 stay Mac-local, served from /Volumes/FOUR)
"""
import argparse, glob, json, os, subprocess, sys, tempfile

SRC = "/Volumes/FOUR/kimi"
TENSORS = ["w1.weight_packed", "w1.weight_scale", "w2.weight_packed",
           "w2.weight_scale", "w3.weight_packed", "w3.weight_scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--dest", required=True, help="dir on the remote node")
    ap.add_argument("--layers", default="", help="e.g. '1' or '1-3' (default: all MoE)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import torch
    from safetensors.torch import safe_open, save_file

    wmap = json.load(open(glob.glob(f"{SRC}/*index.json")[0]))["weight_map"]
    moe_layers = sorted({int(k.split("layers.")[1].split(".")[0])
                         for k in wmap if "block_sparse_moe.experts." in k})
    if a.layers:
        if "-" in a.layers:
            b, e = map(int, a.layers.split("-")); sel = [x for x in moe_layers if b <= x <= e]
        else:
            sel = [int(a.layers)]
    else:
        sel = moe_layers

    ssh = ["ssh", a.node]
    subprocess.run([*ssh, f"mkdir -p {a.dest}"], check=True)
    # config.json + a marker of the owned range (worker validates its shard).
    subprocess.run(["scp", "-q", f"{SRC}/config.json", f"{a.node}:{a.dest}/config.json"], check=True)

    # open each source shard lazily (reused across layers/experts).
    handles = {}
    def get(key):
        sh = wmap[key]
        if sh not in handles:
            handles[sh] = safe_open(os.path.join(SRC, sh), framework="pt")
        return handles[sh].get_tensor(key)

    # The index ALWAYS covers ALL MoE layers of this shard (not just the ones this
    # invocation copies) — so per-layer invocations don't overwrite it with a partial
    # map. Files for not-yet-copied layers just aren't present until their run.
    index = {"metadata": {"format": "pt"}, "weight_map": {}}
    for li in moe_layers:
        fn = f"experts_L{li:03d}_{a.lo}-{a.hi}.safetensors"
        for e in range(a.lo, a.hi):
            for t in TENSORS:
                index["weight_map"][f"language_model.model.layers.{li}.block_sparse_moe.experts.{e}.{t}"] = fn
    total_gb = 0.0
    for li in sel:
        fname = f"experts_L{li:03d}_{a.lo}-{a.hi}.safetensors"
        # resume: skip if already on node.
        exists = subprocess.run([*ssh, f"test -f {a.dest}/{fname}"]).returncode == 0
        pfx = f"language_model.model.layers.{li}.block_sparse_moe.experts"
        names = [f"{pfx}.{e}.{t}" for e in range(a.lo, a.hi) for t in TENSORS]
        if exists:
            print(f"  L{li:03d} already on {a.node}, skip", flush=True); continue
        if a.dry_run:
            print(f"  L{li:03d}: would write {len(names)} tensors → {a.node}:{a.dest}/{fname}", flush=True)
            continue
        tset = {nm: get(nm) for nm in names}
        gb = sum(t.numel() * t.element_size() for t in tset.values()) / 1e9
        total_gb += gb
        with tempfile.TemporaryDirectory() as td:
            local = os.path.join(td, fname)
            save_file(tset, local)
            subprocess.run(["rsync", "-a", "--partial", local, f"{a.node}:{a.dest}/{fname}"], check=True)
        print(f"  L{li:03d}: {gb:.1f} GB → {a.node}  (cum {total_gb:.0f} GB)", flush=True)

    # write the merged index to the node.
    with tempfile.TemporaryDirectory() as td:
        ip = os.path.join(td, "model.safetensors.index.json")
        json.dump(index, open(ip, "w"))
        subprocess.run(["scp", "-q", ip, f"{a.node}:{a.dest}/model.safetensors.index.json"], check=True)
    print(f"done: {a.node} experts [{a.lo},{a.hi}) × {len(sel)} layers = {total_gb:.0f} GB", flush=True)


if __name__ == "__main__":
    main()
