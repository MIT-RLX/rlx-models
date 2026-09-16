"""Concatenate `RLXDN001` tile sets.

    python scripts/merge_datasets.py out.bin a.bin b.bin ...

Scene content is a pure function of its index, so sets generated at different
`OFFSET`s hold disjoint scenes and can simply be appended. The tile side has to
agree across inputs — the network is compiled for one shape.

Keep the *original* validation set when merging training data. Swapping it out
makes the new numbers incomparable with every number measured before, which
costs more than the extra tiles are worth.
"""
import sys

MAGIC = b"RLXDN002"
MAGIC_V1 = b"RLXDN001"

if len(sys.argv) < 3:
    raise SystemExit(__doc__)

out_path, inputs = sys.argv[1], sys.argv[2:]
tile = None
planes = None
total = 0
chunks = []

for path in inputs:
    raw = open(path, "rb").read()
    if raw[:8] == MAGIC:
        t = int.from_bytes(raw[8:12], "little")
        count = int.from_bytes(raw[12:16], "little")
        ins = int.from_bytes(raw[16:20], "little")
        outs = int.from_bytes(raw[20:24], "little")
        header = 24
    elif raw[:8] == MAGIC_V1:
        t = int.from_bytes(raw[8:12], "little")
        count = int.from_bytes(raw[12:16], "little")
        ins, outs, header = 9, 3, 16
    else:
        raise SystemExit(f"{path}: not a {MAGIC.decode()} file")

    if tile is None:
        tile, planes = t, (ins, outs)
    if t != tile:
        raise SystemExit(f"{path}: tile {t}, but {inputs[0]} is {tile}")
    if (ins, outs) != planes:
        raise SystemExit(
            f"{path}: {ins}+{outs} planes, but {inputs[0]} is {planes[0]}+{planes[1]}"
        )

    body = raw[header:]
    per_tile = (ins + outs) * t * t * 4
    if len(body) != count * per_tile:
        raise SystemExit(f"{path}: {len(body)} bytes for {count} tiles of {t}")
    chunks.append(body)
    total += count
    print(f"  {path}: {count} tiles")

with open(out_path, "wb") as f:
    f.write(MAGIC)
    f.write(tile.to_bytes(4, "little"))
    f.write(total.to_bytes(4, "little"))
    f.write(planes[0].to_bytes(4, "little"))
    f.write(planes[1].to_bytes(4, "little"))
    for c in chunks:
        f.write(c)
print(f"wrote {out_path}: {total} tiles of {tile}, {planes[0]}+{planes[1]} planes")
