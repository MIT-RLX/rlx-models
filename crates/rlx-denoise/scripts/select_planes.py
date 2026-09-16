"""Write a dataset with a subset of its input planes.

    python scripts/select_planes.py in.bin out.bin 0-8      # colour+albedo+normal
    python scripts/select_planes.py in.bin out.bin 0-8,9     # ...and depth

Target planes are always kept; the selection applies to inputs only.

The point is not to save space. Deciding whether a guide plane earns its place
means training with and without it on *identical* scenes, and re-rendering to
drop two planes would change the Monte-Carlo noise along with the channel count
— so the comparison would confound the guide with the seed. Dropping planes
from a rendered set keeps every other byte the same.
"""
import sys

MAGIC = b"RLXDN002"
MAGIC_V1 = b"RLXDN001"
HEADER = 24


def parse_selection(text, count):
    """`0-8,10` -> [0..8, 10], validated against the file's input count."""
    chosen = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if lo > hi:
                raise SystemExit(f"range {part} runs backwards")
            chosen.extend(range(lo, hi + 1))
        else:
            chosen.append(int(part))
    if not chosen:
        raise SystemExit("no planes selected")
    for p in chosen:
        if not 0 <= p < count:
            raise SystemExit(f"plane {p} is outside the file's 0..{count - 1} inputs")
    if len(set(chosen)) != len(chosen):
        raise SystemExit("a plane was selected twice")
    return chosen


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, dst, selection = sys.argv[1], sys.argv[2], sys.argv[3]

    raw = open(src, "rb").read()
    if raw[:8] == MAGIC:
        tile = int.from_bytes(raw[8:12], "little")
        count = int.from_bytes(raw[12:16], "little")
        ins = int.from_bytes(raw[16:20], "little")
        outs = int.from_bytes(raw[20:24], "little")
        header = HEADER
    elif raw[:8] == MAGIC_V1:
        tile = int.from_bytes(raw[8:12], "little")
        count = int.from_bytes(raw[12:16], "little")
        ins, outs, header = 9, 3, 16
    else:
        raise SystemExit(f"{src}: not a {MAGIC.decode()} file")

    keep = parse_selection(selection, ins)
    pixels = tile * tile
    plane_bytes = pixels * 4
    per_tile = (ins + outs) * plane_bytes
    body = raw[header:]
    if len(body) != count * per_tile:
        raise SystemExit(f"{src}: {len(body)} bytes for {count} tiles of {tile}")

    out = bytearray()
    out += MAGIC
    out += tile.to_bytes(4, "little")
    out += count.to_bytes(4, "little")
    out += len(keep).to_bytes(4, "little")
    out += outs.to_bytes(4, "little")
    for t in range(count):
        base = t * per_tile
        for p in keep:
            off = base + p * plane_bytes
            out += body[off : off + plane_bytes]
        # Targets follow every input, unchanged.
        off = base + ins * plane_bytes
        out += body[off : off + outs * plane_bytes]

    with open(dst, "wb") as f:
        f.write(out)
    print(
        f"wrote {dst}: {count} tiles of {tile}, "
        f"{len(keep)}+{outs} planes (kept {keep} of {ins})"
    )


if __name__ == "__main__":
    main()
