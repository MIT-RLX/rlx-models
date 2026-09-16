#!/usr/bin/env python3
"""Split the classic-scene test set into one file per scene.

The aggregate over the four classic scenes has hidden a real result once
already: a change that fixed Veach entirely and moved the Cornell box by
nothing read, in the total, as a modest uniform gain. So score them apart.

`denoise_dataset` with CLASSIC=1 cycles the scene through
`index % 4` -> cornell, cornell-with-glass, veach, furnace, and every scene
contributes the same number of tiles, cut by position. So a tile's scene is
its index divided by the tiles-per-scene, and its *kind* is that modulo four.

    python3 split_classic.py /tmp/classic.bin /tmp/cls   # -> /tmp/cls_cornell.bin, ...
"""

import struct
import sys

HEADER = struct.Struct("<8s4I")
NAMES = ["cornell", "glass", "veach", "furnace"]
TILES_PER_SCENE = 4


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, prefix = argv[1], argv[2]

    with open(src, "rb") as f:
        blob = f.read()

    magic, tile, count, inputs, outputs = HEADER.unpack_from(blob, 0)
    if magic != b"RLXDN002":
        print(f"{src} is not RLXDN002")
        return 1

    stride = (inputs + outputs) * tile * tile * 4
    body = HEADER.size
    if len(blob) - body != count * stride:
        print(f"{src} is truncated: {len(blob) - body} bytes for {count} tiles of {stride}")
        return 1

    buckets = {name: [] for name in NAMES}
    for t in range(count):
        kind = NAMES[(t // TILES_PER_SCENE) % len(NAMES)]
        buckets[kind].append(blob[body + t * stride : body + (t + 1) * stride])

    for name, tiles in buckets.items():
        out = f"{prefix}_{name}.bin"
        with open(out, "wb") as f:
            f.write(HEADER.pack(b"RLXDN002", tile, len(tiles), inputs, outputs))
            for chunk in tiles:
                f.write(chunk)
        print(f"{out}  {len(tiles)} tiles")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
