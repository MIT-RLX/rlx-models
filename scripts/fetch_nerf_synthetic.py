#!/usr/bin/env python3
"""Fetch one NeRF-Synthetic scene for `rlx-fgd`'s Section 4.3 example.

The dataset ships as a single 1.27 GB zip, and Section 4.3 uses 24 training
images and every 8th test image of one scene out of eight — about 3% of it. A
zip's central directory is at the end of the file and each entry is
independently addressable, so `zipfile` over an HTTP-range-backed file object
downloads exactly the members asked for: ~38 MB for `ficus`, no archive on disk
and no unpacking step.

    python3 scripts/fetch_nerf_synthetic.py ficus
    python3 scripts/fetch_nerf_synthetic.py lego --out /tmp/nerf --train 100 --test-skip 8

Then:

    just fgd-radiance -- --scene /tmp/nerf_synthetic/ficus --views 24 --steps 200

The mirror is a HuggingFace copy of the archive published with the original
NeRF paper (Mildenhall et al., 2021); the scene layout is the dataset's own
(`transforms_{split}.json` beside `{train,test}/r_N.png`), which is what
`rlx_fgd::scene::load_nerf_synthetic` expects.
"""

import argparse
import io
import os
import re
import sys
import urllib.request
import zipfile

ARCHIVE = (
    "https://huggingface.co/datasets/XayahHina/nerf_synthetic/"
    "resolve/main/nerf_synthetic.zip"
)
SCENES = ("chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship")


class HttpFile(io.RawIOBase):
    """A seekable file-like object served by HTTP range requests."""

    def __init__(self, url):
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.size = int(response.headers["Content-Length"])
            # Follow the CDN redirect once so every range request skips it.
            self.url = response.url
        self.pos = 0
        self.requests = 0
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.pos
        if size <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(request) as response:
            data = response.read()
        self.pos += len(data)
        self.requests += 1
        self.transferred += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", choices=SCENES, help="which scene to fetch")
    parser.add_argument(
        "--out", default="/tmp/nerf_synthetic", help="parent directory (default: %(default)s)"
    )
    parser.add_argument(
        "--train", type=int, default=24, help="training frames to keep (default: %(default)s)"
    )
    parser.add_argument(
        "--test-skip", type=int, default=8, help="keep every Nth test frame (default: %(default)s)"
    )
    args = parser.parse_args()

    source = HttpFile(ARCHIVE)
    print(f"archive {source.size / 1e9:.2f} GB", flush=True)
    archive = zipfile.ZipFile(io.BufferedReader(source, buffer_size=1 << 20))
    entries = [n for n in archive.namelist() if f"{args.scene}/" in n]
    if not entries:
        sys.exit(f"no entries for scene {args.scene!r}")

    wanted = set()
    for split, keep in (
        ("train", lambda i: i < args.train),
        ("test", lambda i: i % max(args.test_skip, 1) == 0),
    ):
        for name in entries:
            match = re.search(rf"/{split}/r_(\d+)\.png$", name)
            if match and keep(int(match.group(1))):
                wanted.add(name)
            elif name.endswith(f"transforms_{split}.json"):
                wanted.add(name)

    destination = os.path.join(args.out, args.scene)
    for name in sorted(wanted):
        relative = name.split(f"{args.scene}/", 1)[1]
        path = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with archive.open(name) as src, open(path, "wb") as dst:
            dst.write(src.read())

    print(
        f"wrote {len(wanted)} files to {destination}; "
        f"{source.requests} range requests, {source.transferred / 1e6:.1f} MB transferred"
    )


if __name__ == "__main__":
    main()
