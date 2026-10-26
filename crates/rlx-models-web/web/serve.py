#!/usr/bin/env python3
"""Serve the repo root for the browser demo (`just web-models-serve`).

The page lives at /crates/rlx-models-web/web/ and fetches weights from
/weights/. `POST /__result` (the self-test page) prints the body and, with
--once, exits — so a headless browser run can be scripted.
"""
import argparse
import http.server
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        # A dev server: never let the browser keep a stale module. Chrome
        # otherwise reuses cached ES modules across reloads, so a page could
        # run a new chat.js against an old pipeline.js. Weights are re-read
        # from the disk cache anyway (Range requests skip this).
        if not self.path.startswith("/weights/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        # `Range: bytes=a-b` (one range) on files, as real hosts honour it:
        # pipeline.js workers fetch only their own tensors of a GGUF.
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not rng.startswith("bytes=") or "," in rng or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        first, _, last = rng[len("bytes="):].partition("-")
        try:
            if first:
                start, end = int(first), min(int(last) if last else size - 1, size - 1)
            else:
                start, end = max(size - int(last), 0), size - 1
        except ValueError:
            return super().send_head()
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        self._range_left = None
        while left > 0:
            buf = source.read(min(left, 1 << 20))
            if not buf:
                break
            outputfile.write(buf)
            left -= len(buf)

    def do_POST(self):
        if self.path != "/__result":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("content-length", 0))).decode()
        self.send_response(204)
        self.end_headers()
        print(body, flush=True)
        if self.server.once:
            os._exit(0 if '"ok":true' in body.replace(" ", "") else 1)

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--once", action="store_true", help="exit after the first POST /__result")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.once, srv.verbose = args.once, args.verbose
    print(f"http://127.0.0.1:{args.port}/crates/rlx-models-web/web/", file=sys.stderr, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
