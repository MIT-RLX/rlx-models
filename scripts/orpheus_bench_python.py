#!/usr/bin/env python3
"""Bench RLX Orpheus vs upstream Python token→code mapping and SNAC decode.

Usage:
  # Token stream parity (no LM — validates stream index rules):
  python3 scripts/orpheus_bench_python.py --simulate-tokens

  # Full SNAC decode from RLX-exported codes file:
  ORPHEUS_DUMP_CODES=/tmp/rust_codes.txt python3 scripts/orpheus_bench_python.py --decode-codes

Requires: pip install snac torch safetensors numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def turn_token_into_id(token_string: str, index: int) -> int | None:
    token_string = token_string.strip()
    start = token_string.rfind("<custom_token_")
    if start == -1:
        return None
    last = token_string[start:]
    if not (last.startswith("<custom_token_") and last.endswith(">")):
        return None
    try:
        n = int(last[14:-1])
        return n - 10 - ((index % 7) * 4096)
    except ValueError:
        return None


def python_codes_from_token_strings(tokens: list[str]) -> list[int]:
    """Mirror orpheus_tts/decoder.py tokens_decoder buffer logic."""
    buffer: list[int] = []
    count = 0
    for tok in tokens:
        code = turn_token_into_id(tok, count)
        if code is None:
            continue
        if code > 0:
            buffer.append(code)
            count += 1
    return buffer


def load_rust_codes(path: Path) -> list[int]:
    text = path.read_text().strip().splitlines()
    if len(text) < 2:
        raise ValueError(f"expected len + codes in {path}")
    n = int(text[0].strip())
    codes = [int(x) for x in text[1].split()]
    if len(codes) != n:
        print(f"warn: header says {n} codes, got {len(codes)}", file=sys.stderr)
    return codes


def decode_snac(codes: list[int], snac_dir: Path) -> tuple[list[float], float]:
    import numpy as np
    import torch
    from snac import SNAC

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cpu()

    def convert_to_audio(multiframe: list[int]):
        if len(multiframe) < 7:
            return None
        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]
        c0, c1, c2 = [], [], []
        for j in range(num_frames):
            i = 7 * j
            c0.append(frame[i])
            c1.extend([frame[i + 1], frame[i + 4]])
            c2.extend([frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])
        codes_t = [
            torch.tensor([c0], dtype=torch.int32),
            torch.tensor([c1], dtype=torch.int32),
            torch.tensor([c2], dtype=torch.int32),
        ]
        with torch.inference_mode():
            audio = model.decode(codes_t)
        return audio.squeeze().cpu().numpy()

    pcm = convert_to_audio(codes)
    if pcm is None:
        return [], 0.0
    peak = float(np.abs(pcm).max())
    return pcm.tolist(), peak


def simulate_stream_index_regression() -> None:
    # Code 0 in slot 0 must NOT advance index — next token uses slot 0 again.
    tok0 = f"<custom_token_{10 + 0}>"  # code 0, slot 0
    tok1 = f"<custom_token_{10 + 1}>"  # code 1, slot 0 if index unchanged
    codes = python_codes_from_token_strings([tok0, tok1])
    assert codes == [1], f"expected [1], got {codes}"
    print("stream_index regression: ok (code 0 does not advance slot)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate-tokens", action="store_true")
    ap.add_argument("--decode-codes", action="store_true")
    ap.add_argument("--codes-file", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=Path("/tmp/orpheus_py_bench.json"))
    args = ap.parse_args()

    if args.simulate_tokens:
        simulate_stream_index_regression()
        return

    if args.decode_codes:
        path = args.codes_file or Path(
            __import__("os").environ.get("ORPHEUS_DUMP_CODES", "/tmp/rust_codes.txt")
        )
        if not path.is_file():
            print(f"missing codes file {path}", file=sys.stderr)
            sys.exit(1)
        codes = load_rust_codes(path)
        pcm, peak = decode_snac(codes, Path("/tmp/rlx-weights/snac"))
        out = {"codes": codes, "pcm_len": len(pcm), "peak": peak}
        args.out_json.write_text(json.dumps(out, indent=2))
        print(f"decoded {len(codes)} codes -> {len(pcm)} samples peak={peak:.4f}")
        print(f"wrote {args.out_json}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
