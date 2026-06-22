#!/usr/bin/env python3
"""Merge Rust echo_bench JSON with Python Speex/NLMS bench JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-json", type=Path, required=True)
    ap.add_argument("--python-json", type=Path, required=True)
    ap.add_argument("--csv-out", type=Path, required=True)
    args = ap.parse_args()

    rust = json.loads(args.rust_json.read_text())
    py = json.loads(args.python_json.read_text())
    rows = []
    for r in rust.get("rows", []):
        rows.append({"source": "rust", **r})
    for r in py.get("rows", []):
        rows.append({"source": "python", **r})

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for row in rows for k in row.keys()})
        with args.csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"merged {len(rows)} rows → {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
