#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Run HF prepare_data.py with a larger codec batch (less GPU idle / I/O round-trips)."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--finetune-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args, unknown = p.parse_known_args()
    rest = list(args.rest) + list(unknown)
    if rest and rest[0] == "--":
        rest = rest[1:]
    script = args.finetune_dir / "prepare_data.py"
    text = script.read_text(encoding="utf-8")
    for old in ("BATCH_INFER_NUM = 32", "BATCH_INFER_NUM=32"):
        if old in text:
            text = text.replace(old, f"BATCH_INFER_NUM = {args.batch}", 1)
            break
    patched = args.finetune_dir / ".prepare_data_patched.py"
    patched.write_text(text, encoding="utf-8")
    sys.argv = [str(patched), *rest]
    runpy.run_path(str(patched), run_name="__main__")


if __name__ == "__main__":
    main()
