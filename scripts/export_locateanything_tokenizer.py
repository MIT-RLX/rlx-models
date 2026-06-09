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

"""Write `tokenizer.json` into the LocateAnything checkpoint (HF-compatible BPE)."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

# Processor import pulls optional `decord`.
sys.modules.setdefault("decord", types.ModuleType("decord"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".cache/locateanything/LocateAnything-3B"),
    )
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    out = args.model_dir / "tokenizer.json"
    tok.backend_tokenizer.save(str(out))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
