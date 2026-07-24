#!/usr/bin/env python3
"""Write weights/asr/manifest.json after GGUF pack/prune (called from just)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    asr = Path(os.environ.get("RLX_ASR_DIR") or (sys.argv[1] if len(sys.argv) > 1 else "weights/asr"))
    src = os.environ.get("RLX_ASR_PACK_SRC", "")
    gguf = asr / "model.gguf"
    man = {
        "format": "rlx-asr-weights-v3",
        "root": str(asr),
        "source": src,
        "gguf": str(gguf) if gguf.is_file() else None,
        "gguf_bytes": gguf.stat().st_size if gguf.is_file() else 0,
        "note": "GGUF-only publish; sidecars embedded in model.gguf",
    }
    (asr / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(
        json.dumps(
            {"dst": str(asr), "gguf_mb": round(man["gguf_bytes"] / (1024 * 1024), 1)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
