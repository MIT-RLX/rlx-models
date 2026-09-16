#!/usr/bin/env bash
# Fetch the released DeepSeek-V4.1 reference sources next to this harness.
#
# They are DeepSeek's own MIT-licensed code and are deliberately NOT vendored:
# the point of the harness is to compare against whatever the upstream repo
# currently says, and a stale vendored copy would quietly stop doing that.
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference"
for f in model.py engram.py vision.py image_processor.py; do
  echo "fetching $f"
  curl -fsSL "$BASE/$f" -o "$f"
done
# `kernel.py` here is OUR CPU transliteration of the tilelang kernels; keep it.
echo "done — run: RLX_REF_NOQUANT=1 python3 dump.py"
