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

"""
Dump a reference logits + token id fixture for the
`tests/gemma4_reference_fixture.rs` parity test.

Outputs a single JSON file at `$FIXTURE_DIR/reference.json` with the
shape the Rust test expects:

```
{
  "prompt":  "...",
  "tokens":  [int, int, ...],   # input ids
  "logits":  [float, float, ...]  # last-token logits, vocab-sized
}
```

Usage:

```bash
pip install torch transformers accelerate
FIXTURE_DIR=$HOME/gemma4-fixture python3 dump_gemma4_reference.py
RLX_GEMMA4_FIXTURE=$FIXTURE_DIR \
    cargo test -p rlx-gemma --test gemma4_reference_fixture --features apple-silicon
```

The fixture directory must also contain the HF `config.json`,
`tokenizer.json`, and either `model.safetensors` or sharded
`model-*.safetensors`. The Rust runner consumes those; this script
only produces the reference dump.
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────

MODEL_ID = os.environ.get("RLX_GEMMA4_REFERENCE_MODEL", "google/gemma-4-12B")
FIXTURE_DIR = os.environ.get("FIXTURE_DIR") or os.environ.get("RLX_GEMMA4_FIXTURE")
PROMPT = os.environ.get(
    "RLX_GEMMA4_REFERENCE_PROMPT",
    "The quick brown fox jumps over the lazy dog.",
)

if not FIXTURE_DIR:
    print("error: FIXTURE_DIR (or RLX_GEMMA4_FIXTURE) must be set", file=sys.stderr)
    sys.exit(2)
os.makedirs(FIXTURE_DIR, exist_ok=True)

# ── Load model + tokenizer ─────────────────────────────────────────

print(f"[dump] loading {MODEL_ID}…", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # f32 for fair parity comparison
    device_map="cpu",
).eval()

# ── Run prefill, capture last-token logits ─────────────────────────

print(f"[dump] tokenizing prompt: {PROMPT!r}", file=sys.stderr)
ids = tokenizer(PROMPT, return_tensors="pt").input_ids
print(f"[dump] {ids.shape[1]} tokens", file=sys.stderr)
with torch.no_grad():
    out = model(ids)
last_logits = out.logits[0, -1].tolist()
print(f"[dump] last-logits len = {len(last_logits)} (vocab)", file=sys.stderr)

# ── Write fixture ──────────────────────────────────────────────────

path = os.path.join(FIXTURE_DIR, "reference.json")
with open(path, "w") as f:
    json.dump(
        {
            "prompt": PROMPT,
            "tokens": ids[0].tolist(),
            "logits": last_logits,
        },
        f,
    )
print(f"[dump] wrote {path}", file=sys.stderr)
