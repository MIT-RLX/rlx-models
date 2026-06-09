#!/usr/bin/env bash
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

# Legacy one-time migration (pre–per-crate layout). Do not run on current tree.
# One-time migration: move monolithic src/ into per-model crates.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"
CRATES="$ROOT/crates"

mkdir -p "$CRATES"

move_file() {
  local src="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$src" ]]; then
    git mv "$src" "$dest" 2>/dev/null || mv "$src" "$dest"
  fi
}

move_dir() {
  local src="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -d "$src" ]]; then
    git mv "$src" "$dest" 2>/dev/null || mv "$src" "$dest"
  fi
}

# ── core ──
CORE="$CRATES/rlx-models-core/src"
mkdir -p "$CORE"
for f in arch_registry config dataprocessing flow_bridge flow_util lm \
         vision_ops_ir weight_loader weight_map; do
  move_file "$SRC/${f}.rs" "$CORE/${f}.rs"
done

# ── sam-ir ──
SAM_IR="$CRATES/rlx-models-sam-ir/src"
mkdir -p "$SAM_IR"
for f in mlp_relu_ir mask_hyper_matmul_ir mask_prompt_ir twoway_transformer_ir; do
  move_file "$SRC/${f}.rs" "$SAM_IR/${f}.rs"
done

# ── tensor (from sam3) ──
TENSOR="$CRATES/rlx-models-tensor/src"
mkdir -p "$TENSOR"
move_file "$SRC/sam3/tensor.rs" "$TENSOR/lib.rs"

# ── model directories ──
move_dir "$SRC/bert" "$CRATES/rlx-models-bert/src" 2>/dev/null || true
move_file "$SRC/bert.rs" "$CRATES/rlx-models-bert/src/bert.rs"
move_file "$SRC/bert_flow.rs" "$CRATES/rlx-models-bert/src/flow.rs"

move_file "$SRC/nomic.rs" "$CRATES/rlx-models-nomic/src/nomic.rs"
move_file "$SRC/nomic_flow.rs" "$CRATES/rlx-models-nomic/src/flow.rs"

move_file "$SRC/vision.rs" "$CRATES/rlx-models-vision/src/vision.rs"
move_file "$SRC/vision_flow.rs" "$CRATES/rlx-models-vision/src/flow.rs"

move_dir "$SRC/dinov2" "$CRATES/rlx-models-dinov2/src"
move_dir "$SRC/embed" "$CRATES/rlx-models-embed/src"
move_dir "$SRC/sam" "$CRATES/rlx-models-sam/src"
move_dir "$SRC/sam2" "$CRATES/rlx-models-sam2/src"
move_dir "$SRC/sam3" "$CRATES/rlx-models-sam3/src"
move_dir "$SRC/qwen3" "$CRATES/rlx-qwen3/src"
move_dir "$SRC/llama32" "$CRATES/rlx-models-llama32/src"
move_dir "$SRC/llada2" "$CRATES/rlx-models-llada2/src/llada2"
move_dir "$SRC/tide" "$CRATES/rlx-models-llada2/src/tide"
move_dir "$SRC/qwen35" "$CRATES/rlx-qwen35/src"
move_dir "$SRC/flux2" "$CRATES/rlx-models-flux2/src"
move_dir "$SRC/vjepa2" "$CRATES/rlx-models-vjepa2/src"
move_dir "$SRC/wav2vec2_bert" "$CRATES/rlx-models-wav2vec2-bert/src"

# ── facade (run + bin) ──
FACADE="$CRATES/rlx-models/src"
mkdir -p "$FACADE/bin"
move_file "$SRC/run.rs" "$FACADE/run.rs"
move_file "$SRC/lib.rs" "$FACADE/lib.rs"
move_file "$SRC/bin/rlx_run.rs" "$FACADE/bin/rlx_run.rs"

echo "done — remaining in src/:"
ls -la "$SRC" 2>/dev/null || echo "(empty or removed)"
