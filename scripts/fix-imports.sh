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

# Legacy one-time import rewrite (monolith → rlx-models-core paths). Do not run on current tree.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

apply() {
  local dir="$1"
  shift
  local exprs=("$@")
  find "$dir" -name '*.rs' | while read -r f; do
    for expr in "${exprs[@]}"; do
      sed -i '' "$expr" "$f"
    done
  done
}

CORE=(
  's/crate::weight_map/rlx_models_core::weight_map/g'
  's/crate::weight_loader/rlx_models_core::weight_loader/g'
  's/crate::config/rlx_models_core::config/g'
  's/crate::flow_bridge/rlx_models_core::flow_bridge/g'
  's/crate::flow_util/rlx_models_core::flow_util/g'
  's/crate::lm/rlx_models_core::lm/g'
  's/crate::arch_registry/rlx_models_core::arch_registry/g'
  's/crate::vision_ops_ir/rlx_models_core::vision_ops_ir/g'
  's/crate::dataprocessing/rlx_models_core::dataprocessing/g'
)

SAM_IR=(
  's/crate::mlp_relu_ir/rlx_models_sam_ir::mlp_relu_ir/g'
  's/crate::mask_hyper_matmul_ir/rlx_models_sam_ir::mask_hyper_matmul_ir/g'
  's/crate::mask_prompt_ir/rlx_models_sam_ir::mask_prompt_ir/g'
  's/crate::twoway_transformer_ir/rlx_models_sam_ir::twoway_transformer_ir/g'
)

apply "$ROOT/crates/rlx-models-bert" "${CORE[@]}"
apply "$ROOT/crates/rlx-models-nomic" "${CORE[@]}"
apply "$ROOT/crates/rlx-models-vision" "${CORE[@]}"
apply "$ROOT/crates/rlx-models-dinov2" "${CORE[@]}"
apply "$ROOT/crates/rlx-models-embed" "${CORE[@]}" \
  's/crate::bert_flow/rlx_models_bert::flow/g' \
  's/crate::nomic_flow/rlx_models_nomic::flow/g' \
  's/crate::BertConfig/rlx_models_core::config::BertConfig/g'
apply "$ROOT/crates/rlx-models-sam-ir" "${CORE[@]}"
apply "$ROOT/crates/rlx-models-sam" "${CORE[@]}" "${SAM_IR[@]}"
apply "$ROOT/crates/rlx-models-sam2" "${CORE[@]}" "${SAM_IR[@]}" \
  's/crate::sam::/rlx_models_sam::/g'
apply "$ROOT/crates/rlx-models-sam3" "${CORE[@]}" \
  's/crate::sam::/rlx_models_sam::/g' \
  's/crate::sam3::tensor/rlx_models_tensor/g'
apply "$ROOT/crates/rlx-qwen3" "${CORE[@]}"
apply "$ROOT/crates/rlx-models-llama32" "${CORE[@]}"
apply "$ROOT/crates/rlx-models-llada2" "${CORE[@]}" \
  's/crate::llada2::/crate::/g'
apply "$ROOT/crates/rlx-qwen35" "${CORE[@]}" \
  's/crate::qwen35::/crate::/g' \
  's/crate::qwen3::/rlx_models_qwen3::/g' \
  's/crate::tide::/rlx_models_llada2::tide::/g'
apply "$ROOT/crates/rlx-models-flux2" "${CORE[@]}" \
  's/crate::sam3::tensor/rlx_models_tensor/g'
apply "$ROOT/crates/rlx-models-vjepa2" "${CORE[@]}" \
  's/crate::sam3::tensor/rlx_models_tensor/g'
apply "$ROOT/crates/rlx-models-wav2vec2-bert" "${CORE[@]}"
apply "$ROOT/crates/rlx-models/src" \
  's/crate::arch_registry/rlx_models_core::arch_registry/g' \
  's/crate::bert_flow/rlx_models_bert::flow/g' \
  's/crate::bert::/rlx_models_bert::bert::/g' \
  's/crate::config/rlx_models_core::config/g' \
  's/crate::dinov2/rlx_models_dinov2/g' \
  's/crate::embed/rlx_models_embed/g' \
  's/crate::flux2/rlx_models_flux2/g' \
  's/crate::flow_bridge/rlx_models_core::flow_bridge/g' \
  's/crate::flow_util/rlx_models_core::flow_util/g' \
  's/crate::lm/rlx_models_core::lm/g' \
  's/crate::llama32/rlx_models_llama32/g' \
  's/crate::llada2/rlx_models_llada2::llada2/g' \
  's/crate::nomic_flow/rlx_models_nomic::flow/g' \
  's/crate::nomic::/rlx_models_nomic::nomic::/g' \
  's/crate::qwen3/rlx_models_qwen3/g' \
  's/crate::qwen35/rlx_models_qwen35/g' \
  's/crate::sam2/rlx_models_sam2/g' \
  's/crate::sam3/rlx_models_sam3/g' \
  's/crate::sam/rlx_models_sam/g' \
  's/crate::tide/rlx_models_llada2::tide/g' \
  's/crate::vision_flow/rlx_models_vision::flow/g' \
  's/crate::vision::/rlx_models_vision::vision::/g' \
  's/crate::vjepa2/rlx_models_vjepa2/g' \
  's/crate::wav2vec2_bert/rlx_models_wav2vec2_bert/g' \
  's/crate::weight_loader/rlx_models_core::weight_loader/g' \
  's/crate::weight_map/rlx_models_core::weight_map/g'

echo "import rewrites done"
