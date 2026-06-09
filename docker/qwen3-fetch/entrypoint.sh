#!/usr/bin/env bash
# Download a Hugging Face Hub model into /weights/<short-name>/.
#
# Args:
#   $1   model id (e.g. "Qwen/Qwen3-0.6B")
#
# Env:
#   HF_TOKEN              optional, required for gated repos
#   ALLOW_PATTERNS        comma-separated file globs to include
#                         (default: "*.safetensors,*.safetensors.index.json,
#                          config.json,tokenizer*,*.json,*.txt,*.model")
#   IGNORE_PATTERNS       comma-separated file globs to exclude
#                         (default: "*.bin,*.gguf,*.pt,*.h5,*.msgpack")

set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-0.6B}"
SHORT_NAME="${MODEL_ID##*/}"
DEST="/weights/${SHORT_NAME}"

ALLOW="${ALLOW_PATTERNS:-*.safetensors,*.safetensors.index.json,config.json,tokenizer*,*.json,*.txt,*.model}"
IGNORE="${IGNORE_PATTERNS:-*.bin,*.gguf,*.pt,*.h5,*.msgpack}"

echo "[fetch] model:   ${MODEL_ID}"
echo "[fetch] dest:    ${DEST}"
echo "[fetch] include: ${ALLOW}"
echo "[fetch] exclude: ${IGNORE}"

mkdir -p "${DEST}"

# Build CLI args from comma-separated globs. `huggingface-cli download`
# expects nargs='+' on --include / --exclude — one flag followed by all
# patterns, NOT repeated --include flags (which would silently honor
# only the last one — we got bit by this).
allow_args=()
IFS=',' read -ra ALLOW_ARR <<< "${ALLOW}"
if [[ ${#ALLOW_ARR[@]} -gt 0 && -n "${ALLOW_ARR[0]}" ]]; then
    allow_args+=(--include "${ALLOW_ARR[@]}")
fi

ignore_args=()
IFS=',' read -ra IGNORE_ARR <<< "${IGNORE}"
if [[ ${#IGNORE_ARR[@]} -gt 0 && -n "${IGNORE_ARR[0]}" ]]; then
    ignore_args+=(--exclude "${IGNORE_ARR[@]}")
fi

token_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    token_args+=(--token "${HF_TOKEN}")
fi

# `--local-dir` puts files directly under DEST without the HF cache
# layout, which is what rlx-models expects (one flat directory with
# model.safetensors + config.json + tokenizer.json next to each other).
huggingface-cli download \
    "${MODEL_ID}" \
    --local-dir "${DEST}" \
    "${allow_args[@]}" \
    "${ignore_args[@]}" \
    "${token_args[@]}"

echo
echo "[fetch] complete. Files in ${DEST}:"
ls -lh "${DEST}"
echo
echo "Run the parity test against this checkpoint with:"
echo "  RLX_QWEN3_WEIGHTS=\"\$PWD/weights/${SHORT_NAME}/model.safetensors\" \\"
echo "  RLX_QWEN3_CONFIG=\"\$PWD/weights/${SHORT_NAME}/config.json\" \\"
echo "  cargo test -p rlx-models --features parity-candle --release qwen3_parity_vs_candle"
