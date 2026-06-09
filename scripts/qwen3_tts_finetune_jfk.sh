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

# Fine-tune Qwen3-TTS-12Hz-0.6B-Base on chunked JFK → CustomVoice speaker "jfk".
#
# BACKEND=metal  — PyTorch SFT on Apple MPS (Metal)
# BACKEND=mlx    — HF prepare (MPS/CPU) + native RLX talker LoRA on MLX
#
# Prereqs: just qwen3-tts-jfk-prep && just fetch-qwen3-tts-base

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JFK_DIR="${JFK_DIR:-$ROOT/.cache/qwen3-tts/jfk}"
BASE_DIR="${RLX_QWEN3_TTS_BASE_DIR:-$ROOT/.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base}"
OUT_DIR="${OUT_DIR:-$ROOT/.cache/qwen3-tts/jfk-checkpoint}"
SPEAKER="${SPEAKER:-jfk}"
BACKEND="${BACKEND:-metal}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREPARE_BATCH="${PREPARE_BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
RAW_JSONL="${RAW_JSONL:-$JFK_DIR/train_raw.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-$JFK_DIR/train_with_codes.jsonl}"
FINETUNE_DIR="${FINETUNE_DIR:-$ROOT/.cache/qwen3-tts/Qwen3-TTS-finetuning}"
VENV="${VENV:-$ROOT/.venv-qwen3-tts-train}"

if [[ ! -f "$RAW_JSONL" ]]; then
  echo "missing $RAW_JSONL — run: just qwen3-tts-jfk-prep"
  exit 1
fi
if [[ ! -d "$BASE_DIR" ]]; then
  echo "missing Base weights — run: just fetch-qwen3-tts-base"
  exit 1
fi

if [[ ! -d "$FINETUNE_DIR" ]]; then
  git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git "$ROOT/.cache/qwen3-tts/Qwen3-TTS-src"
  ln -sfn "$ROOT/.cache/qwen3-tts/Qwen3-TTS-src/finetuning" "$FINETUNE_DIR"
fi

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "qwen-tts" torch accelerate librosa soundfile safetensors transformers
if ! command -v sox >/dev/null 2>&1; then
  echo "[qwen3-jfk] warning: install sox for codec prepare (brew install sox)"
fi

case "$BACKEND" in
  metal|mps)
    export TRAIN_DEVICE="mps"
    ;;
  cuda)
    export TRAIN_DEVICE="cuda:0"
    ;;
  mlx)
    export TRAIN_DEVICE="mps"
    ;;
  *)
    echo "unknown BACKEND=$BACKEND (metal|mlx|cuda)"
    exit 1
    ;;
esac
if [[ -n "${TRAIN_DEVICE_OVERRIDE:-}" ]]; then
  export TRAIN_DEVICE="$TRAIN_DEVICE_OVERRIDE"
fi

echo "[qwen3-jfk] prepare codes device=$TRAIN_DEVICE batch=$PREPARE_BATCH"
PREPARE_SRC="$FINETUNE_DIR/prepare_data.py"
PREPARE_PATCH="$FINETUNE_DIR/.prepare_data_patched.py"
python3 - <<PY
from pathlib import Path
src = Path("$PREPARE_SRC")
text = src.read_text(encoding="utf-8")
text = text.replace("BATCH_INFER_NUM = 32", "BATCH_INFER_NUM = $PREPARE_BATCH", 1)
Path("$PREPARE_PATCH").write_text(text, encoding="utf-8")
PY
python "$PREPARE_PATCH" \
  --device "${TRAIN_DEVICE}" \
  --tokenizer_model_path "${TOKENIZER_MODEL:-Qwen/Qwen3-TTS-Tokenizer-12Hz}" \
  --input_jsonl "$RAW_JSONL" \
  --output_jsonl "$TRAIN_JSONL"

if [[ "$BACKEND" == "mlx" ]]; then
  echo "[qwen3-jfk] native MLX talker LoRA (RLX)"
  N_TRAIN="$(wc -l < "$TRAIN_JSONL" | tr -d ' ')"
  MLX_STEPS="${MLX_STEPS_PER_EPOCH:-$N_TRAIN}"
  export RLX_QWEN3_TTS_TRAIN_GRAD_ACCUM="${GRAD_ACCUM}"
  cargo run -p rlx-qwen3-tts-train --bin rlx-qwen3-tts-train --features mlx,apple-silicon --release -- \
    jfk-lora \
    --model-dir "$BASE_DIR" \
    --train-jsonl "$TRAIN_JSONL" \
    --out-dir "$OUT_DIR" \
    --device mlx \
    --speaker "$SPEAKER" \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$MLX_STEPS" \
    --max-clips 0 \
    --rank 8 \
    --grad-accum "$GRAD_ACCUM" \
    --n-layers 6
  echo "[qwen3-jfk] MLX LoRA done → $OUT_DIR"
  echo "export RLX_QWEN3_TTS_DIR=$OUT_DIR"
  exit 0
fi

echo "[qwen3-jfk] HF SFT device=$TRAIN_DEVICE speaker=$SPEAKER"
export PYTORCH_ENABLE_MPS_FALLBACK=1
(
  cd "$FINETUNE_DIR"
  python "$ROOT/scripts/qwen3_tts_sft_mps.py" "sft_12hz.py" \
    --init_model_path "$BASE_DIR" \
    --output_model_path "$OUT_DIR" \
    --train_jsonl "$TRAIN_JSONL" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$EPOCHS" \
    --speaker_name "$SPEAKER"
)

echo "[qwen3-jfk] checkpoint under $OUT_DIR"
echo "export RLX_QWEN3_TTS_DIR=$OUT_DIR/checkpoint-epoch-$((EPOCHS - 1))"
echo "just qwen3-tts -- --model-dir \$RLX_QWEN3_TTS_DIR --text \"...\" --speaker $SPEAKER --language english --out-wav /tmp/jfk.wav"
