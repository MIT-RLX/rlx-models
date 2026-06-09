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

# End-to-end JFK custom voice: fetch Base → prep transcripts → train (Metal MPS SFT).
#
# Quick smoke (32 clips): MAX_CLIPS=32 EPOCHS=1 bash scripts/qwen3_tts_train_go.sh
# Full run: bash scripts/qwen3_tts_train_go.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RLX_QWEN3_TTS_BASE_DIR="${RLX_QWEN3_TTS_BASE_DIR:-$ROOT/.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base}"
export JFK_DIR="${JFK_DIR:-$ROOT/.cache/qwen3-tts/jfk}"
export OUT_DIR="${OUT_DIR:-$ROOT/.cache/qwen3-tts/jfk-checkpoint}"
export SPEAKER="${SPEAKER:-jfk}"
export BACKEND="${BACKEND:-metal}"
export EPOCHS="${EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export PREPARE_BATCH="${PREPARE_BATCH:-64}"
export MAX_CLIPS="${MAX_CLIPS:-0}"
export JFK_MAX_CLIPS="${JFK_MAX_CLIPS:-$MAX_CLIPS}"

echo "[go] 1/3 fetch Base weights → $RLX_QWEN3_TTS_BASE_DIR"
if [[ ! -f "$RLX_QWEN3_TTS_BASE_DIR/model.safetensors" ]]; then
  just fetch-qwen3-tts-base
else
  echo "[go] Base weights present"
fi

echo "[go] 2/3 prep JFK clips + Whisper transcripts"
bash scripts/qwen3_tts_prep_jfk.sh

echo "[go] 3/3 train custom voice (BACKEND=$BACKEND speaker=$SPEAKER)"
BACKEND="$BACKEND" EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" PREPARE_BATCH="$PREPARE_BATCH" \
  bash scripts/qwen3_tts_finetune_jfk.sh

echo "[go] done. export RLX_QWEN3_TTS_DIR=$OUT_DIR/checkpoint-epoch-$((EPOCHS - 1))"
