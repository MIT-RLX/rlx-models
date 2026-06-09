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

# JFK inaugural clips for Qwen3-TTS fine-tune (24 kHz mono, 6 s segments).
# Reuses voxtral JFK WAVs when present; otherwise downloads and segments fresh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-"$ROOT/.cache/qwen3-tts/jfk"}"
CLIPS_DIR="${CLIPS_DIR:-"$OUT_DIR/wavs"}"
SEGMENT_SEC="${SEGMENT_SEC:-6}"
REF_CLIP="${REF_CLIP:-jfk_0000.wav}"

# Prefer bundled clips under assets/jfk (flat or assets/jfk/wavs), then legacy cache.
resolve_jfk_clips_dir() {
  local d
  for d in "$ROOT/assets/jfk/wavs" "$ROOT/assets/jfk" "$ROOT/.cache/voxtral/jfk/wavs"; do
    if [[ -d "$d" ]] && compgen -G "$d/jfk_*.wav" >/dev/null; then
      echo "$d"
      return 0
    fi
  done
  return 1
}

mkdir -p "$OUT_DIR" "$CLIPS_DIR"

if JFK_SRC="$(resolve_jfk_clips_dir)"; then
  echo "[qwen3-jfk] link clips from $JFK_SRC"
  rm -f "$CLIPS_DIR"/*.wav
  for f in "$JFK_SRC"/jfk_*.wav; do
    ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$CLIPS_DIR/$(basename "$f")"
  done
else
  echo "[qwen3-jfk] no JFK clip dir — run voxtral segmenter (writes .cache/voxtral/jfk/wavs)"
  OUT_DIR="$OUT_DIR" CLIPS_DIR="$CLIPS_DIR" SEGMENT_SEC="$SEGMENT_SEC" \
    bash "$ROOT/scripts/voxtral_prep_jfk.sh"
  rm -f "$CLIPS_DIR"/*.wav
  for f in "$ROOT/.cache/voxtral/jfk/wavs"/jfk_*.wav; do
    ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$CLIPS_DIR/$(basename "$f")"
  done
fi

N="$(ls -1 "$CLIPS_DIR"/jfk_*.wav 2>/dev/null | wc -l | tr -d ' ')"
echo "[qwen3-jfk] $N clips under $CLIPS_DIR"

REF_WAV="$CLIPS_DIR/$REF_CLIP"
if [[ ! -f "$REF_WAV" ]]; then
  REF_WAV="$(ls "$CLIPS_DIR"/jfk_*.wav | head -1)"
fi

MANIFEST="$OUT_DIR/manifest.json"
TRAIN_RAW="$OUT_DIR/train_raw.jsonl"

# reference = slice public-domain inaugural text by clip time (default, best for JFK)
# whisper   = RLX Whisper ASR per clip
# hybrid    = reference unless Whisper looks cleaner
JFK_TRANSCRIPT_MODE="${JFK_TRANSCRIPT_MODE:-reference}"
REFERENCE_FILE="${REFERENCE_FILE:-$ROOT/scripts/qwen3_tts_jfk_reference.txt}"

WHISPER_DIR="${RLX_WHISPER_DIR:-}"
if [[ "$JFK_TRANSCRIPT_MODE" != "reference" ]]; then
  if [[ -z "$WHISPER_DIR" ]] || [[ ! -f "$WHISPER_DIR/model.safetensors" ]]; then
    for cand in "$ROOT/.cache/whisper-base.en" "$ROOT/.cache/whisper-small.en" "$ROOT/.cache/whisper-tiny"; do
      if [[ -f "$cand/model.safetensors" ]]; then
        WHISPER_DIR="$cand"
        break
      fi
    done
  fi
  if [[ -z "$WHISPER_DIR" ]] || [[ ! -f "$WHISPER_DIR/model.safetensors" ]]; then
    echo "[qwen3-jfk] fetching whisper-base.en for optional ASR (JFK_TRANSCRIPT_MODE=$JFK_TRANSCRIPT_MODE)"
    (cd "$ROOT" && just fetch-whisper-base)
    WHISPER_DIR="$ROOT/.cache/whisper-base.en"
  fi
  export RLX_WHISPER_DIR="$WHISPER_DIR"
fi

echo "[qwen3-jfk] build manifest mode=$JFK_TRANSCRIPT_MODE segment=${SEGMENT_SEC}s"
export JFK_TRANSCRIPT_MODE
export SEGMENT_SEC
export JFK_MAX_CLIPS="${JFK_MAX_CLIPS:-0}"
cargo run -p rlx-models --example qwen3_tts_jfk_manifest --release -- \
  --wav-dir "$CLIPS_DIR" \
  --manifest "$MANIFEST" \
  --train-jsonl "$TRAIN_RAW" \
  --ref-wav "$REF_WAV" \
  --reference-file "$REFERENCE_FILE" \
  --segment-sec "$SEGMENT_SEC" \
  --transcript-mode "$JFK_TRANSCRIPT_MODE"

echo "[qwen3-jfk] wrote $MANIFEST and $TRAIN_RAW"
