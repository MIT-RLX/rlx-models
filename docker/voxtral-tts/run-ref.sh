#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG=${RLX_VOXTRAL_TTS_REF_TAG:-rlx-voxtral-tts-ref:gpu}

require_dir() {
  local name=$1
  if [[ -z "${!name-}" ]]; then
    echo "missing env: $name" >&2
    exit 2
  fi
}

CMD=${1:-help}
shift || true

gpu_args=(--gpus all)

case "$CMD" in
  build)
    docker build -t "$TAG" -f "$ROOT/docker/voxtral-tts/Dockerfile.ref" "$ROOT/docker/voxtral-tts"
    ;;
  export-codes)
    require_dir RLX_VOXTRAL_TTS_DIR
    TEXT=${RLX_VOXTRAL_TTS_TEXT:-Hello}
    VOICE=${RLX_VOXTRAL_TTS_VOICE:-neutral_female}
    OUT=${RLX_VOXTRAL_TTS_OUT_CODES:-$ROOT/.cache/voxtral/tts/vllm_codes.txt}
    SEED=${RLX_VOXTRAL_TTS_SEED:-42}
    CFG=${RLX_VOXTRAL_TTS_CFG_ALPHA:-1.2}
    mkdir -p "$(dirname "$OUT")"
    docker run --rm "${gpu_args[@]}" \
      -v "$(realpath "$RLX_VOXTRAL_TTS_DIR"):/model:ro" \
      -v "$(realpath "$(dirname "$OUT")"):/out" \
      -e RLX_VOXTRAL_TTS_TEXT="$TEXT" \
      -e RLX_VOXTRAL_TTS_VOICE="$VOICE" \
      -e RLX_VOXTRAL_TTS_OUT_CODES="/out/$(basename "$OUT")" \
      -e RLX_VOXTRAL_TTS_CFG_ALPHA="$CFG" \
      -e RLX_VOXTRAL_TTS_SEED="$SEED" \
      "$TAG" export-codes
    ;;
  synthesize)
    require_dir RLX_VOXTRAL_TTS_DIR
    TEXT=${RLX_VOXTRAL_TTS_TEXT:-Hello}
    VOICE=${RLX_VOXTRAL_TTS_VOICE:-neutral_female}
    OUT=${RLX_VOXTRAL_TTS_OUT_WAV:-$ROOT/.cache/voxtral/tts/vllm_reference.wav}
    mkdir -p "$(dirname "$OUT")"
    docker run --rm "${gpu_args[@]}" \
      -v "$(realpath "$RLX_VOXTRAL_TTS_DIR"):/model:ro" \
      -v "$(realpath "$(dirname "$OUT")"):/out" \
      -e RLX_VOXTRAL_TTS_TEXT="$TEXT" \
      -e RLX_VOXTRAL_TTS_VOICE="$VOICE" \
      -e RLX_VOXTRAL_TTS_OUT_WAV="/out/$(basename "$OUT")" \
      "$TAG" synthesize
    ;;
  *)
    echo "Usage: run-ref.sh {build|export-codes|synthesize}" >&2
    exit 2
    ;;
esac
