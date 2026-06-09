#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG=${RLX_VOXTRAL_TTS_TOOLS_TAG:-rlx-voxtral-tts-tools:latest}

require_dir() {
  local name=$1
  if [[ -z "${!name-}" ]]; then
    echo "missing env: $name" >&2
    exit 2
  fi
}

CMD=${1:-help}
shift || true

case "$CMD" in
  build)
    docker build -t "$TAG" -f "$ROOT/docker/voxtral-tts/Dockerfile.tools" "$ROOT/docker/voxtral-tts"
    ;;
  tokenize)
    require_dir RLX_VOXTRAL_TTS_DIR
    TEXT=${RLX_VOXTRAL_TTS_TEXT:-Hello}
    VOICE=${RLX_VOXTRAL_TTS_VOICE:-neutral_female}
    OUT=${RLX_VOXTRAL_TTS_OUT:-$ROOT/.cache/voxtral/tts/prompt_tokens.txt}
    mkdir -p "$(dirname "$OUT")"
    docker run --rm \
      -v "$(realpath "$RLX_VOXTRAL_TTS_DIR"):/model:ro" \
      -v "$(realpath "$(dirname "$OUT")"):/out" \
      -e RLX_VOXTRAL_TTS_TEXT="$TEXT" \
      -e RLX_VOXTRAL_TTS_VOICE="$VOICE" \
      -e RLX_VOXTRAL_TTS_OUT="/out/$(basename "$OUT")" \
      "$TAG" tokenize
    ;;
  convert-voices)
    require_dir RLX_VOXTRAL_TTS_DIR
    docker run --rm \
      -v "$(realpath "$RLX_VOXTRAL_TTS_DIR"):/model" \
      "$TAG" convert-voices
    ;;
  *)
    echo "Usage: run-tools.sh {build|tokenize|convert-voices}" >&2
    exit 2
    ;;
esac
