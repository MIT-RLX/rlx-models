#!/usr/bin/env bash
# Sesame CSM CUDA validation on NVIDIA host (set RLX_CUDA_HOST; e.g. ssh your-cuda-host).
# From Mac: scripts/sesame_cuda_validate.sh --remote
# On the CUDA host:   scripts/sesame_cuda_validate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${RLX_CUDA_HOST:?set RLX_CUDA_HOST to your CUDA host, e.g. user@host}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"
FOX='The quick brown fox jumps over the lazy dog.'
LONG='The quick brown fox jumps over the lazy dog. Courage and kindness matter more than cleverness alone when people face hard times together and choose to help each other without waiting for perfect conditions.'

remote_run() {
  echo ">> sync trees to $HOST"
  bash "$ROOT/scripts/matrix/sync_to_remote.sh"
  echo ">> running Sesame CUDA validate on $HOST"
  ssh "$HOST" "cd $REMOTE_MODELS && bash scripts/sesame_cuda_validate.sh --local"
}

hf_download() {
  local repo="$1" dest="$2"
  local py="${HF_PYTHON:-}"
  if [[ -z "$py" ]]; then
    if [[ -x /tmp/rlx-hf-venv/bin/python ]]; then
      py=/tmp/rlx-hf-venv/bin/python
    elif command -v hf >/dev/null 2>&1 && head -1 "$(command -v hf)" 2>/dev/null | grep -qv '/Users/'; then
      hf download "$repo" --local-dir "$dest"
      return
    else
      echo ">> creating /tmp/rlx-hf-venv for huggingface_hub"
      python3 -m venv /tmp/rlx-hf-venv
      /tmp/rlx-hf-venv/bin/pip install -q -U pip huggingface_hub
      py=/tmp/rlx-hf-venv/bin/python
    fi
  fi
  "$py" -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$dest')
"
}

ensure_weights() {
  mkdir -p weights/tts/sesame .cache/mimi .cache/whisper-tiny
  if [[ ! -f weights/tts/sesame/model.safetensors ]]; then
    echo ">> fetching unsloth/csm-1b → weights/tts/sesame"
    hf_download unsloth/csm-1b weights/tts/sesame
  fi
  if [[ ! -f .cache/mimi/model.safetensors ]] && [[ ! -f .cache/mimi/config.json ]]; then
    echo ">> fetching kyutai/mimi → .cache/mimi"
    hf_download kyutai/mimi .cache/mimi
  fi
  if [[ ! -f .cache/whisper-tiny/model.safetensors ]]; then
    echo ">> fetching openai/whisper-tiny → .cache/whisper-tiny (best-effort)"
    hf_download openai/whisper-tiny .cache/whisper-tiny || true
  fi
}

local_run() {
  cd "$ROOT"
  export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH:-}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"

  ensure_weights

  echo "== cargo check -p rlx-sesame --features nvidia-gpu =="
  cargo check -p rlx-sesame --features nvidia-gpu

  echo "== fox CUDA synth (timed WAV) =="
  /usr/bin/time -f 'wall_sec=%e' cargo run -p rlx-sesame --release --features nvidia-gpu -- \
    --model-dir weights/tts/sesame --mimi-dir .cache/mimi \
    --text "$FOX" --device cuda --seed 42 --max-frames 200 \
    --output /tmp/sesame_cuda_fox.wav

  echo "== backend matrix fox cpu,cuda =="
  RLX_DEVICES=cpu,cuda \
  RLX_SEED=42 \
  RLX_CODES_CACHE=/tmp/sesame_fox_codes.json \
  RLX_FORCE_RESYNTH=1 \
  cargo run -p rlx-sesame --release --example backend_matrix --features nvidia-gpu

  echo "== backend matrix long cpu,cuda =="
  RLX_DEVICES=cpu,cuda \
  RLX_SEED=42 \
  RLX_TEXT="$LONG" \
  RLX_CODES_CACHE=/tmp/sesame_long_codes.json \
  RLX_FORCE_RESYNTH=1 \
  cargo run -p rlx-sesame --release --example backend_matrix --features nvidia-gpu

  echo "Done. WAV: /tmp/sesame_cuda_fox.wav"
}

case "${1:-}" in
  --remote) remote_run ;;
  --local|"") local_run ;;
  *)
    echo "usage: $0 [--remote|--local]" >&2
    exit 2
    ;;
esac
