#!/usr/bin/env bash
# Wake-word CUDA validation on NVIDIA host (e.g. ssh msi).
# From Mac: scripts/wake_cuda_validate.sh --remote
# On msi:   scripts/wake_cuda_validate.sh --local
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${MSI_HOST:-msi}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"
FEAT="${WAKE_CUDA_FEATURES:-cuda}"

remote_run() {
  echo ">> sync trees to $HOST"
  bash "$ROOT/scripts/matrix/sync_to_msi.sh"
  echo ">> running wake CUDA validate on $HOST"
  ssh "$HOST" "cd $REMOTE_MODELS && bash scripts/wake_cuda_validate.sh --local"
}

local_run() {
  cd "$ROOT"
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env" 2>/dev/null || true
  export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH:-}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"

  echo "== nvidia-smi =="
  nvidia-smi -L || true

  echo "== cargo check wake crates --features ${FEAT} =="
  for pkg in rlx-wake rlx-openwakeword rlx-nanowakeword rlx-porcupine rlx-voxrt; do
    cargo check -p "$pkg" --features "$FEAT"
  done

  echo "== backend_quick_check (all wake crates) =="
  cargo test -p rlx-wake --test backend_quick_check --features "$FEAT" --release
  cargo test -p rlx-wake --test train_backends --features "$FEAT" --release
  cargo test -p rlx-openwakeword --test backend_quick_check --features "$FEAT" --release
  cargo test -p rlx-nanowakeword --test backend_quick_check --features "$FEAT" --release
  cargo test -p rlx-porcupine --test backend_quick_check --features "$FEAT" --release
  cargo test -p rlx-voxrt --test backend_quick_check --features "$FEAT" --release

  echo "== backend_parity (cpu vs cuda, 100%) =="
  cargo test -p rlx-wake --test backend_parity --features "$FEAT" --release -- --nocapture
  cargo test -p rlx-openwakeword --test backend_parity --features "$FEAT" --release -- --nocapture
  cargo test -p rlx-nanowakeword --test backend_parity --features "$FEAT" --release -- --nocapture
  cargo test -p rlx-porcupine --test backend_parity --features "$FEAT" --release -- --nocapture
  cargo test -p rlx-voxrt --test backend_parity --features "$FEAT" --release -- --nocapture

  echo "== train cnn --device cpu,cuda (synth) =="
  cargo run -p rlx-wake --bin rlx-wake-train --release --features "$FEAT" -- \
    cnn --synth --keyword hey_rlx --out /tmp/wake_cuda_model.safetensors \
    --device cpu,cuda --epochs 8

  echo "== train OWW phrase --device cpu,cuda (synth) =="
  cargo run -p rlx-openwakeword --bin rlx-openwakeword-train --release --features "$FEAT" -- \
    --synth --keyword hey_rlx --out-dir /tmp/wake_cuda_oww --device cpu,cuda --epochs 5

  # 1s silence WAV for CLI sweep
  python3 - <<'PY'
import wave, pathlib
p = pathlib.Path("/tmp/wake_cuda_silence.wav")
with wave.open(str(p), "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 16000)
print(p)
PY

  echo "== CLI --device cpu,cuda (all engines) =="
  cargo run -p rlx-openwakeword --bin rlx-openwakeword --release --features "$FEAT" -- \
    --wav /tmp/wake_cuda_silence.wav --device cpu,cuda
  cargo run -p rlx-nanowakeword --bin rlx-nanowakeword --release --features "$FEAT" -- \
    --wav /tmp/wake_cuda_silence.wav --device cpu,cuda --weights /tmp/wake_cuda_model.safetensors --lite
  cargo run -p rlx-porcupine --bin rlx-porcupine --release --features "$FEAT" -- \
    --wav /tmp/wake_cuda_silence.wav --device cpu,cuda --weights /tmp/wake_cuda_model.safetensors
  cargo run -p rlx-voxrt --bin rlx-voxrt --release --features "$FEAT" -- \
    --wav /tmp/wake_cuda_silence.wav --device cpu,cuda --weights /tmp/wake_cuda_model.safetensors

  echo "== wake CUDA validate OK =="
}

case "${1:-}" in
  --remote) remote_run ;;
  --local|"") local_run ;;
  *)
    echo "usage: $0 [--remote|--local]"
    exit 2
    ;;
esac
