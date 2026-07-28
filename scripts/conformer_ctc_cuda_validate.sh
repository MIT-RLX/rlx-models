#!/usr/bin/env bash
# Conformer-CTC CUDA validation on NVIDIA host (set RLX_CUDA_HOST; e.g. ssh your-cuda-host).
# From Mac: scripts/conformer_ctc_cuda_validate.sh --remote
# On the CUDA host:   scripts/conformer_ctc_cuda_validate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${RLX_CUDA_HOST:?set RLX_CUDA_HOST to your CUDA host, e.g. user@host}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"

remote_run() {
  echo ">> sync trees to $HOST"
  bash "$ROOT/scripts/matrix/sync_to_remote.sh"
  echo ">> sync conformer-ctc assets (nemo + sample wav)"
  ssh "$HOST" "mkdir -p ~/$REMOTE_MODELS/.cache/conformer-ctc"
  rsync -az \
    "$ROOT/.cache/conformer-ctc/stt_en_conformer_ctc_small.nemo" \
    "$ROOT/.cache/conformer-ctc/sample.wav" \
    "$HOST:$REMOTE_MODELS/.cache/conformer-ctc/"
  echo ">> running Conformer-CTC CUDA validate on $HOST"
  ssh "$HOST" "cd $REMOTE_MODELS && bash scripts/conformer_ctc_cuda_validate.sh --local"
}

local_run() {
  cd "$ROOT"
  export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH:-}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"

  NEMO=".cache/conformer-ctc/stt_en_conformer_ctc_small.nemo"
  WAV=".cache/conformer-ctc/sample.wav"
  if [[ ! -f "$NEMO" ]]; then
    echo "missing $NEMO — fetch on Mac then re-sync, or:"
    echo "  hf download nvidia/stt_en_conformer_ctc_small --local-dir .cache/conformer-ctc"
    exit 1
  fi
  if [[ ! -f "$WAV" ]]; then
    echo "missing $WAV"
    exit 1
  fi

  echo "== nvidia-smi =="
  nvidia-smi -L || true

  echo "== cargo check -p rlx-conformer-ctc --features nvidia-gpu =="
  cargo check -p rlx-conformer-ctc --features nvidia-gpu

  echo "== CLI transcribe cuda (cold + warm) =="
  /usr/bin/time -f 'wall_sec=%e' cargo run -p rlx-conformer-ctc --release --features nvidia-gpu -- \
    transcribe --nemo "$NEMO" --wav "$WAV" --device cuda --warm

  echo "== backend matrix cpu,cuda =="
  RLX_DEVICES=cpu,cuda \
  RLX_CONFORMER_CTC_NEMO="$NEMO" \
  RLX_CONFORMER_CTC_WAV="$WAV" \
  cargo run -p rlx-conformer-ctc --release --example backend_matrix --features nvidia-gpu

  echo "== conformer-ctc CUDA validate OK =="
}

case "${1:-}" in
  --remote) remote_run ;;
  --local|"") local_run ;;
  *)
    echo "usage: $0 [--remote|--local]"
    exit 2
    ;;
esac
