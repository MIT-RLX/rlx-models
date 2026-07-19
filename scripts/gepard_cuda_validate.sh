#!/usr/bin/env bash
# Gepard CUDA full-backend validation (run on NVIDIA rig).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WEIGHTS="${RLX_GEPARD_DIR:-weights/tts/gepard}"
FOX='The quick brown fox jumps over the lazy dog.'
LONG='The quick brown fox jumps over the lazy dog. Courage and kindness matter more than cleverness alone when people face hard times together and choose to help each other without waiting for perfect conditions.'

echo "== cargo check nvidia-gpu =="
cargo check -p rlx-gepard --features nvidia-gpu

echo "== fox CUDA =="
/usr/bin/time -f 'wall_sec=%e' cargo run -p rlx-gepard --bin rlx-gepard --release --features nvidia-gpu -- \
  --weights "$WEIGHTS" --text "$FOX" --device cuda --seed 54 --out /tmp/gepard_cuda_fox.wav

echo "== long CUDA seed 4 =="
/usr/bin/time -f 'wall_sec=%e' cargo run -p rlx-gepard --bin rlx-gepard --release --features nvidia-gpu -- \
  --weights "$WEIGHTS" --text "$LONG" --device cuda --seed 4 --out /tmp/gepard_cuda_long.wav

echo "== backend matrix cpu,cuda =="
RLX_DEVICES=cpu,cuda cargo run -p rlx-gepard --release --example backend_matrix --features nvidia-gpu

echo "== timing =="
cargo run -p rlx-gepard --release --example bench_timing --features nvidia-gpu -- --device cuda

echo "Done. Whisper fox/long on host with Whisper Tiny if available."
