#!/usr/bin/env bash
# Bench production vs legacy native allocation (timing + peak RSS on macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

PHRASE="${1:-hello}"
DEVICE="${KITTEN_BENCH_DEVICE:-auto}"
EXTRA=()
DEVICE_ARGS=()
if [[ "$PHRASE" == "long" ]]; then
  EXTRA+=(--long)
  PHRASE_LABEL=long
else
  PHRASE_LABEL=hello
fi

if [[ "$DEVICE" != "auto" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

export KITTEN_RLX_WEIGHTS="${KITTEN_RLX_WEIGHTS:-crates/kitten_tts_mini_rlx/weights}"
export KITTEN_VOICES_NPZ="${KITTEN_VOICES_NPZ:-.cache/kittentts-mini-0.8/voices.npz}"
export KITTEN_RLX_SKIP_FUSION="${KITTEN_RLX_SKIP_FUSION:-1}"
export KITTEN_RLX_PREFER_METAL="${KITTEN_RLX_PREFER_METAL:-0}"

if [[ ! -f "$KITTEN_VOICES_NPZ" ]]; then
  echo "missing voices.npz — run: just fetch-kittentts" >&2
  exit 1
fi

echo "Building native_alloc_bench (native-fast, metal)..."
cargo build -p rlx-kittentts --features native-fast,metal --release --example native_alloc_bench

BIN=target/release/examples/native_alloc_bench

bench_mode() {
  local mode=$1
  echo ""
  echo "======== ${PHRASE_LABEL} / ${mode} ========"
  /usr/bin/time -l env \
    KITTEN_RLX_WEIGHTS="$KITTEN_RLX_WEIGHTS" \
    KITTEN_VOICES_NPZ="$KITTEN_VOICES_NPZ" \
    "$BIN" --mode "$mode" ${EXTRA+"${EXTRA[@]}"} ${DEVICE_ARGS+"${DEVICE_ARGS[@]}"} 2>&1 \
    | tee "/tmp/kitten_bench_${PHRASE_LABEL}_${mode}.log" \
    | rg -e '\[bench\]|\[bench-result\]|\[kittentts\]|maximum resident set size|load\+prewarm|native infer' || true
}

for mode in production legacy_full; do
  bench_mode "$mode" || {
    echo "WARN: ${mode} failed (see log)" >&2
  }
done

echo ""
echo "Logs: /tmp/kitten_bench_${PHRASE_LABEL}_*.log"
