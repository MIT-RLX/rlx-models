#!/usr/bin/env bash
# Bench Bonsai-27B across RLX backends: complex prompt, prefill + per-token.
#
# Local (Apple Silicon):
#   scripts/matrix/bonsai27b_bench.sh
#
# MSI CUDA (after sync):
#   scripts/matrix/bonsai27b_bench.sh --cuda-only
#   # or from Mac: scripts/matrix/bonsai27b_bench.sh --remote-cuda
#
# Env:
#   BONSAI_GGUF, BONSAI_MAX_TOKENS (default 16), BONSAI_BACKENDS,
#   BONSAI_OUT, RLX_KV_CACHE_MAX_RESIDENT (default 1 — one decode arena)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GGUF="${BONSAI_GGUF:-$REPO/weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf}"
MAX_TOKENS="${BONSAI_MAX_TOKENS:-16}"
OUT_DIR="${BONSAI_OUT:-$HERE/out/bonsai27b_bench}"
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export RLX_QWEN35_BENCH=1
export RLX_LOW_MEM_COMPILE=1
export RLX_DEQUANT_CACHE=0
export RLX_CUDA_NO_CUDNN="${RLX_CUDA_NO_CUDNN:-1}"
# One resident decode arena (weights shared across rungs when climbing).
export RLX_KV_CACHE_MAX_RESIDENT="${RLX_KV_CACHE_MAX_RESIDENT:-1}"

COMPLEX_PROMPT="${BONSAI_PROMPT:-$(cat <<'EOF'
You are advising a systems engineer who must choose an inference backend for a 27B hybrid transformer (gated-DeltaNet + full attention, 1-bit Q1_0 weights). Write a structured brief that: (1) contrasts latency vs VRAM tradeoffs for Metal, MLX, CUDA, and WebGPU; (2) explains why packed 1-bit weights change the memory story versus FP16; (3) lists three failure modes when a GatedDeltaNet op falls back to host readback; (4) ends with a concrete recommendation for interactive chat on a 64GB Apple Silicon Mac and on a 16GB RTX 3080 Ti laptop. Use short numbered sections and keep the tone precise.
EOF
)}"

MODE="local"
for arg in "$@"; do
  case "$arg" in
    --cuda-only) MODE="cuda-only" ;;
    --remote-cuda) MODE="remote-cuda" ;;
    --help|-h)
      sed -n '2,16p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p "$OUT_DIR"
cd "$REPO"

if [ "$MODE" = "remote-cuda" ]; then
  echo ">> sync trees to msi"
  bash "$HERE/sync_to_msi.sh"
  echo ">> remote CUDA bench"
  ssh msi "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; \
    BONSAI_GGUF=\$HOME/rlx-models/weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf \
    BONSAI_MAX_TOKENS=$MAX_TOKENS BONSAI_OUT=\$HOME/rlx-models/scripts/matrix/out/bonsai27b_bench \
    bash \$HOME/rlx-models/scripts/matrix/bonsai27b_bench.sh --cuda-only"
  mkdir -p "$OUT_DIR"
  scp msi:rlx-models/scripts/matrix/out/bonsai27b_bench/cuda.log "$OUT_DIR/cuda.log" || true
  scp msi:rlx-models/scripts/matrix/out/bonsai27b_bench/summary.md "$OUT_DIR/summary_cuda.md" || true
  echo ">> pulled CUDA logs to $OUT_DIR"
  exit 0
fi

if [ ! -s "$GGUF" ]; then
  echo "missing GGUF: $GGUF" >&2
  exit 2
fi

# Tokenize length probe uses the runner itself; estimate via wc for log header.
PROMPT_CHARS=${#COMPLEX_PROMPT}
echo ">> prompt chars=$PROMPT_CHARS max_tokens=$MAX_TOKENS gguf=$GGUF"

if [ "$MODE" = "cuda-only" ]; then
  FEATURES="${BONSAI_FEATURES:-cuda}"
  BACKENDS=(cuda)
  BIN_PKG=rlx-qwen35
  BIN_NAME=rlx-qwen35
  # Bound discrete VRAM: keep one large decode bucket resident.
  export RLX_KV_CACHE_MAX_RESIDENT="${RLX_KV_CACHE_MAX_RESIDENT:-1}"
else
  FEATURES="${BONSAI_FEATURES:-apple-silicon,gpu}"
  if [ -n "${BONSAI_BACKENDS:-}" ]; then
    IFS=',' read -r -a BACKENDS <<< "$BONSAI_BACKENDS"
  else
    BACKENDS=(metal mlx gpu)
    # CPU is ~minutes/token on 27B — opt-in only.
    if [ "${BONSAI_INCLUDE_CPU:-0}" = "1" ]; then
      BACKENDS+=(cpu)
    fi
    if command -v vulkaninfo >/dev/null 2>&1; then
      BACKENDS+=(vulkan)
    fi
    if [[ "$(uname -s)" == Darwin ]]; then
      BACKENDS+=(coreml)
    fi
  fi
  BIN_PKG=rlx-qwen35
  BIN_NAME=rlx-qwen35
fi

echo ">> building $BIN_PKG ($FEATURES)"
cargo build --release -p "$BIN_PKG" --bin "$BIN_NAME" --features "$FEATURES" \
  2>&1 | tee "$OUT_DIR/build.log" | tail -8

BIN="$REPO/target/release/$BIN_NAME"
SUMMARY="$OUT_DIR/summary.md"
{
  echo "# Bonsai-27B backend bench"
  echo
  echo "- host: \`$(hostname)\` (\`$(uname -srm)\`)"
  echo "- gguf: \`$GGUF\`"
  echo "- max_tokens: $MAX_TOKENS"
  echo "- prompt_chars: $PROMPT_CHARS"
  echo "- env: RLX_QWEN35_BENCH=1 RLX_KV_CACHE_MAX_RESIDENT=${RLX_KV_CACHE_MAX_RESIDENT:-unset}"
  echo
  echo "| backend | prompt_tok | new_tok | prefill_ms | decode_ms | ms/tok | tok/s | status |"
  echo "|---------|------------|---------|------------|-----------|--------|-------|--------|"
} > "$SUMMARY"

run_one() {
  local dev="$1"
  local log="$OUT_DIR/${dev}.log"
  echo ">> === $dev ==="
  # max_seq = prompt + new tokens + chat-template headroom (keep lean for wgpu 4GiB).
  local max_seq=$((MAX_TOKENS + 256))
  set +e
  "$BIN" --weights "$GGUF" --packed --device "$dev" \
    --max-seq "$max_seq" --max-tokens "$MAX_TOKENS" \
    --temperature 0.0 --seed 0 \
    --prompt "$COMPLEX_PROMPT" \
    >"$log" 2>&1
  local rc=$?
  set +e

  local line
  line="$(grep -E '\[qwen35\]\[bench\]' "$log" | tail -1 || true)"
  if [ -z "$line" ]; then
    echo "| $dev | — | — | — | — | — | — | FAIL (rc=$rc) |" >> "$SUMMARY"
    echo "FAIL $dev rc=$rc (no bench line)"
    tail -25 "$log" || true
    return 0
  fi
  # Example:
  # [qwen35][bench] device=Metal prompt_tokens=12 new_tokens=16 prefill_ms=1.0 decode_ms=2.0 total_ms=3.0 ms/tok=0.1 tok/s=1.234
  local pt nt pref dec mpt tps
  pt="$(printf '%s\n' "$line" | sed -n 's/.*prompt_tokens=\([0-9]*\).*/\1/p')"
  nt="$(printf '%s\n' "$line" | sed -n 's/.*new_tokens=\([0-9]*\).*/\1/p')"
  pref="$(printf '%s\n' "$line" | sed -n 's/.*prefill_ms=\([0-9.]*\).*/\1/p')"
  dec="$(printf '%s\n' "$line" | sed -n 's/.*decode_ms=\([0-9.]*\).*/\1/p')"
  mpt="$(printf '%s\n' "$line" | sed -n 's/.*ms\/tok=\([0-9.]*\).*/\1/p')"
  tps="$(printf '%s\n' "$line" | sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p')"
  local status="OK"
  [ "$rc" -eq 0 ] || status="FAIL (rc=$rc)"
  echo "| $dev | $pt | $nt | $pref | $dec | $mpt | $tps | $status |" >> "$SUMMARY"
  echo "OK $dev: prefill=${pref}ms decode=${dec}ms (${mpt} ms_per_tok, ${tps} tok_per_s)"
}

echo ">> backends: ${BACKENDS[*]}"
for dev in "${BACKENDS[@]}"; do
  run_one "$dev"
done

echo
echo ">> summary → $SUMMARY"
cat "$SUMMARY"
