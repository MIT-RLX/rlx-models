#!/usr/bin/env bash
# Metal↔CUDA Bonsai decode tap compare.
#
# 1) On Mac (Metal), write taps:
#    RLX_QWEN35_TAP_PATH=/tmp/bonsai_metal_tap.jsonl \\
#      scripts/matrix/bonsai_trace_compare.sh metal
#
# 2) On the CUDA host, after sync:
#    scripts/matrix/bonsai_trace_compare.sh cuda
#
# 3) Diff first mismatched step:
#    scripts/matrix/bonsai_trace_compare.sh diff \\
#      /tmp/bonsai_metal_tap.jsonl /tmp/bonsai_cuda_tap.jsonl
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GGUF="${BONSAI_GGUF:-weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf}"
SYSTEM="${BONSAI_SYSTEM:-You are a helpful assistant. Answer clearly in English.}"
PROMPT="${BONSAI_PROMPT:-What is the capital of France? Reply with one short sentence.}"
MAX_SEQ="${BONSAI_MAX_SEQ:-64}"
MAX_TOKENS="${BONSAI_MAX_TOKENS:-8}"
HOST="${RLX_CUDA_HOST:?set RLX_CUDA_HOST to your CUDA host, e.g. user@host}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"

common_env() {
  export RLX_LOW_MEM_COMPILE=1
  export RLX_DEQUANT_CACHE=0
  export RLX_QWEN35_DECODE_TRACE=1
  export RLX_QWEN35_TAP=1
  export RLX_QWEN35_TAP_STEPS="${RLX_QWEN35_TAP_STEPS:-8}"
  export RLX_PHASE_TIMING="${RLX_PHASE_TIMING:-1}"
}

run_local() {
  local device="$1"
  local tap="${2}"
  common_env
  export RLX_QWEN35_TAP_PATH="$tap"
  if [[ "$device" == "cuda" ]]; then
    export RLX_CUDA_PATH_TRACE=1
    export RLX_CUDA_PARITY=1
    export RLX_CUDA_MATMUL_PRECISE=1
    export RLX_CUDA_COMPILE_TIMING=1
    export RLX_KV_CACHE_MAX_RESIDENT=1
  fi
  cd "$ROOT"
  ./target/release/rlx-run bonsai --weights "$GGUF" --packed --device "$device" \
    --max-seq "$MAX_SEQ" --max-tokens "$MAX_TOKENS" --temperature 0.0 --seed 0 \
    --system "$SYSTEM" --prompt "$PROMPT"
}

cmd="${1:-}"
case "$cmd" in
  metal)
    common_env
    export RLX_QWEN35_TAP_PATH="${RLX_QWEN35_TAP_PATH:-/tmp/bonsai_metal_tap.jsonl}"
    rm -f "$RLX_QWEN35_TAP_PATH"
    echo ">> Metal tap → $RLX_QWEN35_TAP_PATH"
    cargo build --release -p rlx-models --bin rlx-run \
      --features 'runner,minicpm5,bonsai,qwen35,metal' 
    run_local metal "$RLX_QWEN35_TAP_PATH"
    ;;
  cuda)
    bash "$HERE/sync_to_remote.sh"
    TAP_REMOTE="${RLX_QWEN35_TAP_PATH:-/tmp/bonsai_cuda_tap.jsonl}"
    ssh "$HOST" bash -s <<EOF
set -euo pipefail
cd $REMOTE_MODELS
export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
cargo build --release -p rlx-models --bin rlx-run \\
  --features 'runner,minicpm5,bonsai,qwen35,cuda'
rm -f $TAP_REMOTE
export RLX_LOW_MEM_COMPILE=1
export RLX_DEQUANT_CACHE=0
export RLX_CUDA_NO_CUDNN=1
export RLX_CUDA_PARITY=1
export RLX_CUDA_MATMUL_PRECISE=1
export RLX_CUDA_PATH_TRACE=1
export RLX_CUDA_COMPILE_TIMING=1
export RLX_KV_CACHE_MAX_RESIDENT=1
export RLX_QWEN35_DECODE_TRACE=1
export RLX_QWEN35_TAP=1
export RLX_QWEN35_TAP_STEPS=${RLX_QWEN35_TAP_STEPS:-8}
export RLX_QWEN35_TAP_PATH=$TAP_REMOTE
export RLX_PHASE_TIMING=1
./target/release/rlx-run bonsai --weights '$GGUF' --packed --device cuda \\
  --max-seq $MAX_SEQ --max-tokens $MAX_TOKENS --temperature 0.0 --seed 0 \\
  --system '$SYSTEM' --prompt '$PROMPT' \\
  2>&1 | tee /tmp/bonsai_cuda_trace.log
EOF
    echo ">> fetch CUDA tap"
    scp "$HOST:$TAP_REMOTE" "${3:-/tmp/bonsai_cuda_tap.jsonl}"
    scp "$HOST:/tmp/bonsai_cuda_trace.log" /tmp/bonsai_cuda_trace.log || true
    ;;
  diff)
    METAL_TAP="${2:-/tmp/bonsai_metal_tap.jsonl}"
    CUDA_TAP="${3:-/tmp/bonsai_cuda_tap.jsonl}"
    python3 - <<PY
import json, sys
from pathlib import Path
a=Path("$METAL_TAP").read_text().splitlines()
b=Path("$CUDA_TAP").read_text().splitlines()
na=len(a); nb=len(b)
print(f"metal_lines={na} cuda_lines={nb}")
n=min(na,nb)
for i in range(n):
    ja=json.loads(a[i]); jb=json.loads(b[i])
    keys=("phase","step","token","kind","checksum","top_ids")
    same=all(ja.get(k)==jb.get(k) for k in keys)
    if not same:
        print(f"FIRST_MISMATCH line={i}")
        print("metal", {k: ja.get(k) for k in keys})
        print("cuda ", {k: jb.get(k) for k in keys})
        sys.exit(1)
print("ALL_MATCHED", n, "tap lines")
PY
    ;;
  *)
    echo "usage: $0 metal | cuda | diff [metal_tap] [cuda_tap]" >&2
    exit 2
    ;;
esac
