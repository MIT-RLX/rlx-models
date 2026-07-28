#!/usr/bin/env bash
# Validate the MLX-packed Qwen3 path (Op::DequantMatMul{MlxAffine}) on the
# NVIDIA/Vulkan host. The CUDA host has no mlx-lm, so each backend is checked
# against the mlx-lm oracle logits synced from the Mac (.mlx-test/oracle_*.npy)
# and against the on-machine CPU run.
#
# From Mac:  scripts/mlx_packed_cuda_validate.sh --remote
# On the CUDA host:    scripts/mlx_packed_cuda_validate.sh --local
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${RLX_CUDA_HOST:?set RLX_CUDA_HOST to your CUDA host, e.g. user@host}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"
MODEL="${MODEL:-.mlx-test/qwen3-0.6b-4bit}"
ORACLE="${ORACLE:-.mlx-test/oracle_prefill_last_logits.npy}"
# Oracle: prompt "The capital of France is" → last-token argmax = 12095 (" Paris").
ORACLE_ARGMAX="${ORACLE_ARGMAX:-12095}"
BACKENDS_DEFAULT="cpu cuda vulkan gpu"

remote_run() {
  echo ">> sync trees to $HOST"
  bash "$ROOT/scripts/matrix/sync_to_remote.sh"
  echo ">> running MLX-packed validate on $HOST"
  ssh "$HOST" "cd $REMOTE_MODELS && bash scripts/mlx_packed_cuda_validate.sh --local"
}

ensure_model() {
  [ -d "$MODEL" ] && return 0
  echo ">> model missing at $MODEL — fetching mlx-community/Qwen3-0.6B-4bit"
  local py=/tmp/rlx-hf-venv/bin/python
  if [ ! -x "$py" ]; then
    python3 -m venv /tmp/rlx-hf-venv
    /tmp/rlx-hf-venv/bin/pip install -q -U pip huggingface_hub
  fi
  "$py" -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-0.6B-4bit', local_dir='$MODEL')"
}

local_run() {
  cd "$ROOT"
  export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH:-}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
  export WGPU_BACKEND="${WGPU_BACKEND:-vulkan}"
  local BACKENDS="${BACKENDS:-$BACKENDS_DEFAULT}"

  ensure_model

  echo "== build rlx-qwen3 mlx_community_run --features cuda,vulkan,gpu =="
  cargo build --release -p rlx-qwen3 --example mlx_community_run --features "cuda,vulkan,gpu"
  local BIN="target/release/examples/mlx_community_run"

  mkdir -p .mlx-test
  local ran=()
  for b in $BACKENDS; do
    echo "== run backend=$b =="
    if RLX_DIAG_OUT=".mlx-test/msi_${b}.bin" "$BIN" "$MODEL" "$b" 2>&1 | grep -viE '^\[' | grep -E 'device|prefill argmax|PREFILL_ARGMAX'; then
      ran+=("$b")
    else
      echo "   backend $b FAILED to run"
    fi
  done

  echo
  echo "== parity report (argmax must be $ORACLE_ARGMAX; cos vs mlx oracle + vs cpu) =="
  python3 - "$ORACLE" "$ORACLE_ARGMAX" "${ran[@]}" <<'PY'
import sys, numpy as np, os
oracle_path, oracle_argmax = sys.argv[1], int(sys.argv[2])
backends = sys.argv[3:]
ref = np.load(oracle_path).astype(np.float64) if os.path.exists(oracle_path) else None
def load(b):
    p = f".mlx-test/msi_{b}.bin"
    return np.fromfile(p, dtype="<f4").astype(np.float64) if os.path.exists(p) else None
cpu = load("cpu")
ok_all = True
for b in backends:
    v = load(b)
    if v is None:
        print(f"  {b:7s}: NO OUTPUT"); ok_all = False; continue
    am = int(v.argmax())
    cos_o = float(ref@v/(np.linalg.norm(ref)*np.linalg.norm(v))) if ref is not None and len(ref)==len(v) else float('nan')
    cos_c = float(cpu@v/(np.linalg.norm(cpu)*np.linalg.norm(v))) if cpu is not None and len(cpu)==len(v) else float('nan')
    ok = (am == oracle_argmax)
    ok_all = ok_all and ok
    print(f"  {b:7s}: argmax={am} {'OK' if ok else 'FAIL'}  cos_oracle={cos_o:.6f}  cos_cpu={cos_c:.6f}")
print("\nRESULT:", "ALL OK" if ok_all else "FAILURES PRESENT")
sys.exit(0 if ok_all else 1)
PY
}

case "${1:-}" in
  --remote) remote_run ;;
  --local|"") local_run ;;
  *) echo "usage: $0 [--remote|--local]" >&2; exit 2 ;;
esac
