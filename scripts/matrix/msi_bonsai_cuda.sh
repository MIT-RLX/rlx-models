#!/usr/bin/env bash
# Run Bonsai-27B Q1_0 on MSI CUDA (16GB). Sync trees first, then build+run.
#   scripts/matrix/msi_bonsai_cuda.sh
#
# Uses rlx-qwen35 --fast (ChatML + no-think + tight prefill_seq).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${MSI_HOST:-msi}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"
GGUF="${BONSAI_GGUF:-weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf}"
PROMPT="${BONSAI_PROMPT:-What is the capital of France?}"
MAX_TOKENS="${BONSAI_MAX_TOKENS:-16}"

bash "$HERE/sync_to_msi.sh"

echo ">> building rlx-qwen35 (cuda) on $HOST"
ssh "$HOST" bash -s <<EOF
set -euo pipefail
cd $REMOTE_MODELS
export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CARGO_BUILD_JOBS=\${CARGO_BUILD_JOBS:-8}
cargo build --release -p rlx-qwen35 --features cuda \
  2>&1 | tee /tmp/bonsai_cuda_build.log
EOF

echo ">> running Bonsai-27B CUDA packed Q1_0 (--fast)"
ssh "$HOST" bash -s <<EOF
set -euo pipefail
cd $REMOTE_MODELS
export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export RLX_LOW_MEM_COMPILE=1
export RLX_DEQUANT_CACHE=0
export RLX_CUDA_NO_CUDNN=1
export RLX_QWEN35_BENCH=1
export RLX_CUDA_ARENA_DEBUG=\${RLX_CUDA_ARENA_DEBUG:-0}
export RLX_CUDA_MATMUL_PRECISE=\${RLX_CUDA_MATMUL_PRECISE:-1}
./target/release/rlx-qwen35 --weights '$GGUF' --packed --device cuda \
  --fast --max-tokens $MAX_TOKENS --temperature 0.0 --seed 0 \
  --prompt '$PROMPT' 2>&1 | tee /tmp/bonsai_cuda_run.log
EOF

echo ">> logs on msi: /tmp/bonsai_cuda_build.log /tmp/bonsai_cuda_run.log"
