#!/usr/bin/env bash
# One command from the Mac: sync -> run the harness on the CUDA host -> pull the report back.
#   scripts/matrix/remote_run.sh [TIER] [ONLY] [BACKENDS] [ALL]
# e.g. scripts/matrix/remote_run.sh 1 qwen3-0.6b "" 0
set -euo pipefail

TIER="${1:-1}"; ONLY="${2:-}"; BACKENDS="${3:-}"; ALL="${4:-0}"
HOST="${RLX_CUDA_HOST:?set RLX_CUDA_HOST to your CUDA host, e.g. user@host}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/sync_to_remote.sh"

# Non-interactive ssh has no cargo/cuda on PATH — export them inline.
REMOTE_ENV="PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH \
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
TIER='$TIER' ONLY='$ONLY' BACKENDS='$BACKENDS' ALL='$ALL' \
BUILD_TIMEOUT='${BUILD_TIMEOUT:-3600}'"

echo ">> running harness on $HOST (TIER=$TIER ONLY=${ONLY:-*} BACKENDS=${BACKENDS:-auto} ALL=$ALL)"
ssh "$HOST" "cd $REMOTE_MODELS && $REMOTE_ENV python3 scripts/matrix/run_matrix.py"

echo ">> pulling report back"
mkdir -p "$HERE/out"
rsync -az "$HOST:$REMOTE_MODELS/scripts/matrix/out/" "$HERE/out/"
echo ">> report: $HERE/out/report.md"
sed -n '1,80p' "$HERE/out/report.md" 2>/dev/null || true
