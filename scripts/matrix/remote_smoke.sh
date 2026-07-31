#!/usr/bin/env bash
# One command from the Mac: sync both working trees to a remote host and run the
# cross-backend smoke sweep there (rlx-kimi-k3 + the other portable crates).
#
#   RLX_REMOTE_HOST=msi scripts/matrix/remote_smoke.sh cpu gpu cuda vulkan
#   RLX_REMOTE_HOST=amd scripts/matrix/remote_smoke.sh cpu gpu rocm vulkan
#
# The host decides which backends it has; you pass the device list. Non-interactive
# ssh has no cargo/cuda/rocm on PATH, so we export them inline.
set -euo pipefail

HOST="${RLX_REMOTE_HOST:-${RLX_CUDA_HOST:?set RLX_REMOTE_HOST to your remote, e.g. msi or amd}}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"
DEVICES="${*:-cpu}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RLX_REMOTE_HOST="$HOST" bash "$HERE/sync_to_remote.sh"

# CUDA and ROCm live in the usual /usr/local prefixes; harmless if a path is absent.
REMOTE_ENV="PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:/opt/rocm/bin:\$PATH \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib"

echo ">> running backend_smoke on $HOST  (devices: $DEVICES)"
ssh "$HOST" "cd $REMOTE_MODELS && $REMOTE_ENV bash scripts/backend_smoke.sh $DEVICES"
