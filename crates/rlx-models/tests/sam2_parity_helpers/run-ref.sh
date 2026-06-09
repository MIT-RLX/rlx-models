#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#
#
# Run the SAM 2 reference dumper inside the rlx-sam2-ref Docker image.
# Called by `tests/sam2_parity.rs` when `RLX_SAM2_DOCKER=1` is set, or
# directly by humans for ad-hoc debugging.
#
# Required env vars (forwarded to the container):
#   RLX_SAM2_WEIGHTS      host path to sam2_hiera_*.safetensors
#   RLX_SAM2_CONFIG       e.g. "sam2_hiera_b+"
#   RLX_SAM2_IMAGE_BIN    host path to the preprocessed f32 image blob
#   RLX_SAM2_OUT_DIR      host directory to receive .f32 outputs
#
# Optional:
#   RLX_SAM2_DEVICE       "cpu" (default) or "cuda"
#   RLX_SAM2_IMAGE_TAG    docker image tag (default: rlx-sam2-ref:cpu
#                         when DEVICE=cpu, rlx-sam2-ref:gpu when cuda)
#   RLX_SAM2_RUN_DECODER  "1" to also dump decoder outputs
#   RLX_SAM2_POINTS       host path to point coords f32 [N,2]
#   RLX_SAM2_LABELS       host path to point labels f32 [N]

set -euo pipefail

require() {
    local name=$1
    if [[ -z "${!name-}" ]]; then
        echo "missing env var: $name" >&2
        exit 2
    fi
}

require RLX_SAM2_WEIGHTS
require RLX_SAM2_CONFIG
require RLX_SAM2_IMAGE_BIN
require RLX_SAM2_OUT_DIR

DEVICE=${RLX_SAM2_DEVICE:-cpu}
case "$DEVICE" in
    cpu)  DEFAULT_TAG=rlx-sam2-ref:cpu  ;;
    cuda) DEFAULT_TAG=rlx-sam2-ref:gpu  ;;
    *)
        echo "RLX_SAM2_DEVICE must be 'cpu' or 'cuda' (got '$DEVICE')" >&2
        exit 2
        ;;
esac
TAG=${RLX_SAM2_IMAGE_TAG:-$DEFAULT_TAG}

# Mount each host file/dir into a stable in-container path. The
# container reads env vars to find the paths, so we rewrite them.
WEIGHTS_HOST=$(realpath "$RLX_SAM2_WEIGHTS")
IMAGE_HOST=$(realpath "$RLX_SAM2_IMAGE_BIN")
OUT_HOST=$(realpath "$RLX_SAM2_OUT_DIR")
mkdir -p "$OUT_HOST"

WEIGHTS_NAME=$(basename "$WEIGHTS_HOST")
IMAGE_NAME=$(basename "$IMAGE_HOST")

mounts=(
    -v "$WEIGHTS_HOST:/mnt/weights/$WEIGHTS_NAME:ro"
    -v "$IMAGE_HOST:/mnt/in/$IMAGE_NAME:ro"
    -v "$OUT_HOST:/mnt/out"
)

# Mount the host's dump_reference.py over the baked-in one. Lets us
# iterate on the script without rebuilding the ~1.6 GB image.
SCRIPT_HOST="$(realpath "$(dirname "$0")")/dump_reference.py"
if [[ -f "$SCRIPT_HOST" ]]; then
    mounts+=(-v "$SCRIPT_HOST:/opt/rlx-sam2/dump_reference.py:ro")
fi

env_args=(
    -e "RLX_SAM2_WEIGHTS=/mnt/weights/$WEIGHTS_NAME"
    -e "RLX_SAM2_CONFIG=$RLX_SAM2_CONFIG"
    -e "RLX_SAM2_IMAGE_BIN=/mnt/in/$IMAGE_NAME"
    -e "RLX_SAM2_OUT_DIR=/mnt/out"
    -e "RLX_SAM2_DEVICE=$DEVICE"
)

if [[ "${RLX_SAM2_RUN_DECODER:-0}" == "1" ]]; then
    require RLX_SAM2_POINTS
    require RLX_SAM2_LABELS
    PTS_HOST=$(realpath "$RLX_SAM2_POINTS")
    LBL_HOST=$(realpath "$RLX_SAM2_LABELS")
    PTS_NAME=$(basename "$PTS_HOST")
    LBL_NAME=$(basename "$LBL_HOST")
    mounts+=(
        -v "$PTS_HOST:/mnt/in/$PTS_NAME:ro"
        -v "$LBL_HOST:/mnt/in/$LBL_NAME:ro"
    )
    env_args+=(
        -e "RLX_SAM2_RUN_DECODER=1"
        -e "RLX_SAM2_POINTS=/mnt/in/$PTS_NAME"
        -e "RLX_SAM2_LABELS=/mnt/in/$LBL_NAME"
    )
fi
if [[ "${RLX_SAM2_RUN_MEMORY_ENCODER:-0}" == "1" ]]; then
    env_args+=(-e "RLX_SAM2_RUN_MEMORY_ENCODER=1")
fi
if [[ "${RLX_SAM2_RUN_MEMORY_ATTENTION:-0}" == "1" ]]; then
    env_args+=(-e "RLX_SAM2_RUN_MEMORY_ATTENTION=1")
fi

gpu_args=()
if [[ "$DEVICE" == "cuda" ]]; then
    # Require NVIDIA Container Toolkit. `--gpus all` is the modern
    # invocation; fall back to the runtime= form for older Docker.
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        gpu_args=(--gpus all)
    else
        echo "warning: RLX_SAM2_DEVICE=cuda but nvidia container runtime not detected" >&2
        gpu_args=(--gpus all)
    fi
fi

exec docker run --rm ${gpu_args[@]+"${gpu_args[@]}"} "${mounts[@]}" "${env_args[@]}" "$TAG"
