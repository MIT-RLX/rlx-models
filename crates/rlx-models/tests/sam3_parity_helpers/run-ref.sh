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
set -euo pipefail

require() {
    local name=$1
    if [[ -z "${!name-}" ]]; then
        echo "missing env var: $name" >&2
        exit 2
    fi
}

require RLX_SAM3_IMAGE_BIN
require RLX_SAM3_OUT_DIR

if [[ -z "${RLX_SAM3_WEIGHTS-}" && "${RLX_SAM3_DOWNLOAD:-0}" != "1" ]]; then
    echo "set RLX_SAM3_WEIGHTS or RLX_SAM3_DOWNLOAD=1" >&2
    exit 2
fi

DEVICE=${RLX_SAM3_DEVICE:-cpu}
case "$DEVICE" in
    cpu)  DEFAULT_TAG=rlx-sam3-ref:cpu ;;
    cuda) DEFAULT_TAG=rlx-sam3-ref:gpu ;;
    *)
        echo "RLX_SAM3_DEVICE must be 'cpu' or 'cuda' (got '$DEVICE')" >&2
        exit 2
        ;;
esac
TAG=${RLX_SAM3_IMAGE_TAG:-$DEFAULT_TAG}

IMAGE_HOST=$(realpath "$RLX_SAM3_IMAGE_BIN")
OUT_HOST=$(realpath "$RLX_SAM3_OUT_DIR")
mkdir -p "$OUT_HOST"
IMAGE_NAME=$(basename "$IMAGE_HOST")

mounts=(
    -v "$IMAGE_HOST:/mnt/in/$IMAGE_NAME:ro"
    -v "$OUT_HOST:/mnt/out"
)

env_args=(
    -e "RLX_SAM3_IMAGE_BIN=/mnt/in/$IMAGE_NAME"
    -e "RLX_SAM3_OUT_DIR=/mnt/out"
    -e "RLX_SAM3_DEVICE=$DEVICE"
)

if [[ -n "${RLX_SAM3_WEIGHTS-}" ]]; then
    WEIGHTS_HOST=$(realpath "$RLX_SAM3_WEIGHTS")
    WEIGHTS_NAME=$(basename "$WEIGHTS_HOST")
    mounts+=(-v "$WEIGHTS_HOST:/mnt/weights/$WEIGHTS_NAME:ro")
    env_args+=(-e "RLX_SAM3_WEIGHTS=/mnt/weights/$WEIGHTS_NAME")
fi

if [[ "${RLX_SAM3_DOWNLOAD:-0}" == "1" ]]; then
    env_args+=(-e "RLX_SAM3_DOWNLOAD=1")
    if [[ -n "${HF_TOKEN-}" ]]; then
        env_args+=(-e "HF_TOKEN=$HF_TOKEN")
    fi
    if [[ -d "${HF_HOME:-}" ]]; then
        HF_HOME_HOST=$(realpath "$HF_HOME")
        mounts+=(-v "$HF_HOME_HOST:/mnt/hf")
        env_args+=(-e "HF_HOME=/mnt/hf")
    fi
fi

for name in RLX_SAM3_TEXT_PROMPT RLX_SAM3_RUN_IMAGE RLX_SAM3_RUN_VIDEO RLX_SAM3_HF_DIR; do
    if [[ -n "${!name-}" ]]; then
        env_args+=(-e "$name=${!name}")
    fi
done

SCRIPT_HOST="$(realpath "$(dirname "$0")")/dump_reference.py"
if [[ -f "$SCRIPT_HOST" ]]; then
    mounts+=(-v "$SCRIPT_HOST:/opt/rlx-sam3/dump_reference.py:ro")
fi

gpu_args=()
if [[ "$DEVICE" == "cuda" ]]; then
    gpu_args=(--gpus all)
fi

exec docker run --rm ${gpu_args[@]+"${gpu_args[@]}"} "${mounts[@]}" "${env_args[@]}" "$TAG"
