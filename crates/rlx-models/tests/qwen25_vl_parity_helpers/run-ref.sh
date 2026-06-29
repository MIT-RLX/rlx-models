#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Run HuggingFace Qwen2.5-VL reference dump in Docker.

set -euo pipefail

require() {
    local name=$1
    if [[ -z "${!name-}" ]]; then
        echo "missing env var: $name" >&2
        exit 2
    fi
}

require RLX_QWEN25_VL_IMAGE
require RLX_QWEN25_VL_OUT_DIR

if [[ -z "${RLX_QWEN25_VL_HF_DIR-}" && "${RLX_QWEN25_VL_DOWNLOAD:-0}" != "1" ]]; then
    echo "set RLX_QWEN25_VL_HF_DIR or RLX_QWEN25_VL_DOWNLOAD=1" >&2
    exit 2
fi

DEVICE=${RLX_QWEN25_VL_DEVICE:-cpu}
case "$DEVICE" in
    cpu)  DEFAULT_TAG=rlx-qwen25-vl-ref:cpu ;;
    cuda) DEFAULT_TAG=rlx-qwen25-vl-ref:gpu ;;
    mps)  DEFAULT_TAG=rlx-qwen25-vl-ref:cpu ;;
    *)
        echo "RLX_QWEN25_VL_DEVICE must be 'cpu', 'cuda', or 'mps' (got '$DEVICE')" >&2
        exit 2
        ;;
esac
TAG=${RLX_QWEN25_VL_IMAGE_TAG:-$DEFAULT_TAG}

IMAGE_HOST=$(realpath "$RLX_QWEN25_VL_IMAGE")
OUT_HOST=$(realpath "$RLX_QWEN25_VL_OUT_DIR")
mkdir -p "$OUT_HOST"
IMAGE_NAME=$(basename "$IMAGE_HOST")

mounts=(
    -v "$IMAGE_HOST:/mnt/in/$IMAGE_NAME:ro"
    -v "$OUT_HOST:/mnt/out"
)

env_args=(
    -e "RLX_QWEN25_VL_IMAGE=/mnt/in/$IMAGE_NAME"
    -e "RLX_QWEN25_VL_OUT_DIR=/mnt/out"
    -e "RLX_QWEN25_VL_DEVICE=$DEVICE"
)

if [[ -n "${RLX_QWEN25_VL_HF_DIR-}" ]]; then
    HF_HOST=$(realpath "$RLX_QWEN25_VL_HF_DIR")
    mounts+=(-v "$HF_HOST:/mnt/hf:ro")
    env_args+=(-e "RLX_QWEN25_VL_HF_DIR=/mnt/hf")
fi

if [[ "${RLX_QWEN25_VL_DOWNLOAD:-0}" == "1" ]]; then
    env_args+=(-e "RLX_QWEN25_VL_DOWNLOAD=1")
    if [[ -n "${HF_TOKEN-}" ]]; then
        env_args+=(-e "HF_TOKEN=$HF_TOKEN")
    fi
    if [[ -d "${HF_HOME:-}" ]]; then
        HF_HOME_HOST=$(realpath "$HF_HOME")
        mounts+=(-v "$HF_HOME_HOST:/mnt/hf-cache")
        env_args+=(-e "HF_HOME=/mnt/hf-cache")
    fi
fi

for name in RLX_QWEN25_VL_PROMPT; do
    if [[ -n "${!name-}" ]]; then
        env_args+=(-e "$name=${!name}")
    fi
done

SCRIPT_HOST="$(realpath "$(dirname "$0")")/dump_reference.py"
if [[ -f "$SCRIPT_HOST" ]]; then
    mounts+=(-v "$SCRIPT_HOST:/opt/rlx-qwen25-vl/dump_reference.py:ro")
fi

gpu_args=()
if [[ "$DEVICE" == "cuda" ]]; then
    gpu_args=(--gpus all)
fi

exec docker run --rm ${gpu_args[@]+"${gpu_args[@]}"} "${mounts[@]}" "${env_args[@]}" "$TAG"
