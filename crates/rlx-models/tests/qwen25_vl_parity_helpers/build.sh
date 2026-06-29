#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Build Qwen2.5-VL HF reference Docker images.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TAG_CPU=${RLX_QWEN25_VL_IMAGE_TAG:-rlx-qwen25-vl-ref:cpu}

docker build -f "$ROOT/Dockerfile" -t "$TAG_CPU" "$ROOT"

if [[ "${BUILD_GPU:-0}" == "1" ]]; then
    docker build -f "$ROOT/Dockerfile" \
        --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
        -t rlx-qwen25-vl-ref:gpu \
        "$ROOT"
fi

echo "built $TAG_CPU"
