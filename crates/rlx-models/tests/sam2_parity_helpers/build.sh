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
# Build the rlx-sam2-ref Docker image(s) used by the SAM 2 parity
# harness. Run once per machine; rebuild only when bumping sam2 /
# torch versions in the Dockerfile.
#
# Usage:
#   ./build.sh cpu       # ~1.6 GB, CPU-only (default)
#   ./build.sh gpu       # ~5 GB, requires NVIDIA Container Toolkit
#   ./build.sh both      # build both tags

set -euo pipefail

cd "$(dirname "$0")"

target=${1:-cpu}

build_cpu() {
    echo "[+] building rlx-sam2-ref:cpu (python:3.11-slim + torch CPU wheel)"
    docker build \
        -t rlx-sam2-ref:cpu \
        -f Dockerfile \
        .
}

build_gpu() {
    echo "[+] building rlx-sam2-ref:gpu (CUDA 12.1 base, torch already in image)"
    docker build \
        --build-arg BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
        --build-arg INSTALL_TORCH=0 \
        -t rlx-sam2-ref:gpu \
        -f Dockerfile \
        .
}

case "$target" in
    cpu)  build_cpu ;;
    gpu)  build_gpu ;;
    both) build_cpu; build_gpu ;;
    *)
        echo "usage: $0 [cpu|gpu|both]" >&2
        exit 2
        ;;
esac

echo "[+] done. example invocation:"
echo "    RLX_SAM2_DOCKER=1 \\"
echo "    RLX_SAM2_DEVICE=${target} \\"
echo "    RLX_SAM2_WEIGHTS=/abs/path/sam2_hiera_base_plus.safetensors \\"
echo "    RLX_SAM2_CONFIG=sam2_hiera_b+ \\"
echo "    cargo test -p rlx-models --features parity-pytorch --release \\"
echo "      sam2_encoder_parity_vs_pytorch -- --nocapture"
