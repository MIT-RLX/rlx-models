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

cd "$(dirname "$0")"

target=${1:-cpu}

build_cpu() {
    echo "[+] building rlx-sam3-ref:cpu"
    docker build \
        -t rlx-sam3-ref:cpu \
        -f Dockerfile \
        .
}

build_gpu() {
    echo "[+] building rlx-sam3-ref:gpu"
    docker build \
        --build-arg BASE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
        --build-arg INSTALL_TORCH=0 \
        -t rlx-sam3-ref:gpu \
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

echo "[+] done. example:"
echo "    RLX_SAM3_DOCKER=1 RLX_SAM3_WEIGHTS=/abs/path/sam3.safetensors \\"
echo "      cargo test -p rlx-models --features parity-pytorch --release \\"
echo "      sam3_patch_embed_parity_vs_pytorch -- --nocapture"
