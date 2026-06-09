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
# Hermetic PyTorch reference container for SAM3 parity testing.

ARG BASE=python:3.12-slim
ARG INSTALL_TORCH=1
# Default to PyPI so arm64 (Apple Silicon under Docker) gets a working wheel.
# The PyTorch /whl/cpu index only ships working wheels for x86_64; arm64
# wheels there were built without the ISA features Apple's virtio CPU
# exposes and crash on import with SIGBUS. Override TORCH_INDEX explicitly
# when building x86 GPU/CPU variants.
ARG TORCH_INDEX=
ARG TORCH_VERSION=

FROM ${BASE}
ARG INSTALL_TORCH
ARG TORCH_INDEX
ARG TORCH_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        TORCH_SPEC="torch${TORCH_VERSION:+==$TORCH_VERSION}"; \
        if [ -n "$TORCH_INDEX" ]; then \
            pip install --no-cache-dir --index-url "$TORCH_INDEX" "$TORCH_SPEC" "torchvision" ; \
        else \
            pip install --no-cache-dir "$TORCH_SPEC" "torchvision" ; \
        fi ; \
    fi && \
    pip install --no-cache-dir \
        "git+https://github.com/facebookresearch/sam3.git" \
        "numpy>=1.26,<2" \
        "safetensors>=0.4" \
        "huggingface_hub[cli]" \
        "pillow" \
        "einops" \
        "psutil" \
        "pycocotools"

COPY dump_reference.py /opt/rlx-sam3/dump_reference.py
WORKDIR /opt/rlx-sam3

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/opt/rlx-sam3/dump_reference.py"]
