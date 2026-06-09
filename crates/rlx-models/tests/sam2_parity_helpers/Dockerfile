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
# Hermetic pytorch reference container for SAM 2 parity testing.
#
# Two build targets, picked via the BASE build-arg + a TORCH_INDEX
# pip flag:
#
#   # CPU-only (default — ~1.6 GB, builds anywhere, runs without nvidia-docker).
#   # Uses the slim python base + pytorch's CPU wheel index.
#   docker build -t rlx-sam2-ref:cpu \
#       -f tests/sam2_parity_helpers/Dockerfile \
#       tests/sam2_parity_helpers
#
#   # GPU (CUDA 12.1, requires NVIDIA Container Toolkit on the host).
#   # Uses the official pytorch/pytorch CUDA image (torch already installed).
#   docker build -t rlx-sam2-ref:gpu \
#       --build-arg BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
#       --build-arg INSTALL_TORCH=0 \
#       -f tests/sam2_parity_helpers/Dockerfile \
#       tests/sam2_parity_helpers
#
# Run via the wrapper at run-ref.sh (handles --gpus passthrough,
# weight mounting, env-var forwarding).
ARG BASE=python:3.11-slim
ARG INSTALL_TORCH=1
# CPU wheel index — overridden to "" by GPU build (torch already in base).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM ${BASE}
ARG INSTALL_TORCH
ARG TORCH_INDEX

# `sam2` pulls in opencv-python (libGL.so.1) and a few other shared
# libs; install them at the system level so the import path is clean.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pin sam2 + numpy + safetensors so parity numbers stay stable on
# container rebuild. torch is conditionally installed: the pytorch
# CUDA base image already ships torch, so INSTALL_TORCH=0 skips the
# duplicate (and avoids replacing CUDA wheels with CPU ones).
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install --no-cache-dir --index-url "$TORCH_INDEX" \
            "torch==2.4.0" ; \
    fi && \
    pip install --no-cache-dir \
        "sam2==1.1.0" \
        "numpy>=1.26,<3" \
        "safetensors>=0.4"

# The reference dumper. Mounted-in at /work in run-ref.sh, but baked
# here too so the image is self-contained for CI runners that pull
# without a workspace mount.
COPY dump_reference.py /opt/rlx-sam2/dump_reference.py
WORKDIR /opt/rlx-sam2

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/opt/rlx-sam2/dump_reference.py"]
