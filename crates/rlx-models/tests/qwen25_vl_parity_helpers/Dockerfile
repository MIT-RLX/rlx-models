# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Hermetic HuggingFace Qwen2.5-VL reference container for Rust parity tests.

ARG BASE=python:3.12-slim
ARG INSTALL_TORCH=1
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
            pip install --no-cache-dir --index-url "$TORCH_INDEX" "$TORCH_SPEC" "torchvision"; \
        else \
            pip install --no-cache-dir "$TORCH_SPEC" "torchvision"; \
        fi ; \
    fi && \
    pip install --no-cache-dir \
        "transformers>=4.49" \
        "accelerate" \
        "numpy>=1.26,<2" \
        "safetensors>=0.4" \
        "huggingface_hub[cli]" \
        "pillow" \
        "qwen-vl-utils"

COPY dump_reference.py aif_probe.py /opt/rlx-qwen25-vl/
WORKDIR /opt/rlx-qwen25-vl

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/opt/rlx-qwen25-vl/dump_reference.py"]
