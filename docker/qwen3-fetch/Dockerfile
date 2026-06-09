# RLX — Qwen3 checkpoint fetcher.
#
# A tiny image whose sole job is to download a Qwen3 (or any HF Hub)
# checkpoint into a host-mounted volume so the rlx-models binary
# tests and benchmarks can use it natively — without polluting the
# host's Python environment.
#
# Build:
#   docker build -t rlx-qwen3-fetch docker/qwen3-fetch
#
# Run (downloads Qwen3-0.6B to ./weights/):
#   mkdir -p weights
#   docker run --rm \
#     -v "$PWD/weights:/weights" \
#     -e HF_TOKEN="$HF_TOKEN" \
#     rlx-qwen3-fetch Qwen/Qwen3-0.6B
#
# The HF_TOKEN env var is only required for gated repos. Set it from
# https://huggingface.co/settings/tokens (read-only is fine).

FROM python:3.11-slim

# pip's resolver is the slow part; pinned deps speed up rebuilds and
# make the image reproducible.
RUN pip install --no-cache-dir \
        "huggingface_hub[cli,hf_transfer]==0.27.1" \
    && rm -rf /root/.cache/pip

# hf_transfer enables Rust-backed parallel downloads (~3-5× faster on
# multi-GB checkpoints). Safe to leave on for small files too.
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Target directory inside the container. Bind-mount a host dir here.
WORKDIR /weights
VOLUME ["/weights"]

# Default: fetch model.safetensors + config.json + tokenizer files.
# Override patterns via the IGNORE_PATTERNS / ALLOW_PATTERNS env vars
# at run time. See entrypoint.sh.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["Qwen/Qwen3-0.6B"]
