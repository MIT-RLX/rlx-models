# Qwen3 checkpoint fetcher

A throwaway Docker image whose sole job is to pull a Qwen3 (or any
Hugging Face) checkpoint into a host-mounted volume so the
`rlx-models` test suite and benchmarks can use it **natively** — no
need to install Python or `huggingface-cli` on the host.

Why a container for downloads only? Because the actual benchmarks
should run on bare metal — on macOS, Docker runs in a Linux VM
(qemu/HVF) and adds 20–40% CPU overhead, exactly wrong for measuring
inference performance. So:

- **Container**: fetches files only. One-shot, then exits.
- **Host**: runs `cargo test`, `cargo bench`, the example binary.

## Build

From the **rlx-models** repo root:

```bash
docker build -t rlx-qwen3-fetch docker/qwen3-fetch
```

## Usage

```bash
# From rlx-models (or any checkout); weights land in ./weights
mkdir -p weights

# Default: Qwen/Qwen3-0.6B (smallest open Qwen3).
docker run --rm -v "$PWD/weights:/weights" rlx-qwen3-fetch

# Specific model:
docker run --rm -v "$PWD/weights:/weights" rlx-qwen3-fetch Qwen/Qwen3-1.7B

# Gated repo (needs https://huggingface.co/settings/tokens — read-only is fine):
docker run --rm \
    -v "$PWD/weights:/weights" \
    -e HF_TOKEN="$HF_TOKEN" \
    rlx-qwen3-fetch Qwen/Qwen3-32B
```

After it finishes:

```
weights/
└── Qwen3-0.6B/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── …
```

## Run things against the fetched checkpoint (native)

Run `cargo` from the sibling **`rlx`** workspace (`cd ../rlx`).

**Numerical parity test vs candle-transformers:**

```bash
cd ../rlx
RLX_QWEN3_WEIGHTS="$PWD/../rlx-models/weights/Qwen3-0.6B/model.safetensors" \
RLX_QWEN3_CONFIG="$PWD/../rlx-models/weights/Qwen3-0.6B/config.json" \
cargo test -p rlx-models --features parity-candle --release qwen3_parity_vs_candle
```

**End-to-end demo (prefill + generate via cached KV path):**

```bash
cargo run -p rlx-models --release --example qwen3_demo -- \
    --config "$PWD/weights/Qwen3-0.6B/config.json" \
    --weights "$PWD/weights/Qwen3-0.6B/model.safetensors" \
    --prompt-ids 1,2,3,4 \
    --new-tokens 32
```

**Benchmarks:**

```bash
cargo bench -p rlx-models --bench qwen3_inference
```

(The bench uses a synthetic config by default — it's measuring kernel
+ framework throughput, not weight quality. Real-weight benches require
wiring the checkpoint path through the bench harness, which is a small
follow-up if you want it.)

## Tuning what gets downloaded

The entrypoint honours two env vars:

- `ALLOW_PATTERNS` — comma-separated include globs.
  Default: `*.safetensors,*.safetensors.index.json,config.json,tokenizer*,*.json,*.txt,*.model`
- `IGNORE_PATTERNS` — comma-separated exclude globs.
  Default: `*.bin,*.gguf,*.pt,*.h5,*.msgpack`

E.g., to grab the GGUF instead of safetensors:

```bash
docker run --rm \
    -v "$PWD/weights:/weights" \
    -e ALLOW_PATTERNS="*.gguf,config.json,tokenizer*" \
    -e IGNORE_PATTERNS="" \
    rlx-qwen3-fetch Qwen/Qwen3-0.6B-GGUF
```

## Why this image is small

- Base: `python:3.11-slim` (~150 MB)
- Adds: `huggingface_hub[cli,hf_transfer]` only
- `hf_transfer` enables the Rust-backed parallel downloader → 3–5× faster on multi-GB checkpoints.
- Total: ~300 MB image, builds in <30 s on a warm pip cache.
