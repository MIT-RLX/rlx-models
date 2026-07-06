# rlx-minicpm5

[MiniCPM5](https://huggingface.co/openbmb/MiniCPM5-1B) edge LMs in RLX. The 1B checkpoint is **Llama-shaped** (`LlamaForCausalLM`, `general.architecture = llama` in GGUF). This crate validates HF/GGUF metadata and delegates inference to [`rlx-llama32`](../rlx-llama32).

**Crate version:** `0.2.11` (depends on `rlx-llama32` 0.2.11; workspace pins upstream `rlx*` 0.2.11). Publish via `scripts/publish.sh` tier 5 before facade `rlx-models` 0.2.11.

## Prerequisites

From the **repo root** (`rlx-models/`):

```sh
# Optional but recommended
brew install just

# Build the CLI (tokenizer feature required for decode)
cargo build -p rlx-minicpm5 --features tokenizer --release
```

GPU backends: add feature flags (`metal`, `mlx`, `cuda`, `all-backends`, …) on `rlx-minicpm5` or use `just features=all-backends minicpm5 -- …`.

## Download weights

**Safetensors** (~2.1 GB, reference parity path):

```sh
just fetch-minicpm5
# → /tmp/rlx-weights/MiniCPM5-1B/  (override with MINICPM5_MODEL_DIR=…)
```

**GGUF** ([openbmb/MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)):

```sh
just fetch-minicpm5-gguf Q4_K_M    # or Q8_0, F16, or `all`
# → /tmp/rlx-weights/MiniCPM5-1B-GGUF/
```

Requires the `hf-download` feature (`just fetch-*` enables it via the facade example).

## CLI (`rlx-minicpm5`)

Same flags as `rlx-llama32` (this binary wraps `rlx_llama32::cli::run` after weight-kind checks).

### Prefill / logits (prompt token ids)

```sh
WEIGHTS=/tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors

just minicpm5 -- \
  --weights "$WEIGHTS" \
  --device cpu \
  --prompt-ids 1,42,314 \
  --max-tokens 0

# Or per-crate binary:
cargo run -p rlx-minicpm5 --features tokenizer --release -- \
  --weights "$WEIGHTS" \
  --device cpu \
  --prompt-ids 1,42,314 \
  --max-tokens 16 \
  --max-seq 512
```

### Greedy decode with HF tokenizer file

```sh
just minicpm5 -- \
  --weights "$WEIGHTS" \
  --tokenizer /tmp/rlx-weights/MiniCPM5-1B/tokenizer.json \
  --device cpu \
  --prompt "What is 2+2? Answer briefly." \
  --max-tokens 32 \
  --no-stream
```

Decoded text is printed after `response:` when `--tokenizer` is set.

### GGUF packed prefill (Q4_K_M / Q8_0 / F16)

```sh
GGUF=/tmp/rlx-weights/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf

just minicpm5 -- \
  --weights "$GGUF" \
  --packed \
  --device cpu \
  --prompt-ids 1,42,314 \
  --max-tokens 8
```

Packed graphs use `Op::DequantMatMul`. **CPU**, **Metal**, and **MLX** run natively on the requested device (Metal uses an MPSGraph workaround; MLX uses host GGUF dequant with lazy eval). **wgpu / CUDA / ROCm** packed prefill still **executes on CPU** until upstream GPU parity lands (see `rlx_core::flow_bridge::packed_gguf_execution_device`). Decode still uses the F32 generator on your `--device`.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--device cpu\|metal\|mlx\|cuda\|…` | Execution device |
| `--packed` | GGUF K-quant / F16 packed matmul path |
| `--max-seq N` | Compile / KV cap (default 512; lower for faster compile on long contexts) |
| `--max-tokens N` | Greedy decode steps after prefill |
| `--no-bucketed-decode` | One-shot decode graphs (try on MLX if output diverges) |
| `--no-stream` | Print full `response:` once at end |
| `--temperature`, `--top-p` | Sampling (default greedy) |

## Chat (HF chat template)

Applies the official MiniCPM5 chat template in Python, then calls `rlx-minicpm5`:

```sh
pip install transformers
just fetch-minicpm5
just minicpm5-chat "What is the capital of France? Reply in one short sentence."
```

Override paths / device:

```sh
MINICPM5_MODEL_DIR=/path/to/MiniCPM5-1B \
RLX_MINICPM5_DEVICE=cpu \
  just minicpm5-chat "Hello" --max-tokens 64
```

Script: [`crates/rlx-models/examples/minicpm5_chat.py`](../rlx-models/examples/minicpm5_chat.py).

## Library API

```rust
use rlx_minicpm5::MiniCpm5Runner;
use rlx_runtime::Device;

let weights = "/tmp/rlx-weights/MiniCPM5-1B/model-00000-of-00001.safetensors";
let mut runner = MiniCpm5Runner::builder()
    .weights(weights)
    .device(Device::Cpu)
    .max_seq(512)
    .build()?;

let prompt = [1u32, 42, 314];
let logits = runner.predict_logits(&prompt)?;
let generated = runner.generate(&prompt, 16, |tok| eprint!(" {tok}"))?;
```

Facade path: `rlx_models::minicpm5::…`.

Full programmatic example: `cargo run -p rlx-models --example run_minicpm5 --release` (set `RLX_MINICPM5_WEIGHTS`).

## Examples and tests (facade `rlx-models`)

| Command / file | What it does |
|----------------|--------------|
| `just fetch-minicpm5` | `examples/minicpm5_download.rs` |
| `just fetch-minicpm5-gguf Q4_K_M` | `examples/minicpm5_gguf_download.rs` |
| `just bench-minicpm5-real --device cpu` | `examples/minicpm5_forward_bench.rs` |
| `just test-minicpm5-parity-full` | RLX vs PyTorch (`minicpm5_parity`) |
| `just test-minicpm5-backends-all` | Synthetic graph, all backends |
| `just test-minicpm5-gguf-backends` | Real Q4_K_M packed prefill |
| `cargo run -p rlx-models --example run_minicpm5 --release` | Builder API walk-through |

## Weights on Hugging Face

| Artifact | Hub |
|----------|-----|
| Safetensors 1B | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) |
| GGUF Q4_K_M / Q8_0 / F16 | [openbmb/MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) |

## Remote CUDA rig (Windows + WSL)

From a Mac/Linux dev machine with SSH to the rig ([`rig.sh`](../../../rlx/rig.sh) in a sibling `rlx/` checkout):

```sh
cd ../rlx
./rig.sh sync-rlx-models              # source → H:/rlx-workspace/rlx-models
./rig.sh sync-rlx-models-weights    # weights → H:/rlx-weights (rig-side download)
./rig.sh test-rlx-models            # CPU + CUDA + WGPU backend matrix (Windows)
./rig.sh sync-minicpm5-gguf         # MiniCPM5 Q4_K_M only (H:/rlx-weights)
./rig.sh test-minicpm5              # MiniCPM5-only subset
```

Everything lives on **H:** (`H:/rlx-workspace`, `H:/rlx-weights`, `H:/rlx-cargo-target`). Weights are downloaded on the rig to **H:/rlx-weights** only — never fetched or scp'd from the Mac.

## See also

- Main repo [README](../../README.md#minicpm5)
- [AGENTS.md](../../AGENTS.md) — `just` recipes and CI commands
