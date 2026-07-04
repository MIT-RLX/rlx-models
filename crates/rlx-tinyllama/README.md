# rlx-tinyllama

[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) in RLX. The 1.1B chat checkpoint is **Llama-shaped** (`LlamaForCausalLM`, `general.architecture = llama` in GGUF). This crate validates HF/GGUF metadata (including 1.1B dims: 2048 hidden, 22 layers) and delegates inference to [`rlx-llama32`](../rlx-llama32).

## Transformers-style quickstart (`pipeline` feature)

Text in, text out — the same shape as `huggingface/transformers`, with auto-download, a cached tokenizer, chat templates, EOS-aware stopping, and streaming all handled for you:

```rust
use rlx_tinyllama::pipeline::{TextGeneration, GenerationConfig, ChatMessage};

// Downloads + caches on first use; also accepts a local dir / .safetensors / .gguf.
let mut pipe = TextGeneration::from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")?;

// Raw completion.
let out = pipe.generate("Once upon a time", &GenerationConfig::default())?;

// Chat (applies the model's chat template).
let reply = pipe.chat(&[ChatMessage::user("Name three primary colors.")],
                      &GenerationConfig::default())?;

// Streaming.
pipe.chat_stream(&[ChatMessage::user("Tell me a joke.")],
                 &GenerationConfig::default().with_max_new_tokens(64),
                 |piece| print!("{piece}"))?;
# anyhow::Ok(())
```

The `transformers` equivalent:

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(pipe("Once upon a time", max_new_tokens=64)[0]["generated_text"])
```

Run the bundled example or the one-liner CLI:

```sh
cargo run -p rlx-tinyllama --example pipeline --features pipeline --release

cargo run -p rlx-tinyllama --bin rlx-tinyllama-pipeline --features pipeline --release -- \
  --prompt "What is the capital of France?"
# local checkpoint + GPU:
cargo run -p rlx-tinyllama --bin rlx-tinyllama-pipeline --features pipeline,metal --release -- \
  --model /tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0 --device metal --prompt "Hi!"
```

`GenerationConfig` maps onto RLX's sampler (`max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `skip_special_tokens`). The lower-level [`TinyLlamaRunner`](src/lib.rs) (token-ids in/out) remains available for advanced control.

## Prerequisites

From the **repo root** (`rlx-models/`):

```sh
brew install just   # optional
cargo build -p rlx-tinyllama --features tokenizer --release
```

GPU backends: add feature flags (`metal`, `mlx`, `cuda`, `all-backends`, …) on `rlx-tinyllama` or use `just features=all-backends tinyllama -- …`.

## Download weights

**Safetensors** (~2.2 GB, 3 shards):

```sh
just fetch-tinyllama
# → /tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0/
```

**GGUF** ([TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)):

```sh
just fetch-tinyllama-gguf Q4_K_M    # or Q8_0, Q6_K, or `all`
# → /tmp/rlx-weights/TinyLlama-1.1B-GGUF/
```

Requires the `hf-download` feature (`just fetch-*` enables it via the facade example).

Manual curl (when Hub API is unavailable):

```sh
mkdir -p /tmp/rlx-weights/TinyLlama-1.1B-GGUF
curl -L -o /tmp/rlx-weights/TinyLlama-1.1B-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
```

## CLI (`rlx-tinyllama`)

Same flags as `rlx-llama32` (this binary wraps `rlx_llama32::cli::run` after weight-kind checks).

### Prefill / logits

```sh
WEIGHTS=/tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0/model-00001-of-00003.safetensors

just tinyllama -- \
  --weights "$WEIGHTS" \
  --device cpu \
  --prompt-ids 1,42,314 \
  --max-tokens 16 \
  --max-seq 512
```

### Greedy decode with HF tokenizer

Provide `tokenizer.json` beside the weights — GGUF-only SentencePiece metadata is not enough for text decode.

```sh
just tinyllama -- \
  --weights "$WEIGHTS" \
  --tokenizer /tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0/tokenizer.json \
  --device cpu \
  --prompt "What is 2+2? Answer briefly." \
  --max-tokens 32 \
  --no-stream
```

### GGUF packed prefill

```sh
GGUF=/tmp/rlx-weights/TinyLlama-1.1B-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

just tinyllama -- \
  --weights "$GGUF" \
  --packed \
  --device metal \
  --prompt-ids 1,42,314 \
  --max-tokens 8
```

Packed graphs use `Op::DequantMatMul`. **CPU**, **Metal**, and **MLX** run natively on the requested device. **CUDA** packed prefill executes on CPU until upstream GPU parity lands; **wgpu** may hit buffer-size limits on some GPUs.

## Library API

```rust
use rlx_tinyllama::TinyLlamaRunner;
use rlx_runtime::Device;

let weights = "/tmp/rlx-weights/TinyLlama-1.1B-Chat-v1.0/model-00001-of-00003.safetensors";
let mut runner = TinyLlamaRunner::builder()
    .weights(weights)
    .device(Device::Cpu)
    .max_seq(512)
    .build()?;

let prompt = [1u32, 42, 314];
let logits = runner.predict_logits(&prompt)?;
let generated = runner.generate(&prompt, 16, |tok| eprint!(" {tok}"))?;
```

Facade path: `rlx_models::tinyllama::…`.

## Tests (facade `rlx-models`)

| Command | What it does |
|---------|----------------|
| `just fetch-tinyllama-gguf Q4_K_M` | Download Q4_K_M GGUF |
| `just test-tinyllama-gguf-backends` | Real Q4_K_M packed prefill vs CPU (all backends) |
| `just test-tinyllama-backends-all` | Synthetic 1.1B-shaped graph, all backends |
| `just test-tinyllama-real` | Safetensors config + runner build (needs `fetch-tinyllama`) |

## Weights on Hugging Face

| Artifact | Hub |
|----------|-----|
| Safetensors 1.1B chat | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| GGUF Q4_K_M / Q8_0 / Q6_K | [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |

## See also

- Main repo [README](../../README.md#tinyllama)
- [AGENTS.md](../../AGENTS.md) — `just` recipes and CI commands
