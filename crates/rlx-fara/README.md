# rlx-fara

Microsoft **Fara1.5** computer-use agent (CUA) for RLX — screenshot in, structured `<tool_call>` out.

| Size | Hugging Face | Base |
|------|--------------|------|
| 4B | [microsoft/Fara1.5-4B](https://huggingface.co/microsoft/Fara1.5-4B) | Qwen3.5-4B |
| 9B | [microsoft/Fara1.5-9B](https://huggingface.co/microsoft/Fara1.5-9B) | Qwen3.5-9B |

Inference is [`rlx-qwen35`](../rlx-qwen35) multimodal (hybrid Gated DeltaNet + vision tower). This crate adds Fara system prompt, tool-call parsing, size presets, and a thin CLI. It does **not** sandbox a browser — pair with MagenticLite (or your own loop) for safe execution.

## Quick start

```bash
just fetch-fara-4b
just fara --model-dir .cache/fara/4b --image shot.png \
  --goal "Book a table for 2 at a sushi place in Sunnyvale for Friday 7pm." \
  --device metal --max-tokens 512
```

Or:

```bash
cargo run -p rlx-fara --release --features apple-silicon,hf-download -- \
  --size 4b --download --image shot.png --goal "Open example.com" --device cpu
```

Train-time grounding resolution is typically **1440×900**.

### Memory

HF Fara1.5 ships BF16 safetensors (~8.5 GiB for 4B). The runner expands to host F32 (~17 GiB) plus Metal buffers. Defaults use `skip_warm` and `max_seq=1024` to avoid OOM on 64 GiB machines; still expect a large peak on first multimodal prefill/decode. Quantized GGUF (when available) is the practical low-RAM path.

## API

```rust
use rlx_fara::{FaraRunner, FaraSize};
use rlx_runtime::Device;

let mut runner = FaraRunner::from_model_dir(".cache/fara/4b", FaraSize::B4, Device::Cpu)?;
let step = runner.step("Find the FAQ", &rgb, w, h, 256, None)?;
for call in &step.tool_calls {
    println!("{:?} {}", call.action(), call.arguments);
}
# Ok::<(), anyhow::Error>(())
```

## Features

| Feature | Role |
|---------|------|
| `tokenizer` (default) | HF `tokenizer.json` |
| `qwen35-vlm` (default) | Image load + multimodal prefill |
| `hf-download` | `huggingface_hub` fetch into `.cache/fara/` |
| `metal` / `mlx` / `cuda` / … | Forwarded to `rlx-qwen35` |

## Notes

- Official weights are bf16 safetensors (~9 GB for 4B). The current path dequantizes to F32 for the RLX graph — plan for host RAM accordingly.
- Critical-points safety (pause before PII / irreversible actions) is in the system prompt; still keep a human in the loop.
