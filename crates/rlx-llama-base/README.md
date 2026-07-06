# rlx-llama-base

Shared **"Llama-shaped" architecture config + GGUF reader** for the family of decoders that all share the same skeleton — GQA + RoPE + RMSNorm + SwiGLU FFN — differing mainly in RoPE base/scaling, sliding-window setting, and tensor-name conventions. This is the config foundation the M4-family runner crates build on; it does not run inference itself.

| Family | llama.cpp arch tag | Notes |
|---|---|---|
| Mistral 3+ | `mistral3` / `mistral4` | Sliding-window (5K for 3.5) |
| Phi 3 / 4 | `phi3` | Default + YaRN (`phi4` reuses `phi3`) |
| Bonsai | `llama` | Ships as llama-tagged GGUF |
| OmniCoder | `qwen3` | Ships as qwen3-tagged GGUF |
| Granite | `granite` | IBM Llama-shaped (attn/embedding scale) |
| Command-R | `command-r` / `cohere2` | Cohere Llama-shaped |

## Public API

```rust
use rlx_llama_base::{LlamaBaseConfig, family_preset};

let cfg = LlamaBaseConfig::from_gguf_path("model.gguf")?;   // GQA/RoPE/RMSNorm dims
if let Some(preset) = family_preset(&cfg.arch_tag) {
    // per-family RoPE-scaling / sliding-window deltas
}
# anyhow::Ok(())
```

## How it fits

The per-model wrappers — [rlx-phi](../rlx-phi), [rlx-mistral](../rlx-mistral), [rlx-cohere](../rlx-cohere), [rlx-granite](../rlx-granite), [rlx-bonsai](../rlx-bonsai), [rlx-omnicoder](../rlx-omnicoder) — validate their arch tag here and run inference via [rlx-llama32](../rlx-llama32).
