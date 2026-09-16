# rlx-translategemma

[Google TranslateGemma](https://huggingface.co/google/translategemma-4b-it) (Gemma 3 MT-tuned) for RLX.

Checkpoints are **gemma3**-shaped. The mlx-community catalog already marks `translategemma-4b-it-*` as validated on the generic / Gemma path. This crate:

- Validates Gemma 3 / TranslateGemma weight metadata
- Delegates inference to [`rlx-gemma`](../rlx-gemma) (all standard RLX backends)
- Adds MT prompt helpers for subtitle-style EN→FR (etc.)

## Backend parity

Reuse Gemma’s existing matrix:

```sh
just features=all-backends test-gemma-backends
cargo test -p rlx-translategemma --lib
```

## Example

```rust
use rlx_translategemma::{TranslateGemmaRunner, TranslatePrompt, target_language_name};

let mut runner = TranslateGemmaRunner::builder()
    .weights("/path/to/translategemma-4b")
    .device(rlx_runtime::Device::Cpu)
    .build()?;
let _ = TranslatePrompt::new("Hello", target_language_name("fr")).render_user();
# anyhow::Ok(())
```
