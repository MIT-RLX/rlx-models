# rlx-qwen3-tts-train

Native **LoRA distillation training** for the [Qwen3-TTS](../rlx-qwen3-tts) talker on RLX (MLX / Metal / CPU) — e.g. fitting a custom voice. Fully in-Rust autodiff: dataset prep, distillation cache, Adam, and adapter export, no Python in the loop.

## Quick start

```bash
cargo run -p rlx-qwen3-tts-train --bin rlx-qwen3-tts-train --release -- --help
```

## Modules

- `dataset` — training-sample prep.
- `distill_cache` — teacher-output caching for distillation.
- `compile` / `backward_prep` — graph compilation + gradient setup.
- `adam` — optimizer.
- `codec_table`, `config`, `device` — supporting config.

## How it fits

- [rlx-qwen3-tts](../rlx-qwen3-tts) — the inference model these adapters plug into.
- [rlx-tune](../rlx-tune) — the generic LoRA/DoRA machinery this specializes for the TTS talker.
- Built on `rlx-autodiff` / `rlx-opt` / `rlx-compile`.
