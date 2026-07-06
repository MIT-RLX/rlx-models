# rlx-cli

Shared CLI plumbing for the per-model `rlx-<family>` binaries (and the optional `rlx-run` multiplexer). It centralizes the argument parsing, device selection, chat templating, output formatting, and weight-inspection helpers so each model crate's binary stays thin and consistent.

## Modules

- `args` / `lm_args` — common flag parsing (`--weights`, `--device`, `--prompt`, sampling, …).
- `device` — `parse_standard_device(...)` maps `cpu|metal|mlx|cuda|wgpu|vulkan|ane` strings to `rlx_runtime::Device`.
- `chat` — chat-template application.
- `format` — token/latency/RTF output helpers.
- `auto_dispatch` — registry that routes `rlx-run <family> …` to the right model runner.
- `inspect` — GGUF/safetensors introspection (also exposed via the `rlx-inspect` binary).

## Binary

```bash
# Inspect a checkpoint's tensors / metadata
cargo run -p rlx-cli --bin rlx-inspect -- <weights.gguf|model.safetensors>
```

## How it fits

Consumed by every model runner crate (e.g. [rlx-llama32](../rlx-llama32), [rlx-qwen3](../rlx-qwen3), [rlx-whisper](../rlx-whisper), [rlx-phi](../rlx-phi)) for their command-line front-ends.
