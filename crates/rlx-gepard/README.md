# rlx-gepard

Gepard (~556M) autoregressive TTS for RLX — Qwen3.5 full-attention backbone + NanoCodec FSQ @ 22.05 kHz.

## Quick start

```bash
just fetch-gepard   # model.safetensors + tokenizer + gepard_config.json
# also place nano_dec_1.89kbps.safetensors under weights/tts/gepard

just gepard-demo TEXT="The quick brown fox jumps over the lazy dog." DEVICE=metal
just gepard-whisper
just gepard-whisper-long
just gepard-backends
just gepard-parity
```

CLI:

```bash
cargo run -p rlx-gepard --bin rlx-gepard --release --features apple-silicon -- \
  --weights weights/tts/gepard \
  --text "Hello from Gepard." \
  --device metal \
  --out /tmp/gepard.wav
```

## Layout

```text
weights/tts/gepard/
  gepard_config.json
  model.safetensors
  tokenizer.json
  nano_dec_1.89kbps.safetensors   # HiFi-GAN decoder (required)
```

Weights: [nineninesix/gepard-1.0](https://huggingface.co/nineninesix/gepard-1.0).

## Inference path

- **Text**: HF `tokenizer.json` + TextRepeater layout + SOS (see MODEL_GUIDE §9.6).
- **AR**: compiled Qwen3.5 via `rlx-qwen35` when `--device != cpu` (or `RLX_GEPARD_COMPILED=1` on CPU). Eager CPU remains the default for `--device cpu`.
- **Decode**: NanoCodec HiFi-GAN on `--device`.
- **Sampling**: temperature `0.4`; seed `54` (short) / `4` (long paragraph) — see `default_seed_for_text`.
- **Prefill**: sequential token steps (match eager numerics; required for long-text Whisper).

Env:

- `RLX_GEPARD_COMPILED=1` — force compiled AR even on CPU.
- Default: compiled AR for non-CPU devices; eager for CPU.

### Apple Silicon note

Metal Qwen3.5 decode currently diverges from CPU; when `--device metal` is selected,
backbone AR is routed to **MLX** (NanoCodec still runs on Metal). Prefill/decode use
`skip_warm` so long sequential AR does not OOM under the CLI.

`cargo test` Whisper gates use **CPU compiled AR** (MLX + the test harness can still
SIGKILL on this machine); validate MLX via:

```bash
cargo run -p rlx-gepard --bin rlx-gepard --release --features apple-silicon -- \
  --weights weights/tts/gepard --device mlx --seed 54 --out /tmp/gepard_fox_mlx.wav
```

CUDA / ROCm / Vulkan / MLX run AR natively on the selected device.

## CUDA (rig)

```bash
just gepard-validate-cuda
just gepard-whisper-cuda-long
just gepard-bench DEVICE=cuda
RLX_DEVICES=cpu,cuda cargo run -p rlx-gepard --release --example backend_matrix --features nvidia-gpu
```

**CUDA memory knobs (8–16 GiB cards):** Gepard uses bucketed decode (not
per-`past_seq` dynamic graphs), skips warm-up, defaults `RLX_GEPARD_MAX_SEQ=1024`,
and drops the prefill arena after the first hidden prefill. Qwen3.5
`prefill_from_hidden` no longer registers the unused F32 `token_embd` table.
If compile still OOMs, lower `RLX_GEPARD_MAX_SEQ` (e.g. `256`) or run AR on CPU.

**AR quality:** compiled Qwen3.5 AR uses **bucketed decode** by default (one
padded graph — matches eager; needed for long paragraphs without OOM). Opt into
per-`past_seq` graphs with `RLX_GEPARD_DYNAMIC_DECODE=1`. Metal routes AR to MLX
when available. Force eager CPU AR with `RLX_GEPARD_EAGER_AR=1`. Fox matrix uses
sampling seed **54**.

## Backends

Build with `--features apple-silicon`, `nvidia-gpu`, or `all-backends`.

## Tests

- `just gepard-whisper` / `just gepard-whisper-long` — fox + long vs Whisper Tiny.
- `just gepard-parity` — eager vs compiled prefill SOS cosine.
- `just gepard-backends` — cross-backend cos + Whisper.
- `just gepard-bench` — wall-clock timing / RTF.
