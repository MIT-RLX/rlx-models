# rlx-vlash

Native [RLX](https://crates.io/crates/rlx-runtime) port of the
[VLASH](https://github.com/mit-han-lab/vlash) **π₀** and **π₀.₅**
Vision-Language-Action robot policies.

Both policies pair a **PaliGemma** backbone (SigLIP-So400m/14 @224 vision tower
+ a Gemma-2B text model) with a **Gemma-300M action expert**. The two Gemma
stacks run through 18 *joint* transformer layers that share one attention over
the concatenated `[image ++ text ++ suffix]` sequence, and actions are produced
by **flow matching** (a short Euler integration of a learned velocity field).

- **π₀** — state is a suffix token; time is fused into the action embeddings;
  standard Gemma RMSNorm.
- **π₀.₅** — action-only suffix; state + time drive **adaptive RMSNorm**
  (adaRMS) in the action expert.

Weights load from the published `lerobot/pi0_base` / `lerobot/pi05_base`
checkpoints (bf16 `model.safetensors`, OpenPI naming; remapped by
[`weights::canonical_key`]), or from ready-to-run RLX bundles.

## Pre-built weights

RLX-native bundles (canonical keys baked in, no remap at load) are published at
**[`eugenehp/rlx-vlash`](https://huggingface.co/eugenehp/rlx-vlash)** — both
variants (`pi0/`, `pi05/`) in GGUF and RLX-package containers at f16 / q8_0 /
q4_K / f32. Point the runner at a directory containing `model.gguf`:

```rust
use rlx_vlash::{VlashRunner, VlashVariant};
use rlx_runtime::Device;

let mut runner = VlashRunner::builder(VlashVariant::Pi05)
    .device(Device::Cpu)
    .num_images(1)
    .prompt_tokens(200)
    .model_dir("path/to/pi05")   // dir with model.gguf (+ tokenizer.json)
    .build()?;
let actions = runner.predict_action_chunk(&[image_nchw], &state, "pick up the cube", None)?;
```

Use **f16** for faithful control — the quantized variants keep per-step outputs
near-exact but the 10-step flow-matching rollout amplifies small errors into the
action chunk (actions-vs-reference cosine: f16 = 1.000, q8_0 ≈ 0.97, q4_K ≈ 0.81;
see the model card). The PaliGemma tokenizer is gated and not bundled — fetch
`google/paligemma-3b-pt-224`'s `tokenizer.json` for prompt tokenization.

## Backends

The full arch (SigLIP tower, joint two-stream attention, RoPE, GQA, adaRMS,
flow-matching denoise loop) builds and runs on every RLX backend. The
`tests/smoke.rs` suite compiles + runs both π₀ and π₀.₅ with a tiny synthetic
checkpoint and, when the backend is present at runtime, on each accelerator:

```bash
cargo test -p rlx-vlash                              # CPU
cargo test -p rlx-vlash --features metal  --test smoke -- --nocapture
cargo test -p rlx-vlash --features mlx    --test smoke -- --nocapture
cargo test -p rlx-vlash --features gpu    --test smoke -- --nocapture   # wgpu
cargo test -p rlx-vlash --features vulkan --test smoke -- --nocapture
cargo test -p rlx-vlash --features cuda   --test smoke -- --nocapture   # NVIDIA
```

Metal / MLX / wgpu / Vulkan are verified to run both variants on Apple silicon;
CUDA / ROCm are wired identically (need NVIDIA / AMD hardware). `all-backends`
enables the lot.

## Preparing weights (`.gguf` / `.rlxp`)

Convert a raw checkpoint into an RLX-native bundle with the canonical key names
baked in (so the runtime loads it with no remap). Schemes: `f16` (default),
`q8_0`, `q4` (Q4_K), `f32`; formats `gguf` / `rlxp`:

```bash
# one scheme/format
cargo run --release -p rlx-vlash --example prep_weights -- \
    --variant pi05 --model <checkpoint_dir> --out model.gguf --format gguf --scheme f16

# every format+precision into a directory (f16/q8_0/q4_K/f32 × gguf/rlxp)
cargo run --release -p rlx-vlash --example prep_weights -- \
    --variant pi05 --model <checkpoint_dir> --out <out_dir> --all-formats
```

The runner loads a directory containing `model.gguf` / `model.rlxp` /
`model.safetensors` transparently (`prep::load_prepped` dispatches on extension;
GGUF/rlxp keep canonical keys, safetensors is remapped on load). GGUF stores
shapes in GGML order internally and round-trips `[out,in]` weights byte-for-byte
at f32 (verified in `prep::tests`).

## Parity vs the original

`scripts/run_parity.py` runs the upstream VLASH implementation and rlx-vlash on
identical fixed inputs and prints a per-stage cosine / max|Δ| table (PASS if
cosine > 0.999):

```bash
python crates/rlx-vlash/scripts/run_parity.py --variant pi05 \
    --checkpoint lerobot/pi05_base \
    --out ~/.cache/rlx-vlash/fixtures/pi05
```

Under the hood: `vlash_ref_dump.py` dumps the reference intermediates (needs a
Python env with upstream VLASH + the pinned `transformers@dcddb97` + `lerobot`),
the `dump_intermediates` example runs rlx-vlash on the same inputs, and the
driver compares `image_features_raw`, `prefix_embeds`, `velocity_step0`, and
`actions_padded`. `tests/parity.rs` performs the same comparison from Rust and
skips gracefully when `RLX_VLASH_{PI0,PI05}_{FIXTURE,MODEL}` are unset.

**Verified:** **both** π₀ (`pi0_base`) and π₀.₅ (`pi05_base`) match the original
VLASH implementation at **cosine = 1.000000** across all four stages — SigLIP
vision features, the image+text prefix, one flow-matching denoise step (joint
attention + adaRMS / state-token path), and the full 10-step action chunk. Max
\|Δ\| ≤ 1.6e-4 (π₀.₅) / ≤ 1.1e-3 (π₀), all attributable to MPS-f32 reference vs
CPU-f32 rlx. If the gated PaliGemma tokenizer is unavailable, pass
`--tokens "2,1596,603,573,2578,108"` to the dump script to parity-check with
fixed token ids (numeric parity only needs identical ids on both sides).

## Status

Release-ready: full architecture, weight loader (safetensors / GGUF / rlxp),
host preprocessing / normalization / tokenization, flow-matching sampler, runner,
CLI, weight-prep tooling, and the parity harness. **18 tests** green; runs on
CPU + Metal / MLX / wgpu / Vulkan; **real-weight parity = cosine 1.0 vs the
original for both variants**. Pre-built weights: `eugenehp/rlx-vlash`.

## License

GPL-3.0-only, matching the RLX workspace. The upstream VLASH project is
Apache-2.0.
