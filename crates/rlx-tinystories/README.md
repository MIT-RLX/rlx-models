# rlx-tinystories

Train a small **nanoGPT / GPT-2-style** language model **from scratch** on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset —
a compact, end-to-end showcase of the **RLX training flow**: the model graph is
written in the [`rlx!`](../../../rlx/crates/core/rlx-tensor) DSL, gradients come
from `rlx-tensor`'s autodiff, and the loop uses **Muon + AdamW** (Muon on the
2-D weight matrices, AdamW on the embeddings/biases/norms) with a warmup-cosine
schedule, running on Apple GPU (Metal) or CPU, with a live progress bar.

TinyStories is purpose-built for this: a few-million-parameter model trained on
it produces fluent, coherent short stories, so a from-scratch run finishes in
minutes on a laptop GPU rather than requiring a cluster.

> Unpublished (`publish = false`) — a reference/demo crate.

## The model

A standard decoder-only transformer (the choice the TinyStories paper and
nanoGPT use):

- **byte-level tokenizer** (vocab = 256) — zero dependencies, maps 1:1 to corpus
  bytes so the loader `mmap`s the file and slices windows directly;
- token + **learned positional** embeddings (added as one-hot `@ table`);
- `n_layer` pre-LN blocks: **causal multi-head attention** + a **4× GELU MLP**,
  each with a residual connection;
- final LayerNorm and a **tied** LM head (`logits = h · wteᵀ`);
- next-token **softmax cross-entropy** loss.

Default config ≈ **2.8M params** (`n_layer=6, n_embd=192, n_head=6, ctx=256`);
a `--smoke` config (~120K params) is used for the CPU test.

## Built with the `rlx!` DSL

The forward pass is expressed in the DSL (`rlx_expr!`), with a Rust loop driving
depth. The whole training objective is **one line**:

```rust
// per block (crates/rlx-tinystories/src/model.rs)
let mlp = rlx_expr!((gelu(hn2 @ w1 + b1)) @ w2 + b2);
h       = rlx_expr!(h + mlp);
// ...
let logits = rlx_expr!(matmul_t(h, wte));           // tied head
rlx_expr!(mean(cross_entropy(logits, tgt)))         // the loss
```

`cross_entropy(...)` and `mean(...)` are DSL sugar added to `rlx!` for training
(see *Framework additions* below), and the loop uses the new all-params,
device-pinned training step:

```rust
let mut model = model::init(model::build(&cfg, cfg.batch, true), &cfg, seed);
let mut opt   = HybridOptimizer::new(adamw_lr, muon_lr, 0.1); // Muon(2-D) + AdamW(rest)
for step in 0..steps {
    let (tok, tgt) = batcher.sample(train_data, &mut rng);
    let feed = &[("tok", &tok), ("pos", batcher.pos()), ("tgt", &tgt)];
    let (next, loss) = model.train_step_all_at_on(dev, &mut opt, &sched, step, feed);
    model = next;                                    // no hand-listed param names
}
```

## Usage

```bash
# Train (auto-downloads the full TinyStories train split from the Hub on first
# run; caches under ~/.cache/huggingface). Metal is used automatically if built.
cargo run --release -p rlx-tinystories --bin rlx-tinystories-train

# Train on a local text file instead, on CPU, a quick smoke config:
cargo run --release -p rlx-tinystories --bin rlx-tinystories-train -- \
    --data corpus.txt --device cpu --smoke --steps 500

# Generate from a checkpoint:
cargo run --release -p rlx-tinystories --bin rlx-tinystories-generate -- \
    --model weights/tinystories/tinystories.rlxts --prompt "Once upon a time" --tokens 400
```

### `rlx-tinystories-train` flags

| flag | default | meaning |
|------|---------|---------|
| `--data FILE` | — | local corpus (skips the download) |
| `--split train\|valid` | `train` | which TinyStories split to download (train ≈ 2 GB, valid ≈ 20 MB) |
| `--steps N` | `2000` | training steps |
| `--lr F` | `3e-4` | AdamW base learning rate (warmup-cosine) |
| `--muon-lr F` | `2e-2` | Muon learning rate (tracks the same schedule) |
| `--grad-clip F` | `1.0` | global-norm gradient clip (cures loss spikes) |
| `--fake-quant SPEC` | off | emulated low-precision QAT: `nvf4`/`f8e4m3`/`bf8`/`f16` or generic `fXmYeZ` (e.g. `f8m3e4`) |
| `--precision f32\|f16\|bf16` | `f32` | compute dtype for the matmuls (STE-emulated, f32-accumulate — see notes) |
| `--device cpu\|metal` | auto | force a backend |
| `--smoke` | off | tiny config (fast CPU) |
| `--batch/--seq/--layers/--embd/--heads N` | — | architecture overrides |
| `--out FILE` | `weights/tinystories/tinystories.rlxts` | checkpoint path |
| `--eval-every / --sample-every N` | `200 / 500` | periodic held-out loss / sample |
| `--max-bytes N` | all | cap corpus bytes (quick runs) |
| `--bpe VOCAB` | `0` (byte-level) | train a from-scratch BPE to `VOCAB` tokens — denser tokens, faster convergence (see notes) |

### Features

- `metal` *(default)* — train on Apple GPU. Drop with `--no-default-features`
  for a pure-CPU build.
- `download` *(default)* — enable the Hub auto-download (`hf-hub`). Without it,
  `--data` is required.

## Test

```bash
cargo test -p rlx-tinystories --no-default-features   # offline CPU smoke test
```

The smoke test trains the tiny config on an in-memory corpus and asserts the
loss falls sharply, then exercises the generation path — no network, no GPU.

## Data path: gather embedding + BPE (`--bpe`)

Training here is **I/O/dispatch-bound**, not FLOP-bound (half-precision compute
gives ~0 speedup; throughput plateaus at ~13k tok/s regardless of batch). Two
changes attack that:

**Gather embedding.** The token embedding is a `wte.gather(ids)` fed `[B*T]`
integer ids, not a `[B*T, V]` one-hot `@ wte`. Targets likewise ship as ids and
their (label-smoothed) one-hot is rebuilt *on-device* by gathering a constant
`[V,V]` table. So ~`V×` less host→device traffic and no fake embedding matmul.
Positions need no input at all — `wpe [T,D]` broadcasts over the batch. This is
bit-for-bit the same objective (verified: gather val 2.81 ≈ one-hot 2.80); the
`wte` gradient now arrives via the gather's scatter-add *plus* the tied head.

**BPE (`--bpe VOCAB`).** A from-scratch byte-level BPE (`src/bpe.rs`, no
`tokenizer.json`, no external crate) trained on the corpus itself. Byte-level
tokens are information-sparse (1 byte/token); BPE packs ~4 bytes/token, so a
fixed-length sequence carries ~4× more text and the model reaches a given
**bits/byte** in far fewer steps. The gather embedding is the enabler — a 2k–50k
one-hot would be GB/step, but as ids it's the same `[B*T]` payload at any vocab.

Measured (identical 4L·128d, batch 16, seq 128, 60 steps, 8 MB corpus):

| tokenizer | bytes/token | bits/byte @ 60 steps |
|---|---|---|
| byte-level | 1.00 | 4.923 |
| `--bpe 2048` | 4.02 | **2.381** |

~2× better bits/byte at the same steps and near-identical wall-clock — and the
BPE samples emit real words within ~10 steps where byte-level still babbles
characters. `bits/byte` is normalized by bytes-per-token, so the two tokenizers
compare on one axis. Bigger vocab shifts cost into the tied head (a 50k head is a
real matmul on a small model), so a moderate vocab (2k–8k) is the sweet spot
here. The trained BPE is embedded in the checkpoint (v2 format), so `generate`
reloads it automatically.

## Any-precision training (`--fake-quant`)

Train at **any** float precision — including formats with no hardware kernel —
via *emulated* precision: weights are round-tripped through the target format's
grid each step (straight-through gradient to f32 masters), so the compute stays
f32 on any backend but the model learns at the emulated precision. The formats
are generated by a Rust macro (`rlx_tensor::lowp::float_format!`) and cover
`fXmYeZ` for any exponent/mantissa split.

```bash
# 8-bit (E4M3), 4-bit NVFP4, or an arbitrary format:
cargo run --release -p rlx-tinystories --bin rlx-tinystories-train -- --fake-quant f8e4m3 …
cargo run --release -p rlx-tinystories --bin rlx-tinystories-train -- --fake-quant nvf4    …
cargo run --release -p rlx-tinystories --bin rlx-tinystories-train -- --fake-quant f8m3e4  …   # generic fXmYeZ
```

Verified on TinyStories: **fp8 → val 2.33**, **nvf4 (4-bit) → val 2.37** (both
from ~5.5, same trajectory as f32). Narrow formats use **per-tensor absmax
scaling** (the MXFP4/NVFP4 microscaling trick) — without it, ~0.02 init weights
round to 0 under nvf4 (min step 0.5) and the net never learns.

### Half-precision compute (`--precision bf16` / `f16`)

`--precision {bf16,f16}` trains in **mixed precision** — the residual stream,
LayerNorm, biases, GELU and attention stay f32 (so a half-quantized row can't
collapse to zero variance and blow LayerNorm's backward up to ±3e38), while the
big transformer **matmuls** see half-precision *inputs* with an **f32 accumulate
and f32 backward**. The matmul inputs are rounded with a straight-through
estimator (`ste_round` in `model.rs`):

```
forward:  x → clamp(±max) → round-to-dtype → f32     (emulates the low-precision forward)
backward: identity in f32                            (the round lives inside .detach())
```

Two reasons this beats a *native* half matmul. (1) **Backward overflow**: f16's
5-bit exponent caps at ±65504, and the grad matmuls here reach ~3e4 — a native
f16 backward overflows to ±inf → NaN by step ~2. STE keeps the whole backward in
f32, so gradients never touch the f16 range. (2) **Forward saturation**: the
`clamp(±max)` makes an out-of-range activation saturate (as saturating-rounded
hardware f16 does) instead of casting to inf. Verified on Metal: f32 / bf16 / f16
all converge together (val **2.68 / 2.69 / 2.68** at 50 steps, 6L·256d, no NaN).

The supported float sizes are declared by **one Rust macro** — `float_sizes!` in
`src/precision.rs`. Each row (`"name" => (DType, max_finite)`) generates the
`--precision` parser, its help/error text, and the range clamp `ste_round` uses,
so adding a float size is a single line:

```rust
float_sizes! {
    "f32"  => (DType::F32,  f64::from(f32::MAX)),
    "bf16" => (DType::BF16, f64::from(f32::MAX)), // 8-bit exponent ⇒ f32 range
    "f16"  => (DType::F16,  65504.0),             // IEEE binary16 max normal
}
```

This is *quality* emulation (the matmul accumulates in f32, so it isn't faster
than f32); it demonstrates numerically-stable low-precision training at any float
size. A backend fix landed alongside it in `../rlx` — the Metal `Sgemm` thunk had
no `a_f16` path, so a mixed `f16 × f32` matmul (e.g. a genuinely f16-activation
backward) silently reinterpreted the 2-byte f16 A operand as f32; it now routes
to a dedicated `sgemm_f16a` kernel that reads A as half with an f32 accumulate.

## Framework additions this crate motivated

To make training first-class in the `rlx!` DSL, `rlx-tensor` / `rlx-macros`
gained (all additive):

- **`Tensor::cross_entropy` / `softmax_cross_entropy_with_logits`** and
  **`mean_all` / `sum_all`** — fused softmax cross-entropy + reduce-to-scalar.
- **`rlx!` sugar**: `cross_entropy(logits, tgt)`,
  `softmax_cross_entropy(logits, labels)`, and 1-arg `mean(x)` / `sum(x)`.
- **`Func` training DX**: `param_names()`, `init_params(closure)`,
  `init_randn(seed, std)`, `value_and_grad_all()`, and all-params /
  device-pinned training steps (`train_step_all`, `train_step_on`,
  `train_step_all_at_on`, …).
- **Gradient clipping + QAT steps**: `train_step_all_at_on_clipped`
  (global-norm clip) and `train_step_all_at_on_qat` (straight-through
  quantization-aware training with any quantizer closure).
- **`rlx_tensor::lowp`**: the `float_format!` macro + generic `fXmYeZ`
  quantizer (RNE, subnormals, saturation), named formats (`f16`, `bf16`,
  `f8e4m3`, `bf8`/`e5m2`, `nvf4`/`e2m1`, …), `parse_format()`, and
  per-tensor-scaled `quantize_slice_scaled` (MXFP4/NVFP4 microscaling).
- **Muon on Apple silicon**: `rlx-optim` routes Muon's Newton–Schulz through
  **Accelerate/AMX** `cblas_sgemm` on macOS — ~7× faster Muon training.
