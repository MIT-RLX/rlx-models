# rlx-tiny

Train a small language model **from scratch** on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) where
**every weight matrix is synthesized from a tiny codebook** instead of stored
dense — *functions, not data*.

It is a near-clone of the sibling [`rlx-tinystories`](../rlx-tinystories) dense
GPT: same dataset, same byte/BPE tokenizer, same Muon + AdamW train loop and
`rlx! { }`-DSL forward. The only thing that changes is **how the weights are
represented**, so the two can be A/B compared on identical data (see
[A/B recipe](#ab-recipe-vs-rlx-tinystories) below).

> Unpublished (`publish = false`) — a reference/demo crate.

## The thesis: minimize DRAM I/O

On a small model, training and inference are **I/O / dispatch bound**, not
FLOP-bound — the bottleneck is moving weight bytes to the compute unit, not the
multiplies. So instead of storing an `[k,n]` weight as `k·n` floats and streaming
all of them every step, rlx-tiny stores a **small trained codebook** and
**reconstructs the weight inside the matmul kernel** from a fixed index table.
The weight is a *function* of a few hundred numbers, not a few hundred thousand.

## The model

A standard pre-LN decoder-only transformer (tied LM head, causal attention),
with three substitutions — all first-class RLX ops:

- **Codebook weight-synthesis** (`Op::SynthMatMul`). A projection `x·W`
  (`W [k,n]`) is not a dense matmul. A fixed u8 index table `[n, k/ED]` selects,
  per weight row, `k/ED` entries from a trained codebook `[NE, ED]`
  (`ED=4`, `NE=256`), and `y = x·Wᵀ` reconstructs `W` on the fly. A `k·n` weight
  becomes just `NE·ED = 1024` trainable numbers.
- **Residual multi-stage VQ** (`--synth-stages`). `W = Σ_s codebook_s[idx_s]`:
  each extra stage is another tiny index table + L1-resident codebook — more
  in-kernel compute, ~no extra DRAM weight bytes. Multiplies codebook degrees of
  freedom by the stage count.
- **Low-rank correction** (`--lora-rank`). `W += A·Bᵀ` (rank `r`) — a small dense
  delta that recovers the degrees of freedom a fixed-assignment codebook
  structurally cannot reach.
- **KAN spline FFN** (`Op::SplineActivation`). The MLP's activation is a
  per-channel **learnable spline** (Gaussian-RBF basis, initialized ≈GELU)
  instead of a fixed GELU. The backward is a **fused** kernel
  (`Op::SplineActivationBackwardX`/`Coeff`) that builds the RBF basis in
  registers — the decomposed VJP materialized a ~25M-element basis to DRAM, so
  fusing it cut the training backward **~56%** (`RLX_DECOMPOSE_SPLINE_BWD=1` to
  A/B). Combined with the default 64×64-tile Metal GEMM (`Simd64`), this brings
  the synth model within ~1.25× of a dense transformer (was ~2.3×).

Token embeddings are a **gather** of `wte` by `[B*T]` integer ids (fed as f32,
cast to i64 in-graph) — ~V× less host→device traffic than a one-hot `@ table` —
and the whole forward is **one `rlx! { }` block** (`src/model.rs`): token
embedding → the `repeat i in 0..n_layer` stack → final norm → tied head → the
next-token loss, with the per-layer weights adopted by a single `bind layers[];`.

Default config: `n_layer=6, n_embd=192, n_head=6, ctx=256`. Run `rlx-tiny-train`
and it prints the **actual** trainable-scalar count next to the dense-equivalent
capacity — the gap is the compression.

## Usage

```bash
# Train (auto-downloads the TinyStories train split from the Hub on first run;
# Metal is used automatically if built with it).
cargo run --release -p rlx-tiny --bin rlx-tiny-train

# Quick CPU smoke run on a local text file:
cargo run --release -p rlx-tiny --bin rlx-tiny-train -- \
    --data corpus.txt --device cpu --smoke --steps 500

# Generate from a checkpoint:
cargo run --release -p rlx-tiny --bin rlx-tiny-generate -- \
    --model weights/tiny/tinystories.rlxts --prompt "Once upon a time" --tokens 400
```

### `rlx-tiny-train` flags

The codebook-specific knobs (the rest — `--steps/--lr/--muon-lr/--grad-clip/
--device/--smoke/--batch/--seq/--layers/--embd/--heads/--out/--eval-every/
--sample-every/--max-bytes` — match `rlx-tinystories`; see `--help`):

| flag | default | meaning |
|------|---------|---------|
| `--synth-stages N` | `2` | residual-VQ stages (`W = Σ_s codebook_s[idx_s]`); more stages ⇒ more codebook capacity, ~no extra weight bytes |
| `--lora-rank N` | `8` | low-rank dense correction `W += A·Bᵀ` (`0` disables) |
| `--init-from DENSE.rlxts` | off | **PQ-init** the codebooks + indices from a trained dense `rlx-tinystories` checkpoint (starts far closer to the dense model — prints per-weight reconstruction error) |
| `--distill DENSE.rlxts` | off | **distill** from a dense teacher: add `α·` soft-CE against the teacher's per-token distribution (no new model params) |
| `--bpe VOCAB` | `0` (byte-level) | train a from-scratch BPE to `VOCAB` tokens (denser tokens ⇒ faster convergence) |
| `--fake-quant SPEC` | off | emulated low-precision QAT of the master weights: `nvf4`/`f8e4m3`/`bf8`/`f16` or generic `fXmYeZ` |

`--init-from` / `--distill` require the synth model to share the dense model's
architecture (shapes must line up 1:1) and byte-level tokens, so they force the
config to match the checkpoint and reject `--bpe`.

> `--precision {f32,f16,bf16}` is accepted for CLI parity with the dense sibling,
> but the synth path runs in **f32** (the codebook *is* the compression), so it
> only labels the run. Use `--fake-quant` for genuinely reduced-precision weights.

## A/B recipe (vs `rlx-tinystories`)

Both crates train the *same* TinyStories data with the *same* loop; only the
weight representation differs. Train each on a shared local corpus and compare
val loss, trainable params, and speed:

```bash
# 0) grab a small corpus (or use --split valid, ~20 MB, on either trainer)
cargo run --release -p rlx-tinystories --bin rlx-tinystories-train -- --split valid --steps 0 \
    --out /tmp/_dense.rlxts   # first run just downloads + caches the corpus
CORPUS=~/.cache/huggingface/**/TinyStoriesV2-GPT4-valid.txt   # path it printed

# 1) dense baseline
cargo run --release -p rlx-tinystories --bin rlx-tinystories-train -- \
    --data "$CORPUS" --steps 2000 --out /tmp/dense.rlxts

# 2) rlx-tiny, random codebook init
cargo run --release -p rlx-tiny --bin rlx-tiny-train -- \
    --data "$CORPUS" --steps 2000 --out /tmp/tiny.rlxts

# 3) rlx-tiny, PQ-initialized from the trained dense model (closes the gap)
cargo run --release -p rlx-tiny --bin rlx-tiny-train -- \
    --data "$CORPUS" --steps 2000 --init-from /tmp/dense.rlxts --out /tmp/tiny_pq.rlxts

# 4) or distill the dense model into rlx-tiny
cargo run --release -p rlx-tiny --bin rlx-tiny-train -- \
    --data "$CORPUS" --steps 2000 --distill /tmp/dense.rlxts --out /tmp/tiny_kd.rlxts
```

Each run prints its trainable-param count and periodic **val loss (bits/byte)** —
normalized so byte-level and BPE runs compare on one axis. Watch: (1) rlx-tiny
has **several× fewer trainable parameters** than the dense model, and (2) how
much of the resulting quality gap `--init-from` / `--distill` recover.

### The honest tradeoff

A fixed codebook assignment is a real constraint: at equal step budget, a random-
init rlx-tiny trains to a **higher** loss than the dense model of the same shape —
you trade quality for far fewer parameters and an I/O-minimal, compute-in-kernel
weight path. `--synth-stages` / `--lora-rank` add capacity back (still
IO-minimal), and `--init-from` / `--distill` start from — or pull toward — the
dense model's learned function to narrow the gap. There is no free lunch; the
point is to make the tradeoff *tunable* and measurable on the same data.

## Data path: gather embedding + BPE

Training here is **I/O/dispatch-bound**. Two things attack that:

- **Gather embedding.** The token embedding is `wte.gather(ids)` fed `[B*T]`
  integer ids, not a `[B*T, V]` one-hot `@ wte`; targets ship as ids too. So ~V×
  less host→device traffic and no fake embedding matmul. Positions need no input —
  `wpe [T,D]` broadcasts over the batch inside the block.
- **BPE (`--bpe VOCAB`).** A from-scratch byte-level BPE (`src/bpe.rs`, no
  external tokenizer) trained on the corpus itself. Byte-level tokens are
  information-sparse (1 byte/token); BPE packs ~4 bytes/token, so a fixed-length
  sequence carries ~4× more text and reaches a given **bits/byte** in far fewer
  steps. The gather embedding is the enabler (ids are the same `[B*T]` payload at
  any vocab). The trained BPE is embedded in the checkpoint, so `generate`
  reloads it automatically.

## Any-precision QAT (`--fake-quant`)

Train the master weights at **any** float precision — including formats with no
hardware kernel — via *emulated* precision: each weight is round-tripped through
the target format's grid every step (straight-through gradient to the f32
masters), so compute stays f32 on any backend but the model learns at the
emulated precision. Formats come from `rlx_tensor::lowp::float_format!` and cover
`fXmYeZ` for any exponent/mantissa split, plus named `nvf4`/`f8e4m3`/`bf8`/`f16`.
Narrow formats use per-tensor absmax scaling (the MXFP4/NVFP4 microscaling trick).

```bash
cargo run --release -p rlx-tiny --bin rlx-tiny-train -- --fake-quant f8e4m3 …
cargo run --release -p rlx-tiny --bin rlx-tiny-train -- --fake-quant nvf4    …
```

## Test

```bash
cargo test -p rlx-tiny --no-default-features   # offline CPU smoke test
```

The smoke test trains the tiny config on an in-memory corpus, asserts the loss
falls sharply, then exercises generation — no network, no GPU. (`tests/dense_init`
additionally checks that PQ dense-init starts below random init; it is a slower,
optional check and can lag when a GELU teacher is product-quantized into a
KAN-spline student.)
