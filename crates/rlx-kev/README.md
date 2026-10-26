# rlx-kev

Jev-style **System One decision models**, native in RLX.

A decision model takes one document (the *state*) plus a set of typed
questions and returns a calibrated probability distribution for each, in a
single forward pass. It does not generate text. This is a port of
[jaredpalmer/kev](https://github.com/jaredpalmer/kev), which reimplements the
architecture reverse-engineered from TypeSafe's hosted Jev in
["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked).

```text
state  ──► Qwen3.5 backbone (+ merged rank-16 LoRA) ──► hidden
                                                         │
question k:  <q> instr <opt> a </opt> <opt> b </opt> <decide>
                             │                 │          │
                             └──── pointer head ──────────┘  ──► p(a), p(b)
```

## Quick start

```bash
# Inspect a checkpoint (no base model needed)
rlx-kev --run runs/kev-4b --info

# Dump the packed encoding — the artifact to diff against kev.model.encode
rlx-kev --run runs/kev-4b --encode-only --request ticket.json

# Answer a request
rlx-kev --run runs/kev-4b --base models/Qwen3.5-4B-Base \
        --device metal --request ticket.json
```

`--run` is a kev checkpoint (`head.pt`, `adapter_model.safetensors`,
`tokenizer.json`); `--base` is the matching `Qwen/Qwen3.5-*-Base` snapshot.
Nothing is downloaded.

## Request / response

The `/v1/systemone` shape, so TypeSafe's SDK can point at it unchanged:

| type | criteria | answer |
|---|---|---|
| `noul` | optional `true` / `false` descriptions | `noul`: probability of yes |
| `choice` | 1–255 named options with a description or `null` | argmax + `probabilities` + `confidence` |
| `score` | 2–255 ordered levels | mean level index + `legend` + `probabilities` |

## How it works

**Row form.** Qwen3.5 interleaves attention with Gated DeltaNet layers, which
are recurrent and ignore attention masks — so the packed block-causal mask
that isolates question branches on an attention-only backbone is a no-op
there. Each question therefore runs as its own row, `state ++ branch_k`.
Isolation is exact by construction.

Within a row the state occupies positions `0..Ls` and the branch continues at
`Ls..`, so position `i` is just `i`. That is why this runs on a stock causal
prefill graph with no custom mask and no runtime position input. Rows are
right-padded; every layer is causal, so a pad can never influence a readout
below it.

**Pointer head.** `z_j = (K(h_opt_j) · Q(h_decide)) / sqrt(256)`, softmax over
options, divided by the checkpoint's fitted temperature. Kept in f32 even when
the backbone is not.

**LoRA is merged, not layered.** The fold happens inside a `WeightLoader`, so
each delta lands as its base tensor is materialized — one f32 copy, no second
pass. In fp32 the merge is exact.

## Three things that look like details and are not

* **Option order changes the answer.** Options interact before the softmax by
  design. `serde_json::Value` sorts object keys, which would silently reorder
  every option list — hence the crate's own order-preserving
  [`json::Json`](src/json.rs). The same applies to a `state` given as an
  object, whose field order reaches the prompt verbatim.
* **Caller text can never emit a delimiter.** `<|name|>` is rewritten to
  `<¦name¦>` before tokenizing, so option boundaries are unforgeable.
* **The temperature is never guessed.** It exists only in `head.pt`. Loading
  fails rather than defaulting to 1.0 — that default leaves every argmax
  unchanged while roughly doubling the confident-error rate, which is the kind
  of wrong that passes a smoke test.

## Testing

```bash
cargo test -p rlx-kev                       # 60 tests; 5 skip without weights

# Against a real checkpoint (~155 MB: head.pt + adapter + tokenizer.json;
# no base model required)
KEV_RUN_DIR=/path/to/kev-4b cargo test -p rlx-kev --test real_checkpoint
```

Two fixtures pin behaviour against the reference implementation itself:

* `tests/fixtures/golden_encoding.json` — the token protocol. A request with a
  buried `<|fim_prefix|>`, a `null` option description, an object-valued
  `instructions` and all three question types, with the ids
  `kev/model.py::encode` produces for it
  (`scripts/kev_golden_encoding.py`).
* `tests/fixtures/golden_probs_kev_0_8b.json` — the **probabilities** kev
  itself returns for `examples/ticket.json`
  (`scripts/kev_reference_probs.py`).

```bash
# Checkpoint only (~155 MB), no base model
KEV_RUN_DIR=/path/kev-0.8b cargo test -p rlx-kev --test real_checkpoint

# Full end-to-end parity (adds a ~1.7 GB base model)
KEV_RUN_DIR=/path/kev-0.8b KEV_BASE_DIR=/path/Qwen3.5-0.8B-Base \
  cargo test --release -p rlx-kev --features metal,mlx --test real_weights
```

## Serving

```bash
cargo run --release -p rlx-kev --features serve,metal -- \
  --run runs/kev-0.8b --base models/Qwen3.5-0.8B-Base --device metal --serve --port 8009

curl -s localhost:8009/v1/systemone -H 'content-type: application/json' \
  -d @crates/rlx-kev/examples/ticket.json
```

`POST /v1/systemone`, `POST /v1/systemone/separate`, `GET /v1/models`,
`GET /health`. Loopback, no auth, one model behind a mutex — requests queue
rather than race, which is honest for a forward pass that owns a compiled
graph and its arena.

It lives here rather than in `rlx-serve` on purpose: that crate is an
OpenAI-compatible server built around a *generative* engine (streaming,
sampling, stop sequences), and wiring a decision model in would also invert
the dependency. kev ships a separate sidecar for the same reason.

`--prefix-cache [N]` enables state-prefix caching (see below).

## Status

**End-to-end parity with the reference implementation**, on real weights:
`jaredpalmer/kev-0.8b` over `Qwen/Qwen3.5-0.8B-Base`, max |Δp| = **1.4e-5**
against kev's own fp32 output.

| backend | max \|Δp\| vs reference |
|---|---|
| CPU | 0.000014 |
| Metal | 0.000014 |
| MLX | 0.000014 |
| CoreML / ANE | refused up front — see below |

Also verified: encoder byte-identical to the reference on the real Qwen3.5
tokenizer; `head.pt` read natively (no libtorch, no Python) off the released
`kev-4b`; peft adapter fp32 merge with every unmerged module reported;
question isolation, sibling-order invariance and liveness on a synthetic
trunk, plus `/v1/systemone/separate` agreeing exactly with the packed request
on real weights.

### CoreML / ANE

MPSGraph **miscompiles** the shape the Gated DeltaNet prologue emits:
a fused projection narrowed at a non-zero offset, passed through
`sigmoid`/`softplus`, then reshaped. MPS rewrites the reshape onto the
narrow's *parent* and drops the offset:

```text
v20 = sigmoid(v17)                      MIL says tensor<fp32, [512, 16]>
v24 = reshape(shape=[4,128,16], x=v20)  MPS sees tensor<4x16x8208xf32>
'mps.reshape' op the result shape is not compatible with the input shape
```

`rlx-coreml` already had an escape for the *direct* `Reshape(Narrow)` form —
lower it as a `gather`, which has no offset attribute to absorb. It did not
fire here, because its detector only looked one hop back and there is a
`sigmoid` in the way. Two fixes went in:

1. The detector now walks back through single-input shape-preserving ops.
2. Gather turns out **not** to survive the elementwise hop (verified: with
   the gather emitted, MPS still reports the reshape input as `4x16x8208`),
   so that case is refused with a named error instead of emitting a module
   that fails verification and `abort()`s the process — which no caller could
   catch, and which used to take the test runner down with it.

`rlx-kev` then refuses `--device ane` up front, before reading 3.5 GB of
weights. `examples/coreml_slice_repro.rs` keeps the two nearby shapes that do
work, so a future fix has something to check against.

### Training

```bash
cargo run --release -p rlx-kev --features train,metal -- --train \
  --run runs/kev-0.8b            # tokenizer, and an adapter to warm-start from
  --base models/Qwen3.5-0.8B-Base \
  --data labelled.jsonl --out runs/mine \
  --epochs 2 --records 1 --accum 8 --lr 2e-5 --head-lr 1e-3 --lora-rank 16
```

Fits a LoRA adapter and the pointer head; base weights stay frozen and out of
the optimizer. Objective is plain cross-entropy over a question's options —
which is what the released checkpoints used (`perm_kl`, `ord_w`, `brier_w`,
`focal_gamma` and `anchor_w` are all `0` in kev-4b's own
`training_config.json`).

Data is JSONL, one request per line plus the correct option **key** per
question:

```json
{"state": "…", "questions": {…}, "labels": {"department": "shipping", "escalate": "true"}}
```

A bad key is an error, not a silent class 0.

The output directory is a working checkpoint — `adapter_model.safetensors`,
`adapter_config.json`, `head.json` + `head.safetensors`, `tokenizer.json` — and
loads straight back for inference. `--temperature` is written as `1.0`; fit the
calibration on held-out data before trusting the probabilities.

### What the tests check

A falling loss proves little, so `tests/train_graph.rs` pins the things that
make training silently useless:

* **analytic vs finite-difference gradient** on the head (worst relative error
  0.032);
* at step 1 only `B` and the head have gradients — with `B = 0`,
  `dL/dA = (dL/dy)·Bᵀ` is *identically zero*, so an all-zero `A` gradient there
  is correct, and pinning it stops someone "fixing" it;
* every trainable moves within two steps — a graph rewriter that duplicates a
  `Param` yields permanently frozen weights while the loss still descends via
  the others, which has happened in this codebase;
* a masked (padded) row cannot change the loss;
* overfitting one batch (1.098 → 0.506 in 30 steps);
* the adapter and head round-trip through the *inference* loaders — the seam
  where two orientation conventions meet (in-graph `[in, out]` so `inject_lora`
  emits `A [in, r]` / `B [r, out]`, versus peft's `A [r, in]` / `B [out, r]`).

## Reproducing the released recipe

```bash
# the frozen suite kev-4b trained on
huggingface-cli download jaredpalmer/kev-suites --repo-type dataset --local-dir suites

cargo run --release -p rlx-kev --features train,metal -- --train \
  --suite suites/v7/decision-v7 \
  --run runs/kev-0.8b                 # tokenizer only; use --init-from to warm start
  --base models/Qwen3.5-0.8B-Base \
  --out runs/repro --device metal \
  --epochs 1 --accum 2 --lr 2e-5 --lora-rank 16 \
  --checkpoint-every 100          # default; 0 disables

# A full run is 1529 steps and hours long, so the adapter and head are exported
# every `--checkpoint-every` steps (0.1s per write), staged through
# `<out>/.staging` and renamed into place — a kill at any instant leaves a
# complete, loadable checkpoint rather than a torn one. There is no true resume
# (Adam moments and the schedule position are not persisted); continue from a
# partial run with `--init-from <out>`, which merges the adapter into the base
# and trains a fresh one on top, and note that OneCycle restarts.

# fit the calibration temperature (a fresh run carries 1.0, which leaves every
# argmax alone while roughly doubling the confident-error rate)
cargo run --release -p rlx-kev --features train,metal -- --calibrate \
  --run runs/repro --base models/Qwen3.5-0.8B-Base \
  --suite suites/v7/decision-v7

# score it the way kev reports
cargo run --release -p rlx-kev --features train,metal -- \
  --run runs/repro --base models/Qwen3.5-0.8B-Base \
  --suite suites/v7/decision-v7 --eval development --limit 200
```

### What the released recipe actually is

`kev-0.8b`'s own `training_config.json` is worth reading before treating any
number here as a reproduction, because it describes a *different experiment*
from a cold start:

* `lr: 2e-05` with `head_lr: 0.0` — and `head_lr` 0 means "use `lr`", so the
  pointer head trains at 2e-5 as well. (An earlier version of the command above
  said `--lr 1e-4 --head-lr 1e-3`: 5x and 50x the reference. Corrected.)
* `init_from: "jaredpalmer/kev-0.8b"` — the released checkpoint is the output of
  a **continuation round** (`out: /runs/r15-08b/...`, i.e. round 15) that
  warm-started from the previous kev-0.8b, not of a single cold run.
* `n_per_source: 1000`, `replay: 6000`, `batch: 4`, `checkpointing: 1`.

So `0.8136` is the score of an iteratively-refined model. A cold one-epoch run
over the same suite is not the same thing and should not be expected to match
it; use `--init-from` to run the recipe as published — that restores both the
adapter and the pointer head, so step 0 starts from the source checkpoint rather
than from a random readout.

### Reproduction status

The first full run — 1529 steps over all 12,576 records, 2.1 h — scored **0.2845**
on the development split against the released kev-0.8b's **0.8136**, i.e. chance.
The training loss looked plausible the whole way (5.12 → 0.709) but it was
sitting on `ln(n_options)` for whichever shape each step drew: 0.709 ≈ `ln 2` for
the yes/no batches, 4.17 ≈ `ln 64` for the many-option ones. It had learned
nothing.

**The cause was not in this crate, and it is now fixed.** Metal returned NaN or
zero for most of the Qwen3.5 trunk's parameter gradients while computing the
forward *bit-identically* to CPU — two separate bugs in Metal's thunk-level
fusions and its conv weight gradient, both written up in the rlx CHANGELOG and
pinned by `rlx-qwen35`'s `trunk_backward_device_parity`. Before the fix, the same
single-record overfit gave `grad_norm NaN` at step 1 on Metal and `grad_norm
255.53` on CPU; after it, Metal gives the same 255.53 and drives the loss to
**0.0095** in 25 steps (CPU: 0.2057), **7.5× faster** than CPU (244 s vs 1818 s).

The lesson worth keeping is the one the guard now encodes: **a finite loss says
nothing about the gradients.** `Trainer::optimizer_step` refuses a non-finite
gradient norm and names the worst parameter, so this class of bug costs seconds
rather than a 2-hour run. And when reading a training log, compare the loss
against `ln(n_options)` — not against zero.

The fix declines those fusions when a value is read again after the fused
window, which a backward graph does and a forward graph does not. An earlier
version of this file claimed that cost ~2× on a Metal training step; that was
**wrong and is retracted**. Instrumenting the fusion showed the pattern firing on
0 of 34 candidate anchors in this graph, and the 2× came from machine contention
— identical work timed 492 s against 1062 s depending only on what else was
running. A training step is compute-bound (10.2 s measured against ~6.2 s of f32
FLOPs). Forward/decode is unaffected.

With the fix, a 154-step run over 1,200 records (a tenth of the reference's data)
scores **0.3389** where the broken full run scored 0.2845, and its loss declines
monotonically (3.02 → 2.27 → 1.46) instead of bouncing between chance floors.
But read 0.3389 against the chance floor of the sample it was scored on, which
is **0.3059** — it is *at chance*, and the gap to the broken run is ~1.8 SE at
SE 0.031. This is **not** a reproduction, and 154 steps against the reference's
1529 was never going to be one. The evidence that the backend is fixed is the
gradient parity, the overfit, and the loss shape — not this number.

**What fixing the backend revealed.** With real gradients, a full-data run
diverges at the OneCycle peak — loss 1.06 at step 50, then 4.33 at step 100 with a
pre-clip grad norm of 1294, and 9.41 by step 150, which is above every
`ln(n_options)` chance floor.

Two causes, both outside the backend. The first: the recipe rebuilds the trainer
on every shape change (28 times an epoch) and carried the adapter and head
*weights* across but not the Adam moments. `rlx-tune`'s `Adam` grew
`take_state`/`restore_state` and the recipe now carries them. That attenuated the
divergence without removing it.

The second is the actual mechanism, and it was in `rlx-tune`'s `Adam` itself:
the bias-correction step counter lived on the optimizer, but `Optimizer::step` is
called once per *parameter*, so it advanced by the number of trainables (~340
here) per optimizer step. After the first step `1 - beta^t` is 1 to machine
precision, so bias correction was **off for the rest of training** — and
uncorrected, `v` is underestimated while it warms up, inflating the effective
learning rate by up to **6.5×** (still 2.65× at step 153, exactly the OneCycle
peak). A recipe asking for `1e-4` was training at several times that through the
whole ramp. The counter now lives in the per-parameter state entry, which is what
PyTorch's `state['step']` is. None of this was visible while the gradients were
NaN — nothing moved at all.

Backend status for *training*:

| | forward | backward |
|---|---|---|
| CPU | ✓ | ✓ |
| Metal | ✓ bit-exact | ✓ (was NaN/zero) |
| MLX | ✓ | ✓ (was: would not compile the graph at all) |

All three now train, bf16 included. MLX took two more upstream fixes: its
`RmsNorm` lowering dropped the flattening that `fuse_rms_norm_reshape` folds in
(so a following `gather` indexed the batch axis), and `grad_with_loss` sized a
zero cotangent constant at 4 bytes per element under a bf16 declaration.

### What is reproduced exactly

* **The data pipeline, byte for byte.** `augment` (option permutation,
  none-of-the-above swap/insert, distractor insert), `none_pair`, and
  `materialize` all match kev's own output on real suite records —
  `tests/recipe_parity.rs` checks 80 cases against `kev/data.py`, including
  the `p_none` branch that *removes the correct option*.
* **CPython's PRNG.** The augmentation's draws depend on stream order, not
  just distribution, so `src/pyrandom.rs` is MT19937 plus CPython's exact
  `random` / `getrandbits` / `_randbelow` / `shuffle` / `choice`, seeded like
  `random.seed(int)` and keyed by `source_seed(seed, "{epoch}:{id}")`.
  Verified against CPython in `tests/pyrandom_parity.rs`.
* **The hyperparameters** from kev-4b's own `training_config.json`: rank 16,
  alpha 32, AdamW `wd` 0.01, clip 1.0, `p_none` 0.1 / `p_none_distract` 0.12 /
  `p_distract` 0.15 / `p_none_pair` 0.25, and OneCycle with `pct_start` 0.1 —
  including torch's off-by-one `step_size_up = pct_start·total − 1`, pinned in
  `one_cycle_matches_torch`.
* **The source policy**: eval-only sources are refused rather than trained on.

### What is not

* **bf16 autocast is implemented and correct but off by default** — it does
  not save memory on CPU yet (see below). Batches are therefore packed under a
  token budget (`KEV_TRAIN_TOKEN_BUDGET`, default 2048 `rows × seq`) instead of
  the reference's flat "4 records per batch".
* **Shape-sorted batch order.** rlx compiles per shape, and `decision-v7`
  mixes 2-option yes/no questions with banking77's 77 intents; a single global
  shape asks for a 68 GB arena. Examples are therefore sorted by shape and
  batched into buckets, so the *data* is identical but the *order* is not the
  reference's shuffle — a real difference in the optimizer trajectory.
* **No soft-target loss.** `decision-v7` contains none (verified: zero
  `target` fields); it only matters for the later `night2` delta data.
* `ssm_alpha`/`ssm_beta` are fused into one matmul by the builder, so they
  share a single adapter rather than getting one each.

### Upstream additions this needed

Two gaps were closed rather than worked around:

* **LoRA dropout** (`rlx-tune`). `LoraSpec::dropout` was accepted and ignored.
  `inject_lora_dropout` now emits it as a host-supplied mask multiplied into
  the adapter branch only — no new op, so it works on every backend, and the
  caller keeps the RNG, which a reproducible run needs. Six tests cover where
  the mask lands (never the frozen base path), that a zero mask removes the
  delta exactly, that an all-ones mask reproduces the undropped graph, and
  that `FuseMode::Fused` decomposes rather than silently dropping the dropout.
* **bf16 autocast** (`TrainConfig::bf16`). The policy machinery
  (`AutoMixedPrecision`, `ArenaWidthPolicy`) already existed; what was missing
  was a policy that is *correct*, and six backend bugs.

  Enabling BF16 one op kind at a time on a Qwen3.5 forward originally gave:
  Elementwise and Reduction finite, **Compute and DataMovement NaN** on CPU
  *and* Metal, with MLX the only backend that survived. That turned out to be a
  single root cause, not a numerical one — see **The bf16 NaN** below. All of it
  is fixed: every op kind and the full `AutoMixedBf16` policy now match the f32
  run on CPU, Metal and MLX (`max|Δ|` ≤ 4e-4, and 0 on Metal/MLX), reproduced by
  `examples/bf16_sweep.rs`:

  ```
                    CPU       Metal     MLX
  elementwise       0.0000    0.0000    0.0000
  reduction         0.0000    0.0000    0.0000
  compute           0.0000    0.0000    0.0000     <- was NaN on CPU + Metal
  datamovement      0.0004    0.0000    0.0000     <- was NaN on CPU + Metal
  AutoMixedBf16     0.0004    0.0000    0.0000     <- was NaN on CPU + Metal
  ```

  **Training still takes `AutoMixedBf16Safe`** (BF16 in Elementwise and
  Reduction only), for two measured reasons rather than a crash:
  under the full policy the head gradient's cosine against the f32 run drops to
  **0.83**, against **1.00000** under the safe one; and CPU and Metal have no
  bf16 kernels at all — `cpu_low_precision::promote_to_f32` rewrites bf16
  *arithmetic* to f32 before planning — so the full policy there costs accuracy
  and saves nothing. With the safe policy, bf16 training reproduces the f32 run:
  **loss 0.5494 vs 0.5494, gradient cosine 1.00000**, and `Param` nodes stay
  F32 — torch's autocast contract, fp32 master weights included.

  #### The bf16 NaN

  One cause, six symptoms. A low-precision tensor lives in the arena in one of
  two layouts — PACKED at 2 B/elem or WIDENED to 4 B — and which one it gets is
  a property of the backend's `ArenaWidthPolicy`, not of its dtype. Every kernel
  that touches it has to agree with the planner about which. They didn't:

  1. **CPU planner** claimed to widen half activations; it did not
     (`ArenaWidthPolicy::Native` packs them). Added
     `ArenaWidthPolicy::NativeHalfWidened` — params and int/bool native, f16/bf16
     activations widened — which is what a thunk executor that computes in f32
     and reads every activation slot as `&[f32]` actually needs.
  2. **CPU `Op::MatMul`** chose the packed dequant GEMM from `Param`-ness and the
     declared dtype. Now driven by the real slot width, with a new
     `Thunk::SgemmHalf` for any operand or destination that is genuinely packed.
     A node with *no* buffer is no longer treated as packed — that was the
     backward graph's "range end index 16 out of range for slice of length 0".
  3. **Metal planner**: same problem, narrower fix. Metal has real `half`
     kernels but `HalfFlag` has no BF16 lane, so *every* Metal kernel reads a
     bf16 tensor as f32 → `ArenaWidthPolicy::NativeBf16Widened` (bf16
     activations only; f16 stays packed).
  4. **Metal `Op::Cast`** to/from BF16 went through a host cast that decoded the
     now-4 B words as packed bf16 pairs. This is the one that made the bug
     legible: every *odd* element of the result was exactly right and every
     even one was mantissa junk. Now rounds in place (a new `round_bf16_f32`
     MSL kernel, the GPU twin of the CPU's `RoundHalf`) or plain-copies.
  5. **Metal MPSGraph** would have been handed `MPSDataTypeBFloat16` for an
     f32-wide buffer, so it now refuses a graph with bf16 activations
     (`graph_has_bf16_activation`) — no fast path is lost, since there are no
     bf16 Metal kernels to lose.
  6. **Metal arena** derived element counts by dividing the slot size by the
     dtype width, which double-counts a widened slot. Now taken from the shape.

  Two latent bugs fell out on the way: Metal's `encode_cast` same-dtype branch
  bound `copy_f32` as if it took two pre-offset buffers when it takes
  `(arena, src_off, dst_off, len)` — it silently wrote **zeros**; and the
  widening has to skip tensors that `Op::Custom` and `Op::KvAppend` address at
  their declared dtype (plus their whole alias families, since a view and its
  root share bytes).

  The NaN surfaced at an innocent `Softplus` whose own input was intact when
  evaluated on its own, which is why `examples/bf16_probe.rs` re-roots the graph
  at one node at a time: the first node that is *wrong* is rarely the first one
  that *looks* wrong.

  Pinned in `rlx-runtime/tests/bf16_activation_layout.rs` (the AMP graph tracks
  f32 on every backend; a bf16 op does not clobber its neighbour; packed bf16
  params still round-trip) and `cpu_half_activation_layout.rs`.

  **Backend support.** The three additions were checked against CPU on every
  backend available here
  (`rlx-runtime/tests/training_features_backend_parity.rs`):

  | | CPU | Metal | MLX | wgpu | CoreML/ANE |
  |---|---|---|---|---|---|
  | LoRA dropout mask | ✓ | ✓ | ✓ | ✓ | ✓ |
  | Checkpointed backward | ✓ | ✓ | ✓ | ✓ | ✓ |
  | bf16 autocast (safe) | ✓ | ✓ | ✓ | ✓ | ✓ |
  | bf16 matmul on activations | ✓ | ✓ | ✓ | ✓ | ✓ |
  | full `AutoMixedBf16` forward | ✓ | ✓ | ✓ | – | – |
  | **bf16 *training*** | ✓ | ✓ | ✓ | – | – |

  bf16 training (forward *and* backward) reproduces the f32 run on **CPU, Metal
  and MLX** — MLX included since the two upstream fixes above; it used to die in
  the backward on `nelems doesn't match shape product`. CUDA, ROCm and Vulkan are
  untested (no hardware here).

  **What it does not do yet: save memory.** 116 GB → 134 GB on the real training
  graph. On CPU and Metal that is now expected rather than a puzzle: neither has
  bf16 kernels, bf16 arithmetic is promoted to f32 before planning, and bf16
  activations are deliberately stored f32-wide — so the inserted casts add
  buffers and nothing shrinks. The win is not a CPU/Metal kernel project after
  all; it is MLX's backward.
* **Gradient checkpointing** (`rlx-autodiff::checkpoint`). Recomputes
  non-boundary activations from the nearest boundary instead of holding them
  across the backward. On a 32-deep chain the planned arena falls **143 KB →
  32 KB**.

  The honest caveat, measured: on *this* model it is a **loss** — 116 GB →
  149 GB — because a residual trunk's skip connections make most nodes cross a
  segment boundary and get kept anyway, so the recomputed duplicates are pure
  addition. Memory is also U-shaped in the segment count, not monotone
  (143 KB → 32 KB at 8 segments → 75 KB at 32). So the trainer plans both and
  keeps whichever the memory planner says is smaller; it is never enabled on
  faith. Segmenting along the model's layer structure rather than node index
  is what would make it pay on transformers, and is not implemented.

### kev-4b specifically

Not runnable here. The released kev-4b is two runs — a v7 base run plus a
delta pass (`--init_from jaredpalmer/kev-4b --data night2/dates_unknowable.jsonl
--replay 2000`) — and 4B in f32 needs roughly 16 GB of weights, a second 16 GB
`params` copy and a third for the arena before any activations, against 64 GB
of shared RAM. The reference trained it in about an hour on an H100 in bf16
with gradient checkpointing. The recipe above runs at 0.8B, which is the size
this machine can hold.

## State-prefix caching

In the row form every question recomputes the state, and a repeated state
recomputes it again on every request. `--prefix-cache [N]` splits the forward
into a state pass that exports the trunk's recurrent + KV state and a
per-question continuation that resumes from it:

| 625-token state, 3 questions, Metal, warm | latency |
|---|---|
| without | 1461 ms |
| with | **265 ms** |

Answers are unchanged — max \|Δp\| vs the uncached path is **1e-6**, and the
uncached path is itself parity-exact against the reference.

Runtime opt-in, because each compiled shape owns a parameter arena and this
adds two more. Gated on a minimum state length (default 384 tokens, kev's own
floor) since the split does not pay for short states. The continuation runs one
question at a time: at `batch > 1` with `seq > 1` the decode flow panics
reshaping the GDN conv state (`[4,4,3,6144] -> [4,3,6144]`), and batching rows
would only amortize weight reads anyway.

## Memory

Peak RSS for Kev-0.8B, f32:

| phase | peak |
|---|---|
| after load | 5.2 GB — mmap (1.7) + f32 bundle (3.5) |
| after forward | **11.8 GB** (was 13.3) |

`load` installs a reloader so the ~3.5 GB weight bundle is dropped after each
graph build rather than held for the model's lifetime; a reshape re-reads it
from disk (~1.5 s) instead. That is the 1.5 GB saving.

What remains is structural: `rlx-qwen35` materializes the bundle, then returns
a second full `params` copy, and the compiled arena makes a third. Collapsing
that needs a streaming parameter path in `rlx-qwen35` — not something this
crate can do from outside. 4B and 9B want a large-memory host.

### Not implemented

* **The packed form** — all questions in one sequence under an additive
  block-causal mask. It is only reachable on attention-only Qwen3-generation
  bases (which kev no longer develops), and needs `MaskKind::Bias` threaded
  through the `rlx-qwen35` prefill graph, which today exposes a mask input on
  the cache/verify path only. Its one real value here was as a cross-check of
  the row form — and end-to-end parity against the reference, plus the
  `separate`-vs-packed agreement, subsumes that. The encoder already emits the
  `seg` / `opt` vectors it would need.
* **Quantized or bf16 arenas** — see the RSS note above.

