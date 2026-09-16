# rlx-jlens — Jacobian lens

Read out what an internal activation is disposed to make the model say.

```text
lens_l(h) = unembed( J_l · h ),   J_l = E[ ∂h_final / ∂h_l ]
```

`J_l` is one `[d_model, d_model]` matrix per layer, estimated as the average
input–output Jacobian over a corpus. It is prompt-independent: fit once, apply
anywhere. Unlike a logit lens — which decodes a mid-layer residual as if the
remaining layers were the identity — the transport accounts for what the rest of
the stack would have done to it.

A native RLX port of the reference implementation accompanying *Verbalizable
Representations Form a Global Workspace in Language Models*
(`../jacobian-lens`, Apache-2.0).

**In a hurry?** `examples/lens_readout.rs` is the shortest text demo (below);
[`examples/vl_report.rs`](#one-command-for-all-of-it) is one command that fits a
vision-language model and writes every figure — per-layer attention, per-word
image masks, patch segmentation, and the transported word readout.

## It works

`examples/lens_readout.rs` runs a prompt through the lens and prints the top word
per layer, with the plain logit lens beside it — the difference between the two
columns is what the transport buys.

```bash
cargo run -p rlx-jlens --features qwen35,qwen35-tokenizer,metal --release \
    --example lens_readout -- --device metal --every 2 \
    --prompt "Fact: The capital of Japan is Tokyo. Fact: The capital city of France is"
```

```text
layer   jacobian lens    logit lens       top-k (jacobian lens)
────────────────────────────────────────────────────────────────────────────────
 8       ____            likewise       " ____" " Paris" "...." "____" " franc"
10      ....            其中之一          "...." " ____" " ...." " Paris" " capitals"
14      .").            门户             ".\")." " Paris" "”." "**." "东京"
16       Paris          capital        " Paris" " París" "巴黎" "Paris" " paris"
18       Paris          capital        " Paris" "巴黎" "Paris" " París" " paris"
20       Paris           Paris         " Paris" "巴黎" "Paris" " paris" " París"
22       Paris           Paris         " Paris" " paris" " Versailles" " Lyon"
────────────────────────────────────────────────────────────────────────────────
23*      Paris                          model's own output
```

" Paris" is already in the top-5 at layer 8, is top-1 by 16, and by 18 shows up
as the same concept in several surface forms — "巴黎", "Paris", " París",
" paris". The plain logit lens is still saying " capital" at 18 and only catches
up at 20. That gap is the whole claim: the transport reads out what the
activation is *disposed* to make the model say, not what it would decode to if
the remaining layers were the identity.

## It generalizes

`J_l` above is fitted on the prompt being read, which is circular for any claim
about generalization. Averaging over a corpus is what makes the lens
prompt-independent, and it holds up:

```bash
# Fit once — resumable, and shardable via JacobianLens::merge.
cargo run -p rlx-jlens --features qwen35,qwen35-tokenizer,metal --release \
    --example fit_corpus -- --device metal --corpus ./docs --prompts 24 \
    --seq 48 --every 2 --checkpoint fit.ckpt --out qwen35.lens.safetensors

# Apply it to prompts it was never fitted on.
cargo run -p rlx-jlens --features qwen35,qwen35-tokenizer,metal --release \
    --example lens_probe -- --device metal --lens qwen35.lens.safetensors \
    --prompt "Fact: The largest planet in our solar system is"
```

A lens fitted over 24 prompts of **this repository's own markdown** — no capitals,
no astronomy — reads factual recall out of prompts from neither domain:

| held-out prompt | model's answer | lens top-1 from | logit lens at 18 |
|---|---|---|---|
| `…capital city of France is` | " Paris" (p 0.99) | **layer 18** | " capital" |
| `The Eiffel Tower is located in the city of` | " Paris" (p 0.52) | **layer 20** | " cities" |
| `Fact: The largest planet in our solar system is` | " Jupiter" (p 0.29) | **layer 18** | " planets" |

That is the prompt-independence claim: the transport was estimated from unrelated
text and still reads out what these activations are disposed to produce, several
layers before the logit lens does.

`--lens` skips fitting entirely — reading through a fitted lens is a forward pass
and a `d × d` matvec, so it runs at batch 1 in about a second. The file is
safetensors keyed `J.{layer}`, so the Python reference can load one fitted here
and vice versa.

## What you can read out of a layer

`examples/lens_probe.rs` is the instrument. Beyond the top word it reports, per
layer, the rank and probability of the model's own final answer, the entropy of
the lens distribution, `KL(lens ‖ final)`, and a **layer × position** grid.

```text
layer  jacobian lens top-1   logit lens top-1   p(answer) rank    entropy  KL→final
16     " .**"                "…**"              0.0155    9         9.47     8.44
18     " planet"             " planets"         0.1393    1         4.87     7.33
20     " Jupiter"            " planets"         0.6302    0         2.32     2.67
22     " Jupiter"            "有一颗"             0.5014    0         2.74     0.76
```

Two things fall out of reading it this way. The collapse is **abrupt** — entropy
9.5 → 4.9 → 2.3 bits and KL 8.4 → 0.8 across three tapped layers, not a gradual
slide. And the lens reads the **category before the instance**: " planet" at
layer 18 (already rank 1 for " Jupiter"), " Jupiter" at 20; " cities" at 16–18
then " Paris" at 20. The position grid shows the same answer appearing at earlier
positions before the one that finally predicts it.

### How many prompts does a fit need?

`fit_corpus` prints `mean shift`, the relative movement of the running mean. That
is *convergence*, not quality — an estimator can settle onto a poor answer, and
this one partly does. `examples/eval_lens.rs` measures quality directly, on ten
held-out prompts, against the only ground truth that needs no labels: the token
the model itself goes on to emit. Snapshots come from `fit_corpus --snapshots`.

```text
              layer 12        layer 18         layer 22
prompts     rank     KL     rank     KL    agreement    KL
      1     1071   11.48       2    6.84      8/10     0.79
      4      356    9.62       2    6.27      8/10     0.71
     16      161    9.12       1    5.35      8/10     0.61
     32      277    8.72       2    5.37      8/10     0.60
     64      321    8.94       2    5.49      9/10     0.61
    160      291    8.97       1    5.65      9/10     0.61
```

KL is the signal to read; median rank over ten prompts is too coarse to resolve
much past the first few. By that measure the fit **converges by 16–32 prompts**
and 160 buys nothing further.

That is not what a smaller corpus suggested, and the difference is the more
interesting result. An earlier 32-prompt fit over `docs/` (127 KB, 5 files) was
still visibly improving; the run above uses `crates/` (1.4 MB, 335 files). At the
*same* 32 prompts:

| corpus at n=32 | layer 12 KL | layer 18 KL | layer 22 KL | layer 12 rank |
|---|---|---|---|---|
| `docs` — 127 KB | 9.88 | 6.10 | 0.65 | 816 |
| `crates` — 1.4 MB | **8.72** | **5.37** | **0.60** | **277** |

So corpus **diversity** buys more than prompt **count**: 32 prompts drawn from a
wide corpus beat 32 drawn from a narrow one on every layer, and adding prompts
from the narrow corpus was chasing a ceiling the corpus itself imposed. `J_l` is
an expectation over the input distribution, so a corpus that samples it narrowly
converges quickly to the wrong thing — which `mean shift` cannot tell you,
because a narrow corpus makes the running mean settle *faster*.

### Long context

`J_l` is `[d_model, d_model]` — it has no sequence axis. Positions are averaged
over during fitting, so **fitting is sequence-bounded but reading is not**: a
lens fitted at `seq` 48 applies to a context of any length.

That matters because the two costs are wildly different. Fitting keeps the
delta-net state history, `(seq + 1) · n²` per `(batch, head)`, so its memory is
linear in `seq` and its time grows faster (24 → 48 measured 2.3×). Reading is one
forward, a `d × d` matvec per probed position, and an unembed — and `lens_probe`
decodes only the positions it reports, because at 152k vocabulary the decode
otherwise dominates everything else.

Retrieval over a 369-token needle-in-a-haystack, read with the `seq` 48 lens
above — 7.7× the length it was fitted at:

```text
layer   jacobian lens top-1   logit lens top-1   p(" dragon")  rank    entropy
8       " bot"                "叫做"                0.0000       7809     11.94
12      " lion"               "叫做"                0.0002        569     11.87
16      " **"                 "～。"                 0.0474          3      7.16
18      " dragon"             " **.**"             0.3314          0      7.50
20      " dragon"             " **.**"             0.9861          0      0.13
```

The needle (`The vault is guarded by a dragon.`) sits ~350 tokens back behind
filler. Layer 12 reads **" lion"** — the right *category* of answer, a guardian
animal, before the specific fact is retrieved at 18. The logit lens is noise
until 22.

One honest caveat: fitting at short `seq` estimates the average Jacobian from
short-range behaviour. It transfers well here, but that is an empirical result on
this model, not a guarantee — if you care about long-range readouts, fit at a
longer `seq` and compare. Note also that `SKIP_FIRST_N_POSITIONS = 16` (attention
sinks) discards a third of a 48-token fit and only an eighth of a 128-token one.

## Why RLX makes this cheap

The Python reference needs forward hooks and a retained autograd tape. Here the
model *is* a graph, and two `rlx-autodiff` primitives cover the whole job:

- `grad_with_loss_wrt` exposes `d_output` as a real graph input shaped like
  `outputs[0]`, so a VJP can be seeded with an **arbitrary cotangent** rather
  than a scalar `1.0`.
- `Wrt::Output(i)` addresses an **intermediate** activation across the
  renumbering that autodiff preparation performs — publish a residual as an
  auxiliary forward output and name it by index.

Together: a VJP at any cut point, with any seed, no hooks and no tape.

## Swapping models

Nothing in the core knows what a Qwen is. Implement `model::LensModel` — hand
back a graph whose input is a residual stream and whose output is a residual
stream — and the estimator, fitting loop and readout work unchanged.

```rust
pub trait LensModel {
    fn name(&self) -> &str;
    fn n_layers(&self) -> usize;
    fn d_model(&self) -> usize;
    /// One residual block: residual in → residual out.
    fn block(&self, layer: usize, batch: usize, seq: usize) -> Result<BlockGraph>;
    /// The whole stack, tapped. Defaults to `Unsupported`.
    fn stack(&self, source_layers: &[usize], target_layer: usize,
             batch: usize, seq: usize) -> Result<StackGraph>;
    /// Final norm + LM head, for decoding a transported residual.
    fn unembed(&self, rows: usize) -> Result<UnembedGraph>;
}
```

Implementations live in `models/`, each behind its own feature, so the core
crate depends on no model crate. Adding a model is a module plus a feature.

```bash
cargo test -p rlx-jlens --features qwen35   # hybrid delta-net/attention, GGUF
cargo test -p rlx-jlens --features qwen3    # dense attention-only, HF safetensors
```

There are two, deliberately: a trait with one implementor is an untested guess.
`models::qwen35` is a delta-net/attention hybrid loaded from GGUF and handed
materialized weights; `models::qwen3` is a dense attention-only decoder that
opens HF safetensors itself. Between them they exercise both axes the interface
was meant to abstract, and adding the second immediately found a hidden
assumption: `residual_stream` walks back from the graph output, which is a
residual in the Qwen3.5 prefix builder but the *final norm* in the Qwen3 full
trunk, so the chain came back one node long. `Qwen3LensModel::stack` steps past
a trailing norm before walking.

`tests/lens_model_api.rs` drives only the trait against synthetic weights;
`tests/qwen3_lens_model.rs` drives it against a second, structurally different
model on a real checkpoint.

### Vision-language: the same lens, no new machinery

`models::qwen25_vl` (feature `qwen25-vl`) taps a Qwen2.5-VL. A Qwen-VL projects
image patches to the LM's hidden width and splices them into the token
sequence, so image content travels the **LM's** residual stream as ordinary
positions — the stream already shown to be identity-dominated, and the one the
lens is known to work on. Nothing about the estimator changes. The readout is
the LM's own vocabulary, so there is no candidate caption list to pick: it says
what the model is disposed to *say*, not which of your words a patch matches.

Two things the interface had to grow, both real:

* **`StackGraph::extra_feeds`.** A trunk is not always a function of its token
  sequence alone. Qwen2.5-VL takes mRoPE `cos`/`sin` as graph *inputs*, because
  they depend on where the image sits in the prompt. They ride along and are
  bound on both halves of the split.
* **f32 lens files.** `save()` emits f16 only when every entry fits; DINOv3's
  `J` reaches 1e10 and silently became `inf`.

`examples/vl_lens.rs` runs it; `--no-fit` does forward + untransported readout
only, and `--ref` checks the whole path against `scripts/qwen25vl_reference_mm.py`.

#### One command for all of it

`examples/vl_report.rs` produces everything the lens and the attention probe can
say about one image, from a single model load and a single fit:

```bash
cargo run -p rlx-jlens --features qwen25-vl,metal --release --example vl_report -- \
    --device metal --out ./report \
    --mmproj .../mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
    --image crates/rlx-locateanything/fixtures/sample.jpg
```

| output | what it shows |
|---|---|
| `1_attention_by_layer.png` | where the answer position looks, **every** layer, shared scale |
| `2_mask_<word>_by_layer.png` | where each probe word is supported, per fitted layer |
| `report.txt` — attention table | `img%`, peak patch, top-third share per layer |
| `report.txt` — transported top-1 | the Jacobian lens proper, at every **text** position |
| `report.txt` — segmentation | which probe word each patch supports most |

Flags worth knowing: `--words " crowd, man, hand"` picks the probes (the leading
space matters — `" man"` and `"man"` are different tokens, and a continuation
would use the first), `--every` sets the layer stride for the fit, `--dim-batch`
trades memory for passes.

Two things about its shape are deliberate rather than oversights. **Attention
covers every layer and the lens covers every `--every`-th**, because attention is
one forward and effectively free while a fit costs `d_model / dim_batch` backward
passes. And there is an explicit `drop(runner)` before the fit: the runner holds
a full f32 copy of the LM, and leaving it alive while `StackLens` builds its two
arenas puts three copies of a 3B model in memory at once — which the OS kills
outright, with no panic and no allocation error.

### What porting to VL found

The VL trunk answered `<|im_end|>` to every image and gibberish to text. CPU and
Metal agreed on the garbage, which rules out a kernel, and the crate's own CLI
was equally broken — so this was never the lens.

The method that localized it in one pass: dump HF's per-layer hidden states
(`scripts/qwen25vl_reference.py`), feed the reference's **own embedding output**
in as `prefill_hidden` so the trunk is isolated from the tokenizer and the
embedding table, tap every layer exit, and compare
(`examples/vl_bisect.rs`). Layer 0 diverged with the *right* magnitude — rms
0.575 against 0.577 — and the wrong direction, cos 0.78. Right magnitude, wrong
direction, at the very first block, is a missing term inside the block, not
plumbing.

Three bugs in `rlx-qwen25-vl`, each silent:

1. **`attention_bias` defaulted to `false`.** A Qwen 2 attention block applies
   Q/K/V bias unconditionally — HF has no flag for it, so `config.json` says
   nothing while the checkpoint ships `self_attn.{q,k,v}_proj.bias`. The graph
   loaded the weights and dropped all three. It sits one line from the
   `qk_norm = false` fix, which is the same omission with the opposite sign.
2. **`image_min_pixels` defaulted to 1024 *tokens*.** No mmproj GGUF on the Hub
   carries `clip.vision.image_min_pixels`, so the default always decides, and a
   1024-token floor is not a floor but a target: `smart_resize` upscales
   anything below it. A 640x360 photo that HF turns into 299 tokens became
   1032, and every vision position sat at the wrong scale.
3. **`qwen25_vl_chatml` dropped every `<|im_end|>`**, closing turns with a bare
   newline. The model then reads one unterminated turn, and the likeliest
   continuation of an unterminated turn is to terminate it.

After the fixes, against HF on the same checkpoint:

```text
text-only     logits cosine 1.000000   top-1 " Paris" 22.0098 vs 22.0098
image prompt  logits cosine 0.999357   top-1 "The", 324 positions vs 324
```

The remaining 6e-4 is the f16 GGUF vision tower against HF's f32 safetensors.

None of this was caught because **every `rlx-qwen25-vl` test is synthetic**, and
synthetic weights cannot expose a dropped bias or a wrong pixel budget: both
sides of a self-consistency check are built from the same wrong config. A lens
port is a good way to find this class of bug precisely because it forces a
comparison against an outside implementation.

### The VL lens, fitted

`Describe the image.` over a 448-side photo — 169 positions, 144 of them image
patches — six layers, `dim_batch` 4, one prompt, 497s on Metal. The model's own
next token is `"The"`; the table reads that token's rank out of each layer.

```text
layer   logit lens        Jacobian lens     rank of the answer
0       "讣"               "XYZ"              44
6       "').\""           "指导意见"           37
12      "遨"               ".\n"              10
18      "洇"               "You"               2
24      "在一个"            "The"               0
30      "抱歉"             "Unfortunately"      5
```

The logit lens never arrives; the Jacobian lens has the answer top-1 by layer
24. That is the paper's claim reproduced on a vision-language model, and from a
**single-prompt** fit — stronger than the text-only result led me to expect,
where the mid-stack was still moving at 32 prompts.

### Attention is a different question from the lens

The lens says what a position is disposed to make the model **say**. Attention
says which positions it **reads** when it says it. `examples/vl_focus.rs` scores
the answer query against every key using the model's own post-RoPE Q and
GQA-expanded K — one forward, no fit, so all 36 layers are affordable.

```text
layer   img%   peak(x,y)   top⅓        layer   img%   peak(x,y)   top⅓
0       3.7%      (1,0)     54%        26       9.5%     (1,0)     43%
9      10.0%     (15,0)     75%        27      10.8%     (9,6)     46%
19     14.2%      (1,0)     72%        33      19.7%    (11,1)     39%
                                       35      22.0%     (8,4)     33%
```

Through layer 26 the peak is pinned at the ends of the *first patch row* —
positional extremes, not content, and the visual attention sits in the top third
of the frame. From layer 27 it relocates to centre and lower-centre, where the
subjects are, and the total visual budget roughly doubles. Attention sinks on
content-free image tokens are a known property of multimodal models, not
something this port introduced.

Set against the lens result above — the answer reaching rank 0 at layer 24 —
this is consistent with the readout committing first and attention then
gathering supporting detail. From one prompt that is a hypothesis, not a
finding.

### Three things that will make these figures lie

The first two were live in the first version of the contact sheet, and all three
are worth knowing before drawing any patch-grid map.

**Per-tile normalisation.** Scaling each layer's overlay to its own peak is
right for a single figure and wrong for a set: a layer holding 2% of the
attention and one holding 22% both saturate the ramp, so darkness — the
strongest visual channel — carries no information, and early layers read as
sharply focused when they are nearly empty. `heatmap::overlay_scaled` takes an
explicit range; `vl_focus` writes a shared-scale sheet **clipped at the 99th
percentile**, because scaling to the true maximum is the opposite failure (one
sink patch blanks every other tile). Both sheets are emitted — a faint layer's
*shape* is only legible under per-tile scaling.

**The patch→cell mapping.** Every spatial claim assumes token `p` is grid cell
`(p % gx, p / gx)`, and Qwen2.5-VL's tower permutes its sequence for window
attention. It does restore raster order: `vision/builder.rs:140` gathers into
window order before the blocks and `:206` scatters back after the merger.

Verifying that *empirically* is harder than it looks, and
`examples/vl_patch_order.rs` records three probes that each look like evidence
against the ordering while really being statements about the probe:

| probe | why it misleads |
|---|---|
| furthest from the batch mean | a ViT mixes patches, so the most unusual token need not be the one that saw the light |
| raw argmax of the per-token delta | the tower's own sink tokens absorb every perturbation and win regardless of where it was |
| z-scored delta | above chance (argmax 44/144 against 1/144; mean rank 17.7 against 71.5) but diffuse |

"Above chance but diffuse" is the expected signature of a *correct* mapping plus
real mixing — the tower runs a full-attention layer every `n_wa_pattern`-th
block, and a bright cell on grey shifts global image statistics. Read the graph
to decide; use the probe to catch a gross permutation, not to certify a fine one.

**Resampling.** `Upsample::Bilinear` between patch *centres* is the default: it
renders the measured field as the continuous quantity it samples, which is what
makes an overlay legible against a photograph rather than a mosaic. It adds no
information — every patch centre still carries exactly its measured value — but
it does hide where the samples are, so the grid size belongs in the caption.
`--patches` switches to `Upsample::Nearest` when the question is "which patch".
The centre offset is the part that is easy to get wrong: patch centres sit at
`((cx + 0.5) / gx, (cy + 0.5) / gy)`, so interpolating on raw cell indices
shifts the whole field by half a patch — 14 px here, enough to move a peak off
the object it belongs to. `bilinear_reproduces_patch_centres` pins it.

**Reading image patches as words does not work, and there is a reason.** Both
lenses decode vision positions to punctuation, newlines and stray identifiers
(`" \n\n"`, `":\n\n"`, `"?\n"`). This is not a transport failure. In a causal
LM the residual at a position is only ever trained to predict the *next*
position, and the next position at an image site is another `<|image_pad|>` —
the model is never asked for a meaningful next-token distribution there, so
there is no trained meaning for a readout to recover. The lens transports the
residual into the final-layer basis correctly; the basis at that position was
never supervised. Image content is legible where the model is actually asked to
speak about it, which is the text positions after the image.

## Status

**Verified against the Python reference.** `tests/reference_parity.rs` runs both
implementations on the same f32 checkpoint (`weights/Qwen3-0.6B`) and the same
token ids, and compares `J_l` entry for entry:

```text
J.3: relF = 2.77e-6  max|diff| = 8.53e-7  mean diag ref 0.4105 vs rlx 0.4105
J.4: relF = 2.02e-6  max|diff| = 7.45e-7  mean diag ref 0.5752 vs rlx 0.5752
J.5: relF = 1.33e-6  max|diff| = 1.01e-6  mean diag ref 0.7488 vs rlx 0.7488
```

That is f32 rounding — the estimators are numerically equivalent. It is also how
a real defect was found: the reference hooks each residual block's forward
**output**, while this crate tapped each block's **input**, so rlx's `J_l`
reproduced the reference's `J_{l-1}` — correct arithmetic under a
one-layer-shifted labelling, which no amount of self-consistency checking would
have caught, and which silently mislabels every layer in a lens file that is
meant to be interchangeable with the reference's. `layer_exit_taps` is the fix.

**Working and verified against finite differences:**

- The VJP-at-a-cut-point primitive (`rlx-autodiff`).
- The estimator: position selection, one-hot cotangent layout, VJP → rows of `J`.
- Per-block Jacobians through `LensModel` (`BlockLens`).
- **Whole-stack Jacobians** (`StackLens`): residual-stream tap discovery, one
  backward yielding every layer's `J_l` at once. Checked by tapping the last
  layer, where `J` reduces to a single block's Jacobian that `BlockLens`
  computes by an independent route — they agree to **6e-8**.
- The readout: transport through `J_l`, decode with the model's own unembedding.
- Qwen3.5/3.6 blocks end to end — **both** full-attention and gated-delta-net.

**Verified upstream as a side effect** — each op the lens depends on, isolated
and checked against finite differences in `../rlx/crates/core/rlx-autodiff/tests/`:
`attention_causal_vjp_fd`, `rope_vjp_fd`, `rms_norm_rank_vjp_fd`,
`gated_delta_net_vjp_fd`, `depthwise_conv1d_vjp_fd`. All pass.

### On real weights

`tests/qwen35_real_weights.rs` runs against **Qwen3.5-0.8B** (24 blocks,
`d_model` 1024, 6 attention / 18 gated-delta-net). It skips itself when the
checkpoint is absent; point `RLX_JLENS_QWEN35_GGUF` at a `.gguf` or drop one in
`weights/Qwen3.5-0.8B-gguf/`.

| block | passes | time | mean diagonal | max off-diagonal | ‖J‖/√d |
|---|---|---|---|---|---|
| layer 0, gated-delta-net | 64 | 8.5 s | 1.0073 | 0.0260 | 1.0079 |
| layer 3, full attention  | 64 | 1.3 s | 1.0352 | 0.1009 | 1.0590 |

The mean diagonal sits at ~1.01–1.04, which is what a residual block's Jacobian
should look like — `J ≈ I` plus the block's own derivative. Attention carries
**35× more off-diagonal energy** than the delta-net (0.0462 vs 0.0013 relative
to the diagonal), matching what the two mechanisms do.

`real_weight_jacobian_matches_finite_differences` is the correctness statement:
two full columns of `J`, **2048 entries checked against central differences,
worst delta 1.32e-4**, with **19/19 source positions usable** — no
ill-conditioning at all. That last number is worth noting on its own: the
finite-difference trouble documented below is entirely an artifact of synthetic
ramp weights, and does not occur on a trained model.

The gated-delta-net block was **19× slower** than attention (47 s vs 2.5 s)
until this crate's profiling traced it to a missing VJP: autodiff was unfusing
`Op::GatedDeltaNet` into 585 per-timestep primitives, ~32× slower than its fused
kernel. Adding `Op::GatedDeltaNetBackward` upstream took the block to 8.4 s and
the gap to 3.1×. See the changelog entry in `../rlx` for the measurements.

### Timing

`BlockLens::timing()` breaks a fit into compile / forward / bind / replay, with
run counts. That split is the point of `rlx_autodiff::split_vjp`: the forward
should run **once per residual**, the gradient half once per pass. If the
forward count ever tracks the pass count, the split has stopped working — the
real-weight tests assert exactly that.

```text
attention, CPU   compile  617 ms | forward   35 ms ×1 | bind   7 ms | replay  1276 ms ×64
attention, Metal compile 1129 ms | forward 3015 ms ×1 | bind   8 ms | replay   287 ms ×64
delta-net, CPU   compile  626 ms | forward  151 ms ×1 | bind   8 ms | replay  8349 ms ×64
```

The split cut the attention block 2.7 s → 1.3 s. It is roughly neutral on the
delta-net block, and the breakdown says why: once `Op::GatedDeltaNetBackward`
made the forward cheap, there was little forward left to hoist — 151 ms against
8.3 s of gradient work.

### Why a whole-stack fit costs what it costs, and what actually helps

A whole-stack fit of Qwen3.5-0.8B is ~44 s per prompt at `seq` 48 on Metal, and
almost all of it is the delta-net backward's state history.

Reconstructing states backwards would mean dividing out `exp(g) < 1`, which
amplifies rounding without bound, so the kernel keeps every state entering every
timestep — `(seq + 1) · n²` floats per `(batch, head)`. The reverse scan then
sweeps that history repeatedly, so the obvious lever is to sweep it fewer times.
Fusing the phases — `dS` updated and consumed in one pass rather than three,
`dk`/`dg`/the `dS` decay sharing a single read of `S_{t-1}`, `v − m` recorded by
the forward, and `A` folded into both its consumers instead of materializing
`P = A ⊙ S` — took 17 n²-tile passes per timestep down to 10, parity unchanged
(worst |CPU − Metal| = 5.96e-8).

**Counting those tile passes as memory traffic badly mispredicts the speedup**,
and that is the useful thing to know here. 17 → 12 was worth 1.21× (62.3 s →
51.6 s); 12 → 10 was worth nothing at all (51.6 s → 51.3 s). The reason is that
one `(batch, head)` tile is only `n² · 4` = 64 KB, so at modest batch the tiles a
sweep touches are *cache-resident* — removing such a sweep removes instructions,
not DRAM traffic. What is genuinely unavoidable is the history itself: written
once by the forward, read twice per timestep by the reverse.

The measurement that shows it is a `dim_batch` sweep, where **total work is
identical at every setting** — each fills the same `d` rows of `J`:

| `dim_batch` | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| fit time | 44.6 s | **44.1 s** | 49.9 s | 57.3 s | ~79 s |
| live n² tiles | 4.2 MB | 8.4 MB | 16.8 MB | 33.6 MB | 67 MB |
| two identical fits differ by | — | **0.003** | 0.005 | — | **0.154** |

Constant work, rising time: the cost tracks the *live working set*, flat while
the tiles fit in cache and climbing once they spill past ~8 MB. So `dim_batch`
defaults to 8, and 62.3 s → 43.5 s came about a third from the kernel fusion and
the rest from simply not spilling.

The remaining forward re-scan is repeated identically by every replay pass.
`split_vjp` cannot hoist it, because the recomputation is *inside* the kernel;
having `Op::GatedDeltaNet` emit its history as a saved activation would remove
it, at the cost of changing the forward op's contract.

### `dim_batch` does not do what it looks like

Raising `dim_batch` looks like free occupancy — 16 passes of batch 64 instead of
64 passes of batch 16, the same total work. It loses on both axes. *Slower*, per
the table above. *Noisier* — at the time this was measured, Metal's backward was
nondeterministic (a missing threadgroup barrier, since fixed; see the open items)
and `StackLens` multiplied it across 24 blocks. The timings still stand.

This masqueraded convincingly as a correctness bug: `J` fitted at `dim_batch` 64
differed from `J` at 16 by 13.6%, and `dim_batch` is pure scheduling, so `J` must
not depend on it. It doesn't. CPU — where every op is deterministic — is
**bit-exact** invariant at block *and* whole-stack level, and two *identical*
batch-64 Metal runs differ by 15.4%, i.e. more than the cross-batch comparison
did. `tests/dim_batch_invariance.rs` pins all of this down: strict equality on
CPU, a documented reproducibility bound on Metal.

Worth keeping in mind when reading a fitted lens: a corpus fit averages over
prompts, so zero-mean noise falls off as `1/sqrt(n)` — 0.5% over 100 prompts is
0.05%. At `dim_batch` 64 it would still be 1.5%.

### On GPU

`RLX_JLENS_DEVICE=metal` (or `mlx`, `cpu`) selects the backend; build with the
matching feature.

**Metal works and agrees with CPU exactly** — same mean diagonal, same
off-diagonal, and the finite-difference check passes with the same worst delta
of 1.32e-4. With the pipeline warm:

| block | CPU | Metal |
|---|---|---|
| gated-delta-net | 8.5 s | **1.5 s** |
| full attention  | 1.3 s | **0.3 s** |

A first-touch Metal run pays ~1–3 s of pipeline compilation on top, so a single
block understates it.

One GPU note:

* `Op::GatedDeltaNetBackward` is claimed by **CPU, Metal and MLX**, so none of
  the three needs a fallback. A backend without it falls back to the unrolled
  decomposition — correct (identical numbers) but slow, and needing 327 saved
  activations instead of 42 — selected by `RLX_GDN_UNFUSE_FOR_AD=1`. That is
  still a process-global env var rather than something derived from the target.
  The generic `decompose_backward_ops_except` cannot absorb it, because the
  fused/unfused choice has to be made *before* autodiff runs; doing it properly
  means emitting the unrolled reverse scan in IR. No backend the lens currently
  runs on needs it.
* **MLX works** and agrees with CPU to ~1e-7 on real weights. It did not when
  this crate was written — the `[transpose] Received 2 axes for array with 3
  dimensions` failure was a genuine MLX rank-reconciliation gap, since fixed
  upstream along with a partial-RoPE gradient bug. See the open items.

## The bug this crate found

Building the lens surfaced a real defect in the fusion pipeline, now fixed
upstream in `rlx-fusion`.

`Rewriter::copy_node` re-copied nodes that `ensure_mapped` had already hoisted
to a fusion site, leaving two `Op::Param` nodes for one weight. Binding is by
name and reaches a single node, so the duplicate kept the arena's zeros and
every value through it vanished — no shape error, no missing-input error.

A **forward** graph never showed it: the fused-away matmul was the weight's only
reader, so the duplicate was dead code. A **backward** graph did — the mirrored
forward and `dX = dY · Wᵀ` both read the weight — so any backward graph whose
forward had several matmuls sharing an input (a fused QKV or in-projection, i.e.
most decoders) silently lost gradient terms.

In a gated-delta-net block that erased *all* cross-position gradient while the
same-position gradient stayed exactly right, which is why it took a
position-structured probe to see. `copy_node` is now idempotent; `Rewriter`
backs ten fusion passes, so the fix is not specific to one of them.

`rlx_ir::verify_unique_leaf_names` was added to catch the class. It is opt-in
rather than part of `verify`, because two other graphs still trip it:
`jvp(hvp(f))` emits a second `tangent_x` — which is *why* that composition
returns zero instead of the third derivative — and Qwen3.5 speculative-decode
graphs emit `last_token_idx` twice. Both are worth their own look.

## Backends

CPU is the arbiter — it is what the Python reference was checked against — and
every accelerator is asserted against it by `tests/dim_batch_invariance.rs`
(`assert_matches_cpu`, bound 1e-5).

| backend | synthetic | real Qwen3.5 weights |
|---|---|---|
| Metal | 6e-9 | 1e-7 |
| MLX | 6e-9 | 1e-7 |
| CUDA | 7e-9 | 1e-7 (delta-net; needs `RLX_CUDA_NO_TF32=1`) |
| ROCm | 8e-9 | 1e-7 (delta-net) |

Getting there found six backend bugs, none of them in the lens and none visible
from a forward pass:

* **Metal — 11 missing `threadgroup_barrier`s.** A reduction leaves its result
  in `partial[0]`, every thread reads it, and the kernel reuses `partial` for a
  second reduction with nothing in between. Affected `rms_norm_bwd`, three
  `softmax_lastax*`, `layer_norm_bwd`, `group_norm_bwd_input` and both
  AdaLayerNorm backwards; the cross-entropy softmaxes already had it. Those
  kernels also halved with `tsize / 2`, dropping the odd element for any row
  width that is not a power of two.
* **MLX — `Op::Transpose` did not reconcile rank**, so a matmul backward that
  addresses a `[B, S, K]` activation as its `[B·S, K]` matrix was rejected
  outright. MLX could not run these graphs at all before this.
* **MLX — `Op::RopeBackward` ignored head packing**, rotating head 0 and passing
  the rest through whenever `n_rot < head_dim`.
* **`rlx_runtime::supports()` returned `true` for every op** on the CUDA / ROCm /
  wgpu family, so it could not be used to decide anything; it now answers from
  each backend's `SUPPORTED_OPS`.
* **ROCm never promoted `AttentionBackward` to rank-4**, which its own kernel
  requires — a hard panic on any transformer block.
* **RoPE table row stride was guessed, differently, on each side.** CPU assumed
  `n_rot/2`, the shared CUDA/ROCm `rope.cu` assumed `head_dim/2`. The layout is a
  per-model choice — Qwen3.5 allocates `[max_pos, head_dim/2]` and uses the
  leading `n_rot/2` columns, DeepSeek-V4 MLA packs `n_rot/2` exactly — so each
  was right for one model and read a wrong-but-valid row for the other. Both now
  take it from the table's own last dimension, and they only ever agreed when
  `n_rot == head_dim`.

### Do the layer representations agree?

Parity on `J` does not by itself say the *readouts* agree, so
`layer_representations_vs_cpu` (`--ignored`, set `RLX_JLENS_DIAG_DEVICE`) takes
the forward on each device, transports and decodes it **on CPU**, and compares
per layer — any difference is then attributable to the forward alone.

| backend | worst residual relF | cosine | top-1 disagreements |
|---|---|---|---|
| Metal | 2.2e-6 | 1.000000 | 0/12 |
| MLX | 2.5e-6 | 1.000000 | 0/12 |
| CUDA | 2.2e-6 | 1.000000 | 0/12 |
| ROCm | **3.7e-1** (layers 0–20: 2.1e-6) | 0.931 at layer 22 | 0/12 |

So the representations are the same to ~2e-6 and every *decision* is identical —
same top-1 token and same rank of the answer at every layer, on every backend.
`lens_probe` prints byte-identical tables on CPU, Metal and MLX.

The ROCm exception was real and is now diagnosed. Its forward diverged at layer
22 (cosine 0.931), deterministically, while layers 0–20 agreed to 2e-6. A
node-level diff put it on the **last two of eighteen** `GatedDeltaNet` nodes,
whose inputs all agreed to 1e-6 and whose outputs were **exact zeros**; shrinking
the trunk to 20 layers made it vanish, so the fault followed the *end of the
graph*, not any layer. Cause: every ROCm `Step` carries `*_byte_off: u32`, and
this arena is **4,668,113,232 bytes** — past what a `u32` can address, so the
offsets of the final nodes wrapped and their kernels wrote elsewhere, leaving the
real destinations untouched.

`rlx-rocm` now refuses an arena over 4 GiB with that number in the message rather
than silently zeroing its tail. The proper fix is widening those offsets to
`u64`: the shared kernels already take `unsigned long long` (CUDA passes 64-bit
for 226 of its own fields), so it is host-side work — but it spans ~278 fields
and some shared kernels still take `unsigned int`, making it a cross-backend ABI
change rather than a local edit.

Note also that a *readout* needs only the forward, so CUDA and ROCm can apply a
fitted lens even though they cannot fit one (the `head_dim > 128` backward gap
below).

Two things are known-unfixed rather than fixed:

* **CUDA/ROCm `AttentionBackward` supports `head_dim <= 128`.** At 256 — which is
  Qwen3.5 — the kernels wrote nothing and returned exact zeros. They now refuse
  it with an explicit message instead, because a silent zero gradient is worse
  than a stop, and the parity test skips that shape with the reason.
  `rlx-cuda/tests/cuda_attention_backward_head_dim.rs` is the reproducer
  (`--ignored`): 32/64/128 match CPU to ~1e-6, 256 gives relF 1.0.
* **CUDA uses TF32 by default**, worth ~1e-4 relative on a delta-net block.
  `RLX_CUDA_NO_TF32=1` brings it to 1e-7. Fine for training, not for parity.

## Open items

1. **Fit scale.** The 32-prompt fit above took 23 min at ~40 s per prompt. Per
   the convergence measurement, that is plenty for deep-layer readouts and *not*
   enough for mid-stack ones. The reference uses 1000 sequences of 128 tokens;
   that is a ~12-hour run here, and the remaining headroom is the forward
   re-scan, not the arithmetic.

2. **(fixed) A fit held three copies of the model.** Both halves of a split VJP
   were compiled with `compile(graph, &params, device)`, binding from a
   *borrowed* map — so the host-side parameter set stayed alive while the save
   arena and the replay arena each filled with their own copy. For a 3B trunk in
   f32 that is ~33 GB resident before a single activation exists, and on a
   unified-memory machine the process was killed outright: no panic, no
   allocation error, just the fit stopping after it printed its timings.
   `compile_pair` consumes the map so each tensor drops as soon as both graphs
   have copied it, capping the peak at the two arenas plus one tensor. Measured
   on the configuration that used to die — 324 positions, `dim_batch` 4 — it now
   completes in 1215 s at **42.7 GB peak RSS** with the right answer.

   The same shape of bug bites callers: an example that fits and *then* builds a
   forward graph for the readout has two trunks live unless the fitted
   `StackLens` is scoped so it drops first.

3. **(fixed) MLX could not run these graphs.** It now does, and agrees with CPU
   to ~1e-7 on real Qwen3.5 weights for both block types. Two MLX bugs were in
   the way, neither specific to the lens:

   * `Op::Transpose` did not reconcile rank. A matmul backward addresses a
     `[B, S, K]` activation as the `[B·S, K]` matrix it multiplies and permutes
     *that*; CPU and Metal read the buffer either way, MLX rejected a 2-element
     `perm` on a 3-D array. `Op::Reshape` and `Op::Narrow` already reconciled the
     same way — this op just did not.
   * `Op::RopeBackward` ignored head structure. RoPE rotates the first `n_rot`
     channels of *every* head, but the lowering sliced the flat last axis at
     `n_rot`, rotating head 0 and passing every other head through. Shapes line
     up, the forward is unaffected, and the gradient is quietly wrong whenever
     `n_rot < head_dim` with more than one head packed in the row — which is
     Qwen3.5 exactly (`head_dim` 256, `n_rot` 64), worth ~1% relative error in an
     attention block's Jacobian. Guarded by
     `rlx-mlx/tests/mlx_rope_backward_partial.rs`, which fails with the fix
     disabled. **Any MLX training through partial RoPE was affected**, not just
     the lens.

4. **(fixed) Metal's backward was nondeterministic.** Two identical block
   backwards differed by ~1e-4, intermittently, and `StackLens` multiplied it
   across every block it composes. The cause was a missing `threadgroup_barrier`:
   several Metal kernels run a reduction, have every thread read `partial[0]`,
   then reuse `partial` for a second reduction. A threadgroup spans several SIMD
   groups, which diverge freely, so one group could overwrite slot 0 while
   another was still reading it. Affected `rms_norm_bwd`, `softmax_lastax`,
   `softmax_lastax_causal`, `softmax_lastax_h`, `layer_norm_bwd`,
   `group_norm_bwd_input` and both AdaLayerNorm backwards — the cross-entropy
   softmax kernels already had the barrier, which is what made the omission
   visible as an inconsistency rather than a design. The same kernels also halved
   with `tsize / 2`, dropping the odd element for any row width that is not a
   power of two; both are fixed upstream. Metal fits are now **bit-reproducible**,
   and `tests/dim_batch_invariance.rs` asserts exactly that rather than a
   tolerance. Guarded upstream by
   `rlx-metal/tests/metal_attention_backward_determinism.rs`, which fails 2-4
   times in 6 with the barrier removed.

5. **The Qwen3.5 readouts are of a quantized model.** `weights/Qwen3.5-0.8B-gguf`
   is Q3_K_S / Q4_K_M / Q6_K only, so every Qwen3.5 number here — the readout
   tables, the convergence curve, the timings — is the Jacobian *of the Q6_K
   model*. That is not a correctness gap: the reference comparison deliberately
   uses `weights/Qwen3-0.6B`, which is f32, and matches to 2.8e-6 there. But a
   figure meant to say something about Qwen3.5 itself wants an f32/bf16
   checkpoint, which is not on this machine.

## A note on finite differences

Qwen3 applies RMSNorm **per attention head**. RMSNorm's gradient carries a
`1/rms` factor, so a head whose components are nearly equal amplifies the
gradient by thousands. At such a point no finite-difference step is valid: small
enough to be a local derivative and f32 rounding swamps it; large enough to
clear rounding and it is a several-hundred-percent perturbation. The autodiff is
right there and the finite difference is not — which reads exactly like a
gradient bug, and cost real time to rule out once.

`tests/qwen35_block_vjp.rs` therefore screens its own oracle: every central
difference is computed at two step sizes, and points where they disagree are
reported as unusable rather than compared, with a floor on how many must remain
usable so the check cannot go vacuous.
