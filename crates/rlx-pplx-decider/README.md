# rlx-pplx-decider

Perplexity's **pplx-decider** classifiers in RLX —
[`perplexity-ai/pplx-decider-flash-gguf-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-flash-gguf-v1.2)
(Qwen3.5-4B, distilled from the 27B deciders).

A decider never generates text. You give it a **state** (text, or any JSON
value), optional **images**, and one or more typed **questions**; one
forward pass returns calibrated probabilities for each:

| type | criteria | answer |
|---|---|---|
| `choice` | `{option: description or null}`, 1–255 options | `choice`, `probabilities`, `confidence` |
| `noul` | optional `{"true": …, "false": …}` | `noul` = P(true) |
| `score` | list of 2–255 levels, lowest first | `score` (expected level), `probabilities`, `legend`, `confidence` |

Uses: request routing, intent detection, moderation and policy checks, agent
guardrails ("should this tool call run?"), triage, rubric grading.

## Run

```sh
just fetch-pplx-decider                    # 5.2 GB Q8_0 GGUF + 672 MB F16 vision projector
just features=metal pplx-decider -- --weights weights/lm/pplx-decider-flash-gguf --device metal \
    --state "My integration keeps failing with a 401 since we rotated our API key." \
    --questions '{"team": {"type": "choice", "criteria": {"billing": null, "support": null, "sales": null}},
                  "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"}}'
```

```sh
just features=metal pplx-decider -- --weights weights/lm/pplx-decider-flash-gguf --device metal \
    --state "Photo attached to a damage claim." --image claim.png \
    --questions '{"damaged": {"type": "noul", "instructions": "Is the item visibly damaged?"}}'
```

`--image PATH` (repeatable) attaches images to a `--state` request; in a
`--request FILE`, put `"images": [path or data:image/...;base64,...]` in the
object. The `mmproj-*.gguf` beside the weights is picked up automatically
(`--mmproj` to choose another). `--request FILE` takes one request object
(`{"state", "questions", "images"}`) or a list of them; `--logits` prints raw readout logits (use the `true - false`
margin of a `noul` for ranking — it does not saturate); `--prompt` prints the
rendered prompt and token ids.

```rust
use rlx_pplx_decider::{Json, PplxDecider};
let mut d = PplxDecider::open("weights/lm/pplx-decider-flash-gguf", rlx_runtime::Device::Metal)?;
let req = d.request(&Json::parse(r#"{"state": "I was charged twice.",
    "questions": {"team": {"type": "choice", "criteria": {"billing": null, "support": null}}}}"#)?)?;
let response = d.decide(&req, None)?; // {"answers": {...}, "usage": {...}}
```

## How it works

- **Protocol v3** (`joint-question-inline-readout-v1`, ported from the
  checkpoint's `pplx_decider/protocol.py`): one sequence holds the state and
  every question; each question ends with a `<|fim_suffix|>` marker, and the
  readout is applied to the final hidden state at each marker. Literal special
  tokens in user content are neutralized with full-width brackets. Requests
  over `max_length` are split into question chunks, each with the full state.
- **Backbone**: Qwen3.5 through `rlx-qwen35`, with the full-attention layers
  made **bidirectional** (`qwen35.attention.causal = false`;
  `Qwen35Flow::bidirectional_attention`). The Gated DeltaNet layers stay causal.
  Sequences are right-padded to a length bucket and an `attn_key_mask` keeps
  the pads out of the bidirectional layers.
- **Readout**: the GGUF stores it as `output.weight`, zero outside the 255
  code tokens; only those rows are dequantized, and they are applied on the
  host to the marker rows.
- **Weights** stay packed (Q8_0 in-kernel dequant); embeddings are gathered
  row by row from the packed table. 6 GB peak on Metal.
- **Tokenizer**: rebuilt from the GGUF, with the pre-tokenizer of the
  checkpoint's own `tokenizer.json` (Qwen2's split, which differs from stock
  Qwen3.5's on combining marks). `--tokenizer` takes a `tokenizer.json` instead.
- The calibrated temperature (1.554) comes from `pplx_decider.temperature`;
  it is never defaulted.
- **Images** follow the checkpoint's processor (`Qwen2VLImageProcessor`,
  torchvision backend) step for step: `smart_resize` to multiples of 32
  within 65,536–1,638,400 pixels (Python's half-to-even `round`), torch's
  int16 fixed-point bicubic antialias resize
  (`rlx_core::image_preprocess::torch_resize_rgb8_aa` — Pillow's 22-bit
  weights differ by one level on some pixels), `(x - 127.5) / 127.5`. The
  tower is `rlx-qwen35`'s Qwen3-VL ViT on the F16 mmproj
  (`Qwen35VisionEncoder::encode_nchw`). Each `<|image_pad|>` expands to the
  image's merged-patch count, and image tokens get Qwen3.5's 3-D M-RoPE
  positions (`[p, p + row, p + col]`, text resuming at `p + max(rows, cols)`)
  in the **interleaved** layout — which a GGUF does not record and
  `Qwen35Config::from_gguf` leaves off; text-only prompts cannot tell the
  difference, image prompts move a logit by 0.24.
- **JPEG**: build with `--features libjpeg-turbo` for pixels identical to the
  reference (PIL decodes with libjpeg-turbo; verified byte-for-byte on
  baseline, progressive, 4:2:0, 4:2:2, 4:4:4 and grayscale files). It builds
  the library from bundled source with cmake (`TURBOJPEG_SOURCE=pkg-config`
  links a system copy). Without it the pure-Rust decoder differs on ~1.4% of
  a photo's bytes (by up to 3 levels), which moved logits by up to 0.026 on
  a test photo — answers unchanged, but ~100x the implementation noise.

## Parity

Against Perplexity's own code (`scripts/pplx_decider_reference.py`, the
exact padded path in float32) over the nine requests in
`tests/fixtures/requests.json` — every question type, JSON states, non-ASCII,
special-token injection, 30 options:

| | CPU | Metal | MLX |
|---|---|---|---|
| prompt text / token ids / markers | identical | identical | identical |
| max abs(Δlogit) vs the dequantized-Q8_0 reference | 4.5e-4 | 4.4e-4 | 4.4e-4 |
| max abs(Δp) vs the same | 2.3e-5 | 2.5e-5 | 2.6e-5 |
| max abs(Δp) vs the bf16 checkpoint | 4.7e-3 | 4.7e-3 | 4.7e-3 |

Images — the five requests in `tests/fixtures/requests_images.json` (an
upscaled 200x120, a 1400x900, a 1240x1754 page that is scaled down, a
two-image request, and a 4:2:0 JPEG photo; 180-1734 tokens), with
`--features libjpeg-turbo`:

| | CPU | Metal | MLX |
|---|---|---|---|
| resized pixels (sha256 vs torchvision) | identical | identical | identical |
| expanded token ids / markers / M-RoPE positions | identical | identical | identical |
| max abs(Δ) vision tower output (values peak near 8) | 1.9e-4 | 2.8e-4 | 2.7e-4 |
| max abs(Δlogit) vs the Q8_0 + mmproj reference | 2.2e-4 | 2.1e-4 | 2.5e-4 |
| max abs(Δp) vs the same | 1.0e-5 | 1.3e-5 | 2.7e-6 |
| max abs(Δp) vs the bf16 checkpoint | 1.6e-2 | 1.6e-2 | 1.6e-2 |

The last row is quantization (the model card reports 0.016 for its own Q8_0
runtime); the mmproj is F16, which holds the bf16 tower exactly (4e-7).

The dequantized-Q8_0 reference runs the same Python code with every backbone
tensor replaced by the GGUF's (llama.cpp's qwen35 conversion inverted, and
each tensor asserted within 0.4% of the bf16 original), so quantization error
cancels and what remains is implementation. The logit tolerance (5e-3) sits
between that noise and a real defect: attending the padding moves a logit by
0.28 — and still passes a bf16-level probability check.

```sh
just test-pplx-decider                  # prompt + protocol, no weights
just test-pplx-decider-parity [DEVICE]  # real weights
```

## Speed and memory

M-series, warm (graphs for the length bucket and image size already
compiled), peak footprint of the whole process:

| | text request | two-image request (1435 tokens) | peak |
|---|---|---|---|
| Metal | 0.3–0.6 s | 6.6 s (2.1 s tower) | 6 GB |
| MLX | 0.25–1.1 s | 3.7 s | 15 GB text, 22 GB images |
| CPU | ~3 s | 16.6 s (4.2 s tower) | 5 GB text, 10 GB images |

A new length bucket (powers of two to 512, then multiples of 512) costs a
2–4 s compile, a new image size ~1.5 s. Quantized weights cannot share one
buffer across compiled shapes on Metal, so one shape is resident at a time.

What it took (each measured on this model):

- **CPU**: no dequant cache for a prefill-only engine (it kept an f32 copy
  of every weight per compiled shape: 19 GB, and thrashed across buckets),
  and `rlx-compile`'s CPU planner no longer pins every activation of a graph
  that merely contains a `Gather` (the vision tower: 24.8 → 4.9 GB, 8 → 4 s).
- **MLX**: Q8_0 runs as native affine 8-bit `quantized_matmul` instead of a
  host f32 copy; Gated DeltaNet runs MLX's fused `fast::gated_delta_update`
  kernel instead of a per-token lazy graph (text suite 144 s → 11 s); and the
  lowering releases each intermediate at its last use, so MLX can free it
  during evaluation (peak 21.7 → 10 GB at 1024 tokens). Fused vision
  attention (52 GB vs 95 GB decomposed).

## Not yet

- Batching several requests into one forward pass.
