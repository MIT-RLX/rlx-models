# rlx-d1

LiquidAI **[d1-3B](https://huggingface.co/LiquidAI/d1-3B)** for RLX: a
*System One* decision model. You give it a state (text, JSON, pictures, or a
mix) and named, typed questions; each answer is a calibrated distribution read
off the logits at the answer slot. It runs one forward pass per question and
decodes nothing (`output_tokens` is always 0).

d1-3B is a post-trained **LFM2.5-VL-3B**, a stock `Lfm2VlForConditionalGeneration`
checkpoint. Only the protocol around it is new, so this crate is mostly that
protocol plus wiring:

| piece | where |
|---|---|
| prompt + readout (d1's `prompt.py`, `api.py`) | `src/prompt.rs`, `answer` in `src/lib.rs` |
| LFM2.5 text model (ShortConv + GQA hybrid) | `rlx_lfm::Lfm2GgufRunner`, which opens LFM2-VL dirs and has a cached embeddings-in prefill |
| LFM2-VL image processor, SigLIP2 NaFlex tower, pixel-unshuffle projector | `src/vision.rs` over `rlx_siglip2::build_naflex_encoder` |
| Python-exact JSON (`json.dumps`, `repr(float)`, key order) | `rlx_core::ordered_json` (shared with `rlx-kev`) |
| torch / Pillow bit-exact resizes | `rlx_core::image_preprocess::{torch_resize_rgb8_aa, pil_resize_rgb8_exact}` |

## Run

```bash
just fetch-d1                       # weights/lm/d1-3b (6.2 GB bf16)
just d1 -- --weights weights/lm/d1-3b \
  --state "I was charged twice this month, please refund one of them." \
  --questions '{"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                "team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": "Charges, refunds, invoices",
                                      "technical": "App or site faults",
                                      "fraud": "Suspected unauthorised use"}}}'
just d1 -- --weights weights/lm/d1-3b --image cats.jpg \
  --questions '{"cats": {"type": "choice", "instructions": "How many cats are there?",
                         "criteria": {"one": "One", "two": "Two", "more": "Three or more"}}}'
```

`--state-json` / `--state-file` take a JSON state, `--questions-file` reads
the questions from a file, and `--timing` adds a warm second call. The output
is the reference's `system_one` JSON.

```rust
use rlx_d1::{D1, Json, RgbImage};
let mut d1 = D1::open("weights/lm/d1-3b", rlx_runtime::Device::Cpu)?;
let out = d1.system_one(&Json::Str("…".into()), &Json::parse(r#"{"q": {"type": "noul", "instructions": "…"}}"#)?, &[])?;
```

Question types: `noul` (yes/no, answer `P(yes)`), `choice` (named options →
pick, confidence, probabilities), `score` (2–10 ordered levels → expected
level, confidence, probabilities, legend). Option order and state key order
are part of the prompt and are kept exactly as given.

## Parity

`just test-d1` checks prompt rendering and answer shapes against d1's own
`prompt.py` with no weights (15 prompts, byte-identical). `just test-d1-parity [DEVICE]`
runs LiquidAI's code (`scripts/d1_reference.py`, fp32 CPU, `transformers` 5.20)
and compares against it:

* **token ids** of every prompt, including the expanded image-token layouts: identical;
* **vision**: capped picture, every tile's patches and the 2x2-unshuffled,
  projected features: pixels exact, features cosine 1.0000000 (max abs diff ≤ 7.5e-5);
* **answers** over 9 cases / 15 questions (text, JSON, 30 options,
  Cyrillic + emoji, one picture, a 10-tile picture, an upscaled tiny picture):
  max abs(Δp) on CPU in the table below.

| device | max abs(Δp), 15 answers | vision features (cos / max abs diff) | peak memory |
|---|---|---|---|
| CPU | 2.83e-6 | 1.0000000 / 7.5e-5 | 12.6 GB (`auto`; ~25 GB as `f32`) |
| Metal | 2.32e-6 | 1.0000000 / 7.2e-5 | 12.9 GB (`auto`; 30 GB as `f32`) |

The reference's own tree-packed and per-row answers differ by up to 2.9e-6, so
both backends are at the noise floor of fp32. MLX, CUDA, wgpu and Vulkan are
wired through the features but not yet measured.

Speed (Metal, M-series, f32): load 11.5 s, first call 15 s (graph compile),
then 0.19 s for two questions over a one-sentence state. LiquidAI quotes 41 ms
for three questions in bf16 on an M5 Pro; the gap is f32 weights and one pass
per question instead of the reference's shared-state tree.

Two input-side traps, both fixed rather than tolerated:

* LFM2-VL's processor resizes with **torchvision**, whose uint8 antialias
  kernel quantizes weights to int16 at a per-axis precision. Pillow and rlx's
  f32 resize each disagree with it on ~0.2% of pixels.
  `torch_resize_rgb8_aa` reproduces it exactly.
* d1 caps pictures at 1 MP with **Pillow** bicubic (22-bit fixed point).
  `pil_resize_rgb8_exact` matches it exactly; the older f32 `pil_resize_rgb8`
  does not (306 bytes off on a 1400x900 downscale).

The remaining input difference is **JPEG decoding**: the `image` crate and
Pillow's libjpeg-turbo disagree on ~1.4% of bytes of the COCO photo. The
parity test therefore feeds the reference's decoded pixels when
`RLX_D1_REF_DIR` is set; in normal use, decode differences move answers at the
1e-4 level.

## Quantized (GGUF)

`LiquidAI/d1-3B-GGUF` runs as shipped: the text GGUF's projections stay
packed (`DequantMatMul` on the Q4_K / Q6_K / Q8_0 blocks, f32 activations),
the vision tower comes from the `mmproj` GGUF (dequantized; its conv-shaped
patch embedding is unpermuted back to HF's linear layout).

```bash
just fetch-d1-gguf     # weights/lm/d1-3b-gguf: Q4_K_M, Q8_0, mmproj Q8_0/F16 + tokenizer/config from d1-3B
just d1 -- --weights weights/lm/d1-3b-gguf --prefer Q4_K_M --device metal --state "…" --questions '…'
```

`--weights` takes the GGUF file itself or the directory (`--prefer` picks the
quant); the mmproj defaults to the one beside it with the same tag, else F16
(`--mmproj` overrides). The GGUF repo has no `tokenizer.json`, so the recipe
copies it (and the small configs) from the HF repo.

Parity, `just test-d1-gguf-parity QUANT [DEVICE]` (`tests/gguf_parity.rs`):

* **vs LiquidAI's code on the dequantized GGUF** (`scripts/d1_reference.py
  --gguf`): the HF model with every text weight replaced by the GGUF's,
  dequantized, in fp32 — the quantized model's function with float
  activations, which is what rlx computes;
* **vs llama.cpp on the same file** (`scripts/d1_llama_reference.py`, Metal,
  llama.cpp `84e66d0`): picks must agree, the gap is reported.

| quant | vs dequantized reference, Metal / CPU | vs llama.cpp, text / pictures | quantization cost vs fp32 | peak RSS, Metal / CPU |
|---|---|---|---|---|
| Q4_K_M | 5.2e-6 / 5.1e-6 | 3.0e-3 / 6.9e-4 | 3.9e-2 | 6.4 GB / 27 GB |
| Q8_0 | 5.7e-6 / 6.3e-6 | 2.9e-4 / 2.0e-3 | 6.9e-3 | 8.0 GB / 32 GB |

On CPU the packed weights are exact but not small: rlx-cpu's exact
`DequantMatMul` path keeps an f32 copy of each dequantized projection (see
`rlx_lfm`'s notes on `RLX_Q4K_FUSED_MIN_N`, which trades that memory for
8-bit activation rounding), and Q4_K is slow there (the 9-case suite: 32 min
Q4_K_M vs 5 min Q8_0). Use Metal for the quantized model on a Mac.

All picks agree with llama.cpp and with fp32. The F16 mmproj reproduces the
HF tower at cosine 1.000000; Q8_0 at 0.9998 / 0.9991.

The rlx-vs-llama.cpp gap is llama.cpp's rounding, not the port: on the one
answer where it reaches 3e-3 (`json_state` severity, a flat 5-way score) rlx
is 5e-6 from the dequantized reference and llama.cpp 3e-3. llama.cpp's
**CPU** path is further off still — it quantizes activations to 8 bits before
every K-quant matmul, which moved d1's `urgency` score by 1.1e-2 (0.366 vs
0.3777) — so its references are taken on Metal.

## Weight storage and memory

Dense weight matrices no longer have to be expanded to f32. `--weight-storage`
(`OpenOptions::weight_storage`, `rlx_core::weight_storage::WeightStorage`)
chooses how they are stored once uploaded; the math stays f32 on every setting.

| setting | dense weights | for |
|---|---|---|
| `auto` (default) | the checkpoint's own width: d1-3B's bf16 stays bf16 (lossless), an f32 tensor stays f32, a quantized GGUF tensor stays packed | memory |
| `f32` | widened to f32 (the old behaviour) | CPU speed |
| `bf16` | rounded to bf16 | memory, f32 / f16 checkpoints |
| `f16` | rounded to f16 (bf16 values above 65504 clip) | — |

GGUF K-quant projections stay packed under every setting; the tied embedding
of a quantized GGUF stays packed too (Q6_K, rows decoded on demand) under
`auto`. Norms, biases and conv taps stay f32.

Measured on d1-3B (9 parity cases, answers unchanged to the digit — keeping
bf16 is exact):

| run | `f32` | `auto` |
|---|---|---|
| HF checkpoint, Metal, peak | 30.3 GB | 12.9 GB |
| HF checkpoint, CPU, peak / suite time | ~25 GB / 61 s | 12.6 GB / 103 s |

On CPU a bf16 weight is widened per matmul call into a reused scratch for the
BLAS GEMM, which costs ~1.7x the f32 time; choose `f32` when speed matters more
than memory. On Metal there is no time cost (an exact bf16 → f32 unpack kernel
feeds the GEMM). On wgpu an `f16` weight sends large matmuls to the
cooperative f16 GEMM, which also rounds activations to f16; `bf16` (or
`RLX_WGPU_MATMUL_F32_ONLY=1`) keeps the math f32. Under `auto`, an F16 mmproj's
tower stays f32 on Metal (MPSGraph casts f16 params on every run: measured
larger and slower than f32).

`--dequant-cache off|unbounded|BYTES` sets rlx-cpu's dequant cache for packed
GGUF weights (process-wide; the CLI owns the process, a library user sets
`RLX_DEQUANT_CACHE` / `RLX_DEQUANT_CACHE_MAX_BYTES`). Measured, Q4_K_M on CPU,
same answer each time:

| `--dequant-cache` | peak | warm call |
|---|---|---|
| default (¼ of RAM) | 13.4 GB | 1.6 s |
| `off` | 3.0 GB | 3.9 s |
| `2G` | 5.3 GB | 28.9 s (LRU thrash — avoid small caps) |

Getting there fixed four silent upstream bugs, each of which turned a
half-precision weight into wrong answers with no error (`rlx-models-core`
`tests/half_weight_matmul.rs` and `tests/lfm2_weight_storage.rs` pin them on
CPU, Metal, MLX and wgpu):

* **wgpu / Vulkan** sized an F16 weight's arena slot at 2 B but widened the
  upload to 4 B (`rlx-compile` `memory.rs`);
* **wgpu** indexed its packed-bf16 side buffer as `arena_off / 2`, so a second
  bf16 weight overlapped the first (`rlx-wgpu` `buffer.rs`, `lower.rs`);
* **Metal / wgpu** misread `Concat` of bf16 weights when fusing sibling
  matmuls (QKV, gate/up) — fusion now leaves bf16 siblings unmerged
  (`rlx-fusion` `shared_input_matmul.rs`, `swiglu_dual.rs`);
* **Metal**'s per-op GEMM read a bf16 weight as f32 in any graph MPSGraph
  cannot plan (anything with attention); it now unpacks into scratch first
  (`rlx-metal` `Sgemm { b_bf16 }`, `bf16_unpack_f32`).

The CPU half-weight GEMM also widens once per call for `m >= 4` rows instead of
re-converting the weight for every row (520 s → 103 s on the suite).

## Notes

* d1's reference packs a state's questions into one *tree* (the state read
  once, the questions as branches). That is the same function as one causal
  pass per question, which is what this crate runs; the reference's own packed
  and per-row answers agree to 3e-6. Tokenization follows the reference path:
  single-question prompts are tokenized whole, multi-question ones as the
  state prefix plus each question.
* Single-file safetensors directories now load lazily (mmap; the eager loader
  converted the whole 6.2 GB checkpoint to 12.5 GB of f32 up front). The text
  graph is compiled for a power-of-two length and kept until a longer input
  arrives or inputs get 4x shorter.
