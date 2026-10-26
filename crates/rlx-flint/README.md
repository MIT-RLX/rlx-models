# rlx-flint

[Recursive Flint](https://huggingface.co/RecursiveSI/flint) on RLX: a 39.8B-total,
~2.5B-active mixture-of-experts coding model (`model_type = flint`,
`FlintForCausalLM`, BF16 safetensors), with the routed experts paged from disk
so the 80 GB checkpoint runs in ~14 GB of RAM.

Recursive publishes no transformers modeling code. The reference is the SGLang
plugin inside the model repo (`sglang/flint_sglang-*.whl`, `model.py`).
[`scripts/flint_reference.py`](../../scripts/flint_reference.py) transcribes
that file's forward pass into plain torch, and this crate is checked against it.

## Architecture

| Piece | Layers | Notes |
|---|---|---|
| GQA attention | all 49 | 32 q / 4 kv heads × 128, per-head QK RMSNorm, **learned per-head sink logit** |
| Sliding window | 35 | sees the current token + 512 before it (`sliding_window = 513` counts the current one) |
| Full attention | 14 | causal, **no positional encoding** |
| RoPE | sliding layers | NeoX pairing, θ = 10000, over the whole head |
| Dense SwiGLU | 0, 1 | 6144 wide |
| Sparse MoE | 2–48 | **sigmoid** router, top-6 of 256 renormalized (`Σ + 1e-20`), 512-wide experts, + an **ungated** 1024-wide shared expert |

Implementation notes:

- **Sinks are not a mask.** They add one logit per head to the softmax
  denominator, so attention rows sum to less than one. `Op::Attention` has no
  sink input, so the attention is decomposed: scores, plus the sink as an extra
  column, softmax, then the column is dropped.
- **The router reuses `rlx-llada2`'s group-limited gate kernel** with a single
  group. Its first input weighs and its second selects, which is the
  reference's routing function including the epsilon. The optional
  `mlp.gate.expert_bias` (absent from the published checkpoint) shifts
  selection only.
- **The mask is a host-built additive input**, computed from absolute
  positions. One code path covers full causal, the window, a ring-buffered
  sliding cache and right-padded chunks.

## Running it

```bash
just fetch-flint                                   # 80 GB → weights/lm/flint
just flint -- --weights weights/lm/flint --prompt "Write a Python function that reverses a string."
just flint -- --weights weights/lm/flint --prompt "…" --sample --timing   # model-card sampling: T 1.0, top-p 0.95, top-k 20
```

```rust
use rlx_flint::{FlintRunner, RunnerOptions, chat};
use rlx_runtime::Device;

let tok = chat::FlintTokenizer::open("weights/lm/flint")?;
let ids = tok.encode(&chat::render(&[chat::Message::new(chat::Role::User, "Hi")], true))?;
let mut run = FlintRunner::open("weights/lm/flint", RunnerOptions::new(Device::Cpu))?;
let out = run.generate_greedy(&ids, 256, |_| true)?;
println!("{}", tok.decode(&out)?);
```

`FlintRunner` implements `rlx_runtime::LmRunner`, and `rlx_models::run::auto_runner`
selects it for a Flint directory (facade feature `flint`).

### Tools

`--tools FILE` takes a JSON array of declarations (OpenAI's `tools` field)
and prints each call in the reply to stderr:

```text
$ just flint -- --weights weights/lm/flint --tools tools.json --prompt "What's the weather in Tokyo, in celsius?"
<think>The user wants the current weather in Tokyo in Celsius. I should call the get_weather tool …</think>I'll get the current weather for Tokyo for you.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>
[rlx-flint] tool call: {"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}
```

In code: `chat::render_with_tools(&messages, &tools, true)` with
`tools::Tool::list_from_json`, assistant turns carrying `tool_calls`, and
`Role::Tool` for results. `tools::parse_reply` splits a reply into reasoning,
content and calls, typing each argument by the tool's schema (a `"string"`
parameter stays text; anything else is read as JSON) as SGLang's `flint`
parser does. Schemas keep their key order: the template writes them with
Python's `json.dumps`, and `tools::Json` reproduces it byte for byte.

### How the runner works

```text
host embed ─► per layer:  layer graph   h += attn(norm h); x = norm h; s = σ(router(x)); partial = h + shared(x)
                          host          ids, w = top-6(s); h = partial + Σ w · expert_ids(x)
                                        (each fired expert read from the checkpoint and computed as stored)
           ─► head graph  lm_head(norm h)
```

- **Experts are picked on the host**, from the scores the layer graph
  produced, by the CPU body of the gate kernel the resident graph runs. The
  host needs the ids to page experts anyway, and no backend needs the custom
  op, so the runner runs wherever the dense ops lower.
- **Routed experts run on the host, from the bytes as read.** Each
  projection is read in 64-row blocks (one contiguous `pread`, HF's `[out, in]`
  layout) and every output is a bf16 dot product with f32 accumulation. There
  is no decode to f32, no transpose and no copy into a graph: per token that
  is 1.8 GB read and nothing else written, where the earlier bank + grouped
  GEMM path wrote ~3.5 GB of f32 banks and copied them again. Work is split
  into (expert, projection, block) tasks, ~100 per pass, so every core is busy.
- **Only fired experts are read.** A decode step reads 6 × 3 matrices per
  sparse layer. Experts are read with `pread`, so the OS page cache acts as the
  expert cache. On a machine with RAM to spare the checkpoint ends up cached.
  An expert cache of our own would not help: over a 114-step generation each
  layer touched ~174 of its 256 experts, and an LRU of 48 experts per layer
  (13 GB) would have hit 43% of the time.
- **The next layer's experts are read ahead** ([`prefetch`](src/prefetch.rs)).
  As soon as a layer has picked its experts, its router input, rescaled by
  the ratio of the two layers' norm gains, goes through the next layer's
  router, and the top 6 per row are handed to the OS as read-ahead
  (`F_RDADVISE` / `posix_fadvise`). The disk then works through the expert
  math and the next attention graph instead of idling. Over 3 680 measured
  layer steps the guess held 80% of what the next layer fired (88% for a top
  8, 94% for a top 12); only the top 6 are requested because every miss is a
  read the disk did not need. It changes timing only, never the output.
  **Opt-in** (`--prefetch`, `RunnerOptions::prefetch_experts`): an 8-run A/B
  on this machine was inconclusive. Load swung from 39 to 211 between runs
  and the page cache warmed from run to run, which moved expert time far
  more than prefetch did (2.46 ms per matrix off, 2.85 ms on, averaged over
  four runs each). Worth re-measuring on a quiet machine.
- **Non-expert weights stay bf16 in the arena.** They are the checkpoint's own
  values, so nothing is lost, at half the memory of f32.
- **The KV cache lives on the host**, stored pre-transposed in the layout the
  matmuls read. Full layers hold `max_context` positions (default 4096). Sliding
  layers use a 512-slot ring, because a sliding layer never needs more.
- **Chunked prefill** (`--prefill-chunk`, default 16) runs a second set of
  graphs. On CPU these hold their own copy of the attention weights; Metal and
  MLX share them.

### Quantized experts

The routed experts can be read from a block-quantized copy instead of the
checkpoint. Only the experts are quantized; attention, norms, router, shared
expert and LM head stay bf16.

```bash
just flint-quantize q4_0 weights/lm/flint-experts-q4_0   # or q8_0; ~2 min, reads the 74 GB once
just flint -- --weights weights/lm/flint --experts weights/lm/flint-experts-q4_0 --prompt "…"
```

The blocks are llama.cpp's Q8_0 and Q4_0 (32 weights, one f16 scale). Each
dot product widens the block in NEON registers against f32 activations, so it
adds no error beyond the stored weights.

| experts | store | read / token | dot cost (1 thread) | weight error | perplexity | top-1 vs bf16 | mean KL vs bf16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| bf16 (checkpoint) | 74 GB | 1.77 GB | 0.12 ns / weight | — | 5.612 | — | — |
| Q8_0 | 40 GB | 0.94 GB | 0.12-0.13 ns | 0.54% | 5.860 | 96.2% | 0.048 |
| Q4_0 | 21 GB | 0.50 GB | 0.14 ns | 8.6% | 5.935 | 89.9% | 0.134 |

Quality is teacher-forced over a 158-token English + Rust passage
(`examples/expert_formats.rs`). Weight error is relative Frobenius error on
real expert matrices. Flint is more sensitive to expert quantization than
the weight error suggests: Q8_0 costs 4.4% perplexity, which is more than
Q8_0 usually costs. `tests/quant_store.rs` checks that the runner computes
exactly what the dequantized weights define, so the loss is the format's.

**Speed is not settled.** The point of quantizing is fewer bytes from the
disk, and at Q4_0 the experts fit in the page cache of a 64 GB machine, which
bf16 experts cannot. But on the shared machine they were measured on, load
ranged from 24 to 98 between runs, and the layer graphs alone (no experts)
took anywhere from 9.6 s to 316 s for 32 tokens. Nine alternating 32-token
decodes could not separate the formats. The fastest decode measured, 1.72
tok/s, was Q8_0 in a run where load dropped to 24.

f16 was not tried: it is the same 2 bytes per weight as bf16, so the same
bytes come off the disk and the widen to f32 costs the same, and f16's
smaller exponent range is a worse fit for bf16-trained weights.

### Measured (M-series, 64 GB, checkpoint on an external disk)

| | CPU | Metal | MLX | Vulkan (MoltenVK) | wgpu |
|---|--:|--:|--:|--:|--:|
| decode | ~1.5 tok/s | ~1.2 tok/s | ~0.95 tok/s | ~0.3 tok/s | ~0.1 tok/s |
| peak RSS | ~14 GB | ~15 GB | | | |

This table predates host-side experts and was taken on a quiet machine.
Vulkan and wgpu are from the 11-token parity run, not a long generation.
CPU is the recommended device. At this model's per-token size, a step is ~98
small graphs, and each GPU backend's per-graph overhead outweighs its compute.

Host-side experts, measured A/B on the same busy machine (load average 30-50
on 14 cores, so absolute numbers are low):

| | before | after |
|---|--:|--:|
| expert time, 11-token parity run, warm page cache | 9.4 s | 2.0-2.5 s |
| CPU decode, 48 tokens | 0.53 tok/s | 0.74 tok/s |

A long generation is now bound by the disk: the experts it has not touched
yet come off the external SSD at ~1.8 GB/s, and a token needs ~1.7 GB of
them. The 80 GB checkpoint does not fit in 64 GB of RAM, so on this machine
the page cache cannot hold all of them.

## Parity

| Check | Result |
|---|---|
| Full checkpoint vs `flint_reference.py`, 11-token chat prompt, every position | max abs(Δlogit) ≤ 3e-4 (logits up to ~100), cosine 1.0000000, top-1 identical everywhere: decode and chunked prefill, on CPU, Metal, MLX, Vulkan and wgpu (worst 2.8e-4 / 3.0e-4 / 3.4e-4 / 2.9e-4 / 4.1e-4) |
| Synthetic checkpoint vs the same script (`tests/reference_parity.rs`, no Python needed) | ≤ 2e-4 at every position; negative controls (window off, RoPE on every layer) fail as they should |
| Paged runner vs the resident whole-model graph (`tests/runner_equivalence.rs`) | ≤ 1e-4 at chunk 1, 3, 5, 16 across the sliding window; bf16 weights vs their f32 upcast |
| Tensor manifest vs the published index | all 36 734 tensors, both directions |
| Chat template vs `chat_template.jinja` through transformers (`tests/chat_template.rs`, fixture from `scripts/flint_chat_reference.py`) | byte-identical on 5 conversations: tools only, a call round trip, several system turns before the preamble, a bare function schema with non-ASCII text, escapes and floats, and no tools |

```bash
just test-flint                 # synthetic: no weights needed
just test-flint-parity [metal]  # real checkpoint; generates the reference on first run
```

## Not done

- **Constrained decoding.** SGLang constrains a request with tools to Flint's
  call grammar (and to the schema for `strict` tools). Here the reply is
  free-running and parsed afterwards, and the parser only reads a finished
  reply, not a stream.
- **CUDA and ROCm** have no custom op left to lower, but were not run: there
  is no NVIDIA or AMD machine with the 80 GB checkpoint here.
