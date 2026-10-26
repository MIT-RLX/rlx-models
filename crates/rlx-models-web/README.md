# rlx-models-web

The small RLX models in a browser tab: WebAssembly (CPU, SIMD), weights
fetched by the page. Not published (`publish = false`); built with `wasm-pack`.

| Class | Model | Files to fetch |
|---|---|---|
| `Transcriber` | Moonshine v2 streaming tiny / small | `config.json`, `model.safetensors`, `tokenizer.json` |
| `Embedder(files, "embeddinggemma")` | EmbeddingGemma-300M | + `2_Dense/`, `3_Dense/model.safetensors`, `sentence_bert_config.json`, `config_sentence_transformers.json` |
| `Embedder(files, "jina-v5", task)` | jina-embeddings-v5-text-nano | + `config_sentence_transformers.json`, `adapters/<task>/adapter_{config.json,model.safetensors}` |
| `BertEmbedder(files, pooling, seq, backend)` | BERT-family encoders: bge, MiniLM, MedEmbed, MedCPT, … (`pooling` `"cls"` or `"mean"`) | `config.json`, `model.safetensors`, `tokenizer.json` (+ `tokenizer_config.json`, `special_tokens_map.json` when present) |
| `TextGenerator(files, backend)` | LFM2.5-230M / 350M | `config.json`, `model.safetensors`, `tokenizer.json` |
| `Qwen35Generator(files, backend, max_seq)` | Qwen3.5-0.8B (GGUF: Q4_K_M or BF16) | `model.gguf`, `tokenizer.json` (`just fetch-qwen35-web Q4_K_M\|BF16`) |

`web/models.js` lists them per model.

```js
import init, { Files, Transcriber } from "./pkg/rlx_models_web.js";
await init();
const files = new Files();
for (const name of ["config.json", "model.safetensors", "tokenizer.json"])
  files.add(name, new Uint8Array(await (await fetch(`${base}/${name}`)).arrayBuffer()));
const asr = new Transcriber(files);   // consumes `files`
console.log(asr.transcribe(pcm16k));  // Float32Array, 16 kHz mono
asr.free();                           // wasm memory never shrinks
```

## Run

```sh
just web-models-serve                     # build + http://127.0.0.1:8787/crates/rlx-models-web/web/
just web-models-selftest jina-v5          # headless Chrome, prints a JSON report
just web-models-parity                    # Node build vs the HF reference JSONs
just test-web-models-memory               # native: in-memory loading == directory loading
```

The demo page runs the models in a Web Worker and serves weights from this
repo's `weights/` (fetch them with the `just fetch-*` recipes).

## What it took

- **No filesystem**: models take an `AssetSource` (`open_source`), and
  `rlx_core::SafetensorsBytesLoader` reads weights from memory. It parses the
  header itself: the `safetensors` crate sizes tensors in *bits* in a `usize`,
  which overflows on wasm32 for any tensor over 512 MB.
- **4 GiB address space**: EmbeddingGemma's 805 MB and jina's 394 MB embedding
  tables are gathered row by row on the host (`rlx_core::TensorRows`) instead of
  living in every compiled graph, and on wasm32 only one compiled graph per
  model is kept (each holds a full copy of the weights).
- **`tokenizers`**: the C regex backend (`onig`) does not build for wasm32;
  the workspace pin no longer forces it, and each crate that needs a backend
  picks `onig` natively and `fancy-regex` (`unstable_wasm`) on wasm32.
- **`std::time::Instant`** panics on wasm32; clock reads stay behind profiling flags.

## WebGPU

`new Transcriber(files, "webgpu")` / `new Embedder(files, model, task, "webgpu")`
after `await init_webgpu()`, then the `*_async` methods (`transcribe_async`,
`embed_async`) — reading results back from the GPU cannot block in a browser.
It is wgpu on the browser's WebGPU; outputs match the CPU path (jina max |Δ|
2.2e-7, EmbeddingGemma 1.3e-7). `TextGenerator` takes `"webgpu"` too, with
`generate_async`. `init_webgpu()` resolves `false` when
the browser grants no adapter — it asks `navigator.gpu.requestAdapter()` first,
because wgpu's own request never settles on a null adapter (headless Firefox).

## WebGL2

Every model also takes `"webgl"` (feature `webgl`): `new Transcriber(files,
"webgl")`, `new Embedder(files, model, task, "webgl")`, `new
TextGenerator(files, "webgl")`. It runs the graph as WebGL2 render-to-texture
shaders — each tensor an `R32F` texture — so any browser with
`EXT_color_buffer_float` works, WebGPU or not. Calls are synchronous (the
`*_async` methods resolve once done). Outputs match the CPU path: jina /
EmbeddingGemma vectors within 1.7e-7, the same transcripts, the same text.
What it took:

- attention has no WebGL kernel, so the compiler lowers it to matmul + mask +
  softmax (`LowerAttention` in `rlx-fusion`, run when a backend lacks the op);
- the planner gained grouped/depthwise convolution (LFM's ShortConv), batched
  matmul, GELU (erf and tanh), softplus and GPT-J (interleaved) RoPE — Moonshine
  used to decode garbage past token 7 because RoPE was NeoX-only;
- every tensor is ONE texture of at most `MAX_TEXTURE_SIZE` (~16384) texels a
  side, laid out `rest × last_dim`. A conv weight `[cout, cin, 1, k]` is
  `cout·cin × k` — 204800 rows. Tensors read only by linear index (uploads,
  gathers, reductions and the element-wise ops between them) are now laid out
  wide instead, and Moonshine's convolutions keep time on the last axis. The
  65536 × 1024 LFM embedding is gathered on the host and the tied LM head
  applied there, for prefill and decode. Natively, `Device::OpenGl` checks the
  same limit (`RLX_WEBGL_MAX_TEXTURE`, default 16384), so a test catches it;
- weights and index textures upload once and stay cached across runs;
  per-run textures are freed. Without that LFM ran at 0.9 tok/s.

It is the slowest GPU path (one draw call per op, a synchronous `readPixels`
per run) but the most widely available.

## Qwen3.5-0.8B

The checkpoint is 0.8B parameters; widened to f32 the weights alone are
3.2 GB, more than a wasm32 page (4 GiB) can hold next to everything else. So
nothing is widened: `Qwen35Generator` takes a GGUF and keeps its weights packed
on the host and on the GPU, which dequantizes inside the matmul.

| File | Download | wasm memory | Output |
|---|---|---|---|
| Q4_K_M | 520 MB | ~0.7 GB | quantized (differs from HF from token 4 on the test prompt) |
| BF16 | 1.5 GB | ~1.7 GB | bf16 kept as bf16 — the HF reference's tokens exactly |

WebGPU (`generate_async`) and CPU (`generate`, slow: per-call dequant on one
thread). Prompts are limited to 128 tokens (the static prefill graph is
compiled at that width to keep a wgpu arena under 2 GiB; see below). Both
files produce the native wgpu text in Chrome.

What it took:

- **Runner from memory.** `Qwen35RunnerBuilder::gguf_loader` takes a GGUF
  parsed with `rlx_gguf::GgufFile::from_bytes`, which keeps the download as
  the data segment (no second copy — 1.5 GB matters here). An in-memory GGUF is
  sized from its tensor table: the "≥ 256 MB → packed" rule used to `stat` the
  file, found none, and loaded a BF16 checkpoint as dense f32.
- **bf16 kept.** `QuantScheme::GgufBF16` (`RLX_QWEN35_KEEP_BF16`): a BF16 GGUF
  tensor goes through `DequantMatMul` as a one-element block — rlx-cpu and
  rlx-wgpu read it, the latter under a backend-local kernel id so no other
  backend claims a kernel it lacks.
- **WebGPU without blocking.** `generate_async` awaits every graph run
  (`generate_stepwise` is its native twin, tested against `generate`). On the
  wgpu side: GGUF matmuls that only submit work (the fused GEMV — Q4_K, Q5_K,
  Q6_K, Q8_0, bf16 now) are allowed in the browser; a schedule ending on one no
  longer falls through to a blocking readback (the page spun forever); f32
  params are no longer parked where reaching them takes a blocking host copy.
- **Chrome's shader compiler (Tint)** rejects a constant-folded infinity or
  NaN (`bitcast<f32>(0x7f800000u)`), which naga accepts: the GGUF GEMV shader
  failed to compile in the browser, every matmul was dropped, and the model
  generated `!!!!`. Those constants are runtime values now.
- **wasm32 arena limit.** A wgpu offset with bit 31 set means "weight buffer"
  on wasm32, so an arena may not reach 2 GiB: past it, weights are split into
  the weight buffer, and the prefill width is capped.

## Two Web Workers

`two_workers.html?model=qwen3.5-0.8b-q4km&workers=2` starts two workers,
each loading the whole model into its **own** wasm instance (its own 4 GiB)
and its own WebGPU device, and generating at the same time. Both work and
produce the same text. They share one GPU, though: 1.4–1.5 tok/s each, ~2.9
together, against 3.3 for one worker alone (machine GPU busy with another
job). So workers buy memory, not GPU throughput.

## One model across N workers

**Chat demo:** `web/chat.html` (after `just web-models-serve`:
http://127.0.0.1:8787/crates/rlx-models-web/web/chat.html). Two ways to use
the workers:

- *One chat, layers split across them* — this pipeline: less memory per
  worker, same speed (every token still visits every worker in turn).
- *Parallel questions, a full model each* — one question per input line,
  answered at the same time. Each answer is slower, but they overlap on the
  GPU: six questions (453 tokens) took 15.6 s on one worker, 11.2 s on two and
  10.4–11.3 s on three — 29 → 40–43 tok/s in total. ~1.9 GB of wasm memory
  per worker, so past two workers it costs memory for little gain.

Models come from this server or straight from the Hugging Face Hub: presets
for `unsloth/Qwen3.5-0.8B-GGUF` (Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0, Q4_0,
IQ4_XS, IQ4_NL, BF16) or any
Qwen3.5 GGUF (`owner/repo/file.gguf`, tokenizer from `Qwen/Qwen3.5-0.8B`).
The Hub answers Range requests with CORS, so each worker streams only its own
tensors (`Qwen35Pipeline.create({ modelUrl, tokenizerUrl })`). Presets
include Qwen3.5-2B Q4_K_M (1.3 GB) and Qwen3.5-4B Q4_K_M (2.7 GB, 2 or more
workers). A worker holds its weights once: downloaded pieces (≤ 32 MB) are
copied straight into the stage's buffer (`GgufFile::begin_partial`), and the
stage's layer renumbering renames in place (`retain_map` keeping every
tensor). That brought 0.8B on 3 workers from 2.36 to 1.09 GB of wasm memory
in all and 2B from 4.8 to 2.0 GB; 4B used to run out (~3× its download per
worker) and now takes 3.4 GB over 2 workers.

Every block format these files use stays packed and runs on the GPU: Q4_K,
Q5_K and Q6_K each have a GEMV kernel, and one more covers the 32-weight
block formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, and IQ4_XS as eight
32-weight sub-blocks), all checked against `rlx_gguf::dequant_typed`
(`cargo test -p rlx-runtime --features gpu --test wgpu_matmul_gemv`). Q8_0,
Q4_0 and IQ4 files used to widen to f32 and not fit. Chrome, 0.8B, 3
workers, greedy: Q4_K_M 42.9 tok/s, Q4_0 42.2, IQ4_XS 41.5, Q5_K_M 40.5,
Q8_0 36.7 (before the in-pass GEMV below); natively each matches the CPU
token for token.

Downloaded pieces are kept in the browser's Cache API (`rlx-gguf-v1`, 32 MB
cells at fixed file offsets, so any worker count or mode reuses them): a
second load reads everything from the cache, and each worker card says how
much came from where. *Clear cache* empties it; `?reload=1` loads, unloads
and loads again in one page.

Decoding is sampled by default (temperature 0.7, top-p 0.8, top-k 20,
presence penalty 1.5, as Qwen suggests for non-thinking chat; `Greedy`
switches it off). Greedy Qwen3.5-0.8B often never ends its turn — "tell me a
funny joke" ran 300 tokens on the CPU reference without `<|im_end|>`; sampled,
it ended in 51–106 tokens over three seeds. Sampling reads the logits back
(`Qwen35Runner::set_sampling`, `Qwen35Stage.set_sampling`), greedy reads one
token id. The sampler chain sorted the whole 248k-token vocabulary twice per
token (top-k, then top-p): 11.8 ms natively; top-k now selects in linear time
and top-p looks only at the survivors, 1.2 ms, same results.

Decode graphs on wgpu form a power-of-two ladder from 256 positions up to the
context (2048 here), each rung compiled when an answer first reaches it. They
used to be one bucket the size of the whole context, so every step attended
over (and fed, masked and copied) 2048 positions however short the answer:
the chat page ran 17–20 tok/s where a 256-token context ran ~39. Now, Chrome on
an idle GPU, Q4_K_M, greedy: 0.8B 57–60 tok/s (3 workers, 300 tokens across a
rung), 2B 42, 4B 23 (2 workers) — they were 36–38, 25–26 and 13–14 before the
GEMV went in-pass (below). The step-at-a-time decode carries its on-GPU state
across rungs (read back on the last step of each); 600 tokens across them
match `generate()` exactly. A rung shares the packed weights of the graphs
before it (one GPU copy for the prefill graph and every rung), so its first
use costs a compile and the state hand-over, ~75 ms once, not a weight
upload.

**Speculate** (`create({ rows: 4 })`, `generate({ speculate: true })`):
every decode step runs 4 positions. Prompts past the prefill go in 4 tokens
per step — a 33-token follow-up turn 540 → 214 ms — and, with greedy
decoding, each step feeds the newest token plus up to 3 guessed from the
conversation (what followed the last time its last 2–3 tokens occurred) and
keeps the guesses the model agrees with. A step whose guesses only partly
hold is cut back (`Qwen35Stage.rows_undo`): each worker restores its
recurrent state, saved on the GPU before the step, and re-runs the kept rows.
The answers are exactly the greedy ones (`decode_rows_match_generate`, CPU
and wgpu, including failed guesses across a rung). Chrome, 0.8B: repeating a
list 58.3 → 125 tok/s (58 of 60 guesses kept); a turn recalling the previous
one 57.6 → 67; plain prose 56 → 41, since a 4-row step costs more than one
row and nothing gets guessed. Off by default.

Each worker's layers, download (live, while it streams) and wasm memory are
shown, and every answer reports prefill time and tok/s. With *Think first*
the reasoning comes first, folded, until the model closes it or a *thinking
budget* (default 512 tokens) runs out; then the page closes it for the model
("…give the solution based on the thinking directly now. `</think>`", Qwen's
budget recipe, via `generate({ stopAt: [</think>] })` and an appended turn)
and the answer follows. Without it greedy 0.8B reasoned in circles until the
2048-token context was full. The *Context* select sets the conversation length
(2048 to 32768 tokens; with the layers split each worker holds the attention
cache of its own layers only). Answers and reasoning render as
Markdown (`web/md.js`, HTML-escaped, nested lists included), stream until end of turn by default, and
have Copy buttons; Stop (or Esc) ends the one in flight. `?model=…&mode=layers|copies&workers=N&ask=…`
(repeat `ask`) loads and asks on its own, for scripted runs; also `spec=1`,
`greedy=1`, `eos=0|1`, `max=N`, `reload=1`, `think=1`, `budget=N`, `ctx=N`.

`web/pipeline.js` splits Qwen3.5 **by layer** across as many workers as you
ask for:

```js
import { Qwen35Pipeline } from "./pipeline.js";
const p = await Qwen35Pipeline.create({
  base: "/weights/lm/qwen3.5-0.8b-q4km", // serves model.gguf + tokenizer.json
  workers: 3,                            // 1 … number of layer groups
  backend: "webgpu",
});
const text = await p.generate(prompt, 64, { onToken: (piece) => out.append(piece) });
p.terminate();
```

Worker *i* runs stage *i* of *N* (`Qwen35Stage`): a contiguous run of whole
layer groups (Qwen3.5 has a full-attention layer every 4, so 0.8B's 24 layers
split into at most 6 stages), renumbered from `blk.0`, plus the token
embedding on the first stage and the final norm and LM head on the last. The
page tokenizes (`TextTokenizer`), sends token ids to the first worker, each
worker's hidden rows (`Float32Array`, 4 KB per decode step) to the next, and
gets the next token back from the last. The workers are chained with
`MessageChannel`s, so hidden rows go straight from one worker to the next
(the buffer is transferred, not copied) and only the first and last talk to
the page: N + 1 messages per token instead of 2N relayed through the page. Each worker fetches only the GGUF
header and its own tensors with HTTP `Range` requests (`Qwen35StageFetch`),
so it never downloads or holds the rest; where a server ignores `Range` it
falls back to the whole file. `serve.py` honours `Range`, as Hugging Face and
ordinary web servers do.

One sequence still decodes token by token through every stage, so more
workers buy memory headroom per worker (a model too big for one wasm32
instance), not speed. Natively, `cargo test -p rlx-qwen35 --test
gguf_in_memory pipeline_stages_match_generate` (with `QWEN35_GGUF_PATH`)
checks that 1, 2, 3 and 6 stages, and 3 stages built from byte ranges only,
generate exactly what the whole model does, on CPU and wgpu. Run it in the
browser with `pipeline.html?model=qwen3.5-0.8b-q4km&workers=3&n=24`.

Measured in headless Chrome (WebGPU, Qwen3.5-0.8B Q4_K_M, 48 tokens, idle
GPU, second call). Every split produced the same text:

| Workers | Downloaded per worker | wasm memory per worker | Decode | Prefill |
|---|---|---|---|---|
| 1 | 513 MB | 1941 MB | 38–42 tok/s | 0.63–0.7 s |
| 3 | 315 / 115 / 315 MB | 946 / 465 / 947 MB | 33–41 tok/s | 0.65–0.9 s |
| 6 | 266 / 65 ×4 / 267 MB | 897 / 247 ×4 / 898 MB | 34–39 tok/s | 0.8–0.9 s |

(Ranges over runs on a shared machine. Relaying hidden rows through the page,
3 and 6 workers decoded at 35 and 34 tok/s against 41 and 39 chained, in
back-to-back runs.)

Before ranged loading, every worker downloaded the whole file first, and
middle workers peaked at 709 MB with 6 workers (wasm memory never shrinks).
The first and last stages stay large because each holds the 248k-row
embedding table (the first for lookups, the last for the tied LM head). Each
token crosses the page once per worker, which is what the decode column
loses with more workers. Every worker keeps its prefill graph between prompts
(`RLX_QWEN35_KEEP_PREFILL`, on by default when the worker's weights are under
1 GiB): rebuilt per prompt, warm prefill took 1.28 s with one worker instead
of 0.63 s, and 4.8 s with 6. On wgpu the decode rungs share the prefill
graph's packed weights (`CompiledGraph::share_params_from`), so keeping it
no longer costs a second copy.

## Open Genmoji prompts (static site)

`web/genmoji.html` runs [rlx-genmoji](../rlx-genmoji)'s prompt side in the tab:
the metaprompt rewrite and the object / animal / person classifier on
Qwen3.5-0.8B, split by layer across N workers, then the FLUX prompt assembled
the way `rlx-genmoji` does it (`web/genmoji_prompt.js`; `node
web/genmoji_prompt_check.mjs` checks it against the Rust `assemble`). There is
no server: the GGUF streams from the Hugging Face Hub and the metaprompt from
the open-genmoji repository (fetched, not vendored). `just genmoji-web-dist
[OUT]` writes a self-contained site (`index.html`, five scripts, the wasm; 11
MB) for any static host; `genmoji.html?local=1` reads `weights/` under `just
web-models-serve` instead, `?auto=1` runs a few descriptions and POSTs a report
(`run_page.sh`). The image itself needs FLUX.1 (12B, ~13 GB at 8 bits), past a
wasm worker's 4 GiB, so the page shows the prompt and the `rlx-genmoji
--fast --direct` command for it.

Two additions it needed, both general:

- **Scoring.** The classifier reads the logits of three words, and the
  pipeline answered with token ids only. `Qwen35Pipeline.score(prompt, ids)`
  runs the whole prompt through the prefill pass and the last stage returns
  those ids' logits (`Qwen35Stage.scores`, `Qwen35Runner::stage_scores` — equal
  to the whole model's `predict_logits`, test `pipeline_stage_scores_match_predict_logits`).
  Prefill, because decode-step logits are not scaled like the full model's.
- **A wider prefill.** The rewrite prompt is ~880 tokens; at the default
  128-wide prefill the rest went 8 per step (~14 s). `create({ prefill })`
  sets the width (default 128). Its activations grow ~3.4 MB per token per
  stage: 512 (~2 GiB) fits Chrome's WebGPU buffers on an M4 Pro and cut the
  prompt to ~5 s; 768 (~2.8 GiB) does not — rlx-wgpu then moves the weights to
  a buffer of their own and stops at an op not wired for that.

## Settings

`set_setting(key, value)` sets an `RLX_*` setting for the module — the
browser has no process environment, and every `RLX_*` read goes through
`rlx_ir::env`. `bench.html` takes them as `&set=RLX_KEY=value`.

## Measured

`just web-models-bench MODEL BACKEND BROWSER` (`web/bench.html`): warm medians of
3 calls after a first call that includes graph compilation. Headless Chrome 155
and Firefox 158 on an M-series Mac shared with other jobs (load average
~40 during this run, so these are pessimistic):

| Model | Chrome · WebGPU | Chrome · CPU | Firefox · CPU | Download |
|---|---|---|---|---|
| Moonshine streaming tiny (5.2 s JFK clip) | 4.4× real time | 2.4× | 2.6× | 178 MB |
| Moonshine streaming small | 1.65× | 0.67× | 0.60× | 562 MB |
| Moonshine streaming medium | 0.97× | 0.18× | — | 1.06 GB |
| jina-v5 (~20 / ~330 tokens) | 73 / 265 ms | 758 / 7023 ms | 696 / 6250 ms | 454 MB |
| EmbeddingGemma (~20 / ~330 tokens) | 238 / 1113 ms | 875 / 7729 ms | 682 / 5511 ms | 1.26 GB |
| LFM2.5-230M (32 tokens) | see below | 9.3 tok/s | 10.0 tok/s | 464 MB |
| LFM2.5-350M | see below | 6.8 tok/s | 6.5 tok/s | 714 MB |

On a quieter machine the same page measured Moonshine tiny at 7.2× real time on
WebGPU and 3.3× on CPU, and jina's long text at 166 ms on WebGPU vs 4.5 s on CPU.
First calls (compile included) on WebGPU: 1.4 s (Moonshine tiny) to 6 s
(medium). Results are identical across browsers on the CPU path, and every
transcript is the reference one. Headless Firefox grants no WebGPU adapter, so
it has no GPU column.

Qwen3.5-0.8B in Chrome, WebGPU (GPU 69% busy with another job): Q4_K_M
2.2 tok/s, BF16 1.8 tok/s (32 tokens; first call 12.5 / 15 s including graph
compilation). Superseded by the wgpu kernel work below.

### wgpu kernels: fewer idle threads

A decode step runs tiny matrices, and several wgpu kernels gave a whole row
to one thread — fine for a big batch, idle GPU for one token. Profiling one
Qwen3.5 decode step (each op timed alone) found attention at 18.9 ms per call
— six calls were ~110 of the step's ~130 ms — because it launched one thread
per (batch, head, query): 8 threads for the whole GPU. Fixed in `rlx-wgpu`,
each change checked against CPU (and the GGUF GEMV against an exact
`rlx_gguf::dequant_typed` reference):

- **Attention:** a 64-thread workgroup per row (`attention_coop`) while there
  are at most 16384 rows; 18.9 → ~1 ms per decode call.
- **GGUF GEMV:** a workgroup per output column (`dequant_gemv_coop`; Q4_K,
  Q6_K, Q8_0, bf16): the 133 GEMVs of a decode step 28 → 15 ms.
- **Gated DeltaNet:** thread 0 alone decayed the whole 128×128 state each
  step while 127 threads waited; each thread now decays its own column, with
  no barriers and the same arithmetic in the same order.
- **Fused residual + norm (tee):** one thread per row → a workgroup per row.
- **One submission per run:** a GGUF GEMV no longer ends the GPU section (a
  submit and wait per weight matrix, ~190 per Qwen3.5 token); it is encoded
  into the same command buffer.

Settings to bisect with: `RLX_WGPU_ATTENTION_SERIAL`, `RLX_WGPU_GGUF_GEMV_SERIAL`,
`RLX_WGPU_GGUF_OWN_SUBMIT`, `RLX_WGPU_GGUF_GEMV_COPY_OUT`, `RLX_WGPU_PACKED_IN_ARENA`. Every model's WebGPU output still equals its CPU
output in Chrome (same transcript and text, embedding cosine 1.000000).
"After" is one run on an idle GPU; "before" is the best earlier figure above,
not all of which had an idle machine, so the ratios are rough:

| Model (Chrome, WebGPU) | Before | After |
|---|---|---|
| Moonshine streaming tiny (JFK clip) | 7.2× real time | 13.7× |
| EmbeddingGemma (~330 tokens) | 1113 ms | 121 ms |
| jina-v5 (~330 tokens) | 166 ms | 103 ms |
| LFM2.5-230M | 17 tok/s | 54 tok/s |
| Qwen3.5-0.8B Q4_K_M decode | ~6 tok/s | 19.9 tok/s (42 with the state kept on the GPU, below) |
| Qwen3.5-0.8B Q4_K_M prefill (128-wide) | 3.5–4.1 s | 1.1–1.7 s |

Natively (wgpu on Metal, `rlx-qwen35` CLI, same tokens): decode 7.8 → 24 tok/s,
prefill 3.8 → 1.55 s.

### Decode state kept on the GPU

A decode step used to upload the whole recurrent state and KV cache and read
them back — ~23 MB in and ~27 MB out per Qwen3.5 token, most of it the Gated
DeltaNet state (18 layers × 16 heads × 128²). The step-at-a-time decode the
browser runs (`generate_async`, `Qwen35Stage`) now keeps both on the device
(`Qwen35Runner::keep_state_resident`): each GDN state input is a GPU handle fed
from its own output after every run, and the KV cache is a handle the graph
appends to in place (`Op::KvAppend`, now on for wgpu) with the new row moved
into position on the GPU. Only the logits come back. Each length bucket is a
separate graph, so the step before a bucket change reads everything back to
seed the next one. `RLX_DISABLE_GPU_KV=1` restores the host round trip.

For the in-place append to run in the browser, a `BufferCopy` that is a plain
copy inside the activation arena (the append's row write) is now encoded into
the surrounding command buffer, like the GGUF GEMV, instead of counting as a
blocking host step.

Chrome, Qwen3.5-0.8B Q4_K_M, idle GPU, same text either way:

| | Decode, 64 tokens | Decode, 200 tokens |
|---|---|---|
| State through the host | 20.9 tok/s | 19.2 tok/s |
| State on the GPU | 42 tok/s | 41.3 tok/s |

### Later rounds

- **GGUF GEMV.** Q4_K and Q6_K take 16 columns per 256-thread workgroup, 16
  lanes each covering 16 weights of a block, read as words; the block scales
  are applied once per sub-block sum instead of per weight (Q4_K: `d·sc·Σ x·q −
  dmin·m·Σ x`). Q8_0 and bf16 take 8 columns × 32 lanes. A decode step's 133
  GEMVs: 14.3 → 7.2 ms; one large Q4_K matrix 53 → 85 GB/s, the Q6_K LM head
  75 → 96 GB/s (Metal's own kernel reaches 163).
- **Prefill rows share weight reads.** The Q4_K / Q6_K kernels take up to 8
  activation rows per workgroup, decoding each weight once for all of them;
  a 128-row prefill used to re-read the whole matrix per row. Chrome prefill
  1.5 → 0.66–0.79 s.
- **One token id back, not the logits.** wgpu decode graphs end with a greedy
  `Op::Sample` (`Qwen35DecodeOpts::greedy_token`) and the step-at-a-time loop
  reads back that id (4 bytes) instead of 248k logits (1 MB). The argmax
  kernel itself was one thread per row — 248k values scanned by one thread —
  and is a 256-thread workgroup per row now. Small: ~5% (41.1 vs 39.1 tok/s).
- **Element-wise chains fused.** Packed GGUF graphs compiled with every
  fusion off, because the matmul / norm pattern fusions broke K-quant
  numerics. `FusionOptions::elementwise_only` keeps just
  `MarkElementwiseRegions`, which leaves matmuls and norms alone; Qwen3.5 on
  wgpu uses it. Native wgpu decode +15% (19.6 → 22.8 tok/s, same tokens); in
  Chrome no change at 42 tok/s — there the small dispatches were not the
  bottleneck.

- **GEMV inside the pass, in place.** Packed weights now live in their own
  buffer even for small graphs, so a GEMV can bind the arena read-write and
  write its output in place: no output buffer, no copy back, bind groups
  built once per step and reused, and the dispatch joins the section's
  compute pass instead of opening its own (`RLX_WGPU_GGUF_GEMV_COPY_OUT`
  reverts). Native 300-token decode 17.6–18.5 → 9.9 s; Chrome 0.8B 36 → 57–60
  tok/s, 2B 26 → 42, 4B 13.5 → 23. Q4_K also decodes its 16 levels once per
  block for every row, and the row reduction skips empty rows.
- **Weights shared between compiled graphs.** Prefill and every decode rung
  adopt one weight buffer (`share_weights_from`; `RLX_WGPU_PACKED_IN_ARENA`
  keeps them per graph): 495 MB saved per extra rung for a whole 0.8B Q4_K_M.

LFM2.5 in Chrome on a later, quieter run (load ~27): 230M at 17 tok/s WebGPU,
11.8 CPU, 4.1 WebGL2; 350M at 13.5, 10.2 and 3.6. All three backends write the
same text.

WebGL2 in Chrome (same page, `backend=webgl`): Moonshine tiny 2.0× real time,
small 0.6×; jina 469 ms and EmbeddingGemma 508 ms on the ~330-token text
(164 / 206 ms short) — between WebGPU and the wasm CPU.

The WebGL2 runs shared the GPU with another user's training job (100%
utilisation when checked), so they are lower bounds.

Single-threaded on the CPU path; threads need cross-origin isolation
(COOP/COEP) and a different build.
