# rlx-llm-bench

Unified LLM benchmark harness for RLX. One driver, three dimensions, **model-agnostic**
over `rlx_runtime::lm::LmRunner` — the seam every rlx LM crate already implements — so a
new model joins the leaderboard by adding one small adapter, not a new harness.

Not published (`publish = false`); it's a driver over the model crates.

## Dimensions

| dimension | what it measures | how |
|---|---|---|
| **speed** | prefill & decode tok/s, TTFT, peak RSS | times `prefill_logits` + a `generate` run |
| **quality** | multiple-choice (MMLU, ARC, HellaSwag, OpenBookQA, WinoGrande), GSM8K (generative), perplexity | reuses [`rlx-eval`] scoring; MC via `1` prefill + `k-1` decodes per choice |
| **parity** | argmax agreement + logit cosine vs a reference | compares against an mlx-lm/HF dump, a saved rlx-CPU dump, or another backend |

### Multiple-choice benches (`mc` task)

```bash
rlx-llm-bench mc --dataset arc_challenge --weights <gguf> --tokenizer <json> --device metal --limit 200
```

`--dataset` ∈ `mmlu, arc_challenge, arc_easy, openbookqa, hellaswag, winogrande` (fetched
from the HF datasets-server, normalized, cached). Each has a natural scoring mode:
`letter` (A/B/C/D single token — **fast bucketed packed path**, great on GPU) for
MMLU/ARC/OpenBookQA, `raw` (score the continuation text) for HellaSwag/WinoGrande.
Override with `--mmlu-mode letter|cloze|raw`.

### Backends & the packed bucketing win

Letter-mode MC needs only the context's last-position logits, so it runs on the
**packed** path, which pads every context to one fixed `max_seq` shape. That means a
compiled backend (Metal MPSGraph, CUDA, ROCm, wgpu, Vulkan) compiles the prefill graph
**once** instead of recompiling per distinct context length — the churn that made MMLU
crawl (an 80-doc Metal run went from *timing out at 10 min* to seconds). The F32 path
(gsm8k, `raw`/`cloze` MC) still compiles per length; force it with `--force-f32`.

### Cross-backend parity (validate CUDA / ROCm against CPU)

```bash
rlx-llm-bench parity --weights <gguf> --device cpu  --save-ref cpu.json   # on any host
rlx-llm-bench parity --weights <gguf> --device cuda --ref cpu.json        # on the GPU host
# -> argmax_match=yes cosine=1.000000  (verified locally CPU→Metal)
```

Every dimension prints a machine-readable `LLMBENCH …` line and a row for the markdown
leaderboard (`--report`).

## Quick start

```bash
# Weightless smoke test — proves the pipeline with no checkpoint:
cargo run -p rlx-llm-bench -- speed --dry-run

# Real qwen3 (F32 CPU), all three dimensions + a report:
cargo run -p rlx-llm-bench --release -- all \
  --model qwen3 --weights weights/Qwen3-0.6B/model.safetensors \
  --tokenizer weights/Qwen3-0.6B/tokenizer.json \
  --device cpu --report LLM_BENCH.md

# One dimension, on Metal, capped to 50 MMLU docs:
cargo run -p rlx-llm-bench --release --features metal -- mmlu \
  --weights <path> --tokenizer <path> --device metal \
  --data data/mmlu.jsonl --limit 50
```

Backends are cargo features that fan out to the runtime and the model adapters:
`--features metal|mlx|cuda|rocm|gpu|vulkan|coreml` (or `all-backends`, `apple-silicon`).

## Datasets

Three sources, in priority order:

1. `--data <file.jsonl>` — your own file.
2. `--fetch` — download the real benchmark set (cached, offline on re-run).
3. built-in synthetic smoke set (default; a warning is printed).

`--fetch` pulls from the HuggingFace **datasets-server** JSON API — no auth, no
parquet, no giant tarball — and normalizes into the JSONL shapes below:

- **MMLU** ← `cais/mmlu` (config `all`, 14 042 test rows)
- **GSM8K** ← `openai/gsm8k` (config `main`, 1 319 test rows)

```bash
# Pre-download once (cached under $RLX_LLM_BENCH_CACHE or .cache/llm-bench):
cargo run -p rlx-llm-bench -- fetch --dataset both            # or mmlu|gsm8k
cargo run -p rlx-llm-bench -- fetch --dataset mmlu --limit 500

# Or fetch inline while scoring:
cargo run -p rlx-llm-bench --release -- mmlu \
  --weights <path> --tokenizer <path> --device metal --fetch --limit 500
```

`--limit` caps both fetched and scored rows (the cache file is limit-suffixed, so
different limits don't collide); `--refetch` forces a re-download.

### JSONL format (one object per line)

- **MMLU / MC** — `{"question": str, "choices": [str, …], "answer": int|str, "subject"?: str}`
  (`answer` is a 0-based index or a letter).
- **GSM8K** — `{"question": str, "answer": "…#### <number>"}`.

## Adding a model

1. Ensure the model crate implements `rlx_runtime::lm::LmRunner` (most do).
2. Add `src/adapters/<model>.rs` with `build(&BuildSpec) -> Result<BenchModel>`.
3. Gate it on a `<model>` cargo feature and add one arm to `adapters::build_model`.

The scoring, timing, parity, reporting, and CLI are all shared — untouched.

## Notes

- Quality tasks force the **F32** path (`--force`-equivalent is automatic) because the
  log-prob scorers use `prefill_logits`/`decode_logits`, which the packed/quantized path
  does not expose.
- Cross-backend parity without an external oracle: run `parity` on CPU with
  `ReferenceDump::save`, then run it on the GPU backend with `--ref` pointing at that dump.

[`rlx-eval`]: ../rlx-eval
