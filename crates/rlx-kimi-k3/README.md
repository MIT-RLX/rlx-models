# rlx-kimi-k3

Kimi-K3 (Moonshot AI) for RLX — a multimodal MoE model.

**Text backbone (`KimiLinear`, 93 layers, hidden 7168):**
- **KDA** (Kimi Delta Attention) — gated delta-net linear attention (short causal
  conv on q/k/v + silu, L2-normed keys, gated-RMSNorm output) via `Op::GatedDeltaNet`.
- **MLA** (NoPE) — DeepSeek-style multi-head latent attention, **no** rotary
  embedding, optional sigmoid output gate. Interleaved with KDA per `linear_attn_config`.
- **LatentMoE** — 896 routed experts (16 active, MXFP4-packed) + shared latent
  up/down projection + 2 shared experts; sigmoid `noaux_tc` grouped-topk router.
- **situ** activation (custom GLU), **Attention Residuals** every `attn_res_block_size`.

**Vision:** ViT tower (patch 14, RMSNorm, gelu-tanh, video-aware pos-emb) + a
patch-merger projector, spliced into the text stream at the media placeholder token.

## Status
Verified on synthetic configs across the RLX backends and on a single real-weight
block; a full 93-layer real-weight forward has produced a next token, but there is
no reference comparison for the whole model (the checkpoint is ~1.6 TB bf16 and the
reference implementation does not run on the hardware used). Training, the
activation/weight dumps and the distributed pipeline are verified on synthetic
checkpoints that use the real tensor layout (below); none has run on real weights.

## Command line

```bash
just kimi-synth /tmp/kimi-synth            # small checkpoint, real tensor layout
just kimi -- info /tmp/kimi-synth
just kimi -- help                          # every subcommand
```

`rlx-kimi-k3 synth --out DIR` writes a checkpoint with Kimi-K3's exact names,
`nn.Linear` layouts and dtypes (bf16 matrices, f32 conv / `A_log` / `dt_bias`,
MXFP4 experts) at toy size — every command below accepts it in place of the real
model.

## Decode speed

```bash
cargo run -p rlx-kimi-k3 --release --example decode_bench -- \
    --hidden 1024 --layers 6 --experts 16 --expert-cache-mb 512 [--dense bf16]
```

- **Layer caches** (`runner::LayerCache`, on by default): resident layer
  weights, compiled KDA decode graphs, and per MoE layer the dense MoE weights
  plus compiled router and experts graphs (experts graphs keyed by compact-bank
  size rounded to a power of two). Without the MoE part every token re-read and
  re-transposed each MoE layer's router / latent / shared weights (~760 MB as
  f32 on the real model) and compiled two graphs. `RLX_KIMI_NO_MOE_CACHE`
  turns it off; `RLX_KIMI_EXPERT_CACHE=<MB>` adds an LRU of dequantized experts.
  On the benchmark above, per decode token (CPU time, interleaved runs, CPU
  device): 128 ms with the previous caches → 87 ms with the MoE cache → 64 ms
  with resident KDA state → 50 ms with a 512 MB expert LRU. Bit-identical
  (tests `moe_decode_cache`, `resident_kda`).
- **Resident KDA state** (on by default, `RLX_KIMI_NO_RESIDENT_KDA` turns it
  off): each KDA layer's `scan` matrix stays in its cached decode graph's input
  buffer between tokens — the carry op updates it in place — instead of being
  passed in and read back every token (6.3 MB per layer each way on the real
  model, ~0.9 GB per token over the 69 KDA layers). Works on CPU, Metal, CUDA,
  ROCm, MLX and Vulkan (wgpu keeps the host path). Several sequences can share
  one cache: a displaced sequence's state is parked on the host and restored.
  `DecodeState` host copies go stale while resident;
  `LayerCache::sync_kda_state` refreshes them (state-log snapshots do this).
  Bit-identical on CPU and Metal (test `resident_kda`, `RLX_TEST_DEVICE=metal`).
  On the benchmark: CPU 86.6 → 63.6 ms per token; on Metal no measurable change
  at this size (2 MB of state per token on unified memory) — the transfer it
  removes only matters at real scale, which was not measured.
- On GPU devices the routed experts take the packed-MXFP4 path, which the MoE
  decode cache does not cover yet: it still rebuilds per token there.
- **Dense residency** (`--dense f32|bf16|int8` on `generate` / `worker`,
  `common::set_dense_residency`): bf16 holds the checkpoint's exact values at
  half the bytes; int8 a quarter, lossy. They cut memory and, at real scale,
  bandwidth. On CPU at benchmark size their dequant-on-the-fly kernels cost more
  CPU time than f32 (42 / 57 / 79 ms for f32 / bf16 / int8), so f32 stays the
  default. bf16 needs an rlx-cpu with the 2-byte `Concat` fix (rlx CHANGELOG,
  Unreleased); before it, bf16 residency computed garbage on CPU.

## Training

Next-token cross-entropy, **full-weight** or **LoRA** (default targets: every
attention / shared-expert / dense-FFN projection; `--target` to choose). Two engines:

| `--engine` | What it does | For |
|---|---|---|
| `stream` (default) | One layer resident at a time: the forward keeps each layer's input, the backward rebuilds each layer and takes its vector–Jacobian product. MoE layers pick their experts with the model's own gate op, then run over a compact bank of only the picked experts. | The real checkpoint |
| `whole` | The whole decoder as one graph through `rlx_tune::Trainer` (ZeRO-1, clipping, LR schedules). | Models that fit in memory |

```bash
# LoRA, streamed, with a merged checkpoint at the end
rlx-kimi-k3 train --model-dir DIR --text corpus.txt --mode lora --rank 16 \
    --seq 512 --steps 1000 --lr 2e-4 --warmup 50 --cosine 0.1 --out runs/a --merge
# resume
rlx-kimi-k3 train … --steps 2000 --out runs/b --resume runs/a
# data-parallel on one machine (RANK/WORLD/PEERS across machines)
rlx-kimi-k3 train … --nnodes 4
```

Outputs under `--out`: `adapter/` (PEFT layout: `adapter_model.safetensors` +
`adapter_config.json`), `checkpoint/` (full mode, or LoRA with `--merge`), `state/`
(trained values + AdamW moments, for `--resume`). An exported checkpoint writes
only the changed tensors to `model-trained.safetensors` and **symlinks** every
other shard from the source, so it costs the size of the update, not 1.6 TB.

What trains: every checkpoint weight except the routed experts and the router's
score-correction bias (full); `--freeze` / `--only` substrings narrow it (e.g.
`--only l92.`); `--train-embed` adds the embedding table. The router trains: its
routing weights are a differentiable gather of its sigmoid scores (the selection
itself stays non-differentiable, as in the reference).

Training graphs use **canonical params** (`common::CanonicalParams`): one graph
param per checkpoint tensor, with the fusions the inference builders do on the
host (KDA's fused input projection, per-tap conv vectors, the AttnRes query)
done in-graph. `export::hf_name` maps each param back to its tensor.

Verification (`just test-kimi-train`):
- the streamed forward reproduces the inference graph's logits;
- streamed layer-by-layer gradients equal whole-graph autodiff gradients for every
  trained param, full and LoRA;
- whole-graph gradients agree with central finite differences on one element of
  each kind of weight;
- both engines reduce the loss;
- reusing compiled graphs across steps (`--cache-layers`) is bit-identical to
  rebuilding them;
- an export with untrained values reproduces the checkpoint through the loader,
  and a merged LoRA checkpoint reproduces the adapted model.

`--cache-layers` keeps each layer's weights **and its compiled route / forward /
backward graphs** across steps, re-uploading only trained values, adapters and
the compact expert bank (2–3× faster on small models). It holds roughly three
copies of every layer's weights, so leave it off for the real checkpoint, where
disk paging dominates anyway.

Library: `train::{TrainSpec, build_train_graph, WholeTrainer}`,
`train_stream::{StreamTrainer, Stage}`, `layer_exec` (per-layer forward / VJP),
`export::{export_full, save_lora_adapter, load_lora_adapter}`,
`source::{CheckpointSource, MemorySource}`, `synth::write_synthetic_checkpoint`.

Known limits: rlx-autodiff has no backward for asymmetric-V attention, so
training/trace graphs run MLA with V zero-padded to the QK width (same value,
more compute). Gradient checkpointing inside a layer does not exist; a layer's
activations at the chosen `batch × seq` must fit.

## Inspecting weights and activations

```bash
rlx-kimi-k3 weights --model-dir DIR                         # every tensor, grouped
rlx-kimi-k3 weights --model-dir DIR --layers 10..12 --stats # per-tensor statistics
rlx-kimi-k3 weights --model-dir DIR --match experts.7.w1 --dump out/  # f32 safetensors
rlx-kimi-k3 trace   --model-dir DIR --tokens 1,42,7 --out trace/ --grads
rlx-kimi-k3 graph   --model-dir DIR --layer 3 --full        # one layer's op graph
```

- `weights` reads headers only unless asked for statistics or a dump; both go one
  tensor at a time. MXFP4 experts appear (and dump, dequantized) under their
  logical `….w1.weight` name. `--max-gb` caps a dump.
- `trace` runs the prompt one layer at a time and records **every named
  intermediate**: layer input, both Attention-Residual mixes, attention input and
  output, KDA q/k/v/decay/beta/scan/gates, MLA q/k/v/attention/gate, router
  scores, expert picks and weights, expert latents, shared experts, FFN input and
  output, final norm, logits. `--grads` adds each layer's input cotangent of the
  next-token loss. Output: `layer-NNN.safetensors`, `head.safetensors`,
  `grad-layer-NNN.safetensors`, `manifest.json` (shape and statistics of every
  entry, experts per layer, top-5 next tokens). `--match` filters, `--stats-only`
  skips tensors, `--layers N` stops early.

Library: `inspect::{inventory, weight_stats, dump_weights, trace, layer_graph_text}`.

## KDA state log (feature `state-log`)

Every KDA recurrent state of a greedy generation, exactly recoverable, without
storing ~465 MB per token. The next scan state is a pure function of the previous
one and four per-token inputs (`k`, `v`, `decay`, `beta`), and the conv windows
are the last three raw q/k/v projections. So the log keeps those inputs per token
plus a full snapshot every *k* tokens, and rebuilds any state by stepping the same
`gated_delta_net_carry_pc` op from the nearest snapshot.

```bash
cargo build -p rlx-kimi-k3 --release --features state-log
rlx-kimi-k3 generate --model-dir DIR --tokens 1,42,7 --max-tokens 256 \
    --state-log log/ --snapshot-every 64 --stepwise-prefill
rlx-kimi-k3 kda-state --log log/ --verify                       # every snapshot, bit for bit
rlx-kimi-k3 kda-state --log log/ --position 100 --dump s.safetensors
```

| Real model (69 KDA layers, f32) | Per token |
|---|---|
| every state stored | 465 MB |
| log inputs (`--no-conv`) | 10.2 MB + 465 MB / *k* |
| log inputs + raw projections (default) | 20.4 MB + 465 MB / *k* |

Layout: `kda_inputs.bin` (fixed-size records, token *t* at `t × record_bytes`),
`snapshots/state_<p>.bin`, `manifest.json` (dims, layout, snapshot positions,
step sizes, tokens, device, config fingerprint, numerics-relevant env flags).
Writes go through a background thread with a few tokens in flight.

- Rebuilt on the device that generated the log, states are bit-identical to the
  ones the generation held (test `state_log`; `kda-state --verify` checks a log).
  Rebuild on the logged device: another backend can differ at ~1e-7.
- A chunked prefill computes the whole prompt in one step, so the states at
  positions inside the prompt never existed; `--stepwise-prefill` makes every
  position's state the run's own (the prompt-end state is always snapshotted).
- Capture costs nothing without a logger: the KDA decode step names its inputs as
  taps, and only a logged run exposes them as graph outputs. Without the feature
  none of this is compiled.
- Single-node generation only for now; pipeline workers do not log.

Library: `state_log::{generate_logged, StateLog}` (`state_at`, `state_from`,
`snapshot`, `record`, `verify`).

## Distributed pipeline

Each worker serves a contiguous layer range from its own checkpoint copy; the
coordinator embeds, relays the boundary state (hidden + every AttnRes snapshot)
and applies the head.

```bash
# on each node
rlx-kimi-k3 worker --addr 0.0.0.0:7000 --model-dir /nvme/kimi --device metal
# coordinator
rlx-kimi-k3 ping       --model-dir DIR --workers a:7000,b:7000,c:7000 --split w:64,24,24
rlx-kimi-k3 generate   --model-dir DIR --workers a:7000,b:7000 --tokens 1,42 --max-tokens 32
rlx-kimi-k3 train-dist --model-dir DIR --workers a:7000,b:7000 --mode lora --accum 8 … --out runs/p
```

- One worker process handles everything: stateless prefix forwards, stateful
  O(1)/token decode (state per sequence id), and training its range.
- `generate --prompts-file` (library `dist::generate_batch`) pipelines several
  sequences through the stages at once — one thread per stage, so `n` stages
  serve up to `n` sequences in the time one takes alone; each sequence's tokens
  equal its single-node greedy output. Each in-flight sequence holds its KDA
  state and MLA cache on every worker until it finishes. Each connection has its own thread,
  so a health probe is never stuck behind an idle coordinator.
- Busy workers send heartbeat frames every 5 s, so the coordinator bounds every
  read (`--timeout`, default 60 s): an unplugged or sleeping node becomes an error
  naming it rather than a hang, while a layer range that takes minutes keeps the
  connection alive. Worker-side errors and panics come back as messages and the
  worker keeps serving.
- Before any work the coordinator checks every worker runs the same model config
  and that the ranges tile `0..num_layers` (a gap or overlap otherwise runs some
  layers twice and others never, with every shape still agreeing).
- `--split` gives layer counts (`30,33,30`) or relative weights (`w:64,24,24`, e.g.
  RAM); `--stages host:port:start:end,…` sets ranges explicitly.
- `train-dist` pipelines micro-batches GPipe-style (one thread per stage): later
  micro-batches go forward while earlier ones come back. Gradients and AdamW
  state stay on the worker that owns each layer, with one global-norm clip across
  all stages. One step matches single-node streamed training (test
  `dist_pipeline`). State saves per stage (`state/stage-SSS-EEE/` on each worker's
  disk, `state/head/` on the coordinator) and resumes with `--resume`.

The `kimi_k3_cluster` example (`--features cluster`) builds again: its placement
planner now gets a full `ModelCost` (expert active fraction, per-layer KV from the
config — recurrent state for KDA, K/V for MLA — and a parameter count).

The transport is `rlx_distributed::rpc` (heartbeat-framed request / reply) and
stage planning `rlx_distributed::stages` (on `LayerSplit`); the sizing roofline
is `rlx_distributed::cluster::sizing`; these need an rlx release that contains
them.

Library: `dist::{serve, Cluster, PipelineTrainer, plan_stages}`. The older
`serve_worker` / `run_distributed_generate` entry points still work on the new
protocol.

## Sizing a cluster

`rlx-kimi-k3 size` estimates how many machines a workload needs and how fast it
runs, per resource: memory (RAM + VRAM), storage, disk reads, network, compute.

```bash
rlx-kimi-k3 size                                   # Kimi-K3 reference dims, Mac mini M4 Pro nodes
rlx-kimi-k3 size --list-nodes                      # presets
rlx-kimi-k3 size --compare --experts resident      # one row per preset
rlx-kimi-k3 size --node pc-rtx4090-128 --ram-gb 256 --disk-gbs 12   # override any field
rlx-kimi-k3 size --parallel expert --target-tps 0.5                  # add nodes until 0.5 tok/s
rlx-kimi-k3 size --mode lora --context 2048 --node server-8xh100-2tb --experts resident
rlx-kimi-k3 size --model-dir DIR --json            # from a config / checkpoint (exact bytes)
```

- **Footprint** (`sizing::Footprint`): every parameter counted from the config.
  For the reference dims that is ≈ 2.78 T parameters: 55.6 B dense, 1.17 B
  embedding, 2.72 T routed experts, of which 104 B are active per token. Stored,
  that is 114 GB of bf16 backbone and 1.45 TB of MXFP4 experts, matching the
  checkpoint as it was on disk. Per sequence: 465 MB of KDA state and 2.9 MB of
  MLA KV per token, as this crate caches it. A test checks the counts against
  the elements a checkpoint stores.
- **Workload:** `--mode infer|lora|full`, experts `paged` (read from disk each
  step, optional `--expert-cache-gb`) or `resident`, `--parallel pipeline|expert`,
  dense precision `bf16|int8|f32`, `--context`, `--batch`.
- **Node count:** the largest count any single resource forces, memory or
  storage, or more for a `--target-tps`. Per-token rates (disk read, network,
  FLOPs) are reported alongside.
- **Speed:** a roofline estimate. Each stage's time is its slowest resource at
  nominal rate × `--efficiency` (default 0.5). A pipeline with *m* = min(batch,
  nodes) micro-batches in flight; expert parallelism spreads expert reads and
  compute across every node, with all-to-all dispatch and combine. A pipeline
  never shortens one sequence's token time; only concurrency or expert
  parallelism does, and the tool says so when a target is out of reach.

It is a model, not a measurement. Preset specs are approximate public figures
(FP32 vector rates; matrix units are faster), routing is assumed uniform (real
skew helps a cache), and the runtime's compile and dequant overheads are not
included. The single-node real-weight runs recorded earlier were well above
these floors for exactly those reasons.

## Cluster inference & expert-compute performance
The disaggregated expert-parallel path (recurrent backbone on one node; stateless MoE
experts fanned out to workers over TCP, one rank per compute engine) and its optimized
MXFP4 expert kernels — parallel paging, fused CPU decode-matmul, and **native CUDA/HIP
register-decode GEMM** — are documented in
[`docs/MOE_EXPERT_OPTIMIZATION.md`](docs/MOE_EXPERT_OPTIMIZATION.md). Fleet launch/measure
tooling is in [`scripts/`](scripts/README.md).
