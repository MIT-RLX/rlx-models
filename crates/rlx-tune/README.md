# rlx-tune

Generic **LoRA / DoRA fine-tuning** for RLX models — model-agnostic, host-side adapters, dataset loaders, graph injection, and a data-parallel trainer. The forward pass uses RLX's first-class `LoraMatMul` op (which lowers on every backend), so training runs on CPU / Metal / MLX like inference.

## Modules

- `adapter` — LoRA/DoRA specs + host-side merge ([`fuse_lora`], [`fuse_dora`]).
- `dataset` — text / chat / completions JSONL loaders with prompt masking (mirrors mlx-lm's `datasets.py`).
- `inject` — graph-rewrite `inject_lora` over a model's forward graph.
- `trainer` — the training loop. [`train`] is the minimal version; [`Trainer`] / [`train_dp`] / [`train_dp_with`] add data parallelism, sharding, overlap, mixed precision, clipping, LR schedules, accumulation, checkpointing, and timing.
- `distributed` — data-parallel collectives ([`GradComm`], [`from_env`]): fused all-reduce, reduce-scatter/all-gather, mixed-precision reduce.
- `cluster` — `--nnodes N` self-spawning launcher ([`launch_or_join`]).
- `dwq` — distilled weight quantization support.

## Distributed training

Data parallel with **no code change**: the trainer takes an optional `comm`, and [`from_env`] builds one from the environment. Everything below is one `cargo run` away — see the `data_parallel` (full) / `data_parallel_min` (minimal) LoRA examples and the `cnn` example.

The trainer is **not LoRA-specific** — it trains any graph whose single output is a scalar loss. The `cnn` example fine-tunes a small all-convolutional image classifier (`conv3×3/s2 → relu → conv3×3/s2 → relu → flatten → linear → softmax-cross-entropy`) through the exact same [`Trainer`] / [`DpConfig`] / `--nnodes` path, streaming a fresh data shard per rank and reporting **throughput**:

```bash
cargo run --release -p rlx-tune --example cnn                                 # 1 rank, ~90k samples/s
cargo run --release -p rlx-tune --example cnn -- --nnodes 4 --shard --overlap --bf16 --accum 4
# → loss 1.36 → 0.0002
```

Throughput scales with ranks (each processes its own data shard concurrently), and `--accum`/`--batch` amortize the per-step and communication overhead.

The `mnist` example is the full thing on **real data**: it auto-downloads MNIST, shards the 60k training set across ranks, trains the same CNN, and reports **test accuracy** as it climbs plus throughput.

```bash
cargo run --release -p rlx-tune --example mnist                  # 1 rank → ~95% acc, ~50k samples/s
RLX_WORKERS=3 cargo run --release -p rlx-tune --example mnist -- --nnodes 4 --overlap
cargo run --release -p rlx-tune --example mnist -- --nnodes 4 --shard --overlap --bf16 --accum 2
cargo run --release -p rlx-tune --example mnist -- --prefetch    # background data prefetch
```

> **Speed:** the CNN examples enable `RLX_FAST_CONV` (im2col + BLAS forward conv)
> by default — **~10× over the naive kernel**, same result (set `RLX_FAST_CONV=0`
> to compare). One fast rank already uses all cores via BLAS, so on a *single*
> host extra `--nnodes` ranks oversubscribe — cap threads with `RLX_WORKERS=<cores/ranks>`.
> Data parallelism pays off across *machines* (each rank gets a whole box) and
> for compute-heavy steps (bigger `--batch` / model), where per-step compute
> dominates the all-reduce.

### GPU training

The trainer is device-configurable — `DpConfig::new(lr).cuda()` / `.metal()` (or
`--device cuda|metal|gpu` in the example) compiles the **forward + backward** to
that GPU backend. Build with the matching feature:

```bash
cargo run --release -p rlx-tune --features cuda --example mnist -- --device cuda --big
```

Verified on an RTX 3080 Ti — MNIST reaches **97.5%**, same as CPU. Whether the GPU
*wins* depends on the compute/overhead ratio: the optimizer runs host-side, so
every step round-trips gradients/params over PCIe. For the default (tiny) model
that overhead dominates and GPU ≈ fast-conv CPU; for a **compute-heavy** model
(`--big`, 64/128 channels) the GPU is **~5× the CPU** (6.2k vs 1.2k samples/s).
Rule of thumb: reach for the GPU when the model is big enough to hide the
per-step host↔device transfer.

#### GPU-resident optimizer — `ResidentTrainer`

That host round-trip is exactly what [`ResidentTrainer`] removes. It **fuses the
Adam(W) update into the graph** (forward + backward + update as one on-device
computation) and keeps params + moments in **device-resident handles** fed back
each step. Only the scalar loss is read back (`run_read_outputs`) — the updated
`p'/m'/v'` never leave the device — so nothing but data + loss crosses the bus.
The fused update is identical to the host optimizer (checked loss-for-loss, incl.
AdamW), with a correct host-chained fallback where handles are unsupported.

```rust
let mut rt = ResidentTrainer::new(&graph, &wrt, &params, &AdamConfig::new(1e-3),
    0.01 /*weight_decay*/, Device::Cuda)?;
for step in 0..steps {
    rt.set_lr(schedule(step));              // lr is a scalar input — no recompile
    let loss = rt.step(&[("x", &x), ("labels", &y)]);
}
let trained = rt.params();                  // reads back from device
```

On the `--big` MNIST CNN on an RTX 3080 Ti (batch 256) the full optimization arc
is **~265×**, same accuracy (98.1%) — and larger batches and optional TF32 push
it further still (see below):

| Path | samples/s | vs baseline |
|---|---|---|
| host optimizer (gradients round-trip over PCIe) | 278 | 1× |
| `--resident` (fused Adam, params/moments on-device) | 1 140 | 4× |
| &nbsp;&nbsp;+ loss-only readback (`run_read_outputs`) | 4 286 | 15× |
| &nbsp;&nbsp;+ **cuDNN conv** (see below) | **~74 000** | **~265×** |

```bash
cargo run --release -p rlx-tune --features cuda --example mnist -- --device cuda --big --resident
```

> **cuDNN is the single biggest GPU lever — make sure it loads.** Without a
> loadable `libcudnn.so`, rlx-cuda silently falls back to an im2col conv that is
> ~10× slower and DRAM-bound (throughput even *drops* as batch grows). rlx-cuda
> now prints a one-time warning when convolutions can't find cuDNN:
>
> ```
> rlx-cuda: cuDNN unavailable (libcudnn.so not loadable) — convolutions use the
> ~10× slower im2col path. Put libcudnn.so on the loader path (LD_LIBRARY_PATH),
> or set RLX_CUDA_NO_CUDNN=1 to silence this.
> ```
>
> The usual fix is to point `LD_LIBRARY_PATH` at a cuDNN install (a PyTorch env
> already ships one, e.g. `.../site-packages/torch/lib`).

**Where the time goes (profiled, RTX 3080 Ti, cuDNN loaded).** With cuDNN the
resident step is **GPU-compute-bound** — dominated by the convolutions, not by
launch or host overhead. Measured knobs that *don't* help materially: CUDA-graph
capture (`RLX_CUDA_EXEC_MODE=graph`, ≤3%), host-side batch `--prefetch` (≤3%).
The optimizer itself (fused Adam) runs on tiny weight tensors and is ~1% of the
step. So once cuDNN is in, the conv *is* the workload; the remaining lever is
**batch size** (more samples per launch → better GPU utilization).

**Scale the batch.** Because the step is compute-bound, larger batches amortize
the fixed per-step overhead and push throughput *up* — the resident CNN goes
from ~74k samples/s at batch 256 to **~118k at batch 2048**, same accuracy
(98.3%). This used to be a **hard cliff**: rlx-cuda asked cuDNN for only its
single fastest conv algo and fell back to the ~10× slower im2col path whenever
that algo's workspace overflowed the scratch budget — which happens at batch
≥ 512 (batch 1024 cratered to ~3.9k samples/s). rlx-cuda now scans the
heuristic's ranked algos and picks the fastest one that *fits*, and the conv
scratch budget was raised to 256 MiB, so large batches stay on cuDNN. The fix
lives in `rlx-cuda` and benefits every batched conv, **inference included**.

**TF32 conv is opt-in (`RLX_CUDA_CONV_TF32=1`), and deliberately so.** TF32
tensor cores give the conv another ≈1.4× (batch 2048 ~172k samples/s), and
they're safe for inference / forward-only work — but unlike the matmul path
(where TF32 is the stable default), TF32-precision convs **destabilize
large-batch Adam training**: a reproducible loss blow-up appears at batch ≥ 1024
(98% → ~11% in one step), and the forward pass alone is enough to trigger it. So
conv defaults to strict FP32 (FMA) for stable training, with TF32 a conscious
opt-in. `RLX_CUDA_NO_TF32` / `RLX_CUDA_PARITY` force FMA everywhere (these now
correctly cover convs, which previously ignored them). The deeper fix — global-
norm gradient clipping inside the resident optimizer — would let TF32 training
be stable too; it isn't wired into `ResidentTrainer` yet.

Adding ranks trains on more data per step (larger effective batch), so accuracy improves — but with the fast conv, one rank already saturates the cores, so throughput scales best across *machines* (each rank a whole host) or with larger batches. Synchronous data parallel runs at the speed of the slowest rank.

### Across physical machines

`--nnodes` co-locates ranks on one host (loopback mesh). To span real machines, launch each rank as an independent process with `RANK` / `WORLD` / `PEERS` set — the identical `from_env` → TCP-mesh → all-reduce path, only the peer IPs change. Each rank binds `PEERS[rank]`; each machine needs the binary built and its own copy of the data (the `mnist` example auto-downloads per node).

```bash
# rank 0 on host A (10.0.0.10), rank 1 on host B (10.0.0.11):
RANK=0 WORLD=2 PEERS=10.0.0.10:29500,10.0.0.11:29500  ./mnist --steps 400 --overlap   # on A
RANK=1 WORLD=2 PEERS=10.0.0.10:29500,10.0.0.11:29500  ./mnist --steps 400 --overlap   # on B
```

`scripts/train_multinode.sh` automates that (rank 0 local, the rest over SSH):

```bash
PEERS="10.0.0.10:29500,10.0.0.11:29500" HOSTS="_,user@10.0.0.11" \
  scripts/train_multinode.sh target/release/examples/mnist --steps 400 --overlap --shard
```

This has been run for real across a **macOS-arm64 + Linux-x86_64** pair (gradients all-reduced over the LAN between the two): MNIST converges identically to a single host, since f32 collectives are bit-compatible across platforms. For zero hand-wired IPs, set `DISCOVER=1` on every rank instead of `PEERS` (rlx-driver's UDP auto-discovery).

### 1. Zero-config — `from_env`

```rust
let comm = rlx_tune::from_env()?;    // None unless WORLD > 1
let cfg  = rlx_tune::DpConfig::new(2e-4);
rlx_tune::train_dp(graph, &wrt, &mut params, &inputs, steps, comm.as_deref(), &cfg, |m| {
    println!("{m}");                 // StepMetrics: Display
})?;
```

```bash
# torchrun / mlx-lm style — WORLD=1 (the default) opens no sockets.
RANK=0 WORLD=2 PEERS=10.0.0.1:29500,10.0.0.2:29500  my-trainer
RANK=1 WORLD=2 PEERS=10.0.0.1:29500,10.0.0.2:29500  my-trainer
```

`from_env` reads `RANK`/`WORLD` (+ `PEERS`, or `DISCOVER=1`; `TOPOLOGY=mesh|star`) via `rlx-driver`'s `Node`, so the wire is TCP, Thunderbolt on Apple Silicon, or in-process — unchanged. **Efficient by default:** every step packs all gradients into one bucket and does a single bandwidth-optimal ring all-reduce (K per-parameter reduces → one), the rank-0 weight sync fused into one broadcast. N-way DP is numerically identical to single-node training on the union of the shards — the crate's tests assert it.

### 2. One command, no hostfile — `--nnodes N`

[`launch_or_join`] makes a binary its own launcher: with `--nnodes N` it reserves loopback ports and re-spawns itself once per rank; each child sees `RANK` and trains.

```rust
match rlx_tune::cluster::launch_or_join()? {
    Role::Launcher => return Ok(()),                    // parent: spawned + awaited
    Role::Worker { rank, world, comm } => {             // train this rank's shard
        rlx_tune::train_dp(g, &wrt, &mut p, &inputs, steps, comm.as_deref(), &cfg, on_step)?;
    }
}
```

```bash
cargo run -p rlx-tune --example data_parallel                  # single process
cargo run -p rlx-tune --example data_parallel -- --nnodes 3    # 3-way DP, one command
cargo run -p rlx-tune --example data_parallel -- --nnodes 4 --shard --overlap --bf16
```

### 3. The knobs — `DpConfig`

A fluent builder (fields stay public, so `DpConfig { .. }` also works):

```rust
use rlx_tune::DpConfig;
let cfg = DpConfig::new(2e-4)   // learning rate
    .shard()                    // ZeRO-1 optimizer-state sharding
    .overlap()                  // hide comm behind the optimizer step
    .bf16()                     // bf16 gradients on the wire
    .clip(1.0)                  // global-norm gradient clipping
    .warmup(100).cosine(0.1)    // 100-step warmup, cosine decay to 10%
    .grad_accum(4)              // 4 micro-batches per step
    .log_every(50);
// cfg.describe() → "lr=2.00e-4 shard overlap(x4) accum=4 bf16 clip=1 warmup=100 cosine→10%"
```

| Knob | Method | What it does |
|------|--------|--------------|
| Sharding (ZeRO-1) | `.shard()` | Each rank keeps Adam moments for only `1/world` of the params (the dominant memory: 2× params); reduce-scatter grads, all-gather weights. Numerically identical to unsharded. |
| Overlap | `.overlap()` / `.chunks(n)` | Reduce the bucket in chunks on a background thread while the optimizer steps already-reduced chunks. Numerically exact. **Composes with `.shard()`** (block-cyclic ZeRO-1). |
| Grad accumulation | `.grad_accum(g)` | Average `g` micro-batches before one reduce + step: `g×` less comm, `g×` larger effective batch. Needs [`train_dp_with`]. |
| Mixed precision | `.bf16()` | Halve the bytes on the wire (native `all_reduce_typed`, f64 accumulation). |
| Gradient clipping | `.clip(x)` | Clip the **global** L2 norm before the step; every rank derives the same scale from the reduced gradient, so ranks never drift (tested bit-identical). |
| LR schedule | `.warmup(n)` `.cosine(r)` `.linear_decay(r)` | Linear warmup then decay to `r × lr`. Effective `lr` is in `StepMetrics`. |
| Timing | `.log_every(n)` | Deliver `StepMetrics { lr, compute_ms, comm_ms, step_ms, compute_fraction(), … }` every `n` steps. |

> The knobs combine — including `overlap` + `shard` — with one interaction: `overlap` is skipped when `clip` is set (a global norm needs the whole reduced bucket before any step).

### 4. Varying data + accumulation — `train_dp_with`

[`train_dp`] repeats one fixed batch; [`train_dp_with`] takes a data provider `next_batch(step, micro) -> inputs` for real training and gradient accumulation:

```rust
train_dp_with(g, &wrt, &mut params, steps, comm.as_deref(), &cfg,
    |step, micro| dataset.batch(step, micro),   // your data loader
    |m| println!("{m}"))?;
```

### 5. Custom loops + checkpointing — `Trainer`

`train_dp*` are `Trainer::new(...).run(...)`. Drive the [`Trainer`] yourself for a custom loop (per-step eval, early stop) or **checkpoint / resume**:

```rust
let mut t = Trainer::new(graph, &wrt, &params, total_steps, comm.as_deref(), &cfg)?;
if let Ok(ck) = Checkpoint::load("run.ckpt") { t.restore(&ck); }   // resume
while t.total_steps_remaining() > 0 {
    let m = t.step(&mut next_batch)?;
    if m.step % 500 == 0 { t.checkpoint().save("run.ckpt")?; }     // periodic save
}
```

A [`Checkpoint`] holds weights + full Adam state (moments + timestep) + step index in `wrt`-bucket order. It is **world-size-agnostic**: `checkpoint()` gathers any sharded optimizer state (collective under sharding — call it on every rank), so a run sharded across N ranks can `restore` on M. `save`/`load` use a compact binary format.

**Background data prefetch:** `Trainer::run_prefetched(next_batch, on_step)` runs `next_batch` on a producer thread (double-buffered) so data loading / augmentation overlaps compute — a throughput win when the data pipeline is non-trivial (real datasets, disk, augmentation). Numerically identical to `run`; the provider must be `Send`.

## LoRA adapters

```rust
use rlx_tune::adapter::{LoraSpec, LoraInit, fuse_lora};

let spec = LoraSpec { rank: 16, alpha: 32.0, /* … */ ..Default::default() };
// inject into a forward graph, train, then fuse the adapter back into base weights:
// fuse_lora(&mut weights, &adapter)?;
```

## How it fits

Backend-agnostic; builds on `rlx-autodiff` + `rlx-optim` and `rlx-driver`'s collectives. Adapters fuse into any RLX model's weights — e.g. the model-specific trainers [rlx-qwen3-tts-train](../rlx-qwen3-tts-train) and [rlx-voxtral-tts-train](../rlx-voxtral-tts-train) use the same machinery.
