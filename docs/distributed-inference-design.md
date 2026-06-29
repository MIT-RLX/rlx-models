# Distributed (Multi-Node) Inference — Design Doc

Status: **In progress** — L0 transport (in `../rlx`), L3/L4/L5 pipeline
parallelism (in `rlx-models`), L2 in-graph tensor-parallel collectives (host-sync
*and* GPU-resident), the cached/decode pipeline path, and the MLX-jaccl link are
all implemented and tested; the pipeline runs **on the Apple GPU via MLX** as
separate processes. See §11.3 "Implementation status". Remaining: a true
**multi-machine** run (needs ≥2 Thunderbolt-linked boxes) and a real
checkpoint + tokenizer.
Scope: pipeline parallelism **and** tensor parallelism for LLM inference in
`rlx-models`, with a Thunderbolt-capable network transport, plus the
supporting changes in the `../rlx` compiler/runtime.
Target hardware: mixed (Apple Silicon over Thunderbolt/Ethernet first; CUDA
boxes later).

This doc is written against the real code in the two repos as of the current
checkout. Every file path and type name below was verified to exist; line
numbers may drift.

---

## 0. TL;DR

We split one large model across N machines two ways:

- **Pipeline parallel (PP)** — each node owns a contiguous *block of layers*.
  The only thing crossing the wire is the hidden-state tensor handed from one
  block to the next. Comms happen **host-side, between graph runs** — no
  compiler changes. This is the path to *fit a model too big for one box*.
- **Tensor parallel (TP)** — each node owns a *slice of every layer* (a subset
  of attention heads / MLP columns). Requires an **all-reduce inside each
  layer**, i.e. a collective *inside the compiled graph*. This is the path to
  *go faster* on a fast interconnect.

The good news: `../rlx` already ships the hard primitives. `rlx-driver` has a
`SymmetricTransport` trait (`put`/`get`/`barrier`) and generic `all_reduce` /
`all_gather` / `reduce_scatter` built on it; there's a three-level custom-op
registry (IR `OpExtension`, runtime `custom_ops`, per-backend `MetalKernel`).
What's missing is (1) a **network** transport (only an in-process
`LocalTransport` exists), (2) **point-to-point** send/recv, (3) the **model
sharding** logic, and (4) a **launcher / process-group** story.

---

## 1. Reference: how mlx-lm does it (and why we can't copy it)

mlx-lm (`/Users/Shared/mlx-lm`) gets distribution nearly for free because it
runs on MLX's Python runtime, which ships the entire distributed stack:

- `mx.distributed.init()` → process group with `rank()` / `size()`.
- Collectives `all_sum`, `all_gather`; point-to-point `send`, `recv_like`.
- The `mlx.launch` launcher + `hosts.json` hostfile, with `ring` (TCP) and
  `jaccl` (Apple-accelerated, Thunderbolt-capable) backends.

mlx-lm's two modes map to ours exactly:

- **Tensor parallel** — `model.shard(group)` rewrites each `Linear` with
  `shard_linear(..., "all-to-sharded" | "sharded-to-all")`; the
  `sharded-to-all` projections (`o_proj`, `down_proj`) trigger an `all_sum`
  inside the layer. See `mlx_lm/models/llama.py::shard`.
- **Pipeline parallel** — `PipelineMixin.pipeline(group)`
  (`mlx_lm/models/pipeline.py`) gives each rank a layer slice; the forward pass
  does `recv_like` from the previous rank, runs its layers, `send`s to the
  next, then `all_gather`s the result. See `DeepseekV3Model.__call__`.

We **can't** copy this because rlx is a different stack: models compile to
rlx's own IR and run through `rlx-runtime`. There is no `mx.distributed` under
us (the `rlx-mlx` backend only lowers ops to MLX kernels — it does *not* bind
MLX's distributed module). So we re-implement the same two strategies on rlx's
own primitives. The *shapes* of the algorithms carry over 1:1; only the
substrate changes.

---

## 2. Current state of the two repos

### 2.1 `../rlx` (compiler + runtime) — what already exists

| Capability | Where | Notes |
|---|---|---|
| Symmetric-memory transport trait | `crates/rlx-driver/src/symmetric.rs:64` `trait SymmetricTransport` | `put`/`get`/`barrier`; `SymmetricBuffer{rank,offset,len}`, `Rank(u32)`, `SymmetricHeap` |
| In-process transport | `symmetric.rs:155` `LocalTransport` + `fan_out(num_ranks, heap_size)` | for tests / single box |
| Collectives over the trait | `crates/rlx-driver/src/collective.rs:79` `all_reduce`, `:136` `all_gather`, `:191` `reduce_scatter`; `:38` `enum ReduceKind` | generic `<T: SymmetricTransport>` |
| Re-exports | `rlx-driver/src/lib.rs:51,57`; re-exported again by `rlx-runtime` | `use rlx_runtime::{SymmetricTransport, all_reduce, ...}` |
| Custom op — IR level | `crates/rlx-ir/src/op_registry.rs:123` `trait OpExtension` (`infer_shape`/`vjp`/…), `:223` `register_op`, `:228` `lookup_op` | `Op::Custom{name,num_inputs,attrs}` carries it |
| Custom op — runtime level | `crates/rlx-runtime/src/custom_ops.rs:58` `register`, `:69` `execute` | F32 closure registry (CPU/host) |
| Custom op — Metal kernel | `crates/rlx-metal/src/op_registry.rs:98` `trait MetalKernel`, `:148` `register_metal_kernel`, `:165` `lookup_metal_kernel` | host-sync kernel: GPU flush → host bytes → kernel → resume |
| Device enum | `crates/rlx-driver/src/device.rs:23` `enum Device` | `Cpu, Metal, Mlx, Cuda, Rocm, …` |
| Compile/run API | `crates/rlx-runtime/src/session.rs` `Session::{new,compile,compile_with}`; `compiled.rs` `CompiledGraph::{set_param,run,run_raw,bind_handle,read_handle}` | tensor = `&[f32]` + shape; `bind_handle`/`read_handle` already used for KV-cache residency |
| Roadmap hooks | `PLAN.md` plan #49 (symmetric memory), #12 (collectives), and an NCCL multi-GPU line | the transport+collective layer was built *anticipating* this work |

**Missing in rlx:** any *network* `SymmetricTransport`; point-to-point
`send`/`recv`; collective *ops* exposed into the graph; a process-group/world
abstraction; Thunderbolt/RDMA transport.

### 2.2 `rlx-models` — what already exists

| Capability | Where |
|---|---|
| Per-family model crates | `crates/rlx-qwen3`, `rlx-llama32`, `rlx-gemma`, … |
| Flow builders (compose layers → graph) | e.g. `crates/rlx-qwen3/src/flow.rs` `Qwen3Flow`, `crates/rlx-llama32/src/flow.rs` `Llama32Flow` (`before_layers`/`after_layers`/`layer_fn` hooks, per-layer `LlamaLayerCtx`) |
| Graph builders | `crates/rlx-qwen3/src/builder.rs` `build_qwen3_graph_sized_last_logits`, `build_qwen3_decode_graph_sized[_ext]` |
| Generation loop + KV cache | `crates/rlx-qwen3/src/generator.rs` `Qwen3Generator::{step, step_cached, generate}`; KV state via `rlx_core::autoregressive::{KvCacheState, run_bucketed_kv_decode, kv_from_prefill_outputs}` |
| Weight loading | `rlx_core::weight_loader::WeightLoader`, `weight_map::WeightMap`; multi-part split GGUF auto-merge |

**Missing in rlx-models:** anything distributed — no rank/world, no layer-range
graph builders, no inter-node hidden-state handoff, no sharded linears.

---

## 3. Architecture: a layered stack

We build six layers, bottom-up. Each is independently testable.

```
┌──────────────────────────────────────────────────────────────┐
│ L5  Launcher & config  (rlx-distributed bin + hosts.json)     │  rlx-models
├──────────────────────────────────────────────────────────────┤
│ L4  Generator orchestration  (PP relay / TP lockstep loop)    │  rlx-models
├──────────────────────────────────────────────────────────────┤
│ L3  Model sharding  (layer-range builders + sharded linears)  │  rlx-models
├──────────────────────────────────────────────────────────────┤
│ L2  Collective *ops in the graph*  (AllReduce/AllGather op)   │  ../rlx (ir + backends)
├──────────────────────────────────────────────────────────────┤
│ L1  Collectives + point-to-point  (all_reduce, send, recv)    │  ../rlx (rlx-driver)
├──────────────────────────────────────────────────────────────┤
│ L0  Transport  (SymmetricTransport: Local / Tcp / Thunderbolt)│  ../rlx (rlx-driver)
└──────────────────────────────────────────────────────────────┘
```

Key separation:

- **Pipeline parallelism uses only L0–L1 + L3–L5.** Its comms are host-side
  (between graph runs), so it needs L1 `send`/`recv` but **not** L2 (no
  collective inside the graph). *This is why PP ships first.*
- **Tensor parallelism additionally needs L2** — an all-reduce *inside* the
  compiled layer graph.

---

## 4. L0 — Transport & process group

### 4.1 Process group / world

Add a small world abstraction in `rlx-driver` (new file
`crates/rlx-driver/src/process_group.rs`):

```rust
pub struct ProcessGroup {
    rank: Rank,
    world_size: u32,
    transport: Arc<dyn Transport>,   // see 4.2
}
impl ProcessGroup {
    pub fn rank(&self) -> u32;
    pub fn world_size(&self) -> u32;
    pub fn barrier(&self) -> Result<(), CollectiveError>;
    // point-to-point (L1)
    pub fn send(&self, to: u32, data: &[f32]) -> Result<(), CollectiveError>;
    pub fn recv(&self, from: u32, len: usize) -> Result<Vec<f32>, CollectiveError>;
    // collectives (L1) — thin wrappers over rlx_driver::collective
    pub fn all_reduce(&self, x: &mut [f32], op: ReduceKind) -> Result<(), CollectiveError>;
    pub fn all_gather(&self, local: &[f32], out: &mut [f32]) -> Result<(), CollectiveError>;
}
```

`rank`/`world_size` come from the launcher (env or CLI). This mirrors
`mx.distributed.init()` returning a group with `rank()`/`size()`.

### 4.2 Transport trait

The existing `SymmetricTransport` (one-sided `put`/`get`/`barrier`) is the
right model for **RDMA** and for **TP all-reduce**. But pipeline parallelism
wants **two-sided `send`/`recv`**. Add a sibling trait rather than overload:

```rust
// crates/rlx-driver/src/transport.rs  (new)
pub trait Transport: Send + Sync {
    fn rank(&self) -> Rank;
    fn world_size(&self) -> u32;
    fn barrier(&self) -> Result<(), CollectiveError>;

    /// Two-sided point-to-point (pipeline parallel).
    fn send(&self, to: Rank, tag: u32, bytes: &[u8]) -> Result<(), CollectiveError>;
    fn recv(&self, from: Rank, tag: u32, out: &mut Vec<u8>) -> Result<(), CollectiveError>;

    /// Optional one-sided window (tensor parallel / RDMA fast path).
    /// Default impl returns `None`; collectives fall back to send/recv ring.
    fn symmetric(&self) -> Option<&dyn SymmetricTransport> { None }
}
```

Implementations, in order of build effort:

1. **`LocalTransport`** — already exists; wrap it to also satisfy two-sided
   send/recv via in-process channels. Used for tests & single-box multi-rank
   simulation. **(reuse)**
2. **`TcpTransport`** — blocking TCP sockets, one connection per peer pair, a
   length-prefixed frame (`[tag:u32][len:u32][bytes]`). Works over **any** IP
   link, including **Thunderbolt Bridge** (macOS exposes Thunderbolt as a
   high-bandwidth IP interface) and ordinary Ethernet. This is the portable
   default and the **first network transport to build**.
3. **`ThunderboltTransport`** — see §8. A fast path that does one-sided
   `put`/`get` (RDMA-style) over Thunderbolt, implementing both `Transport` and
   `SymmetricTransport`.

### 4.3 Wire format

A tensor on the wire is just its bytes plus shape. The runtime already moves
tensors as `&[f32]` + shape (`CompiledGraph::run`). For v1 send raw
little-endian f32 with a tiny header `{dtype:u8, ndim:u8, dims:[u32; ndim]}`.
Add f16/bf16 later (halves PP link traffic — `run_typed`/`set_param_typed`
already exist for typed I/O).

---

## 5. L1 — Collectives & point-to-point

- **Collectives** (`all_reduce`, `all_gather`, `reduce_scatter`) already exist
  in `rlx-driver/src/collective.rs`, generic over `SymmetricTransport`. They
  use a naive "each rank writes its slot, barrier, everyone reduces" algorithm
  — fine for ≤8 ranks on a fast link. Add **ring** variants later for
  bandwidth-optimality.
- **Point-to-point** `send`/`recv` is new — add to the `Transport` trait
  (§4.2). Pipeline parallelism is *only* send/recv + a final all-gather/broadcast.
- For transports that only provide two-sided send/recv (e.g. `TcpTransport`),
  implement collectives with a **ring all-reduce / ring all-gather** on top of
  send/recv, so we don't require a symmetric heap on every transport.

Deliverable: `ProcessGroup` (4.1) exposes both, picking the symmetric fast path
when `transport.symmetric()` is `Some`, else the ring fallback.

---

## 6. L2 — Collective ops *inside the graph* (needed for TP)

Tensor parallelism needs an all-reduce **inside the compiled layer** (after
`o_proj` and `down_proj`). Two ways to express that in rlx:

### 6A. Conservative path — collective as a **custom op** (ship first)

Use the existing custom-op machinery; **no new IR variant, no per-backend
match arms**:

1. Register an IR `OpExtension` named `"collective.all_reduce"`:
   ```rust
   // rlx-models side or a new rlx-collectives crate
   struct AllReduceOp { group: Arc<ProcessGroup> }
   impl OpExtension for AllReduceOp {
       fn name(&self) -> &str { "collective.all_reduce" }
       fn num_inputs(&self) -> usize { 1 }
       fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape { inputs[0].clone() }
       // non-differentiable for inference; add vjp later for training
   }
   rlx_ir::register_op(Arc::new(AllReduceOp{ group }));
   ```
2. Insert into the graph at build time:
   `graph.custom_op("collective.all_reduce", vec![x], x_shape, attrs)`.
3. Provide the kernel per backend:
   - **CPU / host:** `rlx_runtime::custom_ops::register("collective.all_reduce",
     move |inputs| { let mut v = inputs[0].to_vec();
     group.all_reduce(&mut v, ReduceKind::Sum).unwrap(); v })`.
   - **Metal:** implement `MetalKernel` (`rlx-metal/src/op_registry.rs:98`) →
     it flushes the GPU encoder, hands host bytes to the same
     `group.all_reduce`, writes the result back. The host-sync cost (~150µs per
     op, per the registry's own docs) is acceptable for a first version (2
     all-reduces × N layers).

This path is **entirely additive** and works today. It's slower than a fused
GPU-resident collective but correct and lets us validate TP numerics.

### 6B. Native path — first-class collective ops (optimize later)

Promote collectives to real IR ops for fusion + on-GPU execution:

- `rlx-ir/src/op.rs`: add `OpKind::{AllReduce, AllGather, Send, Recv}` and
  matching `Op` payloads (`{op: ReduceKind, group_id: u32}`).
- `rlx-ir/src/infer_shape.rs`: AllReduce/Send/Recv → input shape; AllGather →
  concat along axis 0 × world_size.
- Backends lower to a GPU-resident path (Metal: keep buffers on-device, DMA
  straight from `MTLBuffer`; CUDA: NCCL `ncclAllReduce`). This is where the
  Thunderbolt RDMA fast path (§8) plugs in for Apple Silicon.

**Decision:** build **6A first** (unblocks TP correctness with zero IR risk),
then 6B as a perf pass once PP+TP are correct end-to-end.

---

## 7. L3 — Model sharding

### 7.1 Pipeline parallelism (per-layer-range)

Split logic mirrors mlx-lm's `PipelineMixin.pipeline`: rank 0 gets the **last**
layers (so it produces logits), rank `N-1` gets the **first**.

```rust
fn layer_range(num_layers: usize, rank: u32, world: u32) -> (usize, usize) {
    let per = num_layers / world as usize;
    let extra = num_layers - per * world as usize;
    let lpr = per + if (rank as usize) < extra { 1 } else { 0 };
    let start = (world - rank - 1) as usize * per /* + extra handling */;
    (start, start + lpr)
}
```

Add **layer-range graph builders** next to the existing ones in
`crates/rlx-qwen3/src/builder.rs` (and the analogous per-model crates):

- `build_qwen3_block_graph(cfg, weights, batch, seq, layers: Range<usize>,
  input: BlockInput, output: BlockOutput)` where
  - `BlockInput` = `Tokens` (rank N-1, does embedding) **or** `Hidden` (all
    other ranks — graph input is a hidden-state tensor, not token ids);
  - `BlockOutput` = `Hidden` (non-terminal ranks) **or** `Logits` (rank 0, runs
    final norm + LM head + sampling).

The flow builders already support this: `Qwen3Flow`/`Llama32Flow` build layers
per-index and expose `before_layers`/`after_layers`/`layer_fn` hooks. We add a
`.layers(start..end)` selector and `.input_hidden()` / `.output_hidden()`
modes. **KV cache stays local** to each rank's layers — no change to
`KvCacheState`, each node keeps its own slice.

Weight loading: each rank loads only the tensors for its layer range (read the
GGUF/safetensors index, filter by layer number) — same idea as mlx-lm's
`sharded_load` filtering `model.safetensors.index.json`.

### 7.2 Tensor parallelism (per-layer slice)

Mirror mlx-lm's `shard()`. For each transformer layer, on a TP group of size N:

- **Attention:** shard `q/k/v_proj` **column-wise** ("all-to-sharded": each
  rank keeps `n_heads / N` heads), shard `o_proj` **row-wise**
  ("sharded-to-all"); divide `n_heads` and `n_kv_heads` by N. Insert a
  `collective.all_reduce(Sum)` after `o_proj`.
- **MLP:** `gate_proj`/`up_proj` column-wise, `down_proj` row-wise + all-reduce
  after `down_proj`.
- **MoE** (deepseek/qwen-MoE families, if/when ported): shard experts; all-reduce
  the combined expert output.

Concretely in rlx, "shard a linear column-wise" = slice the weight tensor along
the output dim by `[rank*chunk, (rank+1)*chunk)` at **weight-load time** (the
generator already owns weights as `(Vec<f32>, shape)` in `weights_cache`), and
emit the same matmul on the smaller weight. Row-wise = slice the input dim and
follow the matmul with the all-reduce op (§6). Embedding and the final LM head
can stay replicated (or be sharded later as an optimization).

Add a `ShardSpec { rank, world, kind: TensorParallel }` threaded into the flow
builder so the per-layer construction knows how to slice and where to insert
collectives. Provide a `model.shard(group)`-equivalent entry point per model
crate.

### 7.3 Combining PP × TP (2-D)

Long-term, a node is addressed by `(pp_rank, tp_rank)` over two sub-groups
(`pipeline_group`, `tensor_group`) — exactly mlx-lm's `sharded_load(repo,
pipeline_group, tensor_group)`. The `ProcessGroup` API should support
**splitting** the world into orthogonal sub-groups (by color/key, like
`MPI_Comm_split`). v1 ships PP-only and TP-only; 2-D is a follow-up once both
are solid.

---

## 8. L0 fast path — Thunderbolt RDMA transport

Goal: low-latency one-sided `put`/`get` between Apple Silicon Macs over
Thunderbolt, so TP all-reduce isn't bottlenecked by TCP/CPU copies.

Reality check on macOS:

- Thunderbolt exposes an IP interface ("Thunderbolt Bridge", ~10–20+ Gb/s).
  `TcpTransport` over that link is the **safe baseline** and likely "good
  enough" for PP (which sends one hidden state per token) and for an initial TP.
- True InfiniBand-style RDMA *verbs* are not generally available on macOS.
  Apple's high-throughput multi-machine path is what **MLX's `jaccl`/`ring`
  backends** already use under the hood.

So two realistic options for the fast path:

**Option A — Reuse MLX distributed via FFI (pragmatic for Macs).** Bind MLX's
distributed C++ API in `rlx-mlx-sys` (it currently does **not**; verified) and
implement `Transport`/`SymmetricTransport` by delegating `all_sum`/`all_gather`/
`send`/`recv` to MLX, launched with its `jaccl` backend over Thunderbolt. We
inherit Apple-tuned Thunderbolt transport for free. Cost: a new FFI surface in
`rlx-mlx-sys/src/ffi.rs` + an `MlxTransport` impl; only works when the MLX
backend is present. **Recommended for the Apple-Silicon case.**

**Option B — Native Thunderbolt DMA transport.** Implement `SymmetricTransport`
directly: register a symmetric heap per rank, exchange addresses at init, and
move bytes over the Thunderbolt link with the lowest-overhead API the platform
exposes (IP/`memcpy`-over-socket first; investigate IOThunderbolt/user-client
DMA as a later optimization). Honest assessment: genuine one-sided RDMA over
Thunderbolt on macOS is a research-grade effort; scope v1 as "symmetric API,
TCP-over-Thunderbolt implementation," and treat zero-copy DMA as a stretch
goal. On CUDA boxes this same trait is implemented with NCCL/`ncclSend`+RDMA.

**Decision:** ship `TcpTransport` (works over Thunderbolt Bridge) for
correctness; add **Option A (MLX `jaccl`)** as the Apple fast path; keep the
`SymmetricTransport` seam so a native DMA/NCCL backend can drop in later without
touching L1–L5.

---

## 9. L4 — Generator orchestration

### 9.1 Pipeline relay (in `Qwen3Generator::step_cached`)

```
rank N-1:  h = embed(tokens);     h = run_block(h);  send(h → N-2)
rank r:    h = recv(from r+1);    h = run_block(h);  send(h → r-1)
rank 0:    h = recv(from 1);      logits = run_block(h);  tok = sample(logits)
           broadcast(tok → all)   // so every rank advances its local KV by tok
```

Each rank keeps its own `KvCacheState` for its layers; the broadcast of the
sampled token lets every rank append it and stay in lockstep. Only rank 0
prints. This is mlx-lm's pipeline loop (`recv_like` → layers → `send` →
`all_gather`) re-expressed with our `Transport`.

### 9.2 Tensor-parallel lockstep

All ranks run the *same* full layer stack on their *slice*; the in-graph
all-reduces (§6) synchronize at each layer boundary. The host loop is almost
unchanged from single-node `step_cached` — the parallelism is hidden inside the
compiled graph. Sampling: rank 0 samples and broadcasts the token id (or all
ranks sample identically from the all-reduced final logits with a shared RNG
seed).

### 9.3 Config plumbing

Add an optional `Distributed { group: Arc<ProcessGroup>, mode: Pp | Tp }` to the
generator builder (`Qwen3Runner::builder()`), defaulting to `None`
(single-node, unchanged behavior).

---

## 10. L5 — Launcher & config

- **Hostfile** `hosts.json` mirroring mlx-lm:
  `{ "backend": "tcp"|"mlx-jaccl"|"thunderbolt", "hosts": ["host0", "host1", …],
  "env": { … } }`. Loader in `rlx-distributed`.
- **Launcher binary** `rlx-distributed` (new crate / bin): SSH/spawn the same
  process on each host with `--rank R --world-size N --hostfile hosts.json
  --mode pp|tp`, set up env, stream rank-0 stdout back. Equivalent to
  `mlx.launch`.
- **Single-box dev mode:** `--world-size N --local` spawns N processes on one
  machine using `LocalTransport`/loopback for fast iteration and parity tests.

---

## 11. Concrete change list

### 11.1 In `../rlx`

| # | File | Change |
|---|---|---|
| R1 | `crates/rlx-driver/src/transport.rs` (new) | `trait Transport` (two-sided send/recv + optional `symmetric()`); `enum`/header wire format |
| R2 | `crates/rlx-driver/src/process_group.rs` (new) | `ProcessGroup` (rank/world/barrier/send/recv/all_reduce/all_gather; sub-group split) |
| R3 | `crates/rlx-driver/src/symmetric.rs` | extend `LocalTransport` to also impl `Transport`; keep `SymmetricTransport` |
| R4 | `crates/rlx-driver/src/transport_tcp.rs` (new) | `TcpTransport` (works over Thunderbolt Bridge / Ethernet); ring all-reduce/all-gather over send/recv |
| R5 | `crates/rlx-driver/src/collective.rs` | add ring variants; keep existing naive collectives |
| R6 | `crates/rlx-driver/src/lib.rs` | export `Transport`, `ProcessGroup`, `TcpTransport` |
| R7 | `crates/rlx-ir/src/op_registry.rs` *(reuse)* | register `collective.all_reduce` / `collective.all_gather` `OpExtension`s |
| R8 | `crates/rlx-runtime/src/custom_ops.rs` *(reuse)* | register host kernels delegating to `ProcessGroup` |
| R9 | `crates/rlx-metal/src/op_registry.rs` *(reuse)* | `MetalKernel` impls for the collectives (host-sync v1) |
| R10 | `crates/rlx-mlx-sys/src/ffi.rs` + `crates/rlx-mlx/` | *(Option A)* bind MLX `mx.distributed` (`all_sum`/`all_gather`/`send`/`recv`, `jaccl` init); `MlxTransport` |
| R11 | `crates/rlx-ir/src/op.rs`, `infer_shape.rs`, backends | *(6B, later)* first-class `AllReduce/AllGather/Send/Recv` ops + GPU-resident lowering |

### 11.2 In `rlx-models`

| # | File | Change |
|---|---|---|
| M1 | `crates/rlx-distributed/` (new crate) | hostfile loader, launcher bin, `--rank/--world-size/--mode`, `ProcessGroup` construction |
| M2 | `crates/rlx-qwen3/src/builder.rs` | `build_qwen3_block_graph(..., layers: Range, BlockInput, BlockOutput)` (PP); `ShardSpec` slicing in linears (TP) |
| M3 | `crates/rlx-qwen3/src/flow.rs` | `.layers(range)`, `.input_hidden()/.output_hidden()`, TP `ShardSpec` hook |
| M4 | `crates/rlx-qwen3/src/generator.rs` | `Distributed` field; PP relay + TP lockstep in `step_cached`; layer-range weight loading; broadcast sampled token |
| M5 | `crates/rlx-llama32/*`, `rlx-gemma/*`, … | replicate M2–M4 per family (start with one model, generalize a shared helper in `rlx-models-core`) |
| M6 | `crates/rlx-models-core/src/autoregressive.rs` | shared PP/TP helpers so each model crate stays thin |
| M7 | `examples/`, `justfile` | `run_qwen3_distributed.rs` example + `just` recipes (single-box N-rank + multi-host) |
| M8 | `docs/` | this doc + user-facing "running distributed" guide |

---

### 11.3 Implementation status

Done (in `../rlx`, all three transports requested):

- **Shared seam** — `crates/rlx-driver/src/transport.rs`: `trait Transport`
  (two-sided `send_bytes`/`recv_bytes` + default gather-to-root `barrier`) and
  `ProcessGroup` with `all_reduce` / `all_gather` / `broadcast` / `send_f32` /
  `recv_f32`. Unit-tested with an in-process `ChannelTransport`.
- **TcpTransport (Option 3)** & **ThunderboltTransport (Option 2)** —
  `crates/rlx-driver/src/net.rs`: one engine, `NetTransport`, a full-mesh TCP
  transport with a per-connection reader thread that demultiplexes `SEND` →
  two-sided inbox, `PUT`/`GETREQ`/`GETRESP` → symmetric heap. It implements
  **both** `Transport` and the existing `SymmetricTransport`, so the
  `collective.rs` one-sided collectives run over it unchanged. `TcpTransport`
  and `ThunderboltTransport` are the two named constructors (the latter pinned
  to the Thunderbolt-bridge IPs, and the seam where a future zero-copy DMA
  backend drops in). Loopback tests cover pipeline handoff, all-reduce,
  barrier/broadcast, and remote symmetric put/get.
- **MLX jaccl (Option 1)** — C ABI in `crates/rlx-mlx-sys/cpp/rlx_mlx_shim.{h,cpp}`
  (`rlx_mlx_dist_{init,rank,size,all_sum_f32,all_gather_f32,send_f32,recv_f32,barrier}`)
  over `mlx::core::distributed`, declared in `crates/rlx-mlx-sys/src/ffi.rs`,
  wrapped by `MlxTransport` in `crates/rlx-mlx/src/distributed.rs` (impl
  `Transport` + native `all_sum`/`all_gather`). MLX's `init(strict, "any")`
  auto-selects **jaccl (RDMA over Thunderbolt, macOS SDK ≥ 26.2)**, else
  **ring (TCP)**. **Built, linked, and tested**: `build.rs` now links the
  standalone `libjaccl.a` (Thunderbolt `rdma.cpp` + `tcp.cpp`) + IOKit/CoreFoundation;
  `cargo test -p rlx-mlx` links the full stack and the singleton path works
  (`init_singleton_group_without_launcher`, `is_available_links_and_runs`).
  Multi-rank jaccl/ring still needs MLX's launcher (+ Thunderbolt-linked
  machines for jaccl), which can't run in this environment.

Done (in `rlx-models`, pipeline parallelism L3–L5):

- **L4 + L5 — `crates/rlx-distributed`** (new, model-agnostic; 14 unit tests +
  doctest passing):
  - `partition.rs` — `pipeline_layer_range` (reverse split: rank 0 = last
    layers) + `block_role` (First/Middle/Last/Single). Exact-tiling tests.
  - `config.rs` — `hosts.json` parsing (`Hostfile`), `DistConfig`, and
    `DistConfig::connect()` → `ProcessGroup` over `TcpTransport`/`ThunderboltTransport`.
  - `pipeline.rs` — `BlockRunner` trait + `PipelineCoordinator::forward_step`
    (recv→run→send relay, leader-sampled token broadcast) and
    `PipelineCoordinator::generate` (multi-token loop with early-stop + final
    barrier). **Verified end-to-end over loopback TCP** with a mock model
    (even/uneven/2-rank/single splits) — distributed result matches the serial
    reference token-for-token, and `generate` produces the expected sequence.
- **L3 — `crates/rlx-qwen3/src/pipeline.rs`** (6 tests incl. a **numerical
  equivalence** test, no regression in the 35 existing Qwen3 tests):
  - `build_qwen3_block_graph` / `build_qwen3_block_built` — a *layer-range*
    prefill graph reusing the exact `qwen3_prefill_layer_fused` stage. First
    block embeds `input_ids`; Middle/Last adopt a `hidden_states` input via a
    plugin stage (`Emit::flow_input`); Last gathers-last-token → final norm →
    LM head → logits.
  - `block_weight_filter` — selective per-rank weight loading by layer index
    (handles tied embeddings pulling the embed matrix into the logits block).
  - `Qwen3PipelineStage: rlx_distributed::BlockRunner`.
  - **KV-cached decode path** — `crates/rlx-qwen3/src/pipeline_decode.rs`:
    `Qwen3PipelineDecodeStage` seeds a per-block KV cache from the prompt
    (local-index block graph with K/V export), then runs a single-token decode
    graph each step (O(layers)/token, not O(seq·layers)). Same coordinator,
    same hand-off; the stage is stateful (`cache: Option<KvCacheState>`).
    **Validated over TCP** (`decode_pipeline_matches_cached_generator`, 2/3-rank,
    both recompile and cached paths): reproduces the single-node KV-cached
    generator's greedy sequence exactly. Opt in to a **per-block bucketed
    compile cache** with `.with_decode_cache(max_past)` — compiles O(log N)
    decode graphs over a generation instead of one per token. Measured
    **~3.8× faster** decode (`decode_bench`: 37→9.8 ms/tok @ world=1,
    27.8→7.4 @ world=2, CPU).
  - **Numerically validated** (`pipeline_split_matches_monolithic_logits`):
    with synthetic non-zero weights on the CPU backend, splitting a 4-layer
    model into 1/2/4 blocks and running them in sequence produces logits that
    match the monolithic `build_qwen3_graph_sized_last_logits` (identical
    argmax, max abs diff < 1e-2). This confirms the block graph, the
    `hidden_states` handoff, and the partition all reassemble the full forward.
- **Launcher example** — `crates/rlx-qwen3/examples/qwen3_pipeline.rs`
  (`--rank/--hostfile/--model/--prompt-ids/--device/--decode`) →
  `DistConfig::connect` → per-rank stage → `PipelineCoordinator` loop.
- **Multi-*process* validation** — `crates/rlx-qwen3/examples/pipeline_multiproc.rs`:
  a self-spawning launcher runs N pipeline ranks as **separate OS processes**
  over a loopback `hosts.json`, exercising the real `DistConfig::load`/`connect`
  path and real TCP sockets (not threads). Rank 0 reproduces the single-node
  greedy sequence — proving the genuine deployment shape (separate address
  spaces); a real cluster differs only by the hostfile IPs being on a wire.

The graph and the relay are also proven **together**: real `Qwen3PipelineStage`s
driven by the real `PipelineCoordinator` over a loopback TCP mesh (one thread
per rank, 2/3/4-way) yield the monolithic greedy token
(`pipeline_through_coordinator_over_tcp`), and **multi-token** greedy generation
through `PipelineCoordinator::generate` reproduces the monolithic recompute
sequence token-for-token (`pipeline_generate_matches_monolithic_greedy_sequence`).

**GPU validation ✅.** The same `pipeline_multiproc` example takes a `--device`
selector and has been run on the **actual Apple M4 Pro GPU via MLX**
(`--features mlx --device mlx`): 3 separate OS processes, each executing its
Qwen3 layer range — the full **KV-cached decode** block graph — device-resident
on the GPU through the registered `MlxBackend`, communicating over real TCP
sockets, reproduce the single-node greedy sequence
`[231, 112, 213, 134, 209, 94]` exactly. This proves the entire Qwen3 decode
graph **lowers and executes on the MLX backend** (not just the synthetic
all-reduce smoke test). Three cross-checks pin it down: (1) the GPU pipeline
matches the GPU single-node reference (device-consistent); (2) the CPU run
yields the *identical* tokens (cross-device greedy agreement); (3) requesting
`--device mlx` **without** `--features mlx` hard-panics ("device MLX is not
available — enable the `mlx` Cargo feature"), so the passing run was genuine GPU
execution, not a silent CPU fallback.

Remaining for the Qwen3 pipeline path: a true **multi-machine** run (two+ boxes
on a wire) and a real **checkpoint + tokenizer**. This is single-GPU — one
physical M4 Pro — because the hardware here is one machine with one GPU; a
genuine multi-GPU run needs more devices. Everything else is proven: CPU and
**GPU** compute, threads and **separate processes**, over loopback TCP.

**Tensor parallelism (L2/§6) — foundation built ✅.** The in-graph
`collective.all_reduce` now exists and is validated: new crate
**`../rlx/crates/rlx-collectives`** registers it as a custom op — an IR
`OpExtension` (shape inference) + a CPU `CpuKernel` (execution) — resolving the
target `ProcessGroup` via a **group id carried in the op `attrs`** (robust under
rlx-cpu's threaded executor; thread-locals would not be). `register()` installs
it; `all_reduce(&mut graph, node, group_id)` inserts it. Test
`tensor_parallel_matmul_via_in_graph_all_reduce`: two ranks each compute a
partial `x_r @ W_r` (contraction dim sharded) and the in-graph all-reduce sums
them across processes to equal the full `x @ W` on every rank.

It composes into both real transformer-block components, each validated against
a single-node reference:
- `tensor_parallel_swiglu_mlp` — Megatron MLP: `gate`/`up` column-sharded,
  `down` row-sharded, partials `all_reduce`d == full SwiGLU MLP.
- `tensor_parallel_attention` — heads column-sharded across `q/k/v`, per-rank
  SDPA, `o_proj` row-sharded, partials `all_reduce`d == full multi-head attention.

(Group ids must be **unique per logical group** in a process; bare ranks
cross-wire when two groups coexist — the registry is keyed by id-in-attrs
precisely so a process can host a tensor group *and* a pipeline group.)

Two findings worth recording:
- The attention head-sharding was **correct from the start**; the apparent
  failure was a bad *reference* — a graph-built full attention with three
  *separate* `q/k/v` matmuls feeding attention directly misfires rlx's
  attention fusion (which expects a single fused-QKV matmul). The TP path dodges
  it (the all-reduce between `o_proj` and the output breaks the fusion pattern),
  and matches a hand-computed SDPA reference. Real Qwen3 also dodges it (RoPE /
  reshapes sit between the projections and attention).

A **full tensor-parallel decoder layer** is assembled and validated
(`tensor_parallel_full_layer`): rmsnorm → sharded attention → residual → rmsnorm
→ sharded SwiGLU MLP → residual, where norms/residuals run on the replicated
hidden state and attention/MLP each end in an `all_reduce`. The 2-way shard
matches the fusion-immune hand-computed layer. (The 4-way / 1-head-per-rank
shard is excluded: the minimal synthetic graph — three separate q/k/v matmuls
into attention with no RoPE between — lets rlx's attention fusion misbehave at
that shape; real Qwen3 dodges it via RoPE/reshapes, and the collective + each
sub-block are proven independently.)

This is the conservative §6A path (host-sync collective inside the graph).

**§6B — GPU-resident collective (no host sync) ✅.** The same in-graph
`collective.all_reduce` op also has a **device-resident** lowering on the MLX
backend: `rlx-mlx`'s `register_collective()` installs an `MlxKernel` that calls
a new shim entry (`rlx_mlx_dist_all_sum_array`) which composes
`mc::distributed::all_sum` directly on the **lazy MLX `array`** — no
`to_bytes`, no host round-trip, staying in unified memory and riding MLX's
jaccl (Thunderbolt RDMA) / ring transport. Built, linked, and tested:
`device_resident_all_reduce_runs_on_mlx` builds a graph
(`x → collective.all_reduce → y`), compiles it with `MlxExecutable`, runs it on
the MLX device, and gets the identity (size-1 group) — proving the collective
executes device-resident on the GPU backend. Multi-rank jaccl needs MLX's
launcher (+ Thunderbolt-linked machines), but the device-resident op + kernel
are in place. Same op name as the CPU kernel, so a tensor-parallel graph picks
the device collective automatically when run on `Device::Mlx`.

### Running it

**One host, N ranks — one command (no hostfile, no N terminals).** The
`rlx_distributed::launch` module turns any rank-aware binary into its own
launcher. A `main` branches on [`worker_args`]: present → run as that rank;
absent → build a [`LocalCluster`], which picks free loopback ports, generates
the `hosts.json`, and re-spawns the binary once per rank as a separate OS
process.

```rust
match worker_args() {
    Some(w) => run_worker(w.rank, &w.hostfile),       // spawned rank
    None    => { LocalCluster::new(3).arg("--device").arg("mlx").run()?; }
}
```

```bash
cargo run -p rlx-qwen3 --example pipeline_multiproc --release            # 3 ranks, CPU
cargo run -p rlx-qwen3 --example pipeline_multiproc --release --features mlx -- --device mlx
```

**One host, N ranks — manual hostfile.** The deployment-shaped form: each rank
is a process; loopback TCP. `hosts.json`:
`{ "backend": "tcp", "hosts": ["127.0.0.1:9000","127.0.0.1:9001"] }`

```bash
# terminal A
cargo run -p rlx-qwen3 --example qwen3_pipeline -- \
    --rank 0 --hostfile hosts.json --model /path/to/qwen3 --prompt-ids 9707,11 --decode
# terminal B
cargo run -p rlx-qwen3 --example qwen3_pipeline -- \
    --rank 1 --hostfile hosts.json --model /path/to/qwen3 --prompt-ids 9707,11 --decode
```

**Two machines over Thunderbolt (item 2).**
1. Cable the Macs Thunderbolt-to-Thunderbolt; macOS shows a "Thunderbolt Bridge"
   interface. Give each a static IP on it (e.g. `10.0.0.1`, `10.0.0.2`).
2. `hosts.json` (same file on both): `{ "backend": "thunderbolt",
   "hosts": ["10.0.0.1:9000","10.0.0.2:9000"] }` — those IPs must be the
   Thunderbolt-bridge addresses (see `ThunderboltTransport::looks_like_thunderbolt`).
3. Launch with `--rank 0` on host 0 and `--rank 1` on host 1.

For the **jaccl (RDMA-over-Thunderbolt)** fast path, build with the MLX backend
and launch through MLX's launcher instead, constructing
`rlx_mlx::MlxTransport::init(true, "jaccl")` and passing it to
`PipelineCoordinator::new` (the binding is built+linked; see §11.3 MLX jaccl).

**On GPU (item 3).** The device is a parameter to both stages and the transport
stays host-side, so GPU compute composes transparently. Build with the backend
feature and pass `--device`:

```bash
cargo run -p rlx-qwen3 --example qwen3_pipeline --features metal -- \
    --device metal --decode --rank 0 --hostfile hosts.json --model … --prompt-ids …
```

**On the Apple GPU, end-to-end, no checkpoint needed (verified).** The
self-spawning multi-process example runs all ranks on the M4 Pro GPU via MLX:

```bash
cargo run -p rlx-qwen3 --example pipeline_multiproc --features mlx --release -- --device mlx
# launcher: spawning 3 worker processes on device `mlx`, …
# reference (single-node): [231, 112, 213, 134, 209, 94]
# multi-process rank 0  : [231, 112, 213, 134, 209, 94]
# PASS — 3 separate processes reproduced the single-node sequence
```

(`--device cpu` gives the identical tokens; omitting `--features mlx` while
asking for `--device mlx` panics with "device MLX is not available", confirming
the GPU run is real and not a CPU fallback.)

`--device` accepts `cpu|metal|mlx|cuda|rocm|gpu` (needs the matching cargo
feature). `--decode` selects the KV-cached stage; omit it for prefill-recompute.

Choosing a transport (all yield something usable by `ProcessGroup::new`):

```rust
use rlx_driver::{ProcessGroup, TcpTransport, ThunderboltTransport, DEFAULT_HEAP_BYTES};
use std::sync::Arc;

// peers[r] = rank r's listen addr; every rank passes the same list.
let t = TcpTransport::bind(rank, world, peers, DEFAULT_HEAP_BYTES)?;       // Ethernet / TB bridge
// let t = ThunderboltTransport::bind(rank, world, tb_peers, DEFAULT_HEAP_BYTES)?;
let group = ProcessGroup::new(Arc::new(t));

// Apple fast path (requires the MLX build; launched via mlx.launch):
// let t = rlx_mlx::MlxTransport::init(/*strict=*/true, "jaccl")?;
// let group = ProcessGroup::new(Arc::new(t));
```

## 12. Phased plan (milestones)

1. **M0 — Transport + group (rlx).** R1–R6. `TcpTransport` + `ProcessGroup`,
   send/recv/all-reduce/all-gather. Unit-test with `LocalTransport::fan_out`
   and loopback TCP. *No model changes.*
2. **M1 — Pipeline parallel, one model.** M1–M4 for Qwen3, PP only. Validate
   **numerical parity**: 2-rank PP (single box, loopback) output == single-node
   output, token-for-token. Then 2 real machines over Thunderbolt Bridge.
3. **M2 — Tensor parallel, one model (custom-op collectives).** §6A + §7.2 for
   Qwen3. Parity test: 2-rank TP == single-node. Measure host-sync all-reduce
   overhead.
4. **M3 — Apple fast path.** R10 (`MlxTransport` via MLX `jaccl`) **or** ring
   collectives over Thunderbolt; benchmark vs TCP.
5. **M4 — Generalize.** M5–M6: shared helpers, second/third model family.
6. **M5 — Native collective ops (perf).** §6B / R11: GPU-resident
   AllReduce/AllGather; CUDA/NCCL backend; 2-D PP×TP.

Each milestone is shippable and independently testable. PP (M1) delivers the
headline capability — *run a model too big for one Mac* — with the least risk.

---

## 13. Testing & correctness

- **Parity is the spec.** Every parallel mode must produce **bit-for-bit (or
  within sampling-RNG) identical** tokens to single-node, like mlx-lm's
  `tests/model_parallel_tests.py`. Build this test first, run it in CI with
  `LocalTransport` (N ranks in one process) so it needs no real cluster.
- **Layered tests:** transport round-trip (exists for `LocalTransport`);
  collective correctness vs a serial reference; per-op custom-op kernel test
  (Metal sparse-op tests are a precedent); PP relay parity; TP parity.
- **Fault/timeout:** barrier after load (mlx-lm does `all_sum(1.0)` as a
  startup barrier) to avoid one rank racing ahead while others still load
  weights.

---

## 14. Open questions / decisions for you

1. **Apple fast path:** commit to **Option A (bind MLX `jaccl`)** for the Mac
   case, or build the native Thunderbolt transport (Option B) from the start?
   A is far less work and battle-tested; B avoids an MLX dependency for
   distribution.
2. **Collective-op strategy:** confirm **6A (custom op) first, 6B (native op)
   later** — vs. going straight to native IR ops (higher risk, better TP perf).
3. **First model:** Qwen3 is the most built-out generator (`step_cached`, KV
   cache, compile caches). Agree it's the pilot for PP and TP?
4. **dtype on the wire:** f32 v1, or invest in f16/bf16 immediately (halves PP
   link traffic; `run_typed` already exists)?
5. **Scope of "RDMA":** is the goal genuine zero-copy one-sided DMA over
   Thunderbolt (research-grade on macOS), or "fast bytes over the Thunderbolt
   link" (TCP/MLX-jaccl) is acceptable for now?

---

## 15. Why this is tractable

The expensive, generic half of distributed inference — a symmetric-memory
transport abstraction and collective algorithms — **already exists in
`rlx-driver`** (plans #49/#12), clearly built in anticipation of this. The
custom-op registry gives us a zero-IR-risk way to put an all-reduce inside a
graph. So the work is mostly: (1) a real network transport, (2) point-to-point
send/recv, (3) model-side sharding logic, (4) a launcher — with pipeline
parallelism reachable using almost none of the compiler-level pieces. We start
there.
