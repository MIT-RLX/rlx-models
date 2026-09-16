# rlx-ten-vad

[TEN-VAD](https://github.com/TEN-framework/ten-vad) ported to Rust on RLX.
Weights are embedded (305 KB), so there is **no ONNX Runtime and no
`libten_vad` at run time** — and no model download.

## Two builds

This crate is the **desktop and mobile** one: the model as an rlx graph, fully
fused, on seven backends. There is a separate **embedded** stack that shares
this crate's DSP frontend so feature extraction has one implementation:

| crate | for | inference |
|---|---|---|
| **`rlx-ten-vad`** (this one) | desktop, mobile, server | rlx graph — cpu, metal, mlx, wgpu, cuda, rocm, xdna |
| `rlx-ten-vad-core` | MCUs, `no_std` | f32 scalar, or integer-only (no FPU needed) |
| `rlx-ten-vad-mcu` | bare-metal RISC-V | firmware built on the core |
| `rlx-ten-vad-fpga` | FPGA | SystemVerilog, exported from this crate's rlx-ir graph |

The hardware targets are not a reimplementation. `rlx-fpga`'s sequential target
lowers the same graph this crate executes, and both the MCU weight blob and the
FPGA weight image come from one quantiser, so they are identical integers by
construction. See `crates/rlx-ten-vad-fpga/README.md` for the measured
numeric-format trade-offs.

## Why the FFT is transliterated, not `Op::Fft`

rlx has FFT ops (`Op::Fft`, `Graph::rfft_exact`, `Op::LogMel`) and the mel path
is expressible in about eight graph nodes. `src/ooura.rs` is a hand
transliteration anyway, because `f32` addition is not associative: a
mathematically-equivalent FFT gives a different spectrum, and bit-identity with
the upstream C is this port's headline verified property.

That trade is cheap. `cargo run -p rlx-ten-vad --release --example
frontend_profile`:

| | µs/frame | share of frontend |
|---|---|---|
| 1024-pt FFT | 1.74 | 9% |
| **pitch estimator** | **9.63** | **50%** |
| mel filterbank, window, log, normalise | 7.80 | 41% |
| *frontend total* | *19.2* | *48% of the frame* |
| network (f32 scalar) | 20.9 | — |

So swapping the FFT for graph ops would address **4% of a frame** and forfeit
bit-exactness. And it would not let the frontend move to an accelerator anyway:
the pitch estimator — half the frontend — consumes the *same* power spectrum
(`frontend.rs`: "Pitch runs on the un-emphasised signal but reuses this
spectrum"), and it is order-16 Levinson recursion plus a Viterbi backtrack.
That is sequential control flow, not dataflow; it does not become graph ops.

If frontend speed ever matters, the target is the mel filterbank at 41% — a
scalar 513×40 accumulation that wants SIMD or a matmul, not an FFT op.

Verified stage by stage against **the model TEN-framework publishes** —
`ten-vad.onnx` driven by the DSP in `src/*.cc`:

| checked | vs | result |
|---|---|---|
| **frontend features** | the upstream C DSP | **bit-identical** — 30 750/30 750 values, and 58 548/58 548 on real speech |
| network alone, same features | onnxruntime on `ten-vad.onnx` | **2.4e-7** max, 0 decision flips |
| whole pipeline | the same | **2.4e-7** max, 0 decision flips |

The frontend contributes *exactly zero*: `src/ooura.rs` is a verbatim
transliteration of the reference's Ooura split-radix FFT and `src/pitch.rs`
follows its `f32` arithmetic operation for operation, so the pipeline number is
the network number. That last 2.4e-7 is onnxruntime's `f32` accumulation order
against rlx's — the floor for anything that is not a copy of ORT's kernels.
Every backend lands there: cpu 2.4e-7, metal 1.8e-7, mlx 2.4e-7, wgpu 2.4e-7.

> **Bit-exactness is relative to a stated build on a stated platform.** Two
> things bound it, neither this port's doing. **`-ffp-contract`:** clang on arm64
> contracts `a*b + c` into `fma` by default, and `off` / `on` / `fast` each give
> a different spectrum from the same C source — the fixtures use `off`.
> **`libm` is not bit-portable:** the features go through `ln`, `log10`, `powf`
> and `cos`. On arm64 macOS all 30 750 values match exactly; on x86_64 Linux
> 99.1% still do, and the rest are a last-place `ln` apart except the pitch
> feature, where the difference compounds through Levinson and the Viterbi track
> to ~5.8e-6 in feature units. End to end that is worth nothing: cuda and rocm
> land at 1.8e-7 and 1.2e-7. The test asserts bit-equality on the fixture's own
> platform and an absolute bound everywhere.

> **The prebuilt `libten_vad` is not the same thing as `ten-vad.onnx`.** Its
> binary embeds the `coeff.h` DSP tables byte-for-byte, but contains the weights
> of its own bundled model in no float layout, at no byte alignment, in no
> order — and links no onnxruntime. It sits **9.6e-4** from the published model,
> ~4000× further than this port does. Voice decisions still agree everywhere.
> `tests/fixtures/README.md` has the evidence; the test suite pins the gap so a
> change in it shows up.

## What the model is

41 features per 16 ms hop of 16 kHz audio — 40 log-mel bands plus an LPC-based
pitch estimate — three frames of context, a separable CNN, two 64-unit LSTMs,
and a dense head. About 75 k parameters.

```text
i16 frame ──► pre-emphasis ──► Hann-768 STFT ──► |X|² ──► 40 mel ──► log ──► z-score ─┐
                                    │                                                 ├──► [3, 41]
                                    └──► 18-band LPC ──► inverse filter ──► 4 kHz ────┤       │
                                         ──► x-corr ──► Viterbi ──► pitch Hz ─────────┘       │
                                                                                              ▼
                             sigmoid ◄── dense ◄── LSTM×2 ◄── separable CNN ◄────────────────┘
```

The DSP frontend is a direct port of the upstream C (`stft.cc`, `pitch_est.cc`,
`biquad.cc`, the mel half of `aed.cc`). It is recursive — IIR filters and a
Viterbi pitch track — so it stays on the host. The network is an rlx HIR graph
and compiles for any backend.

## Use

```bash
# speech regions
cargo run -p rlx-ten-vad --release -- --wav audio16k.wav --seconds

# one line per frame: time, probability, flag, pitch
cargo run -p rlx-ten-vad --release -- --wav audio16k.wav --frames

# compare backends on one clip
cargo run -p rlx-ten-vad --release --features all-backends -- \
  --wav audio16k.wav --devices all
```

```rust
use rlx_ten_vad::{TenVad, TenVadConfig};

# let pcm: Vec<i16> = Vec::new();
let mut vad = TenVad::new(TenVadConfig::default())?;
for frame in pcm.chunks_exact(vad.hop_size()) {
    let out = vad.process_i16(frame)?;
    println!("{:.3} {} {:.0} Hz", out.probability, out.voice, out.pitch_hz);
}
# Ok::<(), anyhow::Error>(())
```

`TenVadConfig::default()` is `hop_size = 256`, `threshold = 0.5` — the same
defaults as `TenVad(hop_size=256, threshold=0.5)` upstream. Any hop ≥ 32 works;
the analysis hop stays 256, so smaller hops buffer and larger ones advance the
model several times per call, exactly as `ten_vad_process` does.

For a whole clip, [`TenVadBatch`] scores a full 30 s LSTM-reset window per graph
dispatch instead of one dispatch per frame:

```rust
# let pcm: Vec<i16> = Vec::new();
let mut batch = rlx_ten_vad::TenVadBatch::new(rlx_runtime::Device::Cpu)?;
let probs = batch.probabilities_i16(&pcm)?;
# Ok::<(), anyhow::Error>(())
```

Both paths agree to `~2e-7`.

## Backends

All seven rlx backends compile and produce the same probabilities. Throughput on
a 7.6 s clip (M-series, release):

Throughput on a 7.6 s clip (M-series, release). `TenVadBatch` scores
`chunk_frames` frames per graph dispatch with the LSTM state carried across
chunks, so chunk size is a pure latency/throughput dial — the answer is
identical at every size (`chunk_size_does_not_change_the_answer`).

| device | `TenVad` (per frame) | `TenVadBatch` (default 128) |
|--------|----------------------|------------------------------|
| cpu    | **401× RT**          | 313× RT |
| metal  | 46× RT               | **485× RT** |
| mlx    | 39× RT               | 288× RT |
| wgpu   | 11× RT               | 122× RT |

Network-only, by frames per dispatch — this is what chunking buys, and it is
worth the most exactly where a single 16 ms frame is pure launch latency:

| frames/dispatch | latency added | cpu | metal | mlx | wgpu |
|---|---|---|---|---|---|
| 1   | 0 ms   | 433× | 36×   | 20×  | 20×  |
| 8   | 128 ms | 436× | 427×  | 149× | 87×  |
| 32  | 512 ms | 553× | **1028×** | 378× | 140× |

Use **`TenVad` on CPU** when you need a decision every 16 ms; use
**`TenVadBatch`** (any backend) when you can spend a chunk of latency. Metal
goes from 36× to 1028× — 28× — purely from batching the dispatch.

### Fusion

Both graph shapes compile with **zero missed fusion patterns**, and
`both_graph_shapes_are_fully_fused` holds them there via rlx's
`assert_fusion_clean` — reintroduce an unfusable shape and the compile fails
rather than quietly costing dispatches. `RLX_FUSION_REPORT=1` on any run prints
the tally with reasons.

Getting there needed two shapes that are easy to get subtly wrong, because
`rlx-fusion`'s two bias matchers disagree on purpose:

* **matmul bias must be bare rank-1.** The matmul matcher reads the rank of the
  `Add`'s operand directly, so a `[1, n]` bias — or a rank-1 one behind an
  `Expand` — reports `BiasRankTooHigh` and the whole `matmul → add → act` chain
  stays unfused.
* **conv bias must be rank-1 *behind* `Reshape`+`Expand`.** That matcher peels
  wrappers and *requires* at least one, since a bare rank-1 operand added to an
  NCHW tensor would broadcast along W rather than C.

The LSTM cell also concatenates `x` and `h` so its two gate projections become a
single `[1, in+H] @ [in+H, 4H]` matmul; left as two, the pass sees
`add(matmul, matmul)` and fuses neither.

Precision is rlx's default `AlwaysF32` policy throughout — no mixed-precision
reductions, which is what keeps the network within `f32` noise of onnxruntime.

> **`Op::Lstm { carry: true }` is not used here, deliberately.** It would make
> streaming one dispatch, but its documented in-place `hn`/`cn` writeback is
> honoured only by the CPU backend today: Metal, MLX and wgpu all seed from
> `h0`/`c0` and never advance, and `unfuse_lstm` documents the same gap. Wired
> up, it scores `max|Δ| 0.52` with **64 decision flips** on Metal/MLX/wgpu while
> CPU stays correct — a silent wrong answer. The state is threaded as ordinary
> graph inputs/outputs instead.

## Tests

```bash
just test-ten-vad                    # unit + parity vs the reference library
just test-ten-vad-backends           # the same parity on every compiled backend
just bench-ten-vad-all-devices       # the table above
```

`tests/fixtures/` holds a deterministic 4 s synthetic clip, the features the
upstream C DSP computes for it, the probabilities onnxruntime gives for those
features, and — separately — what the prebuilt library returns. Regenerating any
of them needs the upstream repo; see
[`tests/fixtures/README.md`](tests/fixtures/README.md).

`examples/dump_stages.rs` dumps this crate's power spectrum / pitch / features
per frame, and `--score FEATS` runs a feature file through the network alone.
That is how the frontend and the network were measured separately.

## Regenerating the weights

```bash
git clone --depth 1 https://github.com/TEN-framework/ten-vad /tmp/ten-vad
python3 scripts/export_ten_vad_onnx_weights.py /tmp/ten-vad \
  crates/rlx-ten-vad/weights/ten_vad.safetensors
```

The script does the two layout conversions once, at export time: the ONNX LSTM
gate order `i,o,f,c` becomes rlx's `i,f,g,o`, and the separable kernels move
their length axis from W to H (rlx carries 1-D convs as `[N, C, L, 1]`). It also
carries the Hann-768 window and the per-feature mean/std out of `src/coeff.h`, so
the crate has a single embedded asset.

## Licensing

Powered by [ten-vad](https://github.com/TEN-framework/ten-vad).

Upstream TEN-VAD is Apache-2.0 **with additional conditions** (see its `LICENSE`),
and the pitch estimator is derived from Mozilla's LPCNet (BSD-2-Clause /
BSD-3-Clause; see the upstream `NOTICES` and the header of `src/pitch.rs`). This
crate is a Derivative Work of TEN-VAD and the embedded weights are Agora's, so
both remain subject to those terms — which are **not** the same as this
repository's GPL-3.0. Review [`NOTICE`](NOTICE) before redistributing.

## Related

- [`rlx-vad`](../rlx-vad/README.md) — Earshot / Silero / MarbleNet VAD, same 16 kHz frame grid.
