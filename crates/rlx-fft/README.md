# rlx-fft

Learned butterfly FFT + spectral pipelines (mel, Welch PSD, top-K Welch peaks), compiled via RLX.

**Workspace 0.2.5** — depends on upstream `rlx*` 0.2.5. Publish tier 2 (`scripts/publish.sh`), after `rlx-cli` / `rlx-models-core`.

```bash
cargo run -p rlx-fft --release -- --help
```

## Welch peaks (fast top-K spikes)

Extract **top-K frequency spikes** `(bin, power)` without materializing a full Welch PSD. The fast path uses **2 Welch segments** (vs 8 in full Welch); an **ultra-fast** path uses **1 segment** for minimum latency.

### CLI bench

```bash
# Auto strategy (default) — picks fastest path for batch + device
cargo run -p rlx-fft --release -- bench-welch-peaks \
  --n-fft 256 --batch 32 --k 16 --train-steps 0

# Batch sweep + Metal GPU crossover
cargo run -p rlx-fft --features apple-silicon --release -- bench-welch-peaks \
  --n-fft 256 --batch 32,256,1024,4096,8192 --device metal --train-steps 0 --iters 15

# Force a specific strategy (see table below)
cargo run -p rlx-fft --release -- bench-welch-peaks \
  --n-fft 256 --batch 32 --strategy ultra

cargo run -p rlx-fft --features apple-silicon --release -- bench-welch-peaks \
  --n-fft 256 --batch 8192 --device metal --strategy rlx

# K sweep — plot latency vs top-K (JSON rows tagged with batch + k)
cargo run -p rlx-fft --features apple-silicon --release -- bench-welch-peaks \
  --n-fft 256 --batch 8192 --k 4,8,16,32,64 --device metal --train-steps 0 --iters 15 \
  --strategy rlx --json /tmp/welch-k-sweep.json
```

Sweep output ends with a **`k crossover`** table (rustfft / stream / rlx / picker ms per K). Combine with `--batch` for a full grid, e.g. `--batch 32,8192 --k 4,16,64`.

### Fusion phase bench (IO + latency)

Compare baseline interleaved readback, Phase 1 block layout, and Phase 2 fused `Op::WelchPeaks`:

```bash
cargo run -p rlx-fft --features dev,apple-silicon --release -- bench-fusion-phases \
  --n-fft 256 --batch 8192 --k 16 --device metal --iters 15

# Batch sweep + JSON
cargo run -p rlx-fft --features dev,apple-silicon --release -- bench-fusion-phases \
  --n-fft 256 --batch 32,1024,8192 --k 16 --device metal --iters 15 \
  --json /tmp/fusion-phases.json
```

Output includes **IO profiles** (kernel launches, sync points, host readback bytes) and per-phase speedup vs baseline.

```bash
# WGPU (Vulkan/Metal/DX12 via wgpu)
cargo run -p rlx-fft --features dev,gpu --release -- bench-fusion-phases \
  --n-fft 256 --batch 8192 --k 16 --device wgpu --iters 15

# CUDA (when NVIDIA toolkit + `rlx-runtime/cuda` available)
cargo run -p rlx-fft --features dev,cuda --release -- bench-fusion-phases \
  --n-fft 256 --batch 8192 --k 16 --device cuda --iters 15
```

| Phase | Path | What changes |
|-------|------|--------------|
| baseline | `baseline_interleaved_readback` | Full FFT spectrum readback + host top-K |
| Phase 1 | `phase1_block_layout` | Block-layout FFT output; peaks on host |
| Phase 2 | `phase2_fused_welch_peaks_op` | Fused graph; peaks-only readback (~32× less host_out at batch=8192). Metal runs peaks after a single GPU wait (no mid-graph sync). |

| Flag | Default | Description |
|------|---------|-------------|
| `--n-fft` | `256` | FFT size |
| `--batch` | `32` | Batch size, CSV (`32,1024`), or power-of-two range (`32-8192`) |
| `--k` | `16` | Peaks per row; CSV (`4,8,16,32`) or power-of-two range (`4-64`) for K sweep |
| `--device` | `auto` | `cpu`, `metal`, `cuda`, … |
| `--strategy` | `auto` | `auto`, `ultra`, `fast`, `rlx`, `learned` |
| `--train-steps` | `200` | Train a lightweight learned model (`0` to skip); uses `--k` for peak loss |
| `--iters` | `50` | Timing iterations |
| `--no-compiled` | — | Skip explicit RLX/learned compiled baseline rows |
| `--no-ultra-fast` | — | Skip ultra-fast baseline row |
| `--json PATH` | — | Write JSON report |

Bench output includes a **`welch_peaks_picker_<strategy>`** row using auto or forced selection, e.g.:

```text
[welch-peaks] picker (auto): batch=8192 device=Metal -> rlx_compiled
```

### Strategy picker

Use **`AutoWelchPeaks`** in Rust or **`--strategy`** on the CLI.

| Strategy | Label | When to use |
|----------|-------|-------------|
| **auto** | (resolved at runtime) | Default — picks from batch + device |
| **ultra** | `ultra_fast_rustfft` | Smallest batch, lowest latency (1 segment) |
| **fast** | `fast_streaming_rustfft` | CPU / mid batch; best accuracy vs speed on rustfft |
| **rlx** | `rlx_compiled` | Large batch on GPU (Metal/CUDA/…) |
| **learned** | `learned_compiled` | Large batch + sparse learned gates + trained model |

#### Auto selection (IO-aware picker)

Auto mode estimates each strategy with an **Ayala-style latency–bandwidth model** (`T ≈ L·M + S/W`) using `graph_io` profiles and per-device `BackendCostModel` (CPU rustfft paths vs fused `Op::WelchPeaks` on GPU). Fused GPU estimates apply a calibrated compute scale (~7.5× IO-only on Metal, from `bench-fusion-phases` phase-2); CPU rustfft gets a batch growth penalty when compared on GPU devices. It picks the lowest predicted cost.

| Env | Effect |
|-----|--------|
| `RLX_FFT_PICKER_TRACE=1` | Log per-strategy predicted ms when constructing `AutoWelchPeaks` |
| `RLX_FFT_LEGACY_PICKER=1` | Restore fixed thresholds (`8192` GPU crossover, etc.) |

Calibrate with `bench-fusion-phases` (phase-2 fused vs IO-model line) and `bench-welch-peaks` (picker vs rustfft crossover). Metal/Mlx use unified-memory rustfft penalties; WGPU/Vulkan use lighter CPU rustfft adjustment (tail-host fused path is slower than rustfft until a native GPU `WelchPeaks` kernel lands).

Legacy reference thresholds (used only with `RLX_FFT_LEGACY_PICKER`): `batch ≤ 256` CPU / `≤ 128` GPU → **ultra**; mid batch → **fast**; `batch ≥ 8192` GPU → **rlx**; sparse learned gates → **learned**.

**Reference peaks** for training/bench error always use full 8-segment Welch; student paths use 1–2 segments.

### Rust API

```rust
use rlx_fft::{
    AutoWelchPeaks, WelchPeaksPickMode, WelchPeaksStrategy,
    parse_welch_peaks_strategy, pick_welch_peaks_strategy,
};

// Auto (recommended)
let mut picker = AutoWelchPeaks::new(batch, n_fft, k, Some("auto"))?;
println!("strategy: {}", picker.strategy_label());

// Force a strategy
let mut picker = AutoWelchPeaks::with_strategy(
    batch, n_fft, k, Some("metal"), WelchPeaksStrategy::RlxCompiled,
)?;

// Parse CLI-style string
let mode = parse_welch_peaks_strategy("fast")?; // Force(FastStreaming)
let mut picker = AutoWelchPeaks::with_options(
    batch, n_fft, k, Some("cpu"), None, mode,
)?;

// With learned model (for learned strategy or auto sparse-gate path)
let mut picker = AutoWelchPeaks::with_learned(
    batch, n_fft, k, Some("metal"), Some(&model),
)?;

// signal: [batch × full_welch_frame] — 8-segment layout buffer
let peaks = picker.welch_peaks_batch(&signal)?; // [batch, k, 2] packed (bin, power)
```

**Strategy string aliases** (for `parse_welch_peaks_strategy` / `--strategy`):

| Input | Maps to |
|-------|---------|
| `auto` | Auto pick |
| `ultra`, `ultra-fast`, `1seg` | UltraFast |
| `fast`, `streaming`, `rustfft`, `2seg` | FastStreaming |
| `rlx`, `compiled`, `gpu` | RlxCompiled |
| `learned`, `learned_compiled` | LearnedCompiled |

### Performance notes (n=256, Apple Silicon reference)

| Batch | Best auto pick (typical) | vs full Welch |
|-------|--------------------------|---------------|
| 32 | ultra (~0.04 ms CPU) | ~4–5× faster |
| 1024 | fast streaming | ~3× faster |
| 8192 | rlx Metal (~40 ms) | ~2× faster than rustfft fast at this batch |

RLX compiled paths need **large batch** to amortize GPU launch; rustfft wins at small batch.

### Training peaks into the learned model

End-to-end training includes a peak-matching loss on the fast 2-segment path. **`--k` / `--peak-k`** sets how many spikes are matched during training and at inference (learned, compiled-learned, and picker `learned` strategy).

```bash
cargo run -p rlx-fft --release -- train-e2e \
  --n-fft 256 --batch 8 --peak-k 16 --peak-weight 2.0 --steps 2000

# bench-e2e: same K for WelchPeaks pipelines + teacher training
cargo run -p rlx-fft --release -- bench-e2e \
  --n-fft 256 --batch 8 --peak-k 8 --train-first --steps 500
```

At inference, `FastLearnedFftModel::welch_peaks_batch` accepts any `WelchPeakParams::fast_for_n_fft(n_fft, k)` — K is not baked into weights, but training with the target K improves peak accuracy.

### Tests

```bash
cargo test -p rlx-fft welch_peaks_picker --release
cargo test -p rlx-fft peak --release
```

### Modules

| Module | Role |
|--------|------|
| `peak` | `WelchPeakParams`, streaming top-K, `WelchPeaksScratch` |
| `welch_peaks_picker` | `AutoWelchPeaks`, auto/forced strategy |
| `welch_peaks_cost` | Ayala IO cost model, `estimate_welch_peaks_costs` |
| `welch_peaks_compile` | `CompiledRlxWelchPeaks`, `CompiledLearnedWelchPeaks` |
| `bench_welch_peaks` | CLI bench + batch / K sweep |
| `bench_fusion_phases` | Fusion phase bench (`--features dev`; baseline vs block layout vs fused op) |
