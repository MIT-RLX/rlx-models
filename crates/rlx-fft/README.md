# rlx-fft

Learned butterfly FFT + spectral pipelines (mel, Welch PSD, top-K Welch peaks), compiled via RLX.

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

#### Auto selection rules

| Condition | Picked strategy |
|-----------|-----------------|
| `batch ≤ 256` on CPU, `≤ 128` on GPU | **ultra** (1 segment) |
| Mid batch | **fast** (2 segments, streaming top-K) |
| `batch ≥ 8192` on GPU | **rlx** (compiled `Op::Fft`) |
| GPU + `batch ≥ 8192` + learned model with **&lt;25% active gates** | **learned** |

Threshold helpers: `rlx_crossover_batch(device)` → `8192` on GPU; `ultra_fast_max_batch(device)` → `256` CPU / `128` GPU.

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
| `welch_peaks_compile` | `CompiledRlxWelchPeaks`, `CompiledLearnedWelchPeaks` |
| `bench_welch_peaks` | CLI bench + batch sweep |
