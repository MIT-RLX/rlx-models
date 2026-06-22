# rlx-tsac

Native Rust TSAC for RLX plus optional wrapper around [Fabrice Bellard's TSAC](https://bellard.org/tsac/) Linux binary — very low bitrate neural audio (~5–8 kb/s @ **44.1 kHz**).

The **native codec** (vendored [tsac-ng](https://github.com/Hope2333/tsac-ng) via FFI) runs on the host with RLX device tags (`cpu`, `metal`, `cuda`, `vulkan`, …). **Reference binaries** (original Bellard `tsac` + standalone `tsac-ng`) run in a small Docker image for linux/amd64 perf and parity on any host.

## Quick start

```bash
just fetch-tsac
just tsac -- --in-wav speech.wav --out-wav /tmp/out.wav --device cpu
```

## Perf bench (RLX host vs Docker refs)

RLX runs on the host; Docker only runs the original Bellard binary and standalone `tsac-ng`:

```bash
bash crates/rlx-tsac/docker/run.sh build

cargo run -p rlx-tsac --example perf_bench --release --features native-codec -- \
  --in-wav speech.wav --quality 9 --fast --device metal
```

Prints encode/decode timing for all three engines plus PCM correlation vs the Docker references.

Tolerances: `RLX_TSAC_PARITY_MIN_CORR` (default `0.92`), `RLX_TSAC_PARITY_MAX_MSE` (default `0.0025`).

## Bellard parity (Linux x86_64 host, no Docker)

On Linux x86_64 you can compare native RLX against the Bellard binary directly:

```bash
just bench-tsac-parity
just bench-tsac-parity-example -- --in-wav speech.wav --quality 9 --fast
```

## Library

```rust
use rlx_tsac::{TsacCodec, TsacBackendKind, TsacOptions, bench_perf, PerfOptions};
use rlx_runtime::Device;

let native = TsacCodec::open_native(".cache/tsac", TsacOptions {
    backend: TsacBackendKind::Native,
    device: Device::Metal,
    ..Default::default()
})?;

let report = bench_perf(".cache/tsac", "speech.wav", &PerfOptions::default())?;
report.print_summary(&PerfOptions::default());
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `RLX_TSAC_DIR` | `.cache/tsac` | Weight dir for native codec (`*.bin` models) |
| `RLX_TSAC_BIN` | `$RLX_TSAC_DIR/tsac` | Override Bellard binary path (host parity only) |
| `RLX_TSAC_REF_IMAGE` | `rlx-tsac-ref` | Docker image for reference benches |
| `RLX_TSAC_DOCKER_PLATFORM` | `linux/amd64` | Docker platform for reference image |

## Options (passed to `tsac`)

| Flag | Maps to |
|------|---------|
| `-q N` | Quality / bitrate (1–12 joint stereo, 1–9 otherwise) |
| `-f` | Fast mode — disable Transformer (higher bitrate) |
| `-s` | Separate stereo channels (default is joint stereo) |
| `-c N` | Force input channel count |
| `-v` | Verbose statistics |

## Tests

```bash
cargo test -p rlx-tsac --release

# Linux x86_64 Bellard parity (host, no Docker):
RLX_TSAC_PARITY=1 cargo test -p rlx-tsac --test bellard_parity --release --features fetch,native-codec -- --nocapture
```
