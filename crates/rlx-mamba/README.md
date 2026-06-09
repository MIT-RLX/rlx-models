# rlx-mamba

Native **Mamba1** (selective state-space) block for the RLX stack.

Algorithmically aligned with `burn-mamba::mamba1` (in_proj → causal conv1d → SiLU → SSM → SiLU gate → out_proj). The **SSM core** runs through [`rlx-ssm`](../rlx-ssm/) flow stages compiled with `rlx-runtime` (not a hand-written scan loop).

Requires sibling [`rlx`](https://github.com/MIT-RLX/rlx) for `rlx-flow`, `rlx-ir`, and `rlx-runtime`.

## SSM integration

| Path | Mechanism |
|------|-----------|
| [`Mamba1Block::forward`](src/block.rs) | [`selective_scan_flow`](src/scan.rs) → `MambaScanStage` → `Op::SelectiveScan` |
| [`Mamba1Block::step`](src/block.rs) | [`selective_scan_step_flow`](src/scan.rs) → `Mamba1StepStage` → `mamba1_step` |
| [`mamba1_forward`](src/driver.rs) | Same scan via [`MambaBackend::selective_scan`](src/backend.rs) on every backend |

GPU backends compile the scan graph on **CUDA** / **wgpu** / **ROCm** when the feature and hardware are available; **Metal** / **MLX** use the CPU reference path for scan (matmuls still run on device).

## Backends

| Backend | Matmul / conv | SSM scan |
|---------|---------------|----------|
| **CPU** | `rlx-cpu` BLAS | `rlx-runtime` CPU (`SelectiveScan`) |
| **wgpu** | Cached matmul graphs | `SelectiveScan` on GPU when adapter available |
| **CUDA** | Cached matmul graphs | `SelectiveScan` on GPU (NVIDIA host) |
| **ROCm** | Cached matmul graphs | `SelectiveScan` on GPU (AMD host) |
| **Metal** | `rlx-metal` sgemm | CPU scan via unified-memory buffer |
| **MLX** | MLX `matmul` | CPU scan (readback) |

Enable features on the crate: `metal`, `mlx`, `cuda`, `wgpu`, `rocm` (each forwards the matching `rlx-runtime` backend feature).

## Public API

```rust
use rlx_mamba::{
    CpuBackend, Mamba1Block, Mamba1Config, Mamba1ResidentBlock, mamba1_forward,
    ensure_ssm_ops_registered,
};

ensure_ssm_ops_registered();

let cfg = Mamba1Config::new(128);
let block = Mamba1Block::random_for_bench(cfg, 0);

// Direct block API (SSM via rlx-ssm flow)
let y = block.forward(&input, batch, seq)?;

// Or backend driver (linears on backend, SSM via same flow)
let mut backend = CpuBackend::new();
let resident = Mamba1ResidentBlock::upload(&mut backend, &block)?;
let y = mamba1_forward(&mut backend, &resident, &input, batch, seq)?;
```

Low-level scan helpers (custom graphs, tests):

```rust
use rlx_mamba::{selective_scan_flow, selective_scan_step_flow, selective_scan_on_device};
use rlx_runtime::Device;
```

## Tests

```bash
cargo test -p rlx-mamba
cargo test -p rlx-mamba --features wgpu --test wgpu_backend
cargo test -p rlx-mamba --features metal --test metal_backend   # Apple
cargo test -p rlx-mamba --features mlx --test mlx_backend       # Apple
cargo test -p rlx-mamba --features cuda --test cuda_backend     # NVIDIA host
```

## CUDA rig (remote)

From the sibling `rlx` repo, `rig.sh` syncs and runs tests on a Windows/WSL NVIDIA host:

```bash
cd ../rlx
./rig.sh sync
./rig.sh --wsl run -- bash -c \
  "cd /mnt/d/rlx-workspace/rlx-models && \
   cargo test -p rlx-mamba --features cuda --release --test cuda_backend"
```

## ROCm compile check (Docker)

Validates HIP link without AMD hardware:

```bash
docker build --platform linux/amd64 \
  -f crates/rlx-mamba/Dockerfile.rocm-check \
  -t rlx-mamba-rocm-check ..
```

## Benchmarks

```bash
cargo bench -p rlx-mamba
cargo bench -p rlx-mamba --features "metal mlx cuda wgpu rocm" --bench mamba1
```

## vs burn-mamba

Workspace-excluded `rlx-mamba-bench` compares against `burn-mamba` (`max_abs ≈ 9e-13` on shared weights). See that crate's README for commands.

## See also

- Main repo [README](../../README.md#whats-here)
- [AGENTS.md](../../AGENTS.md) — Mamba tests and backend recipes
