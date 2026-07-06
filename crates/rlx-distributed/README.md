# rlx-distributed

Multi-node **distributed inference** (pipeline / tensor parallel) for RLX models. It layers on `rlx-driver`'s transports (`TcpTransport`, `ThunderboltTransport`, and `rlx-mlx`'s `MlxTransport`) and `ProcessGroup` to run one model split across several hosts, coordinating per-rank layer blocks with a model-agnostic relay.

## Modules

- `config` — `hosts.json` parsing + process-group construction ([`DistConfig::connect`]).
- `partition` — pipeline-parallel layer assignment ([`pipeline_layer_range`], [`block_role`]).
- `pipeline` — the model-agnostic relay ([`PipelineCoordinator`]) driving per-rank [`BlockRunner`]s.
- `launch` — local multi-process cluster helpers ([`LocalCluster`], [`worker_args`]).

A model family plugs in by implementing [`BlockRunner`] for its layer block.

## Public API

```rust
use rlx_distributed::{DistConfig, ParallelMode, PipelineCoordinator, pipeline_layer_range};

let cfg = DistConfig::from_hostfile("hosts.json", /*rank*/ 0)?;
let group = cfg.connect()?;                     // ProcessGroup over the chosen transport
let (start, end) = pipeline_layer_range(rank, world, num_layers);
// build a BlockRunner for [start,end) and hand it to PipelineCoordinator
# anyhow::Ok(())
```

## Quick start

```bash
cargo run -p rlx-distributed --example transport_bench
```

## How it fits

Built on `rlx-driver` (`ProcessGroup`, transports). Model crates such as [rlx-qwen3](../rlx-qwen3) provide the per-rank block runners.
