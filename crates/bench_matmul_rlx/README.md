# bench_matmul_relu (RLX native, decomposed from ONNX)

Auto-generated from `/tmp/bench_matmul_relu.onnx`.

## Layout

- `src/graph.rs` — hand-lowered ONNX graph as RLX HIR builder
- `src/weights.rs` — load `model.safetensors`
- `weights/` — exported tensors
- `decompose_report.json` — op coverage / unsupported ops

## Usage

```rust
use bench_matmul_relu::{compile, GraphOptions};
use rlx_runtime::Device;

let compiled = compile(
    Device::Cpu,
    std::path::Path::new("weights"),
    &GraphOptions::default(),
)?;
```

## Coverage

- Lowered nodes: 3
- Skipped: 0
- Unsupported ops: {}

Extend `graph.rs` (or re-run decompose after extending `rlx-onnx-import`) for missing ops.
