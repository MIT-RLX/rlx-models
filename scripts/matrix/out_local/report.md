# Cross-backend harness report

- host: `Darwin`  backends: `cpu`, `metal`, `mlx`, `wgpu`, `coreml`

| model | kind | cpu | metal | mlx | wgpu | coreml |
|---|---|---|---|---|---|---|
| `kokoro` | tts | ✅ | ❌ -0.05094 | ❌ | ✅ 0.98381 | ❌ |
| `supertonic` | tts | ✅ | ✅ 1.0 | ✅ 1.0 | ✅ 1.0 | ✅ 1.0 |

## Needs attention

- ❌ `kokoro` / metal: cosine -0.051 vs cpu
- ❌ `kokoro` / mlx: panic: MLX run_typed failed: mlx error: lower <unnamed> (NodeId(823), Reshape, op=Reshape { new_shape: [1, 512, 384] }; inputs=[NodeId(822):<unnamed>:Gather:Shape { dims: [Static(192), Static(384)], d
- ❌ `kokoro` / coreml: /private/var/folders/9_/pjm86g5j44l4cdv5mld3wd9c0000gn/T/rlx-coreml-cache/fe959d3c775dfa94.mlmodelc/model.mil:7768:12: error: 'mps.reshape' op the result shape is not compatible with the input shape
