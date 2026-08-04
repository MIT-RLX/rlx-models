# Cross-backend harness report

- host: `Darwin`  backends: `cpu`, `metal`, `mlx`, `wgpu`, `coreml`

| model | kind | cpu | metal | mlx | wgpu | coreml |
|---|---|---|---|---|---|---|
| `moss-nano` | tts | ✅ | ✅ 1.0 | ✅ 1.0 | ✅ 1.0 | ✅ 1.0 |
| `luxtts` | tts | ✅ | ✅ 0.97268 | ✅ 0.97268 | ❌ | ✅ 0.97268 |

## Needs attention

- ❌ `luxtts` / wgpu: panic: attempt to calculate the remainder with a divisor of zero
