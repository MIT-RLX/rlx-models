# rlx-hoct

Higher-Order Cell Tracking Transformer ([royerlab/hoct](https://github.com/royerlab/hoct), [arXiv:2607.11754](https://arxiv.org/abs/2607.11754)) for RLX.

End-to-end tracking:

```text
labels[/images] → regionprops (19-d) → kNN candidate graph
  → HOCT edge scores → parental softmax → HiGHS ILP → CTC / GEFF
```

Weights are TorchScript `general_v0.pt` exported to safetensors (156 tensors). Eager forward matches the JIT graph (gated attention, 3D RoPE, line-to-line distance bias). The score head compiles on RLX backends (CPU / Metal / MLX / CUDA / …).

## Quick start

```bash
just fetch-hoct
# → .cache/hoct/general_v0.safetensors

just hoct -- track \
  -m .cache/hoct/general_v0.safetensors \
  --labels labels.raw \
  -o /tmp/hoct-out -f ctc \
  --max-distance 300 --neighbors 5 --max-dt 3 --window 5

# Score head on GPU (body stays eager CPU):
just hoct -- track -m .cache/hoct/general_v0.safetensors \
  --labels labels.raw -o /tmp/out -d cuda   # or metal | mlx | gpu
```

Manual weight export:

```bash
curl -fsSL -o general_v0.pt \
  https://github.com/royerlab/hoct/releases/download/weights-v0/general_v0.pt
python3 crates/rlx-hoct/scripts/export_jit_safetensors.py general_v0.pt -o general_v0.safetensors
```

### CLI

| Subcommand | Role |
|------------|------|
| `track` | Labels → graph → model → ILP → write CTC/GEFF |
| `predict` | Random-batch forward (logit shape check) |

Common flags: `-m`/`--weights`, `--labels`, `-o`/`--out`, `-f ctc|geff`, `--max-distance`, `--neighbors`, `--max-dt`, `--window`, `-d`/`--device`.

Label raw format: `[ndim:u32][dims…][u32 voxels…]` little-endian.

| `-f` | Output |
|------|--------|
| `ctc` (default) | `res_track.txt` (`L B E P`) + `maskNNNN.raw` |
| `geff` | Minimal `tracks.json` (nodes / edges / tracks) |

## Public API

```rust
use rlx_hoct::{HoctRunner, OutputFormat};
use ndarray::Array3;

let runner = HoctRunner::builder()
    .weights(".cache/hoct/general_v0.safetensors")
    .window_size(5)
    .build()?;

let labels: Array3<u32> = /* (T, Y, X) */;
let (solution, nodes) = runner.track_labels(&labels, None)?;
rlx_hoct::write_solution("/tmp/out", &labels, &nodes, &solution, OutputFormat::Ctc)?;
# anyhow::Ok(())
```

Lower-level pieces:

| Type / fn | Role |
|-----------|------|
| `HoctModel` | Eager forward (parity reference) |
| `HoctDeviceRunner` | Eager body + compiled score head on a device |
| `HoctCompiled` / `HoctFlow` | Padded `(N,E)` contract + edge-head ModelFlow |
| `load_hoct_weights` | Safetensors → typed weights |
| `features` / `graph` / `softmax` / `ilp` / `io` | Pipeline stages |

## Architecture (`general_v0`)

| | |
|---|---|
| Input | 19-d regionprops, standardized (`FEATURE_MEAN` / `FEATURE_STD`) |
| Hidden | 288; 4 heads × 72 |
| Blocks | 4 node self-attn + 4 edge (block 0 cross-attn, then self) |
| Norms | RMSNorm in blocks; LayerNorm → Linear score head |
| Orphan logits | Zeros (`exp(0)=1` parental constant) |
| ILP | Appendix B via HiGHS (`good_lp`); ties may differ from Gurobi |

Forward signature:

```text
(node_features[B,N,19], node_pos[B,N,3], edge_pos[B,E,3],
 edge_indices[B,E,2], node_mask[B,N], edge_mask[B,E])
  → (edge_logits[B,E,1], node_h[B,N,288], edge_h[B,E,288], orphan[B,N,1])
```

## Tests

```bash
just test-hoct-parity                          # weights + JIT fixtures + pipeline
just features=apple-silicon test-hoct-backends # Metal / MLX / wgpu score head
just features=cuda test-hoct-backends          # CUDA (e.g. on msi)
```

| Check | Reference | Gate |
|-------|-----------|------|
| Edge logits | TorchScript | max abs ≈ `5e-7` (`< 1e-4`) |
| Hidden states | TorchScript | fp32 abs/rel |
| Features / softmax | Python fixtures | `< 1e-5` |
| Score head | Eager CPU | backends `< 1e-4` |

Checked-in fixtures: `tests/fixtures/jit_ref/`, `tests/fixtures/pipeline/`.
