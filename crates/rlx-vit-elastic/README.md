# rlx-vit-elastic

**SnapViT** elastic structured pruning + **GLARE** continual self-supervised
pre-training for Vision Transformers, on the RLX backends (CPU / Metal / MLX /
wgpu / CUDA).

- **SnapViT** — *Elastic ViTs from Pretrained Models without Retraining*
  ([arXiv:2510.17700](https://arxiv.org/abs/2510.17700), NeurIPS 2025)
- **GLARE** — *Enhancing Semantic Segmentation with Continual Self-Supervised
  Pre-training* ([arXiv:2509.17816](https://arxiv.org/abs/2509.17816), TMLR 12/2025)

Both run on a generic, **differentiable** ViT forward built directly at the
`rlx_ir::Graph` level (so it composes with `rlx-autodiff` + `rlx-tune`), with two
wired backbones — **DINO ViT-B/16** (`facebook/dino-vitb16`, the papers' SSL
backbone) and **UNI2-h** (`MahmoodLab/UNI2-h`) — plus a tiny synthetic topology
for tests. The hand-built forward is validated **cos > 0.9999 vs the
bit-exact-vs-timm `rlx-uni2` reference**.

## SnapViT — retraining-free, label-free structured pruning

`P = diag((1/N_D) Σ‖∇_θ L^SSL‖²) ⊙ M·c` (Eq. 8):

1. **Local Hessian diagonal** — the mean squared gradient of a head-free DINO
   cross-view loss (`grad_with_loss` over the block weights), aggregated to
   per-attention-head / per-FFN-channel scores (`snapvit::compute_local_scores`).
2. **Global scaling `c`** — learned by an NES (`snapvit::optimize`) whose
   label-free fitness is the PCA-192 cosine of pruned-vs-original embeddings,
   averaged over sparsities (Eq. 6, forward-only on any backend).
3. **Elastic pruning** — one prunability score → a continuum of sub-networks at
   any sparsity (`snapvit::run` → masks + retention + FLOP accounting).

The key efficiency trick: **one maskable graph** with `head_mask`/`ffn_mask`
inputs, so every candidate is a mask vector fed to the same compiled graph — no
recompilation.

```
rlx-vit-elastic snapvit elastic --backbone synthetic
rlx-vit-elastic snapvit prune   --backbone dino-vitb16 --weights dino.safetensors \
                                --data imgs/ --sparsity 0.4 --device metal
```

## GLARE — continual SSL pre-training (adapter-only)

Trains **only** a UniAdapter (`x' = x + s·ReLU(x·W_down)·W_up`, after each
attention block) + a cross-attention module + a shared DINO head; the backbone
stays frozen and the teacher is their EMA. Three consistency losses
(`L = L_glob + L_reg + L_loc`, Eq. 10): global ([CLS] DINO), regional
(cross-attention over regions, Eqs. 5–7), local (patch strong-blur, §4.3). Driven
by `rlx-tune`'s `Trainer`.

```
rlx-vit-elastic glare train --backbone dino-vitb16 --weights dino.safetensors \
                            --data imgs/ --steps 200 --device metal
```

## Backends (measured)

The **full pipeline** (SnapViT local-Hessian backward + xNES fitness forward,
and GLARE adapter training) runs **natively on every backend** — verified
`cos > 0.999` / identical fitness vs CPU (`baseline=0.9969 best=0.9981`), on both
the plain-ViT and UNI2-SwiGLU topologies:

| Backend | Forward / inference | Autodiff backward (training + pruning gradient) |
|---|---|---|
| CPU | ✅ bit-exact reference | ✅ |
| Metal (Apple) | ✅ | ✅ |
| MLX (Apple) | ✅ | ✅ |
| wgpu | ✅ | ✅ |
| CUDA (RTX 3080 Ti) | ✅ | ✅ |
| Vulkan (RTX 3080 Ti) | ✅ | ✅ |
| ROCm | untested (no hardware) | — |

Getting all six required two **core `rlx` fixes** (in the sibling
`/Users/Shared/rlx` tree):
- **CUDA** — `rlx-cuda/src/unfuse.rs` promoted rank-3 attention → rank-4 only for
  the *forward* `Op::Attention`; the autodiff `Op::AttentionBackward` reached the
  rank-4-only kernel and panicked. Added the mirror `expand_attention_backward_rank3`.
- **Metal** — the one-pass `layer_norm` variance `E[x²]−E[x]²` could go slightly
  negative on large-magnitude inputs (float cancellation) → `rsqrt(neg)=NaN`,
  poisoning the whole forward (input-dependent, hence intermittent). Clamped to
  `fmax(0, …)` in the five one-pass LayerNorm kernels (`rlx-metal/src/kernels.rs`).

MLX/wgpu/Vulkan already ran the backward correctly. `backward_device()` no longer
routes anything to CPU; `snapvit run` / `glare train` run end-to-end on
`--device {cpu,metal,mlx,gpu,cuda,vulkan}`.

## Documented deviations (tractable scale)

Full paper reproduction (100-epoch GLARE, 500-iter full-covariance xNES over
ViT-B on 7 datasets, 8×A100) is out of scope for a shared-RAM workstation. The
**algorithms are faithful**; the following are documented simplifications, each
behind a config knob so it can be scaled toward the paper:

- **SnapViT global term**: separable NES (per-block variance) rather than the
  paper's full-covariance xNES (cross-block off-diagonals). PCA is optional
  (`pca_dim`).
- **SnapViT export**: the elastic sub-networks are realized by **masking** the
  shared graph (correct outputs + FLOP/param accounting); structural weight
  shrinking with per-layer variable dims is a follow-up.
- **GLARE**: fixed contiguous **regions** (not attention-aware sampling) and
  spatially **aligned** two-view correspondence (photometric distortion only) in
  place of crop back-tracking.

## Tests

```
cargo test -p rlx-vit-elastic                          # CPU
cargo test -p rlx-vit-elastic --features metal         # + Metal backward/forward
```

- `vit_parity` — forward matches the `rlx-uni2` reference (cos > 0.9999).
- `dino_smoke` — DINO loss differentiates + trains; head/crops/teacher sanity.
- `snapvit_local` — local scores finite, non-degenerate, rank heads sensibly.
- `snapvit_elastic` — end-to-end elastic continuum, xNES ≥ baseline, pruned runs.
- `glare_smoke` — adapter-only training reduces the loss; EMA teacher tracks.

## Weights

- DINO ViT-B/16: `facebook/dino-vitb16` (HF `ViTModel`; keys are remapped to the
  timm-canonical `qkv`/`proj`/`fc1`/`fc2` layout automatically).
- UNI2-h: `MahmoodLab/UNI2-h` (gated, **CC-BY-NC-ND 4.0**; obtain separately).

License: GPL-3.0-only.
