# rlx-neuralhash

Apple **NeuralHash** perceptual image hashing, implemented natively on RLX.

Same inputs and the same 24-character hex output as the widely-used
[reference implementation](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX),
with the descriptor network read directly from the vendor's own model container
and built as an rlx-ir graph. No ONNX runtime is involved.

```text
image ──resize 360×360 (PIL bicubic), /255·2−1, NCHW──▶ [1, 3, 360, 360]
      ──descriptor CNN (native rlx-ir graph)──────────▶ 128 f32
      ──seed matrix dot product ([96, 128])───────────▶ 96 f32
      ──binary step (score ≥ 0)──────────────────────▶ 96-bit hash
```

## Status

Validated against the model macOS installs at
`/System/Library/Frameworks/Vision.framework/Versions/A/Resources/`
(verified on macOS 26.4):

| check | result |
|---|---|
| 225-layer graph derived from the vendor container | ✅ all 226 activation shapes match Apple's own `.espresso.shape` sidecar |
| hash vs. an independent PyTorch implementation | ✅ **bit-identical (96/96) on every image tried** |
| descriptor vs. that implementation | ✅ cos ≈ 1.0, relative error 1e-7 … 2e-4 |
| declared preprocessing (`transform_params`) | ✅ matches this crate's `x·2/255 − 1` RGB chain exactly |
| PIL preprocessing vs. Pillow | ✅ 388 729 / 388 800 elements bit-exact (99.98%), rest ±1 8-bit step |
| all 7 rlx backends | ✅ identical hashes (see Backends) |

## Native, not an ONNX shim

| stage | module | what it does |
|-------|--------|--------------|
| `NeuralHashv3b_fp16-current.espresso.{net,shape,weights}` | `espresso` | decompresses the LZFSE `pbze` container, parses the JSON layer list, the blob-shape sidecar and the `{u64 count}{index,size}×N{payloads}` weight table |
| → `NeuralHashSpec` | `spec` | normalizes Espresso layers into rlx-level ops with resolved NCHW shapes; serializable to JSON |
| → `WeightMap` | `weights` | f16 conv kernels, f32 biases, interleaved instance-norm parameters, `inner_product` transposed to `[in, out]` |
| → `rlx_ir::Graph` | `flow` | native ops: `Conv2d` (incl. depthwise), `GroupNorm`, `HardSwish`/`HardSigmoid`, `Pool`, `Clamp`, `MatMul`; lowering is device-aware (see Performance) |
| → `CompiledGraph` | `model` | compiles for CPU / Metal / MLX / CUDA / ROCm / wgpu / Vulkan |

`Cargo.toml` has no ONNX dependency by default. The optional `onnx-parity`
feature adds a **validation-only** `parity` module for diffing against the
community `model.onnx`; nothing in `NeuralHasher` can reach it.

### The architecture

MobileNetV3-shaped, 225 Espresso layers (→ 150 rlx ops after fusion):

* 54 convolutions (3×3 / 5×5 depthwise, 1×1 pointwise, stride 1–2, SAME/VALID)
* 35 **instance** norms — Espresso spells them `batchnorm`, but
  `training_instancenorm = 1` means statistics come from the input at runtime
* 118 elementwise ops, most of them explicit hard-swish chains
  (`+3` → `clamp(0,6)` → `×1/6` → `×x`) and squeeze-excite gates
* 10 global average pools, 6 ReLUs, 2 fully-connected heads (1280→500→128)

Instance normalization is why NeuralHash shrugs off exposure and contrast
changes — see the invariants below.

## Model files

This crate contains and redistributes **no model data**. On macOS both files
are already installed:

```console
$ ls /System/Library/Frameworks/Vision.framework/Versions/A/Resources/ | grep -i neuralhash
NeuralHashv3b_fp16-current.espresso.net
NeuralHashv3b_fp16-current.espresso.shape
NeuralHashv3b_fp16-current.espresso.weights
neuralhash_128x96_seed1.dat
```

Earlier OS versions name the model `NeuralHashv3b-current` and store the
`.net`/`.shape` as plain JSON; both forms are accepted. No `coremltools` or
ONNX conversion step is needed.

`neuralhash_128x96_seed1.dat` is a 128-byte header plus a `[96, 128]`
little-endian f32 matrix — 49 280 bytes exactly.

## Usage

```console
$ R=/System/Library/Frameworks/Vision.framework/Versions/A/Resources

# Hash one image (equivalent to `python3 nnhash.py model.onnx seed1.dat image.jpg`)
$ rlx-neuralhash --net $R/NeuralHashv3b_fp16-current.espresso.net \
                 --seed $R/neuralhash_128x96_seed1.dat --image photo.jpg
e5bf5a36875588b358477ce3

# Batch on the GPU, with the pairwise distance matrix
$ rlx-neuralhash --net ... --seed ... --device metal \
                 --image a.jpg --image b.jpg --image c.png --distances

# Distance between two hashes — no model needed
$ rlx-neuralhash --compare ab14febaa837b6c1484c35e6 ab14febaa837b6c1484c3de6
3
```

`--inspect` prints the Espresso layer-type histogram, the derived native op
list with shapes, and validates the I/O contract. Run it first against a model
you have not used before: an Espresso layer type outside the supported set is
reported by name there rather than silently skipped.

```console
$ rlx-neuralhash --net $R/NeuralHashv3b_fp16-current.espresso.net --inspect
$ rlx-neuralhash --net $R/... --export-spec neuralhash.json
```

### Library

```rust,no_run
use rlx_neuralhash::{NeuralHash, NeuralHasher};
use rlx_runtime::Device;

let r = "/System/Library/Frameworks/Vision.framework/Versions/A/Resources";
let mut hasher = NeuralHasher::open_espresso(
    format!("{r}/NeuralHashv3b_fp16-current.espresso.net"),
    format!("{r}/neuralhash_128x96_seed1.dat"),
    Device::Metal,
)?;

let a: NeuralHash = hasher.hash_image("photo.jpg")?;
let b = hasher.hash_image("photo_resaved.jpg")?;
println!("{a} vs {b}: {} bits differ", a.hamming(&b));
# Ok::<(), anyhow::Error>(())
```

## Backends

Identical 96-bit hashes on all seven standard rlx backends:

| host | backends | fixture |
|---|---|---|
| Apple M-series (macOS 26.4) | `cpu`, `metal`, `mlx`, `gpu` (wgpu), `vulkan` | Apple's real weights **and** synthetic |
| NVIDIA RTX 3080 Ti (Linux) | `cpu`, `cuda`, `gpu`, `vulkan` | synthetic |
| AMD gfx908 (Linux) | `cpu`, `rocm`, `gpu`, `vulkan` | synthetic |

`Vision.framework` does not exist on Linux, so the CUDA and ROCm hosts run
`synth::synthetic_net` — deterministic weights, but the same `[1, 3, 360, 360]`
→ 128 contract and every layer form the real model uses (depthwise 3×3/5×5,
SAME padding, instance norm, hard-swish chains, squeeze-excite broadcast,
residual add, dual FC head). A backend that miscompiles anything the real model
needs fails there too. Reproduce with:

```console
$ cargo test -p rlx-neuralhash --release --test backends --features all-backends
```

## Performance

The Espresso graph is 225 layers, but a naive emitter expands it to far more rlx
nodes — instance norm into a nine-op statistics chain, every conv bias into a
materialized `Expand`, every hard-swish into four elementwise ops. Three changes
cut the graph by 58% with **bit-identical hashes**:

| change | effect |
|---|---|
| instance norm → one `GroupNorm` (one group per channel) | −8 full-tensor passes × 35 norms |
| bias / squeeze-excite gating via implicit broadcast instead of `Expand` | stops materializing a feature map per conv |
| Espresso's `+3 → clamp(0,6) → ×1/6 → ×x` chains → native `HardSwish` / `HardSigmoid` | 4 ops → 1 (or 2 for a gate), ×28 |

Measured on the real model, interleaved best-of-8 (the machine was loaded, so
read the ratios rather than the absolute times):

| backend | before | after | speedup |
|---|---|---|---|
| cpu | 1109 nodes, 239 ms | 460 nodes, 83 ms | **2.9×** |
| metal | 1109 nodes, 139 ms | 460 nodes, 95 ms | **1.5×** |
| gpu (wgpu) | 1109 nodes, 167 ms | 460 nodes, 12 ms | **14×** (incl. the upstream kernel) |
| mlx | 1109 nodes, 52 ms | 460 nodes, 47 ms | **1.1×** |

`--bench <n>` reports graph size, build time and steady-state latency;
`--no-fuse` emits the literal Espresso chains for A/B timing (same hash).

### The wgpu GroupNorm fix

`rlx-wgpu` lowered `Op::GroupNorm` to `Step::GroupNormHost` — a
device→host→device round-trip of the **whole arena**, per norm. On this model
that is 35 × ~83 MB each way per forward, and in a chain it also disagreed with
the other backends (34 bits; an isolated `GroupNorm` matched CPU to 1e-7, but a
truncation sweep put the first divergence immediately after a norm — the
signature of a staging fault, not bad arithmetic).

Both halves are fixed upstream in `../rlx`:

* a native WGSL kernel (`crates/backends/rlx-wgpu/src/kernels/group_norm.wgsl`)
  — one workgroup per `(batch, group)`, shared-memory tree reduction, the same
  stable two-pass variance as `layernorm.wgsl`. wgpu goes from **1096 ms →
  12.4 ms (88×)** on this model;
* the staging fault itself. Whole-arena host steps rewrite the arena directly
  but never invalidated `HostTensorCache`, so a following cache-aware host step
  could serve a pre-step copy of a slot the norm had just overwritten. It needed
  a dead tensor whose slot got reused *and* both kinds of host step adjacent —
  which is why no minimal repro found it and only the full graph did.

So `RLX_WGPU_HOST_NORM=1` and virtually-sharded arenas are correct now too.

`RLX_NEURALHASH_NO_FUSED_NORM=1` / `RLX_NEURALHASH_NO_BROADCAST=1` force this
crate's fallbacks, for bisecting a future divergence.

## Correctness notes

Four details decide whether a hash matches, and every one of them produces a
well-formed but *wrong* hash if you get it subtly wrong. Measured drift when
each is reverted:

| detail | if misread | bits wrong |
|---|---|---|
| instance-norm params are interleaved `[γ, β, mean, var]` per channel | read as four contiguous blocks | **45** |
| `training_instancenorm` ⇒ runtime statistics | use the stored mean = 0 / var = 1 | **28** |
| `avg_or_max`: **0 = average**, 1 = max | treat 0 as max | **53** |
| `pad_mode`: 1 = SAME (explicit `pad_*` are all 0 and must be ignored) | treat as VALID | graph fails to build |

Plus the three from the reference pipeline:

1. **Resize.** `Image.resize([360, 360])` ignores aspect ratio and defaults to
   **BICUBIC**, and Pillow's resampler is antialiased (it widens the filter
   support by the downscale ratio). `tests/pil_parity.rs` pins this against
   Pillow at the 64 elements where filters disagree most.
2. **Threshold.** The binary step is `score >= 0`, so an exactly-zero score
   sets the bit.
3. **Bit order.** `int(hash_bits, 2)` makes `scores[0]` the most significant
   bit; byte *i* holds scores `8i..8i+8`.

### Perceptual behaviour

Measured on the real model — this is the profile NeuralHash is designed for:

| transform | Hamming distance |
|---|---|
| rescale (½×, 1.5×, 3×) | **0** |
| JPEG quality 35 | **0** |
| contrast ×0.6 | **0** |
| 20 px crop | **0** |
| brightness ×1.45 | 5 |
| horizontal flip | 27 |
| unrelated images | 41–53 (≈ the 48-bit random baseline) |

### Reproducibility

NeuralHash is not bit-stable across *implementations*, and the reference
implementation says the same. Descriptor floats land arbitrarily close to zero,
so any change in floating-point association flips that bit. Compare with
`NeuralHash::hamming`, not equality, unless both hashes came from the same
build. `--scores` prints the smallest `|score|` so you can see how close the
nearest bit was to flipping.

This is a **perceptual** hash — a similarity descriptor over image content. It
is not a cryptographic hash and offers no integrity or authenticity guarantee.

## Testing

```console
$ cargo test -p rlx-neuralhash --release --features all-backends
```

* `tests/real_model.rs` — Apple's installed model: graph derivation against the
  `.shape` sidecar, declared `transform_params`, determinism, and the
  perceptual invariants. Skips cleanly when the model is absent.
* `tests/backends.rs` — cross-backend agreement, synthetic and real.
* `tests/pil_parity.rs` — preprocessing against Pillow ground truth.
* `tests/end_to_end.rs` — full pipeline on a synthetic container.
* Unit tests cover the container parser, shape derivation, weight layouts and
  hash arithmetic against hand-computed values.

## Features

`metal`, `mlx`, `cuda`, `rocm`, `gpu` (wgpu), `vulkan`, plus the aggregates
`all-backends`, `apple-silicon`, `nvidia-gpu`, `amd-gpu`, `portable-gpu`.
`onnx-parity` enables the validation-only ONNX cross-check.
