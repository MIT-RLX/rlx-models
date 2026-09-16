# rlx-denoise

Monte-Carlo render denoiser: a guided U-Net over colour, albedo and normal,
trainable end to end in RLX.

A path tracer converges as `1/sqrt(n)`, so the last halving of its noise costs
three quarters of the render. Every production renderer stops early and
reconstructs instead, and the two that matter both do it with a trained
network — Cycles calls OpenImageDenoise, and OptiX has one inside the driver.

## Why not port one

**OptiX.** The denoiser is reached only through `optixDenoiserCreate` /
`optixDenoiserInvoke`. Inspecting driver 595.84 says precisely what is and is
not there.

The architecture is not hidden. `libnvoptix.so` exports two symbols, but its
device code is not stripped, and the kernel names in the `optix_exp` namespace
name every stage: `k_setInput16` and `k_autoexposureInput` for an fp16 input
pass with autoexposure; `k_MaxPooling_NHWC` and `conv_ampere_fpool_*` for
downsampling, pooling fused into the convolution; `k_Scale2x_NHWC`,
`k_bilScale2x_NHWC` and `k_concatSkip_NHWC` for a U-Net decoder with
concatenated skips; `k_SpaceToDepth_NCHW<2>` and `k_DepthToSpace_NCHW<2>` for
pixel shuffle; `k_warpImage`, `k_motionDifference` and `k_warpDisocclusion` for
the temporal variant. An error string — `kpn filter: input, output or weight
layer null` — puts it in the kernel-predicting family: the network emits a
per-pixel filter rather than a colour.

Nothing there is beyond this crate's reach. Every one of those ops is in RLX,
and all but pixel shuffle lower on Metal today.

What cannot be reused is the training. The weights are not in the library at
all — no section approaches the entropy of packed tensors. They are in
`/usr/share/nvidia/nvoptix.bin`, 46.5 MB at entropy 7.392, a container of
chained `(offset, size)` blobs. They are NVIDIA proprietary, not
redistributable, and the OptiX licence forbids reverse engineering them. So the
barrier is legal, not technical.

There is also no portable code to lift. `.nv_fatbin` is 27.6 MB of 306 cubins
covering sm_75 through sm_120 and **zero** PTX entries. OptiX *programs* — the
ones a renderer writes — are PTX, JIT-linked at `optixModuleCreate`, which is
what `rlx-cuda` already replicates through NVRTC. The driver's own kernels ship
as architecture-specific machine code only.

**OpenImageDenoise.** Apache-2.0 down to the weights, so it can legitimately be
read. It is an Intel-shaped U-Net in a container of its own, and tens of
megabytes of parameters.

This crate is the same idea built natively, in ops that already carry autodiff
rules and backend coverage.

## Shape

```text
  in 9 ── enc0 W0 ─────────────────────────── concat ── dec1 W0 ── out 3
             └─ down W1 ──────────── concat ── dec2 W1 ─┘
                   └─ down W2 ── concat ── dec3 W2 ─┘
                         └─ down W3 ── bottleneck ─┘
```

Nine channels in — the render's colour plus the albedo and normal a path tracer
produces at its first useful hit — and three out. Default widths `32/48/64/80`,
about 320k parameters.

Downsampling is a stride-2 convolution and upsampling a stride-2 transposed
convolution, rather than pooling and resize. That keeps the whole network inside
`conv2d` / `conv_transpose2d` / `relu` / `concat`, so the same definition trains
on CUDA and runs on Metal, MLX, ROCm, Vulkan or wgpu with no lowering path of
its own.

The final convolution predicts a *correction* to the input colour and starts at
zero, so an untrained network is the identity and training begins at the
render's own error rather than at noise.

## Training

Pairs cost nothing but time: render a scene at a low sample count and again at a
high one. No corpus needs collecting.

```rust
use rlx_denoise::{Batch, DenoiseNet, TrainConfig, Trainer, Widths};
use rlx_runtime::Device;

let net = DenoiseNet::new(Widths::default());
let mut trainer = Trainer::new(net, 8, 128, 128, Device::Cuda, TrainConfig::default())?;
let loss = trainer.step(&Batch { n: 8, h: 128, w: 128, input: &input, target: &target })?;
```

The loss is relative L2 — `mean((y - t)² / (t² + ε))`. A render is high dynamic
range, and plain L2 in linear radiance is decided almost entirely by the
brightest pixels: a network trained on it polishes highlights and leaves grain
across everything darker.

## Results

Two distributions appear below and they are not comparable with each other. The
**narrow** one is an open floor under a single overhead panel — almost all
direct light. The **wide** one adds enclosed rooms a third of the time, where
indirect illumination dominates. Every figure is on 256 tiles from 64 scenes
that no run trained on and no checkpoint was selected against, relative L2 in
the compressed range with squared errors summed across the set and a single root
at the end.

### Narrow scenes

| | test error | |
|---|---|---|
| unfiltered render | 0.16441 | |
| edge-avoiding À-Trous, parameters fitted | 0.12765 | 1.29x |
| `kpn`, 420 training tiles | 0.05984 | 2.75x |
| `direct`, 420 tiles | 0.05739 | 2.87x |
| `oidn`, 420 tiles | 0.05445 | 3.02x |
| `oidn`, 1860 tiles | 0.05107 | 3.22x |
| `oidn` at `--widths wide`, 1860 tiles | 0.05110 | 3.22x |
| `oidn`, 1860 tiles, four mirrored passes | **0.05131** | 3.20x |
| OpenImageDenoise 1.4.3 | 0.04672 | **3.52x** |

### Wide scenes

| | test error | |
|---|---|---|
| unfiltered render | 0.22856 | |
| this network, trained on the *narrow* set | 0.05329 | 4.29x |
| OpenImageDenoise 1.4.3 | 0.03856 | **5.93x** |

Those two tables are the whole story. On its own distribution this network
finishes 8% behind OIDN; on scenes unlike its training set the same weights fall
38% behind, while OIDN's *absolute* error improves. That gap is what
[Scene variety](#scene-variety) is about, and it is why the generator now builds
rooms.

A caution the numbers above are arranged to make: they were all measured twice.
Regenerating the same scene indices — same code, same seeds, on the GPU rather
than the CPU — moves every one of them, because the Monte-Carlo noise differs.
OIDN scores 0.04561 on the first set of renders and 0.04672 on the second,
against an identical unfiltered 0.16441. A model measured on one and a reference
measured on the other are not comparable, however small the difference looks.

`scripts/oidn_reference.py` reproduces the OIDN rows. It is given the same
128x128 tiles, smaller than the whole frames it is built for, so its figure here
is if anything modest. The À-Trous row is the same measurement on the same
pixels, given guides the network never sees — depth, and the per-pixel variance
the renderer measured. It is the filter at its best.

Trained on CUDA, then evaluated from the same weights file on every backend:

```text
Cuda    0.05430
Cpu     0.05430
Metal   0.05430
```

## Architecture

Two choices are separable, and `--arch` crosses them:

| `--arch` | head | resampling | after |
|---|---|---|---|
| `direct` | residual | strided | this crate's original |
| `kpn` | kernel-predicting | strided | Bako et al. 2017 |
| `oidn` | residual | pooled | OpenImageDenoise |
| `optix` | kernel-predicting | pooled | NVIDIA's OptiX denoiser |

Crossing them gives a factorial, so each choice can be read on its own. Full
widths, scored on the disjoint test set:

| 420 tiles | strided | pooled | pooling buys |
|---|---|---|---|
| **direct head** | 0.05739 | **0.05445** | -5.1% |
| **kernel head** | 0.05984 | 0.05748 | -3.9% |
| *kernel head costs* | *+4.3%* | *+5.6%* | |

At 1860 tiles the pooled pair keeps the same sign: `oidn` 0.05107 against
`optix` 0.05261, the kernel head costing 3.0%.

So both effects hold at both levels of the other factor and at both dataset
sizes. Pooled resampling is worth 4-5%; the kernel head costs 3-6%.

Pooling winning is the smaller surprise — a stride-2 transposed convolution
with a 2x2 kernel tiles evenly and so avoids the usual checkerboard, but it
still spends parameters learning a resampling that does not need learning, and
those parameters do more good in a 3x3 that mixes channels.

The kernel head losing needs the curves, because at any budget short of
convergence it wins:

| epoch | `direct` | `kpn` | `oidn` |
|---|---|---|---|
| 5 | 0.10432 | **0.06851** | 0.10238 |
| 25 | 0.07655 | **0.06086** | 0.07468 |
| 50 | 0.06362 | **0.05717** | 0.06400 |
| 100 | 0.05580 | 0.05828 | **0.05529** |
| 300 | 0.05583 | 0.05614 | **0.05511** |

The kernel head reaches by epoch 5 what direct prediction needs about 50 to
reach, leads by 10% at epoch 50, then flattens and is overtaken near epoch 100.
It converges roughly ten times faster and plateaus slightly higher.

Both halves follow from the same constraint. A softmax over taps makes the
output a convex combination of pixels the renderer actually produced, so an
untrained network is already a sensible filter and gradient descent starts from
somewhere useful — but it also means the network can only ever blend, never
sharpen, and once the easy noise is gone that bound is what limits it. Direct
prediction starts worse and keeps going.

So the head is a training-budget choice rather than a quality one: worth taking
when epochs are scarce, worth dropping when they are not. `--radius` widens the
support if the plateau is what you want to push on.

The lesson about measurement is the sharper one. A short run at reduced width
ranks these architectures in the *opposite* order to a full one, because it
measures how fast a network leaves its initialisation and not where it ends up.

## Guides

The albedo and normal are what separate a denoiser from a blur — they carry no
Monte-Carlo error, because every sample of a pixel agrees about which surface it
hit. Guides taken at the *first* hit describe a mirror rather than the image in
it, so a renderer feeding this should follow the path through specular and
transmissive surfaces before recording them, as Cycles does.

Nine planes went in first — colour, albedo, normal. Adding the two the
renderer already measures and the hand-tuned À-Trous filter already uses, depth
and the per-pixel standard error, made the network **worse**:

| 1860 tiles, same architecture | test error |
|---|---|
| 9 planes | **0.05107** |
| 11 planes, absolute standard error | 0.05268 |

Not neutral — 3.2% worse. The standard error was stored as an absolute
quantity, and its magnitude is set by the scene's brightness and the sample
count rather than by anything local. Across a tile set it varied **17x more
between tiles than within them**: it told the network which render it was
looking at, which the colour already says. Widening `enc0`'s fan-in at the same
channel width then took capacity from every other input, so a plane carrying
nothing was actively harmful rather than merely wasted.

The check is one line and worth running before training on any new plane —
mean within-tile standard deviation over the between-tile spread of tile means:

| plane | within/between |
|---|---|
| absolute standard error | 0.06 |
| depth | healthy |
| relative error | **2.16** |

`Film::resolve_error()` in the renderer already returns the standard error
divided by the pixel's own mean — what adaptive sampling thresholds on, and 36x
more per-pixel signal by that measure. The plane is stored that way now, and
[`Guides`] records which definition a set of weights was trained against:
switching it changes no tensor shape, so weights from one encoding would load
against the other without complaint and denoise quietly wrong.

One honest limit: below two samples a pixel's variance is not knowable from
that pixel, so a quarter of the set — the 1-sample renders — still gets a
constant plane. That is true rather than useful.

## What did not help

Measured on a disjoint test set, never on validation — every one of these looked
different on validation at least once.

| change | effect |
|---|---|
| depth + per-pixel error as extra input planes | **−3.5%** |
| the same, with the error encoded per-pixel rather than absolute | −3.6% |
| 4x the capacity (1.47M parameters against 370k) | none |
| cosine learning-rate decay, against a constant rate | none on quality |
| a kernel-predicting head | −3 to −6% |

Four of five predicted gains returned nothing or less than nothing. The guides
are the sharpest case: they are what the hand-tuned À-Trous filter uses, the
renderer already computes them, and adding them made the network *worse* — two
extra planes widen `enc0`'s fan-in at the same channel width, so every other
input gets a thinner share of the same capacity. Fixing the encoding to carry
36x more per-pixel signal did not change the answer.

The decay is kept despite being neutral: it reaches the same error in 2.5x
fewer epochs, which is worth having even though it is not worth quoting.

What did help was more data, mirrored inference, and — by far the largest —
what the data contained. See [Sample counts](#sample-counts) and below.

## Free at inference

A convolution stack is not symmetric: it answers a mirrored image slightly
differently, and that difference is error rather than signal. Averaging the four
axis mirrors cancels part of it — measured at **2.7%** on a held-out render, for
four times the inference and no retraining at all. Against a path trace that
took minutes, four network passes are free.

The renderer side implements it (`Passes::Mirrored` in threers). Two things have
to be right or it silently misbehaves: the average is taken in the compressed
range before the inverse, because `y/(1-y)` is convex and averaging after would
brighten every frame; and the mirror is applied on read and undone on write, so
the tiling and feathering never see it.

## Scene variety

A denoiser can only generalise over scenes it has been shown, and this is the
one thing that mattered more than everything in "What did not help" put
together.

The first training sets were all one layout: open floor, one overhead panel,
spheres and boxes. Almost all direct light. On that distribution the network
finished 7.9% behind OpenImageDenoise. Tested on a *harder* one — enclosed
rooms, indirect light dominating — the same weights fell **38%** behind:

| diverse test set | error | over unfiltered |
|---|---|---|
| unfiltered | 0.22856 | — |
| this network, trained on the narrow set | 0.05329 | 4.29x |
| OpenImageDenoise 1.4.3 | 0.03856 | 5.93x |

That is the whole diagnosis. It also explains why more data kept returning less
— 4.4x the tiles bought 6%, another 2x bought 3% — because every extra tile was
another sample of the same layout.

So the generator now builds an enclosed room a third of the time, picks shapes
at random from spheres, boxes, tori, cylinders and cones, and adds a second
dimmer light a third of the time. Rooms are the important part and it is about
light transport rather than looks: walls send most of the energy round at least
twice, so indirect illumination dominates and the variance at a given sample
count is far higher — which is the regime a denoiser is bought for, and was
entirely missing.

Worth noting from the table above: on the harder scenes OIDN's *absolute* error
improves (0.03856 against 0.04672 on the easy set) while its ratio nearly
doubles. Noisier input is not harder input for a denoiser, if the noise is the
removable kind.

## Sample counts

A denoiser can only learn noise that is there. With next-event estimation,
multiple importance sampling and a Sobol sequence, a modern path tracer at 32
samples is already within a fraction of a percent of converged on a simple
scene — a dataset built at that rate teaches the network that the identity is
optimal, and it obligingly learns exactly that and nothing else. The pairs here
are 1 to 8 samples, and firefly clamping is loose enough to leave the outliers
in.

## Tests

```sh
cargo test -p rlx-denoise                   # architecture + CPU convergence
cargo test -p rlx-denoise --features cuda   # adds a training run on the device
cargo test -p rlx-denoise --features metal  # adds CPU/Metal agreement, op by op
```

`tests/backend_agreement.rs` runs each op the kernel head introduces on both the
CPU and the accelerator and requires them to match. It was written because they
did not: `Op::Softmax` on any axis but the innermost returned a plausible,
silently wrong tensor on Metal, and the kernel head — which softmaxes the tap
axis of an `[N, taps, H, W]` tensor — trained to convergence on the CPU and
diverged on the GPU. Every backend's softmax kernel reduces *contiguous* runs,
so it can only ever have been the innermost axis; `rlx_fusion::LowerSoftmaxAxis`
now rewrites the rest to transpose / softmax / transpose, the same treatment
`LowerScatterAddAxis` already gave scatter.

The CUDA test trains and requires the loss to fall, which needs every forward
op and every backward rule behind them present on the device — compiling for a
backend is not the same as running on it.
