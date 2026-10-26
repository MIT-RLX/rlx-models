# rlx-upscale

Single-image super-resolution on RLX — fifteen architectures, native Rust, no
PyTorch and no ONNX runtime anywhere on the inference path.

Community upscaler checkpoints are bare state dicts: no config, no architecture
tag, nothing but tensor names and shapes. Point this crate at one and it works
out what the model *is*, builds the matching rlx graph, and runs it tiled so
peak memory is set by the tile rather than by the image.

```console
$ rlx-upscale --model 4xNomos2_realplksr_dysample.pth --image photo.jpg
RealPLKSR ×4 (64 dim, 28 blocks, k17, dysample) on Cpu
  photo.jpg: 1920×1280 → 7680×5120 (tile 512, 431 graph nodes)
photo_upscaled.png
```

## Architectures

**Tier 1 — efficient CNNs.** Sub-megabyte to ~10M parameters, well under a
gigabyte of working set, fast enough for video.

| Arch | What it is |
|---|---|
| `ESRGAN` | RRDBNet — **by volume the most deployed SR architecture there is**: `4x-UltraSharp`, `4x_foolhardy_Remacri`, BSRGAN, RealSR and most of OpenModelDB |
| `Compact` | `SRVGGNetCompact`, the Real-ESRGAN compact net |
| `SPAN` | Swift Parameter-free Attention Network (NTIRE 2024) |
| `SPANV2` | Team XiaomiMM's **NTIRE 2026 Efficient SR challenge winner** |
| `PLKSR` | Partial Large Kernel SR (`CCM` / `ICCM` / `DCCM` mixers) |
| `RealPLKSR` | PLKSR retuned for real degradation, with optional `DySample` |
| `SAFMN` | Spatially-Adaptive Feature Modulation — receptive field from pooling rather than from attention or large kernels |
| `RealCUGAN` | `upcunet_v3` — two valid-padded U-Nets in series; **the anime upscaler** |

**Tier 2 — window-attention transformers.** Higher fidelity, 20–35× slower,
bounded in memory only because the image is tiled.

| Arch | What it is |
|---|---|
| `SwinIR` | The window-attention SR baseline |
| `Swin2SR` | SwinIR rebuilt on Swin **V2** blocks — cosine attention, a learned temperature, an MLP-generated position bias and res-post-norm |
| `HAT` | Hybrid Attention Transformer (+ channel attention, + overlapping cross-attention) |
| `DRCT` | Dense-Residual-Connected Transformer |
| `DAT` | Dual Aggregation Transformer (rectangular windows + transposed channel attention) |
| `MambaIRv2` | Attentive State Space — window attention + a selective scan over semantically sorted tokens |
| `OmniSR` | Omni Self-Attention — MaxViT-style: block attention, **grid** attention and channel attention interleaved |

Every upsampler variant is covered: `pixelshuffle`, `pixelshuffledirect`,
`nearest+conv`, `DySample`, and the bare-`conv_last` **residual** tail SwinIR
uses for denoising and JPEG-artifact reduction — where the network predicts a
correction added to the input rather than a magnification. Scales ×1
(restoration), ×2, ×3, ×4 and ×8.

## Transparency

A source with an alpha channel keeps it. This is worth stating because the easy
mistake is loud in its silence: `image::open(..).to_rgb8()` will flatten a logo,
icon or sprite sheet onto an opaque rectangle and report success.

`upscale_rgba8` splits the mask off, and `--alpha` chooses what happens to it:

| mode | what it does | when |
|---|---|---|
| `upscale` *(default)* | runs the mask through the same network | edges of mask and colour agree, which is what stops fringing |
| `resize` | bicubic resample | ~free next to a second network pass, and cannot invent detail in the mask |
| `discard` | drops it, returns RGB | you want the rectangle |

A fully opaque image skips the second pass entirely.

None of these premultiply. A PNG's fully-transparent pixels usually hold
arbitrary RGB, and upscaling mixes that into the edge where alpha ramps up.
Premultiplying would avoid it but feeds the network colours it was never trained
on. Straight alpha is what these models expect; clean the asset, not the
upscaler.

## Usage

```rust,no_run
use rlx_upscale::{Upscaler, UpscaleOptions};
use rlx_runtime::Device;

let mut up = Upscaler::open("4xNomos2_hq_dat2.pth", Device::Cpu, UpscaleOptions::default())?;
println!("{}", up.config().summary());          // DAT ×4 (180 dim, depths [6, 6, 6, 6, 6, 6], split [8, 32])

let (pixels, w, h) = up.upscale_rgb8(&rgb, 640, 480)?;
# anyhow::Ok(())
```

`--inspect` prints the recovered configuration as JSON without running anything,
which is the fastest way to find out what an unlabelled `.pth` actually is.

## Examples

Five runnable programs in `examples/`, each answering a question you actually
have when you download an upscaler.

```console
# What are all these files? Walks a folder of unlabelled checkpoints.
$ cargo run --release -p rlx-upscale --example identify -- ~/models
file                                   architecture scale   tile peak est tensors
4xBHI_small_hat-l.pth                  HAT             ×4     80   2025 MB    1710
                                       HAT ×4 (180 dim, depths [6 × 12], win 16)
MambaIRV2L_DFLIP_X4.pth                MambaIRv2       ×4    160   1699 MB    2519
                                       MambaIRv2 ×4 (174 dim, depths [6 × 9], win 16, d_state 16)
```

```console
# Just upscale something. The whole API in ~30 lines.
$ cargo run --release -p rlx-upscale --example upscale -- \
    --model model.pth --image photo.jpg
```

```console
# What tile should I use? Measures the real trade-off on this machine, each
# configuration in a fresh process so peak RSS is not a stale high-water mark.
$ cargo run --release -p rlx-upscale --example tile_sweep -- \
    --model model.pth --image photo.png
  tile   est peak   peak RSS est/meas      time    halo
   197     511 MB     169 MB    3.02×     1.14s    1.4×
   295    1147 MB     294 MB    3.90×     0.79s    1.3×
   394    2046 MB     485 MB    4.22×     1.33s    1.2×
```

```console
# Build a labelled comparison sheet: every model, one image, side by side.
# --crop runs the models only on the region the sheet shows.
$ cargo run --release -p rlx-upscale --features metal --example gallery -- \
    --models ~/models --images ~/pics --out ./sheets --crop 160x120 --device metal
```

```console
# Which model? Runs them all over one image, with fidelity and cost.
$ cargo run --release -p rlx-upscale --example compare_models -- \
    --models ~/models --image photo.png --out ./compare
model                              scale   tile      time  est peak fidelity
2xHFA2kSPAN                           ×2    160     0.16s    506 MB   0.9519
realesr-general-x4v3                  ×4    160     0.17s    675 MB   0.9985
team22_spanv2_c2                      ×4    160     0.03s    337 MB   0.9998
```

`compare_models`' fidelity column reads as *how much does this model change the
picture*, not *how good is it* — see the note on correlation below. Look at the
PNGs.

## How it works

**Detection.** `detect` identifies a family by a key only it has, then solves
for its dimensions. Window size falls out of the relative-position bias table
(it has exactly `(2W−1)²` rows); HAT's overlap ratio falls out of its second
table; DySample's group count falls out of the width of its offset convolution.
Three reference hyperparameters genuinely leave no trace in the weights and are
documented as assumptions rather than guessed silently — see `detect`'s module
docs.

**Tiling.** rlx graphs are shape-static, which here is a feature: the network is
compiled **once** for a single tile shape and every tile — edge tiles included —
is that exact shape, padded by edge replication. A 6000×4000 photo and a 256×256
thumbnail cost the same per step.

**The tile is derived from the model's memory cost, not its family**, because
the two vary by two orders of magnitude *within* a family. Peak RSS measured
against tile area on a 512×384 input:

| model | per input pixel | derived tile |
|---|---|---|
| SPANV2 ×4 (3×3, 32 feat) | 0.0027 MB | 394 |
| Compact ×4 (3×3, 64 feat) | 0.020 MB | 278 |
| RealPLKSR ×2 (**17×17**, 64 dim) | 0.25 MB | 98 |
| HAT-L ×4 (window 16 + overlap) | — | 80 |

RealPLKSR is ~90× SPANV2 per pixel for the same output width, because a 17×17
kernel over 16 channels is a 4624-wide im2col row against SPANV2's 288. Between
transformers the split is just as wide: HAT-L and SwinIR-M are both 180-dim with
six groups of six, but HAT's window is twice as wide and its overlapping
cross-attention reads a 2.25× larger key neighbourhood.

And the split does not follow the tier. MambaIRv2 is a transformer that
convolves like a conv net — two `ConvFFN`s per layer, each a depthwise 5×5 at
twice the embedding width, 108 of them across nine groups of six — so it is
costed with the conv-net live-buffer count. Treating it as a Swin-family model
under-estimated a real MambaIRv2-L by 5× and put it at 10.6 GB.

`ModelConfig::peak_working_set_bytes` estimates both terms and `tile_for_budget`
shrinks along the window grid until it fits. The multipliers in it are **fitted
to measured RSS**, not derived — how well the arena reuses a transient belongs
to the backend's memory planner. HAT-L's three measured points fit
`0.200 MB/px² · tile² + 1.12 GB` to within 1%, and the estimate tracks them to
0.8–1.3×. Treat it as a planning aid with a factor-of-two error bar.

**A smaller tile is not free.** Only the `(tile − 2·overlap)²` interior survives
the crop, so shrinking the tile multiplies redundant work — HAT-L on a 512×384
input took 539 s at tile 176 but **1028 s at tile 128**, nearly twice as long
for 40% less memory. The default budget is 2 GiB, aimed at a consumer GPU; on a
machine with headroom raise it with the workspace-wide `RLX_MAX_RAM_BYTES`
rather than living with the default:

| `RLX_MAX_RAM_BYTES` | HAT-L tile |
|---|---|
| unset (2 GiB) | 80 |
| 4 GiB | 112 |
| 8 GiB | 160 |
| 16 GiB | 224 |

`--inspect` prints the tile a checkpoint will actually use, and each run reports
its halo overhead (`2.2× halo work`). `--tile` pins it outright.

Each output block is produced from a source region grown by a halo and only the
block's own pixels are kept. For a conv net the halo covers the receptive field,
so the tiled result is *identical* to an untiled pass and there is no seam to
blend. For a deep transformer it cannot be: shifted windows propagate half a
window per block, so across twelve groups the receptive field exceeds any halo
worth paying for. One window is used — enough that seams are not visible, and
honest about not being exact.

**Weights are not kept twice.** Opening from a file hands the tensors to the
graph and then drops them, remembering only the path. A recompile at a different
tile re-reads from disk — slow, and rare — rather than paying a full extra copy
of the weights for the object's whole life, which is 231 MB on DRCT-L and 158 MB
on HAT-L. `holds_weights()` reports the state; an `Upscaler` built from tensors
in memory has nowhere to re-read from and keeps them.

**Compile-time folding.** Anything that does not depend on the input is resolved
while the graph is built rather than on every tile:

* SPAN's `Conv3XC` trains as a 1×1 → 3×3 → 1×1 expansion plus a 1×1 skip and
  collapses to a single 3×3. The fold is recomputed from the branches, not read
  from the checkpoint's `eval_conv` — measured on a release checkpoint, the
  stored copy is **11% stale**, because the reference recomputes it on every
  forward.
* The Swin lineage's relative-position bias is a gather of a frozen table
  through a fixed index; it becomes a constant. Keeping the mask at
  `[1, nW, 1, N, N]` instead of the `[B·nW, nH, N, N]` a fused attention op
  would demand is the difference between 590 KB and 340 MB for a 192×192 tile.
* DAT's `DynamicPosBias` MLP and every `BatchNorm2d` are evaluated on the host.

**MambaIRv2 is deterministic here; upstream is not.** The reference routes
tokens with `F.gumbel_softmax(..., hard=True)` and has **no `self.training`
guard anywhere in the file**, so the released model draws fresh Gumbel noise on
every forward — the token ordering, the scan and therefore the image differ run
to run, and `torch.sort(..., stable=False)` over `num_tokens` distinct values
adds a second source through tie-breaking. This port takes the `argmax` of the
routing logits with a stable sort. That is a deliberate deviation: a
nondeterministic upscaler is not useful, and matching PyTorch's RNG stream *and*
its unstable-sort tie-breaking is not achievable in any case. Two exact
simplifications fall out — the trailing `LogSoftmax` cannot change an argmax so
it is skipped, and the prototype table is a product of two frozen parameters so
it folds to a constant.

**Bug-compatibility.** Upstream HAT's overlapping-cross-attention index is
shifted by `ws − ows + 1` where centring would need `(ows − ws)/2 + ws − 1`, so
it goes negative and PyTorch wraps it from the end of the table. The released
weights were trained that way, so the wrap is reproduced deliberately — see
`oca_rpe_index`. "Fixing" the centring would silently permute the learned bias.

## Weights

Nothing is bundled. `.pth` / `.pt` / `.bin` (via
[`rlx-torch-ckpt`](../../../rlx/crates/io/rlx-torch-ckpt)) and `.safetensors`
all load, and the usual trainer wrappers (`params`, `params_ema`, `state_dict`,
`net_g`, …) are unwrapped. Models live at [OpenModelDB](https://openmodeldb.info)
and [Phhofm/models](https://github.com/Phhofm/models).

A checkpoint tensor the graph never reads is a **hard error**, not a warning: it
almost always means detection picked the wrong architecture, and the alternative
is an image that is wrong in a way that looks like a bad model.

## Testing

`tests/architectures.rs` builds and executes all fifteen families — every mixer,
upsampler, normalization variant and shift schedule — on synthesized weights.
That does not check numerics (nothing can, without the trained weights), but it
does exercise every shape, permutation and window partition, which is where
porting bugs actually live.

`tests/real_weights.rs` runs release checkpoints if `RLX_UPSCALE_MODELS` points
at a directory of them, and checks that downscaling a ×N result by N lines up
with the input. It measures **correlation**, not PSNR, because restoration
models are *trained* to change the image:

**38 release checkpoints**, covering all fifteen architectures, every scale and
every upsampler tail:

| model | tail | r | PSNR |
|---|---|---|---|
| **OmniSR ×3** — official DIV2K | **grid + block + channel attention** | 0.9999 | 53.8 dB |
| OmniSR ×4 — official DIV2K | pixelshuffle | 0.9998 | 50.3 dB |
| OmniSR ×2 — official DIV2K | pixelshuffle | 0.9997 | 49.8 dB |
| SPANV2 ×4 — NTIRE 2026 winner | pixelshuffle | 0.9998 | 50.6 dB |
| **Swin2SR** ×2 — lightweight | pixelshuffledirect | 0.9998 | 51.7 dB |
| Swin2SR ×2 — classical | pixelshuffle | 0.9998 | 50.4 dB |
| Swin2SR ×4 — classical | pixelshuffle ×2 ×2 | 0.9996 | 48.1 dB |
| Swin2SR ×4 — real-world BSRGAN | nearest+conv | 0.9992 | 45.5 dB |
| SwinIR-M ×4 — classical | pixelshuffle ×2 ×2 | 0.9998 | 49.6 dB |
| SwinIR-S ×2 — lightweight | pixelshuffledirect | 0.9998 | 51.4 dB |
| SwinIR-M ×2 — classical | pixelshuffle | 0.9998 | 51.0 dB |
| SwinIR-M ×3 — classical | pixelshuffle ×3 | 0.9997 | 49.8 dB |
| **SwinIR-M ×1 — colour denoising** | **residual** | 0.9997 | 49.1 dB |
| HAT-L ×4 | pixelshuffle | 0.9997 | 47.5 dB |
| MambaIRv2-L ×4 | pixelshuffle | 0.9996 | 47.6 dB |
| SAFMN ×2 / ×4 — official DF2K | pixelshuffle | 0.9996 | 48.4 dB |
| SAFMN-L ×4 — official DF2K | pixelshuffle | 0.9996 | 47.6 dB |
| DRCT-L ×4 | pixelshuffle | 0.9996 | 47.3 dB |
| DAT-2 ×4 — `4xNomos2_hq` | pixelshuffle | 0.9994 | 42.0 dB |
| DAT-2 ×4 — `4xBHI_otf` | pixelshuffle | 0.9984 | 41.0 dB |
| PLKSR ×2 / ×4 — official DF2K | pixelshuffle | 0.9997 | 49.1 dB |
| ESRGAN ×4 — `RealESRGAN_x4plus` | upconv | 0.9909 | 32.4 dB |
| ESRGAN ×4 — `..._anime_6B` | upconv | 0.9906 | 33.0 dB |
| ESRGAN ×2 — `RealESRGAN_x2plus` (unshuffle) | upconv | 0.9671 | 27.7 dB |
| PLKSR-tiny ×4 — official DF2K | pixelshuffle | 0.9997 | 48.5 dB |
| Compact ×4 — `realesr-general-x4v3` | pixelshuffle | 0.9977 | 38.3 dB |
| RealPLKSR ×2 — dysample + layernorm | **DySample** | 0.9963 | 38.5 dB |
| **Real-CUGAN ×2** — `2xHFA2k` | **valid U-Net + transposed conv** | 0.9897 | 34.4 dB |
| **Real-CUGAN ×4** | **valid U-Net + pixelshuffle** | 0.9781 | 29.6 dB |
| SAFMN-L ×4 — real-world LSDIR | pixelshuffle | 0.9785 | 30.2 dB |
| OmniSR ×1 — `1xDeJPG` (artifact removal) | **residual-free ×1** | 0.9979 | 41.1 dB |
| OmniSR ×2 — `2xEvangelion` (anime GAN) | pixelshuffle | 0.9746 | 22.9 dB |
| Compact ×4 — `realesr-animevideov3` | pixelshuffle | 0.9928 | 32.2 dB |
| **SwinIR-M ×4 — real-world GAN** | **nearest+conv** | 0.9890 | 33.6 dB |
| SPAN ×2 — anime restoration | pixelshuffle | 0.9496 | 13.8 dB |

`RLX_UPSCALE_DEVICE=metal` makes the tier-2 half practical to run routinely —
it is minutes per model on CPU.

The anime SPAN row is the point: it scores worst on PSNR by a wide margin while
producing a visibly better picture than models scoring 50 dB, because it is
*supposed* to change the image. Correlation is invariant to the per-channel gain
and offset a restoration model applies, and is destroyed by any spatial
scrambling — so it separates "renders differently" from "is broken".

## Measured

512×384 input, M-series CPU unless noted, at the auto-derived tile:

| model | tile | time | peak RSS |
|---|---|---|---|
| SPANV2 ×4 — NTIRE 2026 winner | 394 | 1.5 s | 487 MB |
| SPAN ×2 | 321 | 2.4 s | 781 MB |
| Compact ×4 | 278 | 8.8 s | 1.58 GB |
| RealPLKSR ×2 — 17×17 kernel | 98 | 50 s | 2.53 GB |
| SPANV2 ×4 — Metal | 256 | 0.5 s | 531 MB |

Tile size is the memory dial, and it barely costs throughput — SPANV2 at tile
512 / 256 / 128 on the same input is 783 / 235 / 109 MB at 1.31 / 1.06 / 0.83 s.
Smaller tiles are, if anything, *faster* on CPU, because the working set starts
fitting in cache.

## Backends

CPU by default; `metal`, `mlx`, `cuda`, `rocm`, `gpu` (wgpu), `vulkan` and
`coreml` (Neural Engine) are feature flags, with `all-backends` and
`macos-gpu` as bundles.

**All fifteen architectures run on all six locally-testable backends.**
`tests/backends.rs` runs every backend compiled into the build against CPU and
requires agreement:

| backend | worst \|Δ\| vs CPU, over all 15 |
|---|---|
| Metal | 4.2e-7 |
| MLX | 4.8e-7 |
| wgpu | 1.8e-7 |
| Vulkan | 1.8e-7 |
| ANE (CoreML) | 3.6e-7 |

CUDA and ROCm are **not** numerically verified — there is no such hardware here
and the test reports them skipped rather than passing them silently. What *is*
established, for every backend in the RLX tree, is op coverage — decidable
without the hardware:

```
python3 scripts/upscale_backend_matrix.py          # the matrix
python3 scripts/upscale_backend_matrix.py --check  # exit 1 on a NEW gap
```

`--check` is a regression guard, not a demand that every cell be green: gaps on
a backend with no FP32 datapath, and the one documented WebGL exception, are
expected. Anything else fails — verified by removing `Conv` from wgpu's op set
and confirming it fires.

| backends | coverage |
|---|---|
| CPU, Metal, MLX, wgpu, CUDA, ROCm, Vulkan, ANE, OneAPI, TPU | **all 15** |
| WebGL | 14/15 — MambaIRv2 needs `ArgSort` |
| QNN (Hexagon), XDNA (AIE-ML) | int8/fp16 NPUs with no FP32 datapath; not a target for an f32 upscaler |
| eGPU, Cerebras, Cortex-M, FPGA | no declared op set — transport stub, synthesis backends, int8 MCU |

The script accounts for *every* crate under `crates/backends/`, so "all
backends" has a definite meaning: `--check` fails if a new backend appears that
the matrix does not classify.

### Reachable is not the same as claimed

That script exists because reading `SUPPORTED_OPS` alone reports false gaps. An
op is runnable on a backend by three routes, and two of them are invisible in
the claim set:

1. the backend claims it in `SUPPORTED_OPS`;
2. the legalize loop rewrites it when the target does not claim it — this is how
   `Pad` runs everywhere despite only Metal and CUDA having a kernel;
3. it carries `OpCaps::FUSED`, so the unfuse pass decomposes it.

Route 2 is also where a gap gets *closed*. `Op::ConvTranspose2d` had a native
kernel only on Metal and CUDA, so Real-CUGAN — which uses four of them — was
Metal/CUDA-only. Adding `rlx_fusion::LowerConvTranspose2d` (dilate → flip →
ordinary `Conv`) made it reachable on every backend at once, and the eight
backends that do have the kernel still use it.

Route 3 is why MambaIRv2 runs on the ANE at all: CoreML leaves `SelectiveScan`
*deliberately* unclaimed so the rewriter emits the compact `Op::Scan` form,
because the Metal shader compiler is superlinear in program size and the unrolled version
takes 307 s to compile at length 1000. A claim-only matrix would call that a gap.

### Rank 5 is a hard ceiling on CoreML

Window partition and pixel shuffle are naturally rank-6 permutations
(`[N, C, r, r, H, W]`), and MIL rejects any tensor above rank 5. Every such op
here is therefore written with the batch axis **elided** — it is structurally 1,
since the graph runs one tile at a time — which costs nothing and is the
difference between the ANE working and not compiling at all. `Builder::rank5`
checks the assumption rather than trusting it.

### Metal's arena: fixed, 7.5× smaller

Metal re-plans its arena without the output-ancestor pin — which is what lets
slots be reused — but it only *tried* that when the pinned plan overflowed a
hard limit (`maxBufferLength`, or the 4 GiB MPSGraph bind cliff). A graph that
fit under both kept the pinned plan forever, however wasteful it was. Measured
on this crate:

| model (tile 128) | was | now |
|---|---|---|
| OmniSR ×4 | 2986 MB | **394 MB** |
| Real-CUGAN ×4 | 957 MB | **209 MB** |

CPU and CoreML plan the same OmniSR graph at ~400 MB and ~330 MB, which is how
the gap showed up at all: Metal was the only backend needing gigabytes for a
1.6 MB model. The fix makes the re-plan a memory optimization as well as an
overflow escape hatch, keeping the existing 25% "substantial saving" floor and
the host-indexing guard. `rlx-metal`'s `unpin_tests` pin the decision table.

### Measured: Metal's cost here is dispatch, not kernels

On OmniSR ×4, `RLX_METAL_THUNK_PROFILE=1` accounts for **351 ms of GPU kernel
time in a 6.4 s wall-clock run** — `batched_sgemm` 49%, `conv2d` 26%,
`transpose` 6%. The kernels are not the problem; encoding and compiling a
1758-node graph is. The shape of it:

| model | nodes | Metal | MLX |
|---|---|---|---|
| Compact ×4 | ~50 | **0.14 s** | 0.27 s |
| SAFMN-L ×4 | | 3.59 s | 1.38 s |
| SwinIR-M ×4 | | 4.76 s | 0.71 s |
| OmniSR ×4 | 1758 | 4.38 s | **0.36 s** |

Metal is the *fastest* backend on a small conv-only graph and the slowest on a
large one, while MLX barely notices the difference — so pick by graph size, not
by family. This is a property of the Metal backend's per-dispatch overhead, not
of anything in this crate; it is recorded here because the naive reading
("Metal is the native macOS backend, so use Metal") is wrong for tier 2.

On MLX, MambaIRv2 falls back from compiled to `MlxMode::Lazy` — `ArgSort` cannot
be evaluated inside a traced graph — and still agrees. The fallback is automatic
and logged.

## License

GPL-3.0-only, as the rest of this workspace.
