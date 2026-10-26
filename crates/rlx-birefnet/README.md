# rlx-birefnet

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (Zheng et al., **MIT**), bilateral-reference
dichotomous image segmentation and the strongest open background-removal model, for RLX on
every backend. It reads the published `ZhengPeng7/BiRefNet*` `model.safetensors` directly.

```bash
just fetch-birefnet            # BiRefNet_lite (Swin-T) + BiRefNet (Swin-L)
```

```rust
let mut m = rlx_birefnet::BiRefNet::open("weights/image/birefnet/BiRefNet_lite".as_ref(), 512, device)?;
let alpha = m.alpha(&rgb)?;     // 0..1 at the image's size
```

## Architecture (one HIR graph, `src/graph.rs`)

- **Encoder:** Swin-v1 (window 7, or 12 for Swin-L, read from the relative-position table) run
  on the image and on a half-size copy. Stage features are concatenated, the shallower ones are
  resized onto the deepest as context, then a squeeze block.
- **Decoder:** four blocks (conv → **deformable ASPP** with 1×1/3×3/7×7 DCNv2 → conv), each fed
  the image as block space-to-depth patches, lateral 1×1 skips and a gradient-attention gate,
  then a 1×1 logit head. BatchNorm is folded into the convs at load.
- **Deformable convolution** (torchvision `deform_conv2d`, `2·sigmoid` modulation) is built from
  `grid_sample2d`. All `k²` taps' positions (`base + learned offset`) form one grid, one bilinear
  sample gathers them, the modulator scales them, and one matmul against the `[O, C·k²]` kernel
  contracts taps and channels.

## Parity vs the reference code (`scripts/birefnet_reference.py`)

| model | input | backbone stages | logits |
|---|---|---|---|
| BiRefNet_lite | 512 | ≤ 4.6e-6 | 6.4e-7 |
| BiRefNet_lite | 1024 | ≤ 4.6e-6 | 1.4e-6 |
| BiRefNet (Swin-L) | 1024 | ≤ 2.2e-5 | 2.1e-6 |

Relative L2 error, Metal; CPU is the same. Getting the 1024 rows exact on Metal required an
upstream rlx-metal fix: the tiled transpose kernel indexed tiles with a 16-bit grid position and
silently re-read rows past 2²¹.

```bash
python3 scripts/birefnet_reference.py --model weights/image/birefnet/BiRefNet_lite --image IMG --size 512 --out REF
just features=metal test-birefnet-parity REF weights/image/birefnet/BiRefNet_lite metal
```

## Compared with the vendor's subject lifting

`rlx-genmoji/examples/matte_compare.rs` scores against Vision's own masks (Vision = reference):

| set | backend | mean \|Δα\| | IoU | boundary F |
|---|---|---|---|---|
| emoji on white | Vision model (rlx-subject-lift) | 0.0027 | 0.9990 | 0.9976 |
| emoji on white | **BiRefNet_lite @512** | **0.0021** | 0.9962 | **0.9995** |
| emoji on white | BiRefNet @1024 | 0.0028 | 0.9958 | 0.9916 |
| dark/coloured + photo | Vision model | 0.0038 | 0.9999 | 0.9996 |
| dark/coloured + photo | BiRefNet_lite @1024 | 0.0048 | 0.9913 | 0.9800 |
