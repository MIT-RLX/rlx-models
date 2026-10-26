# rlx-subject-lift

The vendor's **subject lifting** network, the model behind Vision's
`VNGenerateForegroundInstanceMaskRequest` ("lift subject from background" in Photos and the
background removal Genmoji uses), run natively by RLX.

**No vendor weights ship with this crate.** It reads the model macOS installs, in place:
`/System/Library/Frameworks/Vision.framework/…/subject_lifting_gen1_rev5_…_multihead_int8.espresso.net`
(macOS 14+). On any other system `SubjectLift::open` reports the model missing.

```rust
let mut lift = rlx_subject_lift::SubjectLift::open(None, device, Default::default())?;
let out = lift.run(&rgb)?;                 // saliency 512², gating confidence
let alpha = out.alpha(rgb.width(), rgb.height());
```

## What the model is

A vendor int8 network read with the `rlx-neuralhash` network-file reader:

- **Input:** 512×512 RGB.
- **Backbone:** EfficientNet-style, with SiLU written as `x·sigmoid(x)`, squeeze-excite via
  global average pooling, depthwise convs and ReLU6.
- **Decoder:** ×2 bilinear upsampling with concat.
- **Two heads:** `saliency` (soft foreground at 512²) and `gating_confidence` (whether there's
  a liftable subject).
- **Weights:** per-output-channel quantized u8, dequantized at load as `w = q·scale + bias`.
- **What Vision does next:** it scales the soft mask to the image
  (`_upsampleLowResMask:reference:`). Measured, that's a slight *feather*, not an edge snap: a
  Gaussian of σ = 0.75 network pixels, then a bilinear resize (`LiftOptions::feather`, `FEATHER`).

## Parity

The settings the file doesn't record were fixed against the Vision framework itself. On nine
images (emoji on white, dark and saturated backgrounds, a photo), `[-1, 1]` RGB with bilinear
upsampling matches Vision's masks at **correlation 0.998, mean |Δα| 0.0038**. Every other
encoding or filter is clearly worse.

Almost all of what was left sat in a band around the subject's edge, with zero mean. Vision's
masks had 2.5× more partial-alpha pixels than ours, and the band's width scaled with the image,
so it is set at the network's 512². Feathering at σ = 0.75 before the resize (mean |Δα| against
`generateScaledMaskForImage`, `rlx-genmoji --example matte_compare`):

| images | bilinear | feathered |
|---|---|---|
| 4 emoji on white, 640 and 1280 px | 0.0027 | **0.0003** |
| the same on dark / teal gradients | 0.0027 | **0.0003** |
| a photo (640×360, 512, 1024) | 0.0031 | **0.0009** |

σ 0.6 and 0.9 both give 0.0010. An edge-aware colour guided-filter upsample, the obvious
reading of `reference:`, moved *away* from Vision (0.0035–0.008), so it isn't what Vision does.

```bash
swiftc -O scripts/vision_subject_lift.swift -o /tmp/vision_subject_lift
/tmp/vision_subject_lift /tmp/lift img1.png img2.png
just features=metal test-subject-lift-parity --device metal img1.png /tmp/lift/img1-vision-mask.png …
```
