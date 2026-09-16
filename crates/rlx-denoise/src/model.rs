// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The network: a guided U-Net from `[N, 9, H, W]` to `[N, 3, H, W]`.
//!
//! Eleven input channels: the render's colour, and the guides a path tracer
//! already produces alongside it — first-hit albedo, shading normal and depth,
//! plus the per-pixel standard error it accumulated. The guides are what let a
//! denoiser tell a texture edge from noise: they carry no Monte-Carlo error,
//! because every sample of a pixel agrees about what surface it hit.
//!
//! # Shape
//!
//! Four resolutions, halving each time, with skip connections across:
//!
//! ```text
//!   in 9 ── enc0 W0 ─────────────────────────── concat ── dec1 W0 ── out 3
//!              └─ down W1 ──────────── concat ── dec2 W1 ─┘
//!                    └─ down W2 ── concat ── dec3 W2 ─┘
//!                          └─ down W3 ── bottleneck ─┘
//! ```
//!
//! Downsampling is a stride-2 convolution and upsampling a stride-2 transposed
//! convolution, rather than pooling and nearest-neighbour resize. Both choices
//! keep the whole graph inside `conv2d` / `conv_transpose2d` / `relu` /
//! `concat` — ops with autodiff rules and coverage on every backend — so the
//! same definition trains on CUDA and runs on Metal or wgpu without a lowering
//! path of its own.
//!
//! # Residual output
//!
//! The final convolution predicts a *correction* to the input colour, not the
//! image. Initialised at zero it makes the untrained network the identity, so
//! training starts from "leave the render alone" and the loss can only fall
//! from the raw error. Predicting the image directly means starting from noise
//! and learning to reproduce the input before learning to improve it.

use rlx_ir::{DType, Graph, GraphExt, NodeId, PadMode, Shape};

/// Colour, albedo, normal, depth and the renderer's own standard error.
///
/// The last two are what the hand-tuned À-Trous filter has always used and the
/// network did not — leaving them out gave the fixed filter strictly more to
/// work with. The standard error is the important one: it is the renderer
/// stating how far each pixel can be trusted, which the network otherwise has
/// to infer from the very noise it is removing.
pub const IN_CHANNELS: usize = 11;

/// Colour, albedo and normal — the guides every denoiser here has always had.
///
/// A network can be built at this width instead, which is what makes "do depth
/// and the error plane earn their channels" answerable: the two sets are
/// otherwise the same renders, so the comparison isolates the guides rather
/// than confounding them with a fresh seed.
pub const CORE_CHANNELS: usize = 9;
/// Denoised colour.
pub const OUT_CHANNELS: usize = 3;
/// Convolution kernels are 3×3 throughout, except the stride-2 transposed ones.
const K: usize = 3;

/// What the guide planes beyond colour, albedo and normal actually contain.
///
/// The weights depend on this and nothing about their shape does: a network
/// trained when plane 10 held one quantity, run against a renderer that fills
/// it with another, loads without complaint and denoises slightly wrong
/// forever. So it is recorded next to the architecture and checked on load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Guides {
    /// Plane 9 depth, plane 10 the *absolute* standard error.
    ///
    /// Measured across a tile set this varied 17x more between images than
    /// within one — its size is set by the scene's brightness and sample count,
    /// so it told the network which render it was looking at rather than which
    /// pixel. Kept only so older weights still identify themselves.
    AbsoluteError,
    /// Plane 9 depth, plane 10 the standard error *relative to the pixel's own
    /// mean* — what adaptive sampling thresholds on, and 36x more per-pixel
    /// signal by the same measure.
    #[default]
    RelativeError,
}

impl Guides {
    pub fn code(self) -> u32 {
        match self {
            Guides::AbsoluteError => 0,
            Guides::RelativeError => 1,
        }
    }

    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Guides::AbsoluteError),
            1 => Some(Guides::RelativeError),
            _ => None,
        }
    }
}

/// How the last layer turns features into a colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    /// Predict a correction and add it to the input colour.
    Direct,
    /// Predict a normalised `(2r+1)²` filter per pixel and apply it to the
    /// input colour — a kernel-predicting network, after Bako et al. 2017.
    ///
    /// The output is a convex combination of pixels the renderer actually
    /// produced, because a softmax makes the taps non-negative and sum to one.
    /// That is a real constraint rather than a regulariser: the network cannot
    /// invent a colour that is not already in the neighbourhood, so it cannot
    /// hallucinate detail, and it cannot overshoot into a value no sample
    /// supports. Direct prediction has neither guarantee, which is why it
    /// tends to smear on high dynamic range input.
    ///
    /// NVIDIA's OptiX denoiser is in this family — driver 595.84 carries the
    /// error string `kpn filter: input, output or weight layer null`.
    Kernel { radius: usize },
}

impl Head {
    /// Channels the final convolution has to emit.
    fn out_channels(&self) -> usize {
        match self {
            Head::Direct => OUT_CHANNELS,
            Head::Kernel { radius } => {
                let side = 2 * radius + 1;
                side * side
            }
        }
    }
}

/// How the network changes resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampling {
    /// Halve with a stride-2 convolution, double with a stride-2 transposed
    /// one. The resampling is learned, and costs parameters.
    Strided,
    /// Halve with a 2×2 average pool, double with nearest-neighbour, both
    /// fixed. OpenImageDenoise and the OptiX denoiser are both built this way —
    /// `k_MaxPooling_NHWC`, `conv_ampere_fpool_*` and `k_Scale2x_NHWC` are all
    /// named in `libnvoptix`.
    ///
    /// The resampling itself carries no weights, so a level's parameters go
    /// entirely into a 3×3 that mixes channels at one resolution. It also
    /// avoids the checkerboard a stride-2 transposed convolution produces when
    /// its kernel does not divide evenly into its stride.
    Pooled,
}

/// A network's shape: widths, output head and resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arch {
    /// Input planes the network takes. [`IN_CHANNELS`] with the guides,
    /// [`CORE_CHANNELS`] without.
    ///
    /// Part of the architecture because it sets `enc0`'s shape — a network
    /// built at one count cannot load weights from another, and the checkpoint
    /// catches that through the tensor sizes.
    pub inputs: usize,
    pub widths: Widths,
    pub head: Head,
    pub sampling: Sampling,
    /// What the guide planes hold. Does not change any tensor's shape, which
    /// is exactly why it has to be recorded rather than inferred.
    pub guides: Guides,
}

impl Default for Arch {
    fn default() -> Self {
        Self {
            inputs: IN_CHANNELS,
            widths: Widths::default(),
            head: Head::Direct,
            sampling: Sampling::Strided,
            guides: Guides::default(),
        }
    }
}

impl Arch {
    /// The shape distilled from NVIDIA's OptiX denoiser: pooled resampling and
    /// a kernel-predicting head.
    pub fn optix_like(widths: Widths) -> Self {
        Self {
            inputs: IN_CHANNELS,
            widths,
            head: Head::Kernel { radius: 2 },
            sampling: Sampling::Pooled,
            guides: Guides::default(),
        }
    }

    /// The shape distilled from OpenImageDenoise: pooled resampling, direct
    /// prediction.
    pub fn oidn_like(widths: Widths) -> Self {
        Self {
            inputs: IN_CHANNELS,
            widths,
            head: Head::Direct,
            sampling: Sampling::Pooled,
            guides: Guides::default(),
        }
    }
}

/// Channel widths at the four resolutions.
///
/// Scaled down from OpenImageDenoise's `32/48/64/80`, which is the shape this
/// follows. The default here is the same at the top two levels and narrower
/// below, which is where the parameters concentrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widths {
    pub level0: usize,
    pub level1: usize,
    pub level2: usize,
    pub level3: usize,
}

impl Default for Widths {
    fn default() -> Self {
        Self {
            level0: 32,
            level1: 48,
            level2: 64,
            level3: 80,
        }
    }
}

impl Widths {
    /// A quarter-width network, for tests and for checking a training loop
    /// converges before spending a GPU on it.
    pub fn tiny() -> Self {
        Self {
            level0: 8,
            level1: 12,
            level2: 16,
            level3: 20,
        }
    }

    /// Double width, so about four times the parameters.
    ///
    /// Whether this helps is a question about the dataset, not the network: on
    /// too few scenes the extra capacity is spent memorising them. Measured
    /// worthless at a 1.51x train/val gap and worth 5% at 1.08x, which is the
    /// same change tested either side of fixing the data.
    pub fn wide() -> Self {
        Self {
            level0: 64,
            level1: 96,
            level2: 128,
            level3: 160,
        }
    }

    /// Roughly the size of the filter this is measured against.
    ///
    /// OpenImageDenoise's `RT` filter carries on the order of four million
    /// weights; [`Widths::wide`] is 1.47 million. Capacity is not the whole
    /// difference between the two — the corpus is — but comparing a model to one
    /// three times its size and attributing the gap elsewhere is not a
    /// comparison worth making.
    ///
    /// Costs about 2.6x `wide` per epoch, and wants the larger corpus to go
    /// with it.
    pub fn huge() -> Self {
        Self {
            level0: 96,
            level1: 160,
            level2: 224,
            level3: 288,
        }
    }
}

/// One convolution's parameter: name and `[out, in, kh, kw]`.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: &'static str,
    pub shape: [usize; 4],
}

impl ParamSpec {
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    /// Fan-in of the convolution this weight belongs to, for initialisation.
    pub fn fan_in(&self) -> usize {
        self.shape[1] * self.shape[2] * self.shape[3]
    }
}

/// The network's architecture: widths, and the parameter list they imply.
#[derive(Debug, Clone)]
pub struct DenoiseNet {
    arch: Arch,
    params: Vec<ParamSpec>,
}

impl DenoiseNet {
    /// The default architecture at these widths: direct head, strided
    /// resampling.
    pub fn new(widths: Widths) -> Self {
        Self::with_arch(Arch {
            widths,
            ..Arch::default()
        })
    }

    pub fn with_arch(arch: Arch) -> Self {
        let Widths {
            level0: w0,
            level1: w1,
            level2: w2,
            level3: w3,
        } = arch.widths;
        // Pooled resampling does not learn its own upsample, so each `up` is an
        // ordinary 3×3 applied after a fixed nearest-neighbour double; strided
        // resampling folds the widening into a stride-2 transposed convolution,
        // whose weights are `[in, out, kh, kw]`.
        let up = |from: usize, to: usize| -> [usize; 4] {
            match arch.sampling {
                Sampling::Strided => [from, to, 2, 2],
                Sampling::Pooled => [to, from, K, K],
            }
        };
        // Order matters: it is the order gradients come back in.
        let params = vec![
            ParamSpec {
                name: "enc0",
                shape: [w0, arch.inputs, K, K],
            },
            ParamSpec {
                name: "down1",
                shape: [w1, w0, K, K],
            },
            ParamSpec {
                name: "down2",
                shape: [w2, w1, K, K],
            },
            ParamSpec {
                name: "down3",
                shape: [w3, w2, K, K],
            },
            ParamSpec {
                name: "bottleneck",
                shape: [w3, w3, K, K],
            },
            ParamSpec {
                name: "up3",
                shape: up(w3, w2),
            },
            ParamSpec {
                name: "dec3",
                shape: [w2, w2 * 2, K, K],
            },
            ParamSpec {
                name: "up2",
                shape: up(w2, w1),
            },
            ParamSpec {
                name: "dec2",
                shape: [w1, w1 * 2, K, K],
            },
            ParamSpec {
                name: "up1",
                shape: up(w1, w0),
            },
            ParamSpec {
                name: "dec1",
                shape: [w0, w0 * 2, K, K],
            },
            ParamSpec {
                name: "out",
                shape: [arch.head.out_channels(), w0, K, K],
            },
        ];
        Self { arch, params }
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    pub fn widths(&self) -> Widths {
        self.arch.widths
    }

    /// Input planes this network takes.
    pub fn inputs(&self) -> usize {
        self.arch.inputs
    }

    pub fn params(&self) -> &[ParamSpec] {
        &self.params
    }

    /// Total trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.params.iter().map(ParamSpec::elems).sum()
    }

    /// Both spatial dimensions must be divisible by this, because the encoder
    /// halves the resolution three times and the skip connections have to line
    /// up on the way back.
    pub const TILE_MULTIPLE: usize = 8;

    /// Add the forward pass to `graph`, returning the denoised colour.
    ///
    /// `input` is `[n, 9, h, w]`; the result is `[n, 3, h, w]`. Parameters are
    /// declared on the graph under the names in [`Self::params`], so a caller
    /// binds them with `set_param` and differentiates against the same list.
    pub fn forward(
        &self,
        graph: &mut Graph,
        input: NodeId,
        n: usize,
        h: usize,
        w: usize,
    ) -> NodeId {
        assert!(
            h.is_multiple_of(Self::TILE_MULTIPLE) && w.is_multiple_of(Self::TILE_MULTIPLE),
            "denoise: {h}x{w} is not a multiple of {}; the encoder halves three times",
            Self::TILE_MULTIPLE
        );
        let _ = n;
        let p: Vec<NodeId> = self
            .params
            .iter()
            .map(|s| {
                let dims: Vec<usize> = s.shape.to_vec();
                graph.param(s.name, Shape::new(&dims, DType::F32))
            })
            .collect();

        let same = [K / 2, K / 2];
        let one = [1, 1];
        let two = [2, 2];
        let Widths {
            level0: w0,
            level1: w1,
            level2: w2,
            level3: w3,
        } = self.arch.widths;

        // Encoder. Each level halves the resolution and widens — as one
        // stride-2 convolution, or as a fixed average pool followed by a
        // stride-1 3x3.
        let down =
            |g: &mut Graph, x: NodeId, weight: NodeId, channels: usize| match self.arch.sampling {
                Sampling::Strided => conv_relu(g, x, weight, [K, K], two, same),
                Sampling::Pooled => {
                    let pooled = avg_pool2x(g, x, channels);
                    conv_relu(g, pooled, weight, [K, K], one, same)
                }
            };
        let enc0 = conv_relu(graph, input, p[0], [K, K], one, same);
        let enc1 = down(graph, enc0, p[1], w0);
        let enc2 = down(graph, enc1, p[2], w1);
        let enc3 = down(graph, enc2, p[3], w2);
        let bottleneck = conv_relu(graph, enc3, p[4], [K, K], one, same);

        // Decoder. Each stage doubles the resolution, concatenates the encoder
        // output at the matching resolution, and mixes the two with a 3x3.
        let up =
            |g: &mut Graph, x: NodeId, weight: NodeId, channels: usize| match self.arch.sampling {
                Sampling::Strided => {
                    let y = g.conv_transpose2d(x, weight, [2, 2], two, [0, 0], one, [0, 0], 1);
                    g.relu(y)
                }
                Sampling::Pooled => {
                    let doubled = nearest2x(g, x, channels);
                    conv_relu(g, doubled, weight, [K, K], one, same)
                }
            };
        let u3 = up(graph, bottleneck, p[5], w3);
        let d3 = graph.concat_(vec![u3, enc2], 1);
        let d3 = conv_relu(graph, d3, p[6], [K, K], one, same);

        let u2 = up(graph, d3, p[7], w2);
        let d2 = graph.concat_(vec![u2, enc1], 1);
        let d2 = conv_relu(graph, d2, p[8], [K, K], one, same);

        let u1 = up(graph, d2, p[9], w1);
        let d1 = graph.concat_(vec![u1, enc0], 1);
        let d1 = conv_relu(graph, d1, p[10], [K, K], one, same);

        let predicted = graph.conv2d(d1, p[11], [K, K], one, same, one, 1);
        let colour = graph.narrow_(input, 1, 0, OUT_CHANNELS);
        match self.arch.head {
            // Residual: the network corrects the colour it was given rather
            // than reproducing it. `out` starts at zero, so an untrained
            // network is the identity and training begins at the raw error.
            Head::Direct => graph.add(colour, predicted),
            Head::Kernel { radius } => {
                apply_predicted_kernel(graph, colour, predicted, radius, n, h, w)
            }
        }
    }
}

/// Fixed 2×2 average pool, as a depthwise convolution with constant weights.
///
/// A constant rather than a parameter, so it contributes no gradient and no
/// storage — the point of pooled resampling is that the resampling itself is
/// not learned.
fn avg_pool2x(graph: &mut Graph, x: NodeId, channels: usize) -> NodeId {
    let weight = graph.full(&[channels, 1, 2, 2], 0.25, DType::F32);
    graph.conv2d(x, weight, [2, 2], [2, 2], [0, 0], [1, 1], channels)
}

/// Fixed nearest-neighbour 2× upsample, as a depthwise transposed convolution
/// whose 2×2 kernel is all ones: every input pixel becomes a 2×2 block of
/// copies of itself.
fn nearest2x(graph: &mut Graph, x: NodeId, channels: usize) -> NodeId {
    let weight = graph.full(&[channels, 1, 2, 2], 1.0, DType::F32);
    graph.conv_transpose2d(x, weight, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], channels)
}

/// Softmax the predicted logits into a per-pixel filter and apply it to
/// `colour`.
///
/// `logits` is `[n, (2r+1)², h, w]`. The softmax runs over the tap axis, so
/// every pixel gets non-negative weights summing to one and the result is a
/// convex combination of its own neighbourhood.
///
/// A constant is added to the centre tap so that an untrained network — whose
/// `out` weights are zero, making every logit zero — starts as a near-delta
/// rather than as the uniform box blur a flat softmax would give. That keeps
/// the property the residual head has: training begins at the render's own
/// error.
fn apply_predicted_kernel(
    graph: &mut Graph,
    colour: NodeId,
    logits: NodeId,
    radius: usize,
    n: usize,
    h: usize,
    w: usize,
) -> NodeId {
    /// `exp(8) / (exp(8) + 24)` ≈ 0.992 of the weight on the centre tap.
    const CENTRE_LOGIT: f32 = 8.0;

    let side = 2 * radius + 1;
    let taps = side * side;
    let centre = taps / 2;

    let before = graph.full(&[n, centre, h, w], 0.0, DType::F32);
    let middle = graph.full(&[n, 1, h, w], CENTRE_LOGIT, DType::F32);
    let after = graph.full(&[n, taps - centre - 1, h, w], 0.0, DType::F32);
    let bias = graph.concat_(vec![before, middle, after], 1);
    let logits = graph.add(logits, bias);
    let weights = graph.sm(logits, 1);

    // Replicate rather than zero-pad: a zero border would drag the edge of the
    // image toward black, which is a visible frame on a dark render.
    let padded = graph.pad_(
        colour,
        vec![[0, 0], [0, 0], [radius, radius], [radius, radius]],
        PadMode::Replicate,
    );

    let mut sum: Option<NodeId> = None;
    for dy in 0..side {
        for dx in 0..side {
            let rows = graph.narrow_(padded, 2, dy, h);
            let shifted = graph.narrow_(rows, 3, dx, w);
            let tap = graph.narrow_(weights, 1, dy * side + dx, 1);
            // One weight plane drives all three colour channels: the filter is
            // a geometric statement about which neighbours belong to the same
            // surface, and that does not differ per channel.
            let tap = graph.concat_(vec![tap, tap, tap], 1);
            let term = graph.mul(shifted, tap);
            sum = Some(match sum {
                None => term,
                Some(acc) => graph.add(acc, term),
            });
        }
    }
    sum.expect("a kernel head has at least one tap")
}

fn conv_relu(
    graph: &mut Graph,
    x: NodeId,
    weight: NodeId,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> NodeId {
    let y = graph.conv2d(x, weight, kernel, stride, padding, [1, 1], 1);
    graph.relu(y)
}

/// Deterministic parameter initialisation.
///
/// He scaling — `sqrt(2 / fan_in)` — because the activations are ReLU, which
/// discards half the signal and so halves the variance at every layer. The
/// output convolution starts at exactly zero, which is what makes the untrained
/// network the identity.
///
/// The generator is a counter-based hash rather than a seeded stream, so a
/// given parameter gets the same values whatever order the layers are built in.
pub fn init_params(net: &DenoiseNet, seed: u64) -> Vec<Vec<f32>> {
    net.params()
        .iter()
        .enumerate()
        .map(|(layer, spec)| {
            if spec.name == "out" {
                return vec![0.0; spec.elems()];
            }
            let scale = (2.0 / spec.fan_in() as f32).sqrt();
            (0..spec.elems())
                .map(|i| {
                    let h = mix64(seed ^ ((layer as u64) << 40) ^ i as u64);
                    // Two uniforms into a triangular distribution: cheap, and
                    // closer to Gaussian than a single uniform is. Its variance
                    // is 1/6, so sqrt(6) is what brings the result to He's
                    // 2/fan_in — sqrt(3) would leave it at half.
                    let a = (h >> 40) as f32 / 16_777_216.0;
                    let b = ((h >> 16) & 0xff_ffff) as f32 / 16_777_216.0;
                    (a + b - 1.0) * scale * 2.449_5
                })
                .collect()
        })
        .collect()
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count_matches_the_declared_shapes() {
        let net = DenoiseNet::new(Widths::default());
        let by_hand: usize = net
            .params()
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert_eq!(net.parameter_count(), by_hand);
        // The default shape is a few hundred thousand parameters — small enough
        // to train on one GPU, large enough to hold a denoiser.
        assert!(
            (200_000..600_000).contains(&net.parameter_count()),
            "unexpected size: {}",
            net.parameter_count()
        );
    }

    #[test]
    fn skip_connections_line_up_with_their_encoder_stages() {
        let net = DenoiseNet::new(Widths::default());
        let w = net.widths();
        let find = |name: &str| {
            net.params()
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .shape
        };
        // Each decoder convolution consumes the upsampled tensor concatenated
        // with the encoder output at that resolution, so its input channel
        // count has to be exactly twice the level width.
        assert_eq!(find("dec3")[1], w.level2 * 2);
        assert_eq!(find("dec2")[1], w.level1 * 2);
        assert_eq!(find("dec1")[1], w.level0 * 2);
        // And each upsample has to produce the level width it is concatenated
        // against.
        assert_eq!(find("up3")[1], w.level2);
        assert_eq!(find("up2")[1], w.level1);
        assert_eq!(find("up1")[1], w.level0);
    }

    #[test]
    fn the_output_layer_starts_at_zero() {
        let net = DenoiseNet::new(Widths::tiny());
        let init = init_params(&net, 7);
        let out = init.last().expect("params");
        assert!(
            out.iter().all(|v| *v == 0.0),
            "the residual layer must start at zero so the network begins as the identity"
        );
        // Everything else must not be zero, or nothing can learn.
        assert!(init[0].iter().any(|v| *v != 0.0));
    }

    #[test]
    fn initialisation_is_deterministic_and_he_scaled() {
        let net = DenoiseNet::new(Widths::default());
        let a = init_params(&net, 11);
        let b = init_params(&net, 11);
        assert_eq!(a, b);
        assert_ne!(a[0], init_params(&net, 12)[0]);

        for (spec, values) in net.params().iter().zip(&a) {
            if spec.name == "out" {
                continue;
            }
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let var =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
            let want = 2.0 / spec.fan_in() as f32;
            assert!(
                (var / want).clamp(0.5, 2.0) == var / want,
                "{}: variance {var} is far from He's {want}",
                spec.name
            );
        }
    }
}
