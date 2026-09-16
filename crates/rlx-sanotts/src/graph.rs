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

//! rlx-ir graph path: the frame-rate acoustic stack and the whole `piperlite`
//! decoder as one HIR graph, compiled per device so it runs on every RLX backend.
//!
//! Backend-portability choices, same as the other conv vocoders in this repo:
//! - 1-D convs are NCHW convs with time in the H axis (`[N, C, L, 1]`, kernel `[K, 1]`).
//! - `ConvTranspose1d` is rewritten as zero-insertion + a regular conv, because
//!   several backends have no native transposed-conv kernel. The zero insertion
//!   itself is a `Pad` on the trailing axis plus a reshape, so nothing
//!   materializes a zeros tensor proportional to the utterance length.
//! - `leaky_relu(x, a)` is `relu(x) - a * relu(-x)`, which is bit-exact on both
//!   sides of zero rather than the algebraically-equal-but-lossy `a*x + (1-a)*relu(x)`.
//!
//! # Length bucketing
//!
//! rlx-ir shapes are static, so a graph is only valid for the length it was
//! built for. Rather than compile one per utterance, graphs are built at a
//! bucketed *capacity* and the input is zero-padded up to it.
//!
//! Zero-padding alone would not be sound: the first conv's bias turns the padded
//! tail into a nonzero constant, and each later conv drags that contamination
//! left by its receptive field until it reaches real frames. So every conv's
//! bias add is followed by a multiply with a validity mask, which pins the tail
//! back to exactly zero. Every other op in the graph maps 0 to 0 —
//! `leaky_relu`, `silu`, `tanh`, scalar multiplies, residual adds of masked
//! tensors, and the transposed conv's zero-insertion — so masking the bias adds
//! is sufficient.
//!
//! What survives is reduction-order noise only: the padded and unpadded graphs
//! convolve over different total lengths, so a backend may tile them
//! differently. Measured on CPU the latent is bit-identical, the waveform
//! differs by ~1e-7, and the region next to the padding — where a leaking mask
//! would show up first — is exactly zero. `bucketing_is_exact` gates that.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::PadMode;
use rlx_ir::{DType, HirGraphExt, HirModule, Op, Shape};
use rlx_runtime::{AotCache, CompileOptions, Device};

use crate::config::{AcousticConfig, DecoderConfig};
use crate::model::{STAGES, check_decoder};
use crate::ops::Mat;
use crate::voicepack::TensorStore;

/// Residual-bank dilations, mirroring [`crate::model`].
const BANK_DIL1: [usize; 3] = [1, 2, 3];
const BANK_DIL2: [usize; 3] = [2, 6, 12];

/// Samples per frame at each resolution the graph works at: frame rate, then
/// after each of the three upsampling stages (8, then 8, then 4).
const LEVEL_FACTOR: [usize; 4] = [1, 8, 64, 256];

/// Bump on any structural change so the AOT cache never serves stale LIR.
const GRAPH_VERSION: &str = "v2";

/// Default bucket granularity in frames. 32 frames is ~0.37 s of audio at the
/// 22.05 kHz voices, so the padding overhead is small while the number of
/// distinct graphs stays tiny.
pub const DEFAULT_BUCKET_FRAMES: usize = 32;

/// Bucket granularity, overridable with `RLX_SANOTTS_BUCKET` (0 or 1 disables
/// bucketing and compiles an exact-length graph per utterance).
pub fn bucket_frames() -> usize {
    std::env::var("RLX_SANOTTS_BUCKET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BUCKET_FRAMES)
        .max(1)
}

/// Round `frames` up to the next bucket boundary.
pub fn bucket_capacity(frames: usize) -> usize {
    let b = bucket_frames();
    frames.div_ceil(b) * b
}

struct Builder<'a> {
    ac: &'a TensorStore,
    ac_cfg: &'a AcousticConfig,
    de: &'a TensorStore,
    de_cfg: &'a DecoderConfig,
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
    /// Validity mask per resolution level, `[1, 1, capacity * factor, 1]`.
    masks: Vec<HirNodeId>,
    /// Padded length at each resolution level.
    lens: Vec<usize>,
}

impl Builder<'_> {
    fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    fn param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>(), "param {name}");
        let id = HirMut::new(&mut self.hir).param(name, Shape::new(shape, DType::F32));
        self.params.insert(name.to_string(), data);
        id
    }

    /// A broadcastable scalar constant.
    fn scalar(&mut self, name: &str, v: f32) -> HirNodeId {
        self.param(name, vec![v], &[1, 1, 1, 1])
    }

    /// Conv weight `[c_out, c_in, k]` (or `[c_out, c_in]` for a 1×1 stored as a
    /// linear) → NCHW `[c_out, c_in, k, 1]`. Returns `(node, c_out, k)`.
    fn conv_w(&mut self, store: Store, key: &str) -> Result<(HirNodeId, usize, usize)> {
        let ts = self.store(store);
        let (data, sh) = ts.get(key)?;
        let (data, sh) = (data.clone(), sh.clone());
        let (c_out, c_in, k) = match sh.len() {
            3 => (sh[0], sh[1], sh[2]),
            2 => (sh[0], sh[1], 1),
            _ => bail!("{key}: unexpected conv weight rank {:?}", sh),
        };
        let id = self.param(
            &format!("{}{key}.w", store.prefix()),
            data,
            &[c_out, c_in, k, 1],
        );
        Ok((id, c_out, k))
    }

    fn conv_b(&mut self, store: Store, key: &str, c_out: usize) -> Result<HirNodeId> {
        let data = self.store(store).data(key)?.to_vec();
        if data.len() != c_out {
            bail!("{key}: bias has {} entries, expected {c_out}", data.len());
        }
        Ok(self.param(
            &format!("{}{key}.b", store.prefix()),
            data,
            &[1, c_out, 1, 1],
        ))
    }

    fn store(&self, which: Store) -> &TensorStore {
        match which {
            Store::Acoustic => self.ac,
            Store::Decoder => self.de,
        }
    }

    /// Add the bias, then pin the padded tail back to zero.
    ///
    /// This is the only place the tail can become nonzero, so it is the only
    /// place that needs a mask — see the module docs.
    fn bias_and_mask(&mut self, y: HirNodeId, b: HirNodeId, level: usize) -> HirNodeId {
        let biased = self.m().add(y, b);
        let mask = self.masks[level];
        self.m().mul(biased, mask)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dil: usize,
        t_in: usize,
    ) -> (HirNodeId, usize) {
        let t_out = (t_in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        let out = Shape::new(&[1, c_out, t_out, 1], DType::F32);
        let id = self.m().add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![pad, 0],
                dilation: vec![dil, 1],
                groups: 1,
            },
            vec![x, w],
            out,
        );
        (id, t_out)
    }

    /// Same-padded conv + bias + mask, reading kernel size from the stored weight.
    fn conv_same(
        &mut self,
        x: HirNodeId,
        store: Store,
        key: &str,
        dil: usize,
        level: usize,
    ) -> Result<(HirNodeId, usize)> {
        let t = self.lens[level];
        let (w, c_out, k) = self.conv_w(store, &format!("{key}.weight"))?;
        let (y, t_out) = self.conv(x, w, c_out, k, 1, dil * (k / 2), dil, t);
        debug_assert_eq!(t_out, t);
        let b = self.conv_b(store, &format!("{key}.bias"), c_out)?;
        Ok((self.bias_and_mask(y, b, level), c_out))
    }

    /// `relu(x) - slope * relu(-x)`.
    fn leaky(&mut self, x: HirNodeId, slope: f32, tag: &str) -> HirNodeId {
        let pos = self.m().relu(x);
        let neg_in = self.m().neg(x);
        let neg = self.m().relu(neg_in);
        let s = self.scalar(&format!("__slope.{tag}"), slope);
        let scaled = self.m().mul(neg, s);
        self.m().sub(pos, scaled)
    }

    /// `x + scale * conv2(silu(conv1(x)))`.
    fn residual_conv_block(
        &mut self,
        x: HirNodeId,
        store: Store,
        prefix: &str,
        level: usize,
    ) -> Result<HirNodeId> {
        let (h, _) = self.conv_same(x, store, &format!("{prefix}.net.0"), 1, level)?;
        let h = self.m().silu(h);
        let (u, _) = self.conv_same(h, store, &format!("{prefix}.net.2"), 1, level)?;
        let scale = self.store(store).scalar(&format!("{prefix}.scale"))?;
        let s = self.scalar(&format!("{}{prefix}.scale", store.prefix()), scale);
        let su = self.m().mul(u, s);
        Ok(self.m().add(x, su))
    }

    /// `ConvTranspose1d(x, W, stride, pad)` as `Conv1d(zero_insert(x, stride), W')`
    /// with `W'[oc, ic, kk] = W[ic, oc, K-1-kk]` and `pad' = K - 1 - pad`.
    fn conv_transpose(
        &mut self,
        x: HirNodeId,
        key: &str,
        k: usize,
        stride: usize,
        pad: usize,
        in_level: usize,
    ) -> Result<(HirNodeId, usize)> {
        let t_in = self.lens[in_level];
        let out_level = in_level + 1;
        let (raw, sh) = self.de.get(&format!("{key}.weight"))?;
        let (c_in, c_out) = (sh[0], sh[1]);
        if sh[2] != k {
            bail!("{key}: kernel {} != expected {k}", sh[2]);
        }
        let mut wrev = vec![0f32; c_out * c_in * k];
        for ic in 0..c_in {
            for oc in 0..c_out {
                for kk in 0..k {
                    wrev[(oc * c_in + ic) * k + kk] = raw[(ic * c_out + oc) * k + (k - 1 - kk)];
                }
            }
        }
        let wid = self.param(&format!("de.{key}.w.conv"), wrev, &[c_out, c_in, k, 1]);

        // Zero-insert: [1,c,T,1] --pad W by (stride-1)--> [1,c,T,stride]
        // --reshape--> [1,c,T*stride,1] --narrow--> (T-1)*stride+1.
        let l_up = (t_in - 1) * stride + 1;
        let up = if stride > 1 {
            let mut m = self.m();
            let padded = m.pad_(
                x,
                vec![[0, 0], [0, 0], [0, 0], [0, stride - 1]],
                PadMode::Constant(0.0),
            );
            let flat = m.reshape_(padded, vec![1, c_in as i64, (t_in * stride) as i64, 1]);
            m.narrow_(flat, 2, 0, l_up)
        } else {
            x
        };
        let (y, t_out) = self.conv(up, wid, c_out, k, 1, k - 1 - pad, 1, l_up);
        if t_out != self.lens[out_level] {
            bail!(
                "{key}: transposed conv produced {t_out} steps, level {out_level} expects {}",
                self.lens[out_level]
            );
        }
        let b = self.conv_b(Store::Decoder, &format!("{key}.bias"), c_out)?;
        Ok((self.bias_and_mask(y, b, out_level), c_out))
    }

    /// `PiperResidualBank`: mean over active branches of a two-conv residual pair.
    fn residual_bank(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        branches: &[usize],
        level: usize,
    ) -> Result<HirNodeId> {
        if branches.is_empty() {
            bail!("{prefix}: empty residual-bank branch list");
        }
        let mut acc: Option<HirNodeId> = None;
        for &branch in branches {
            if branch >= BANK_DIL1.len() {
                bail!("{prefix}: branch index {branch} out of range");
            }
            let tag = format!("{prefix}.{branch}");
            let a = self.leaky(x, 0.1, &format!("{tag}.a"));
            let (u, _) = self.conv_same(
                a,
                Store::Decoder,
                &format!("{prefix}.blocks.{branch}.conv1"),
                BANK_DIL1[branch],
                level,
            )?;
            let y1 = self.m().add(u, x);
            let b = self.leaky(y1, 0.1, &format!("{tag}.b"));
            let (u2, _) = self.conv_same(
                b,
                Store::Decoder,
                &format!("{prefix}.blocks.{branch}.conv2"),
                BANK_DIL2[branch],
                level,
            )?;
            let y2 = self.m().add(u2, y1);
            acc = Some(match acc {
                None => y2,
                Some(a) => self.m().add(a, y2),
            });
        }
        let summed = acc.expect("non-empty branches");
        let inv = self.scalar(&format!("__inv.{prefix}"), 1.0 / branches.len() as f32);
        Ok(self.m().mul(summed, inv))
    }

    /// `tanh(audio + scale * out_conv(residual stack over in_conv(audio)))`.
    fn post_filter(&mut self, audio: HirNodeId) -> Result<HirNodeId> {
        let level = 3;
        let (mut r, _) = self.conv_same(audio, Store::Decoder, "post_filter.in_conv", 1, level)?;
        for layer in 0..self.de_cfg.post_filter_layers {
            let scale = self
                .de
                .scalar(&format!("post_filter.units.{layer}.scale"))?;
            let a = self.leaky(r, 0.1, &format!("pf{layer}.a"));
            let (u, _) = self.conv_same(
                a,
                Store::Decoder,
                &format!("post_filter.units.{layer}.conv1"),
                1 + layer,
                level,
            )?;
            let b = self.leaky(u, 0.1, &format!("pf{layer}.b"));
            let (u2, _) = self.conv_same(
                b,
                Store::Decoder,
                &format!("post_filter.units.{layer}.conv2"),
                1,
                level,
            )?;
            let s = self.scalar(&format!("__pf.scale.{layer}"), scale);
            let su = self.m().mul(u2, s);
            r = self.m().add(r, su);
        }
        let (out, _) = self.conv_same(r, Store::Decoder, "post_filter.out_conv", 1, level)?;
        let s = self.scalar("__pf.mix", self.de_cfg.post_filter_scale);
        let mixed = self.m().mul(out, s);
        let summed = self.m().add(audio, mixed);
        Ok(self.m().tanh(summed))
    }

    /// Returns `(hir, params, latent_channels)`.
    fn build(mut self, capacity: usize) -> Result<(HirModule, HashMap<String, Vec<f32>>, usize)> {
        let in_ch = self.ac_cfg.hidden + 3;
        self.lens = LEVEL_FACTOR.iter().map(|f| capacity * f).collect();

        let x = HirMut::new(&mut self.hir).input(
            "frame_input",
            Shape::new(&[1, in_ch, capacity, 1], DType::F32),
        );
        // One mask per resolution. Kept as inputs (not params) so a single
        // compiled graph serves every true length that fits in the bucket.
        self.masks = (0..LEVEL_FACTOR.len())
            .map(|level| {
                let len = self.lens[level];
                HirMut::new(&mut self.hir).input(
                    format!("mask{level}"),
                    Shape::new(&[1, 1, len, 1], DType::F32),
                )
            })
            .collect();

        // -- acoustic, frame half --
        let (mut h, _) = self.conv_same(x, Store::Acoustic, "frame_input_proj", 1, 0)?;
        for i in 0..self.ac_cfg.depth {
            h = self.residual_conv_block(h, Store::Acoustic, &format!("frame_blocks.{i}"), 0)?;
        }
        let (latent, latent_ch) = self.conv_same(h, Store::Acoustic, "output", 1, 0)?;
        if latent_ch != self.ac_cfg.out_channels {
            bail!(
                "acoustic output has {latent_ch} channels, config says {}",
                self.ac_cfg.out_channels
            );
        }

        // -- decoder --
        let (mut y, _) = self.conv_same(latent, Store::Decoder, "pre", 1, 0)?;
        for (stage, &(k, stride, pad, up_name, bank_prefix)) in STAGES.iter().enumerate() {
            let a = self.leaky(y, 0.1, &format!("up{stage}"));
            let (up, c_out) = self.conv_transpose(a, up_name, k, stride, pad, stage)?;
            let want = self.de_cfg.channels[stage + 1];
            if c_out != want {
                bail!("{up_name} produced {c_out} channels, config says {want}");
            }
            y = self.residual_bank(
                up,
                bank_prefix,
                &self.de_cfg.stage_branches(stage),
                stage + 1,
            )?;
        }

        let a = self.leaky(y, 0.01, "post");
        let (audio, _) = self.conv_same(a, Store::Decoder, "post", 1, 3)?;
        let mut audio = self.m().tanh(audio);
        if self.de_cfg.post_filter_channels > 0 {
            audio = self.post_filter(audio)?;
        }

        HirMut::new(&mut self.hir).set_outputs(vec![latent, audio]);
        Ok((self.hir, self.params, latent_ch))
    }
}

#[derive(Clone, Copy)]
enum Store {
    Acoustic,
    Decoder,
}

impl Store {
    fn prefix(self) -> &'static str {
        match self {
            Store::Acoustic => "ac.",
            Store::Decoder => "de.",
        }
    }
}

fn aot_root() -> PathBuf {
    std::env::var("RLX_SANOTTS_AOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("rlx-sanotts-aot"))
}

/// The frame-rate acoustic stack plus the decoder, compiled for one bucketed
/// frame capacity.
///
/// Accepts any true length up to [`FrameGraph::capacity`]; shorter inputs are
/// zero-padded and masked, which the module docs argue — and the tests check —
/// is identical to a graph built for that exact length.
pub struct FrameGraph {
    compiled: rlx_runtime::CompiledGraph,
    device: Device,
    capacity: usize,
    in_ch: usize,
    latent_ch: usize,
}

impl FrameGraph {
    /// Build and compile for up to `capacity` acoustic frames on `device`.
    pub fn compile(
        ac: &TensorStore,
        ac_cfg: &AcousticConfig,
        de: &TensorStore,
        de_cfg: &DecoderConfig,
        capacity: usize,
        device: Device,
    ) -> Result<Self> {
        if !rlx_runtime::is_available(device) {
            bail!("device {device:?} is not available in this build");
        }
        check_decoder(de_cfg)?;
        if capacity == 0 {
            bail!("frame graph needs at least one frame");
        }
        let in_ch = ac_cfg.hidden + 3;

        let builder = Builder {
            ac,
            ac_cfg,
            de,
            de_cfg,
            hir: HirModule::new("sanotts_frames"),
            params: HashMap::new(),
            masks: Vec::new(),
            lens: Vec::new(),
        };
        let (hir, params, latent_ch) = builder.build(capacity)?;

        let pf = de_cfg.post_filter_channels;
        let key = format!("sanotts_frames_{GRAPH_VERSION}_{device:?}_c{in_ch}_t{capacity}_pf{pf}");
        let cache = AotCache::new(aot_root());
        let mut compiled = cache
            .compile_hir_cached(&key, device, hir, &CompileOptions::default())
            .map_err(|e| anyhow!("compile sanotts frame graph: {e}"))?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Ok(Self {
            compiled,
            device,
            capacity,
            in_ch,
            latent_ch,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Largest frame count this graph accepts.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// `[hidden + 3, frames]` → `(latent [out_channels, frames], audio [frames * 256])`.
    pub fn forward(&mut self, input: &Mat) -> Result<(Mat, Vec<f32>)> {
        let frames = input.cols;
        if input.rows != self.in_ch {
            bail!(
                "frame input has {} channels, graph was compiled for {}",
                input.rows,
                self.in_ch
            );
        }
        if frames == 0 || frames > self.capacity {
            bail!(
                "frame input has {frames} frames, graph capacity is {}",
                self.capacity
            );
        }
        let cap = self.capacity;

        // Zero-pad the input up to capacity, channel-major.
        let padded: Vec<f32> = if frames == cap {
            input.data.clone()
        } else {
            let mut p = vec![0.0f32; self.in_ch * cap];
            for c in 0..self.in_ch {
                p[c * cap..c * cap + frames].copy_from_slice(input.row(c));
            }
            p
        };

        let masks: Vec<Vec<f32>> = LEVEL_FACTOR
            .iter()
            .map(|&f| {
                let mut m = vec![0.0f32; cap * f];
                m[..frames * f].fill(1.0);
                m
            })
            .collect();

        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(1 + masks.len());
        inputs.push(("frame_input", &padded));
        for (level, m) in masks.iter().enumerate() {
            inputs.push((MASK_NAMES[level], m));
        }
        let mut outs = self.compiled.run(&inputs);
        if outs.len() < 2 {
            bail!("frame graph returned {} outputs, expected 2", outs.len());
        }
        let audio = outs.pop().expect("2 outputs");
        let latent = outs.pop().expect("2 outputs");
        if latent.len() != self.latent_ch * cap {
            bail!(
                "latent has {} values, expected {}",
                latent.len(),
                self.latent_ch * cap
            );
        }
        let samples = cap * crate::HOP;
        if audio.len() != samples {
            bail!("audio has {} samples, expected {samples}", audio.len());
        }

        // Crop the padded tail back off.
        let mut latent_out = Mat::zeros(self.latent_ch, frames);
        for c in 0..self.latent_ch {
            latent_out
                .row_mut(c)
                .copy_from_slice(&latent[c * cap..c * cap + frames]);
        }
        let mut audio_out = audio;
        audio_out.truncate(frames * crate::HOP);
        Ok((latent_out, audio_out))
    }
}

/// Input names for the per-level validity masks — `&'static str` so
/// [`FrameGraph::forward`] can borrow them without allocating per call.
const MASK_NAMES: [&str; 4] = ["mask0", "mask1", "mask2", "mask3"];
