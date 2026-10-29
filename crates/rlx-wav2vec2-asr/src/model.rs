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

//! torchaudio's `Wav2Vec2Model` with a CTC head, as the
//! `WAV2VEC2_ASR_LARGE_LV60K_960H` bundle builds it (the English align model
//! WhisperX uses): `extractor_mode="layer_norm"` (a LayerNorm after every
//! conv), a pre-LN (`layer_norm_first`) 24-layer encoder closed by the
//! transformer LayerNorm, and the 29-label `aux` head (the bundle drops
//! fairseq's rows 1..=3). The bundle also normalizes the waveform
//! (`F.layer_norm(x, x.shape)`) before the model.
//!
//! Weights are torchaudio's own `wav2vec2_fairseq_large_lv60k_asr_ls960.pth`,
//! read with `rlx-torch-ckpt` (no Python, no libtorch).

use anyhow::{Context, Result, bail};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_torch_ckpt::PtModel;
use std::collections::HashMap;
use std::path::Path;

const F32: DType = DType::F32;

/// `(out_channels, kernel, stride)` of the conv front end.
pub const CONV_LAYERS: [(usize, usize, usize); 7] = [
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
];
const HIDDEN: usize = 1024;
const HEADS: usize = 16;
const FFN: usize = 4096;
const LAYERS: usize = 24;
const POS_K: usize = 128;
const POS_GROUPS: usize = 16;
const EPS: f32 = 1e-5;

/// `WAV2VEC2_ASR_LARGE_LV60K_960H.get_labels()`.
pub const LABELS: [char; 29] = [
    '-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G',
    'Y', 'P', 'B', 'V', 'K', '\'', 'X', 'J', 'Q', 'Z',
];
/// fairseq dictionary rows the bundle removes from `aux` (`_remove_aux_axis`).
const REMOVE_AUX: [usize; 3] = [1, 2, 3];

/// Frames the conv front end emits for `samples` of 16 kHz audio.
pub fn frames_for(samples: usize) -> usize {
    let mut t = samples;
    for &(_, k, s) in &CONV_LAYERS {
        if t < k {
            return 0;
        }
        t = (t - k) / s + 1;
    }
    t
}

/// `F.layer_norm(x, x.shape)`: zero mean, unit (biased) variance, eps 1e-5.
pub fn normalize_waveform(x: &[f32]) -> Vec<f32> {
    let n = x.len().max(1) as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + EPS as f64).sqrt();
    x.iter()
        .map(|&v| ((v as f64 - mean) * inv) as f32)
        .collect()
}

pub struct Wav2Vec2Ctc {
    ckpt: PtModel,
    device: Device,
    /// Compiled graphs by input length (segments repeat lengths rarely, but a
    /// caller aligning one file twice should not recompile).
    cache: HashMap<usize, CompiledGraph>,
    /// The length whose graph owns the uploaded weights; every other
    /// length's graph shares them (`share_params_from`) — all parameters
    /// are length-independent.
    base: Option<usize>,
    /// Build graphs without reading weights (their values come from `params`).
    dry: std::cell::Cell<bool>,
    /// The prepared parameter values (transposed linears, fused pos-conv,
    /// trimmed aux head), read from the checkpoint once.
    params: std::cell::RefCell<Option<Vec<(String, Vec<f32>)>>>,
}

struct Ctx<'a, 'b> {
    g: &'a mut HirMut<'b>,
    params: Vec<(String, Vec<f32>)>,
}

impl Ctx<'_, '_> {
    fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        assert!(data.is_empty() || data.len() == shape.iter().product::<usize>());
        let name = format!("w{}", self.params.len());
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.push((name, data));
        id
    }

    /// `y = x Wᵀ + b` over the last axis of `x` (`[rows.., in]`).
    fn linear(
        &mut self,
        x: HirNodeId,
        w: &[f32],
        b: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> HirNodeId {
        let mut wt = vec![0f32; w.len()];
        if !w.is_empty() {
            for o in 0..out_dim {
                for i in 0..in_dim {
                    wt[i * out_dim + o] = w[o * in_dim + i];
                }
            }
        }
        let wp = self.param(wt, &[in_dim, out_dim]);
        let bp = self.param(b.to_vec(), &[1, out_dim]);
        let shape = self.g.shape(x).clone();
        let rank = shape.rank();
        let lead: Vec<i64> = (0..rank - 1)
            .map(|i| shape.dim(i).unwrap_static() as i64)
            .collect();
        let rows: i64 = lead.iter().product();
        let x2 = self.g.reshape_(x, vec![rows, in_dim as i64]);
        let y = self.g.mm(x2, wp);
        let be = self.g.expand_(bp, vec![rows, out_dim as i64]);
        let y = self.g.add(y, be);
        let mut out = lead;
        out.push(out_dim as i64);
        self.g.reshape_(y, out)
    }

    fn ln(&mut self, x: HirNodeId, w: Vec<f32>, b: Vec<f32>, d: usize) -> HirNodeId {
        let gp = self.param(w, &[d]);
        let bp = self.param(b, &[d]);
        self.g.ln(x, gp, bp, EPS)
    }
}

impl Wav2Vec2Ctc {
    /// Open torchaudio's checkpoint (`wav2vec2_fairseq_large_lv60k_asr_ls960.pth`).
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let ckpt = PtModel::open(path).with_context(|| format!("opening {}", path.display()))?;
        let aux = ckpt
            .shape_of("aux.weight")
            .context("checkpoint has no aux head")?;
        if aux != [32, HIDDEN] {
            bail!("aux head {aux:?}, expected [32, {HIDDEN}] (fairseq dictionary)");
        }
        Ok(Self {
            ckpt,
            device,
            cache: HashMap::new(),
            base: None,
            dry: std::cell::Cell::new(false),
            params: std::cell::RefCell::new(None),
        })
    }

    fn get(&self, name: &str) -> Result<Vec<f32>> {
        if self.dry.get() {
            return Ok(Vec::new());
        }
        Ok(self
            .ckpt
            .tensor(name)
            .with_context(|| format!("tensor {name}"))?
            .data)
    }

    /// `weight_g * weight_v / ||weight_v||` with the norm over every axis but
    /// 2 (`weight_norm(conv, dim=2)`), in double like torch's `_weight_norm`
    /// would not — the difference is far below f32 resolution of the result.
    fn pos_conv_weight(&self) -> Result<Vec<f32>> {
        if self.dry.get() {
            return Ok(Vec::new());
        }
        let g = self.get("encoder.transformer.pos_conv_embed.conv.weight_g")?;
        let v = self.get("encoder.transformer.pos_conv_embed.conv.weight_v")?;
        let (d0, d1, d2) = (HIDDEN, HIDDEN / POS_GROUPS, POS_K);
        let mut out = vec![0f32; v.len()];
        for j in 0..d2 {
            let mut sq = 0f64;
            for a in 0..d0 * d1 {
                sq += (v[a * d2 + j] as f64).powi(2);
            }
            let s = g[j] as f64 / sq.sqrt();
            for a in 0..d0 * d1 {
                out[a * d2 + j] = (v[a * d2 + j] as f64 * s) as f32;
            }
        }
        Ok(out)
    }

    fn compile(&self, samples: usize) -> Result<CompiledGraph> {
        let frames = frames_for(samples);
        if frames == 0 {
            bail!("{samples} samples is too short for the conv front end");
        }
        let mut hir = HirModule::new("wav2vec2_ctc");
        let mut g = HirMut::new(&mut hir);
        let mut c = Ctx {
            g: &mut g,
            params: Vec::new(),
        };
        let x = c.g.input("pcm", Shape::new(&[1, 1, samples, 1], F32));

        // Conv front end: conv -> LayerNorm over channels -> GELU.
        let (mut cur, mut c_in, mut t) = (x, 1usize, samples);
        for (i, &(c_out, k, s)) in CONV_LAYERS.iter().enumerate() {
            let p = format!("feature_extractor.conv_layers.{i}");
            let t_out = (t - k) / s + 1;
            let w = c.param(self.get(&format!("{p}.conv.weight"))?, &[c_out, c_in, k, 1]);
            cur = c.g.add_node(
                Op::Conv {
                    kernel_size: vec![k, 1],
                    stride: vec![s, 1],
                    padding: vec![0, 0],
                    dilation: vec![1, 1],
                    groups: 1,
                },
                vec![cur, w],
                Shape::new(&[1, c_out, t_out, 1], F32),
            );
            let b = c.param(self.get(&format!("{p}.conv.bias"))?, &[1, c_out, 1, 1]);
            let be = c.g.expand_(b, vec![1, c_out as i64, t_out as i64, 1]);
            cur = c.g.add(cur, be);
            let ntc = c.g.reshape_(cur, vec![1, c_out as i64, t_out as i64]);
            let ntc = c.g.transpose_(ntc, vec![0, 2, 1]);
            let ntc = c.ln(
                ntc,
                self.get(&format!("{p}.layer_norm.weight"))?,
                self.get(&format!("{p}.layer_norm.bias"))?,
                c_out,
            );
            let ntc = c.g.gelu(ntc);
            let nct = c.g.transpose_(ntc, vec![0, 2, 1]);
            cur = c.g.reshape_(nct, vec![1, c_out as i64, t_out as i64, 1]);
            c_in = c_out;
            t = t_out;
        }
        let feats = c.g.reshape_(cur, vec![1, c_in as i64, t as i64]);
        let feats = c.g.transpose_(feats, vec![0, 2, 1]); // [1, T, 512]

        // Feature projection.
        let fp = "encoder.feature_projection";
        let h = c.ln(
            feats,
            self.get(&format!("{fp}.layer_norm.weight"))?,
            self.get(&format!("{fp}.layer_norm.bias"))?,
            512,
        );
        let mut h = c.linear(
            h,
            &self.get(&format!("{fp}.projection.weight"))?,
            &self.get(&format!("{fp}.projection.bias"))?,
            HIDDEN,
            512,
        );

        // Convolutional positional embedding: pad K/2 both sides, grouped conv,
        // drop the trailing frame (even kernel), GELU; added residually.
        let pad = POS_K / 2;
        let hc = c.g.transpose_(h, vec![0, 2, 1]);
        let hc = c.g.reshape_(hc, vec![1, HIDDEN as i64, frames as i64, 1]);
        let zl = c.param(vec![0.0; HIDDEN * pad], &[1, HIDDEN, pad, 1]);
        let zr = c.param(vec![0.0; HIDDEN * pad], &[1, HIDDEN, pad, 1]);
        let padded = c.g.concat_(vec![zl, hc, zr], 2);
        let t_out = frames + 2 * pad - POS_K + 1;
        let wp = c.param(
            self.pos_conv_weight()?,
            &[HIDDEN, HIDDEN / POS_GROUPS, POS_K, 1],
        );
        let mut pos = c.g.add_node(
            Op::Conv {
                kernel_size: vec![POS_K, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: POS_GROUPS,
            },
            vec![padded, wp],
            Shape::new(&[1, HIDDEN, t_out, 1], F32),
        );
        let pb = c.param(
            self.get("encoder.transformer.pos_conv_embed.conv.bias")?,
            &[1, HIDDEN, 1, 1],
        );
        let pbe = c.g.expand_(pb, vec![1, HIDDEN as i64, t_out as i64, 1]);
        pos = c.g.add(pos, pbe);
        let pos = c.g.narrow_(pos, 2, 0, frames);
        let pos = c.g.gelu(pos);
        let pos = c.g.reshape_(pos, vec![1, HIDDEN as i64, frames as i64]);
        let pos = c.g.transpose_(pos, vec![0, 2, 1]);
        h = c.g.add(h, pos);

        // Pre-LN encoder layers.
        let hd = HIDDEN / HEADS;
        for l in 0..LAYERS {
            let p = format!("encoder.transformer.layers.{l}");
            let get = |n: &str| self.get(&format!("{p}.{n}"));
            let x = c.ln(
                h,
                get("layer_norm.weight")?,
                get("layer_norm.bias")?,
                HIDDEN,
            );
            let q = c.linear(
                x,
                &get("attention.q_proj.weight")?,
                &get("attention.q_proj.bias")?,
                HIDDEN,
                HIDDEN,
            );
            let k = c.linear(
                x,
                &get("attention.k_proj.weight")?,
                &get("attention.k_proj.bias")?,
                HIDDEN,
                HIDDEN,
            );
            let v = c.linear(
                x,
                &get("attention.v_proj.weight")?,
                &get("attention.v_proj.bias")?,
                HIDDEN,
                HIDDEN,
            );
            let shp = vec![1, frames as i64, HEADS as i64, hd as i64];
            let (q, k, v) = (
                c.g.reshape_(q, shp.clone()),
                c.g.reshape_(k, shp.clone()),
                c.g.reshape_(v, shp),
            );
            let a = c.g.attention_kind(
                q,
                k,
                v,
                HEADS,
                hd,
                rlx_ir::op::MaskKind::None,
                Shape::new(&[1, frames, HEADS, hd], F32),
            );
            let a = c.g.reshape_(a, vec![1, frames as i64, HIDDEN as i64]);
            let a = c.linear(
                a,
                &get("attention.out_proj.weight")?,
                &get("attention.out_proj.bias")?,
                HIDDEN,
                HIDDEN,
            );
            h = c.g.add(h, a);
            let x = c.ln(
                h,
                get("final_layer_norm.weight")?,
                get("final_layer_norm.bias")?,
                HIDDEN,
            );
            let f = c.linear(
                x,
                &get("feed_forward.intermediate_dense.weight")?,
                &get("feed_forward.intermediate_dense.bias")?,
                FFN,
                HIDDEN,
            );
            let f = c.g.gelu(f);
            let f = c.linear(
                f,
                &get("feed_forward.output_dense.weight")?,
                &get("feed_forward.output_dense.bias")?,
                HIDDEN,
                FFN,
            );
            h = c.g.add(h, f);
        }
        let h = c.ln(
            h,
            self.get("encoder.transformer.layer_norm.weight")?,
            self.get("encoder.transformer.layer_norm.bias")?,
            HIDDEN,
        );

        // aux head without fairseq's <pad>/</s>/<unk> rows.
        let (aw, ab) = (self.get("aux.weight")?, self.get("aux.bias")?);
        let keep: Vec<usize> = (0..32).filter(|r| !REMOVE_AUX.contains(r)).collect();
        let (w, b): (Vec<f32>, Vec<f32>) = if aw.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (
                keep.iter()
                    .flat_map(|&r| aw[r * HIDDEN..(r + 1) * HIDDEN].iter().copied())
                    .collect(),
                keep.iter().map(|&r| ab[r]).collect(),
            )
        };
        let logits = c.linear(h, &w, &b, LABELS.len(), HIDDEN);
        let logits =
            c.g.reshape_(logits, vec![frames as i64, LABELS.len() as i64]);

        let params = std::mem::take(&mut c.params);
        hir.set_outputs(vec![logits]);
        let graph = Graph::from_hir(hir)
            .map_err(|e| anyhow::anyhow!("wav2vec2: HIR lowering failed: {e}"))?;
        let tc = std::time::Instant::now();
        let mut compiled = Session::new(self.device).compile(graph);
        if std::env::var_os("W2V_DEBUG").is_some() {
            eprintln!(
                "wav2vec2: compile {:.2}s (dry {})",
                tc.elapsed().as_secs_f64(),
                self.dry.get()
            );
        }
        let tp = std::time::Instant::now();
        if self.dry.get() {
            let cached = self.params.borrow();
            let cached = cached
                .as_ref()
                .context("wav2vec2: dry build without cached parameters")?;
            for (name, data) in cached {
                compiled.set_param(name, data);
            }
        } else {
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            *self.params.borrow_mut() = Some(params);
        }
        compiled.finalize_params();
        if std::env::var_os("W2V_DEBUG").is_some() {
            eprintln!(
                "wav2vec2: set_param+finalize {:.2}s",
                tp.elapsed().as_secs_f64()
            );
        }
        Ok(compiled)
    }

    /// `log_softmax(model(normalize(pcm)))`: `[frames, 29]` log-probabilities
    /// for 16 kHz mono `pcm`, as WhisperX computes its emission matrix.
    /// Inputs shorter than 400 samples are zero-padded to 400 (WhisperX also
    /// passes `lengths`, which only masks the padded tail).
    pub fn emissions(&mut self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mut x = pcm.to_vec();
        if x.len() < 400 {
            x.resize(400, 0.0);
        }
        let x = normalize_waveform(&x);
        let n = x.len();
        if !self.cache.contains_key(&n) {
            // Every length after the first shares the first graph's weights;
            // fall back to a full build if the backend cannot share.
            self.dry.set(self.base.is_some());
            let tb = std::time::Instant::now();
            let g = match self.compile(n) {
                Ok(g) => g,
                Err(e) if self.dry.get() => {
                    if std::env::var_os("W2V_DEBUG").is_some() {
                        eprintln!("wav2vec2: weight sharing unavailable ({e}); full build");
                    }
                    self.dry.set(false);
                    self.compile(n)?
                }
                Err(e) => return Err(e),
            };
            self.dry.set(false);
            if std::env::var_os("W2V_DEBUG").is_some() {
                eprintln!("wav2vec2: build total {:.2}s", tb.elapsed().as_secs_f64());
            }
            self.cache.insert(n, g);
            self.base.get_or_insert(n);
        }
        let out = self
            .cache
            .get_mut(&n)
            .unwrap()
            .run(&[("pcm", &x)])
            .into_iter()
            .next()
            .context("no output")?;
        let frames = frames_for(n);
        let v = LABELS.len();
        let mut lp = out;
        for row in lp.chunks_mut(v) {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = row.iter().map(|&z| (z - m).exp()).sum();
            let ls = s.ln() + m;
            for z in row.iter_mut() {
                *z -= ls;
            }
        }
        Ok((lp, frames))
    }
}
