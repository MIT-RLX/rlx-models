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

//! WhisperX's default voice activity detection: the pyannote segmentation
//! model it bundles (`whisperx/assets/pytorch_model.bin`, PyanNet: SincNet,
//! a 4-layer BiLSTM, two linear layers and a 3-speaker sigmoid head), run
//! as pyannote.audio 4.0.7's `Inference` does — 5 s windows every 0.5 s, the
//! last one zero-padded, `max` over speakers, Hamming-weighted overlap-add
//! onto the receptive-field frame grid, loose crop to the audio — then
//! WhisperX's own `Binarize` (hysteresis, onset 0.5 / offset 0.363, min-cut
//! at 30 s) and `merge_chunks` into ≤ 30 s chunks.
//!
//! The network is one rlx graph (all windows as one batch); aggregation and
//! binarization run on the host with numpy's and Python's precisions.

use anyhow::{Context, Result};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, BinaryOp, Op, ReduceOp};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{Device, Session};
use rlx_torch_ckpt::PtModel;
use std::path::Path;

const F32: DType = DType::F32;
pub const SAMPLE_RATE: usize = 16_000;
/// `Inference`: 5 s windows (`get_num_samples(5.0)`), `round(0.5 * 16000)` step.
pub const WINDOW: usize = 80_000;
pub const STEP: usize = 8_000;
const CHUNK_DURATION: f64 = 5.0;
const CHUNK_STEP: f64 = 0.5;
/// Model receptive field (`Model.receptive_field`): 991 samples every 270.
const RF_DURATION: f64 = 991.0 / 16000.0;
const RF_STEP: f64 = 270.0 / 16000.0;
const FRAMES_PER_CHUNK: usize = 293;
const HIDDEN: usize = 128;

pub struct PyannoteSegmentation {
    ckpt: PtModel,
    device: Device,
    /// Compiled networks by batch size (32, and the last partial batch).
    graphs: std::cell::RefCell<std::collections::HashMap<usize, rlx_runtime::CompiledGraph>>,
}

/// A speech region in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub start: f64,
    pub end: f64,
}

struct Ctx<'a, 'b> {
    g: &'a mut HirMut<'b>,
    params: Vec<(String, Vec<f32>)>,
}

impl Ctx<'_, '_> {
    fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        let name = format!("p{}", self.params.len());
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.push((name, data));
        id
    }
    fn leaky(&mut self, x: HirNodeId, dims: &[usize]) -> HirNodeId {
        let k = self.param(vec![0.01], &[1; 4][..dims.len()]);
        let ke = self.g.expand_(k, dims.iter().map(|&d| d as i64).collect());
        let kx = self.g.mul(x, ke);
        self.g.add_node(
            Op::Binary(BinaryOp::Max),
            vec![x, kx],
            Shape::new(dims, F32),
        )
    }
}

impl PyannoteSegmentation {
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let ckpt = PtModel::open(path).with_context(|| format!("opening {}", path.display()))?;
        Ok(Self {
            ckpt,
            device,
            graphs: Default::default(),
        })
    }

    fn get(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self
            .ckpt
            .tensor(&format!("state_dict.{name}"))
            .with_context(|| format!("tensor {name}"))?
            .data)
    }

    /// asteroid `ParamSincFB(80, 251, min_low_hz=50, min_band_hz=50).filters()`.
    fn sinc_filters(&self) -> Result<Vec<f32>> {
        let low_hz = self.get("sincnet.conv1d.0.filterbank.low_hz_")?;
        let band_hz = self.get("sincnet.conv1d.0.filterbank.band_hz_")?;
        let window = self.get("sincnet.conv1d.0.filterbank.window_")?;
        let n = self.get("sincnet.conv1d.0.filterbank.n_")?;
        let (half, k) = (n.len(), 2 * n.len() + 1);
        let cut = low_hz.len();
        let mut out = vec![0f32; 2 * cut * k];
        for f in 0..cut {
            let low = 50.0f32 + low_hz[f].abs();
            let high = (low + 50.0 + band_hz[f].abs()).clamp(50.0, 8000.0);
            let band = high - low;
            for (kind, base) in [(0, f * k), (1, (cut + f) * k)] {
                let mut left = vec![0f32; half];
                for i in 0..half {
                    let (fl, fh) = (low * n[i], high * n[i]);
                    let num = if kind == 0 {
                        fh.sin() - fl.sin()
                    } else {
                        fl.cos() - fh.cos()
                    };
                    left[i] = (num / (n[i] / 2.0)) * window[i];
                }
                let center = if kind == 0 { 2.0 * band } else { 0.0 };
                for i in 0..half {
                    out[base + i] = left[i] / (2.0 * band);
                    let r = if kind == 0 {
                        left[half - 1 - i]
                    } else {
                        -left[half - 1 - i]
                    };
                    out[base + half + 1 + i] = r / (2.0 * band);
                }
                out[base + half] = center / (2.0 * band);
            }
        }
        Ok(out)
    }

    /// Raw `[n, 293, 3]` speaker activations for `n` 5 s windows.
    pub fn infer(&self, chunks: &[f32], n: usize) -> Result<Vec<f32>> {
        let mut graphs = self.graphs.borrow_mut();
        if !graphs.contains_key(&n) {
            let g = self.build(n)?;
            graphs.insert(n, g);
        }
        graphs
            .get_mut(&n)
            .unwrap()
            .run(&[("pcm", chunks)])
            .into_iter()
            .next()
            .context("no output")
    }

    fn build(&self, n: usize) -> Result<rlx_runtime::CompiledGraph> {
        let mut hir = HirModule::new("pyannote_segmentation");
        let mut g = HirMut::new(&mut hir);
        let mut c = Ctx {
            g: &mut g,
            params: Vec::new(),
        };
        let x = c.g.input("pcm", Shape::new(&[n, 1, WINDOW, 1], F32));
        let gw = c.param(self.get("sincnet.wav_norm1d.weight")?, &[1]);
        let gb = c.param(self.get("sincnet.wav_norm1d.bias")?, &[1]);
        let mut h = c.g.group_norm(x, gw, gb, 1, 1e-5);
        // (channels, kernel, stride) per block; the sinc block's conv has no bias.
        let mut len = WINDOW;
        let blocks = [
            (80usize, 251usize, 10usize, 1usize),
            (60, 5, 1, 80),
            (60, 5, 1, 60),
        ];
        for (i, &(co, k, s, ci)) in blocks.iter().enumerate() {
            let w = if i == 0 {
                self.sinc_filters()?
            } else {
                self.get(&format!("sincnet.conv1d.{i}.weight"))?
            };
            let wp = c.param(w, &[co, ci, k, 1]);
            let lo = (len - k) / s + 1;
            h = c.g.add_node(
                Op::Conv {
                    kernel_size: vec![k, 1],
                    stride: vec![s, 1],
                    padding: vec![0, 0],
                    dilation: vec![1, 1],
                    groups: 1,
                },
                vec![h, wp],
                Shape::new(&[n, co, lo, 1], F32),
            );
            if i == 0 {
                h = c.g.add_node(
                    Op::Activation(Activation::Abs),
                    vec![h],
                    Shape::new(&[n, co, lo, 1], F32),
                );
            } else {
                let b = c.param(
                    self.get(&format!("sincnet.conv1d.{i}.bias"))?,
                    &[1, co, 1, 1],
                );
                let be = c.g.expand_(b, vec![n as i64, co as i64, lo as i64, 1]);
                h = c.g.add(h, be);
            }
            let lp = lo / 3;
            h = c.g.add_node(
                Op::Pool {
                    kind: ReduceOp::Max,
                    kernel_size: vec![3, 1],
                    stride: vec![3, 1],
                    padding: vec![0, 0],
                },
                vec![h],
                Shape::new(&[n, co, lp, 1], F32),
            );
            let nw = c.param(self.get(&format!("sincnet.norm1d.{i}.weight"))?, &[co]);
            let nb = c.param(self.get(&format!("sincnet.norm1d.{i}.bias"))?, &[co]);
            h = c.g.group_norm(h, nw, nb, co, 1e-5);
            h = c.leaky(h, &[n, co, lp, 1]);
            len = lp;
        }
        let t = len;
        debug_assert_eq!(t, FRAMES_PER_CHUNK);
        let h = c.g.reshape_(h, vec![n as i64, 60, t as i64]);
        let h = c.g.transpose_(h, vec![0, 2, 1]);
        // 4-layer bidirectional LSTM, packed (layer, direction) blocks.
        let (mut wih, mut whh, mut bias) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..4 {
            for sfx in ["", "_reverse"] {
                wih.extend(self.get(&format!("lstm.weight_ih_l{l}{sfx}"))?);
                whh.extend(self.get(&format!("lstm.weight_hh_l{l}{sfx}"))?);
                let bi = self.get(&format!("lstm.bias_ih_l{l}{sfx}"))?;
                let bh = self.get(&format!("lstm.bias_hh_l{l}{sfx}"))?;
                bias.extend(bi.iter().zip(&bh).map(|(a, b)| a + b));
            }
        }
        let (nih, nhh, nb) = (wih.len(), whh.len(), bias.len());
        let wih = c.param(wih, &[nih]);
        let whh = c.param(whh, &[nhh]);
        let bias = c.param(bias, &[nb]);
        let mut h = c.g.inner().lstm(
            h,
            wih,
            whh,
            bias,
            HIDDEN,
            4,
            true,
            Shape::new(&[n, t, 2 * HIDDEN], F32),
        );
        let mut din = 2 * HIDDEN;
        for (i, dout) in [(0usize, 128usize), (1, 128)] {
            h = linear(
                &mut c,
                h,
                self.get(&format!("linear.{i}.weight"))?,
                self.get(&format!("linear.{i}.bias"))?,
                dout,
                din,
                n * t,
            );
            let h4 = c.g.reshape_(h, vec![n as i64, t as i64, dout as i64, 1]);
            let h4 = c.leaky(h4, &[n, t, dout, 1]);
            h = c.g.reshape_(h4, vec![n as i64, t as i64, dout as i64]);
            din = dout;
        }
        let h = linear(
            &mut c,
            h,
            self.get("classifier.weight")?,
            self.get("classifier.bias")?,
            3,
            din,
            n * t,
        );
        let h = c.g.add_node(
            Op::Activation(Activation::Sigmoid),
            vec![h],
            Shape::new(&[n, t, 3], F32),
        );
        let params = std::mem::take(&mut c.params);
        hir.set_outputs(vec![h]);
        let graph = Graph::from_hir(hir)
            .map_err(|e| anyhow::anyhow!("pyannote: HIR lowering failed: {e}"))?;
        let mut compiled = Session::new(self.device).compile(graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Ok(compiled)
    }

    /// `Inference.slide` + `pre_aggregation_hook` (max over speakers) +
    /// `aggregate(hamming=True)` + loose crop: frame scores on the receptive
    /// field grid (`start 0, duration 991/16000, step 270/16000`).
    pub fn frame_scores(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let n = audio.len();
        let num_chunks = if n >= WINDOW {
            (n - WINDOW) / STEP + 1
        } else {
            0
        };
        let has_last = n < WINDOW || (n - WINDOW) % STEP > 0;
        let total = num_chunks + has_last as usize;
        let mut chunks = vec![0f32; total * WINDOW];
        for c in 0..num_chunks {
            chunks[c * WINDOW..(c + 1) * WINDOW]
                .copy_from_slice(&audio[c * STEP..c * STEP + WINDOW]);
        }
        if has_last {
            let tail = &audio[num_chunks * STEP..];
            chunks[num_chunks * WINDOW..num_chunks * WINDOW + tail.len()].copy_from_slice(tail);
        }
        // Batches of 32, as pyannote runs them (one graph per batch size).
        let mut raw = Vec::with_capacity(total * FRAMES_PER_CHUNK * 3);
        let mut at = 0;
        while at < total {
            let b = (total - at).min(32);
            raw.extend(self.infer(&chunks[at * WINDOW..(at + b) * WINDOW], b)?);
            at += b;
        }
        Ok(aggregate(&raw, total, n, has_last))
    }

    /// The merged chunks WhisperX transcribes (`merge_chunks(..., 30, 0.5, 0.363)`).
    pub fn whisperx_chunks(&self, audio: &[f32]) -> Result<Vec<Region>> {
        let scores = self.frame_scores(audio)?;
        Ok(merge_chunks(&binarize(&scores, 0.5, 0.363, 30.0), 30.0))
    }
}

fn linear(
    c: &mut Ctx,
    x: HirNodeId,
    w: Vec<f32>,
    b: Vec<f32>,
    dout: usize,
    din: usize,
    rows: usize,
) -> HirNodeId {
    let mut wt = vec![0f32; w.len()];
    for o in 0..dout {
        for i in 0..din {
            wt[i * dout + o] = w[o * din + i];
        }
    }
    let shape = c.g.shape(x).clone();
    let lead: Vec<i64> = (0..shape.rank() - 1)
        .map(|i| shape.dim(i).unwrap_static() as i64)
        .collect();
    let wp = c.param(wt, &[din, dout]);
    let bp = c.param(b, &[1, dout]);
    let x2 = c.g.reshape_(x, vec![rows as i64, din as i64]);
    let y = c.g.mm(x2, wp);
    let be = c.g.expand_(bp, vec![rows as i64, dout as i64]);
    let y = c.g.add(y, be);
    let mut out = lead;
    out.push(dout as i64);
    c.g.reshape_(y, out)
}

/// `SlidingWindow.closest_frame` on the receptive-field grid.
fn closest_frame(t: f64) -> i64 {
    ((t - 0.0 - 0.5 * RF_DURATION) / RF_STEP).round_ties_even() as i64
}

/// `Inference.aggregate(hamming=True, missing=0)` after the max-over-speakers
/// hook, then `crop(Segment(0, n / sr), "loose")` when a padded last window
/// exists. numpy's dtypes: float32 accumulators updated from float64 terms.
pub fn aggregate(raw: &[f32], num_chunks: usize, num_samples: usize, has_last: bool) -> Vec<f32> {
    let m = FRAMES_PER_CHUNK;
    let ham: Vec<f64> = (0..m)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (m - 1) as f64).cos())
        .collect();
    let num_frames = closest_frame(
        0.0 + CHUNK_DURATION + (num_chunks as f64 - 1.0) * CHUNK_STEP + 0.5 * RF_DURATION,
    ) as usize
        + 1;
    let mut out = vec![0f32; num_frames];
    let mut count = vec![0f32; num_frames];
    for c in 0..num_chunks {
        let chunk_start = 0.0 + c as f64 * CHUNK_STEP;
        let sf = closest_frame(chunk_start + 0.5 * RF_DURATION) as usize;
        for j in 0..m.min(num_frames.saturating_sub(sf)) {
            let r = &raw[(c * m + j) * 3..(c * m + j) * 3 + 3];
            let s = r[0].max(r[1]).max(r[2]);
            out[sf + j] = (out[sf + j] as f64 + (s as f64 * 1.0) * ham[j] * 1.0) as f32;
            count[sf + j] = (count[sf + j] as f64 + 1.0 * ham[j] * 1.0) as f32;
        }
    }
    let mut avg: Vec<f32> = out
        .iter()
        .zip(&count)
        .map(|(&a, &n)| a / n.max(1e-12))
        .collect();
    if has_last {
        let end = num_samples as f64 / SAMPLE_RATE as f64;
        let j = ((end - 0.0) / RF_STEP).floor() as i64;
        let hi = ((j + 1).max(0) as usize).min(avg.len());
        avg.truncate(hi);
    }
    avg
}

/// WhisperX's `Binarize(onset, offset, max_duration)` (no padding, no
/// min-duration filtering) followed by `get_timeline()` (sorted, empty
/// segments dropped, duplicates merged).
pub fn binarize(scores: &[f32], onset: f64, offset: f64, max_duration: f64) -> Vec<Region> {
    let ts: Vec<f64> = (0..scores.len())
        .map(|i| {
            let s = 0.0 + i as f64 * RF_STEP;
            0.5 * (s + (s + RF_DURATION))
        })
        .collect();
    let (on32, off32) = (onset as f32, offset as f32);
    let mut regions = Vec::new();
    if scores.is_empty() {
        return regions;
    }
    let mut start = ts[0];
    let mut active = scores[0] > on32;
    let mut cs: Vec<f32> = vec![scores[0]];
    let mut ct: Vec<f64> = vec![start];
    let mut t = start;
    for i in 1..scores.len() {
        t = ts[i];
        let y = scores[i];
        if active {
            let dur = t - start;
            if dur > max_duration {
                let after = cs.len() / 2;
                let mut mi = after;
                for k in after..cs.len() {
                    if cs[k] < cs[mi] {
                        mi = k;
                    }
                }
                regions.push(Region { start, end: ct[mi] });
                start = ct[mi];
                cs = cs[mi + 1..].to_vec();
                ct = ct[mi + 1..].to_vec();
            } else if y < off32 {
                regions.push(Region { start, end: t });
                start = t;
                active = false;
                cs.clear();
                ct.clear();
            }
            cs.push(y);
            ct.push(t);
        } else if y > on32 {
            start = t;
            active = true;
        }
    }
    if active {
        regions.push(Region { start, end: t });
    }
    regions.retain(|r| r.end - r.start > 1e-6);
    regions.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap()
            .then(a.end.partial_cmp(&b.end).unwrap())
    });
    regions.dedup();
    regions
}

/// `Vad.merge_chunks(segments, chunk_size)`.
pub fn merge_chunks(segs: &[Region], chunk_size: f64) -> Vec<Region> {
    if segs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut curr_end = 0.0f64;
    let mut curr_start = segs[0].start;
    for s in segs {
        if s.end - curr_start > chunk_size && curr_end - curr_start > 0.0 {
            out.push(Region {
                start: curr_start,
                end: curr_end,
            });
            curr_start = s.start;
        }
        curr_end = s.end;
    }
    out.push(Region {
        start: curr_start,
        end: curr_end,
    });
    out
}
