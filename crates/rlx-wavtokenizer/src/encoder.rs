// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// WavTokenizer encoder = a 512-dim causal `encodec_24khz` SEANet (ELU, reflect
// pad) + 2-layer LSTM + final conv(512→512) + single euclidean VQ. Conv stacks
// run on the graph; LSTM + VQ on the host.

use anyhow::{Context, Result};
use rlx_core::audio_ops_ir::{
    AudioGraph, CompiledAudioGraph, Conv1d, PadMode, compile, finish_graph, noncausal_pad,
};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_runtime::Device;
use safetensors::SafeTensors;

const NF: usize = 32;
const DIM: usize = 512; // latent = lstm dim = output dim
const RATIOS: [usize; 4] = [2, 4, 5, 8]; // reversed upsampling ratios
const PREFIX: &str = "feature_extractor.encodec.encoder.model";

#[derive(Clone)]
struct ConvW {
    weight: Vec<f32>,
    bias: Vec<f32>,
    c_out: usize,
    c_in: usize,
    k: usize,
    stride: usize,
}
#[derive(Clone)]
struct ResnetW {
    conv1: ConvW,
    conv2: ConvW,
    shortcut: ConvW,
}
#[derive(Clone)]
struct StageW {
    resnet: ResnetW,
    downsample: ConvW,
}
#[derive(Clone)]
struct LstmLayer {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

#[derive(Clone)]
pub struct EncoderWeights {
    stem: ConvW,
    stages: Vec<StageW>,
    lstm: Vec<LstmLayer>,
    final_conv: ConvW,
    codebook: Vec<f32>, // [4096, 512]
}

fn tf(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
    let raw = t.data();
    match t.dtype() {
        Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        dt => anyhow::bail!("{name}: {dt:?}"),
    }
}
fn cv(
    st: &SafeTensors<'_>,
    p: &str,
    c_out: usize,
    c_in: usize,
    k: usize,
    stride: usize,
) -> Result<ConvW> {
    Ok(ConvW {
        weight: tf(st, &format!("{p}.conv.conv.weight"))?,
        bias: tf(st, &format!("{p}.conv.conv.bias"))?,
        c_out,
        c_in,
        k,
        stride,
    })
}

impl EncoderWeights {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse wavtok encoder")?;
        let stem = cv(&st, &format!("{PREFIX}.0"), NF, 1, 7, 1)?;
        let mut stages = Vec::new();
        let mut d = NF;
        for (i, &r) in RATIOS.iter().enumerate() {
            let base = 1 + 3 * i;
            let h = d / 2;
            let resnet = ResnetW {
                conv1: cv(&st, &format!("{PREFIX}.{base}.block.1"), h, d, 3, 1)?,
                conv2: cv(&st, &format!("{PREFIX}.{base}.block.3"), d, h, 1, 1)?,
                shortcut: cv(&st, &format!("{PREFIX}.{base}.shortcut"), d, d, 1, 1)?,
            };
            let downsample = cv(&st, &format!("{PREFIX}.{}", base + 2), d * 2, d, 2 * r, r)?;
            stages.push(StageW { resnet, downsample });
            d *= 2;
        }
        let lp = format!("{PREFIX}.13.lstm");
        let lstm = (0..2)
            .map(|l| {
                Ok(LstmLayer {
                    w_ih: tf(&st, &format!("{lp}.weight_ih_l{l}"))?,
                    w_hh: tf(&st, &format!("{lp}.weight_hh_l{l}"))?,
                    b_ih: tf(&st, &format!("{lp}.bias_ih_l{l}"))?,
                    b_hh: tf(&st, &format!("{lp}.bias_hh_l{l}"))?,
                })
            })
            .collect::<Result<_>>()?;
        let final_conv = cv(&st, &format!("{PREFIX}.15"), DIM, DIM, 7, 1)?;
        let codebook = tf(
            &st,
            "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed",
        )?;
        Ok(Self {
            stem,
            stages,
            lstm,
            final_conv,
            codebook,
        })
    }
}

fn causal_conv(
    ag: &mut AudioGraph,
    x: HirNodeId,
    t: usize,
    w: &ConvW,
) -> (HirNodeId, usize, usize) {
    let (pl, pr) = noncausal_pad(t, w.k, w.stride, 1);
    let mode = if pl == 0 && pr == 0 {
        PadMode::Constant
    } else {
        PadMode::Reflect
    };
    ag.conv1d(
        x,
        t,
        &Conv1d {
            weight: &w.weight,
            bias: Some(&w.bias),
            c_out: w.c_out,
            c_in: w.c_in,
            k: w.k,
            stride: w.stride,
            dilation: 1,
            groups: 1,
            pad_left: pl,
            pad_right: pr,
            pad_mode: mode,
        },
    )
}
fn resnet(ag: &mut AudioGraph, x: HirNodeId, c: usize, t: usize, r: &ResnetW) -> HirNodeId {
    let (sc, _, _) = causal_conv(ag, x, t, &r.shortcut);
    let h = ag.elu(x, c, t);
    let (h, hc, ht) = causal_conv(ag, h, t, &r.conv1);
    let h = ag.elu(h, hc, ht);
    let (h, _, _) = causal_conv(ag, h, ht, &r.conv2);
    ag.add(sc, h)
}

fn build_pre(
    w: &EncoderWeights,
    in_len: usize,
) -> Result<(rlx_ir::Graph, Vec<(String, Vec<f32>)>, usize, usize)> {
    let mut hir = HirModule::new("wavtok_enc_pre");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let x = ag.input("pcm", &[1, 1, in_len, 1]);
    let (mut h, mut c, mut t) = causal_conv(&mut ag, x, in_len, &w.stem);
    for s in &w.stages {
        h = resnet(&mut ag, h, c, t, &s.resnet);
        h = ag.elu(h, c, t);
        let (hn, cn, tn) = causal_conv(&mut ag, h, t, &s.downsample);
        h = hn;
        c = cn;
        t = tn;
    }
    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, h)?, params, c, t))
}
fn build_post(w: &EncoderWeights, in_t: usize) -> Result<(rlx_ir::Graph, Vec<(String, Vec<f32>)>)> {
    let mut hir = HirModule::new("wavtok_enc_post");
    let mut g = HirMut::new(&mut hir);
    let mut ag = AudioGraph::new(&mut g);
    let z = ag.input("z", &[1, DIM, in_t, 1]);
    let h = ag.elu(z, DIM, in_t);
    let (out, _, _) = causal_conv(&mut ag, h, in_t, &w.final_conv);
    let params = std::mem::take(&mut ag.params);
    Ok((finish_graph(hir, out)?, params))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn lstm_forward(layers: &[LstmLayer], x: &[f32], dim: usize, t: usize) -> Vec<f32> {
    let h = dim;
    let mut seq: Vec<Vec<f32>> = (0..t)
        .map(|ti| (0..dim).map(|c| x[c * t + ti]).collect())
        .collect();
    for l in layers {
        let in_d = l.w_ih.len() / (4 * h);
        let (mut hp, mut cp) = (vec![0f32; h], vec![0f32; h]);
        let mut out = Vec::with_capacity(t);
        for xt in &seq {
            let mut g = vec![0f32; 4 * h];
            for (r, gr) in g.iter_mut().enumerate() {
                let mut a = l.b_ih[r] + l.b_hh[r];
                let ih = &l.w_ih[r * in_d..r * in_d + in_d];
                for (k, &xv) in xt.iter().enumerate() {
                    a += ih[k] * xv;
                }
                let hh = &l.w_hh[r * h..r * h + h];
                for (k, &hv) in hp.iter().enumerate() {
                    a += hh[k] * hv;
                }
                *gr = a;
            }
            let (mut hn, mut cn) = (vec![0f32; h], vec![0f32; h]);
            for j in 0..h {
                let cc = sigmoid(g[h + j]) * cp[j] + sigmoid(g[j]) * g[2 * h + j].tanh();
                cn[j] = cc;
                hn[j] = sigmoid(g[3 * h + j]) * cc.tanh();
            }
            hp = hn.clone();
            cp = cn;
            out.push(hn);
        }
        seq = out;
    }
    let mut res = vec![0f32; dim * t];
    for ti in 0..t {
        for c in 0..dim {
            res[c * t + ti] = seq[ti][c] + x[c * t + ti]; // SLSTM residual skip
        }
    }
    res
}

/// Single-codebook euclidean VQ. `latent` `[512, T]` → codes `[T]`.
fn vq_encode(codebook: &[f32], latent: &[f32], dim: usize, t: usize) -> Vec<u32> {
    let cbsize = codebook.len() / dim;
    let e_sq: Vec<f32> = (0..cbsize)
        .map(|i| (0..dim).map(|d| codebook[i * dim + d].powi(2)).sum::<f32>())
        .collect();
    (0..t)
        .map(|ti| {
            let mut best = 0usize;
            let mut bs = f32::NEG_INFINITY;
            for i in 0..cbsize {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += latent[d * t + ti] * codebook[i * dim + d];
                }
                let s = 2.0 * dot - e_sq[i];
                if s > bs {
                    bs = s;
                    best = i;
                }
            }
            best as u32
        })
        .collect()
}

/// Encode mono PCM → (pre-VQ features `[512, T]`, codes `[T]`).
pub struct WavtokEncoder {
    w: EncoderWeights,
    device: Device,
}

impl WavtokEncoder {
    pub fn new(w: EncoderWeights, device: Device) -> Self {
        Self { w, device }
    }

    pub fn encode(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize, Vec<u32>)> {
        let (g, p, _c, t) = build_pre(&self.w, pcm.len())?;
        let mut pre: CompiledAudioGraph = compile(self.device, g, p, DIM, t, "pcm");
        let pre_out = pre.run(pcm)?.0;
        let post_in = lstm_forward(&self.w.lstm, &pre_out, DIM, t);
        let (g2, p2) = build_post(&self.w, t)?;
        let mut post = compile(self.device, g2, p2, DIM, t, "z");
        let emb = post.run(&post_in)?.0; // [512, T]
        let codes = vq_encode(&self.w.codebook, &emb, DIM, t);
        Ok((emb, t, codes))
    }
}
