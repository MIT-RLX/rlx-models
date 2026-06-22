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

//! RNN-T decoding, host side. The acoustic encoder runs in the RLX graph;
//! the prediction network (a single-layer LSTM) and joint network are
//! small and token-serial, so they run here in exact f32 — the standard
//! pattern for transducer greedy search. Language conditioning fuses a
//! one-hot language vector into the encoder features before the joint.

use anyhow::{Result, ensure};
use rlx_flow::WeightSource;

use crate::config::AsrConfig;
use crate::weights::keys;

/// A single PyTorch-layout LSTM cell (gate order i, f, g, o).
#[derive(Debug, Clone)]
pub struct LstmCell {
    pub hidden: usize,
    pub input: usize,
    w_ih: Vec<f32>, // [4H, input]
    w_hh: Vec<f32>, // [4H, hidden]
    b_ih: Vec<f32>, // [4H]
    b_hh: Vec<f32>, // [4H]
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl LstmCell {
    pub fn new(
        hidden: usize,
        input: usize,
        w_ih: Vec<f32>,
        w_hh: Vec<f32>,
        b_ih: Vec<f32>,
        b_hh: Vec<f32>,
    ) -> Result<Self> {
        ensure!(w_ih.len() == 4 * hidden * input, "lstm w_ih shape");
        ensure!(w_hh.len() == 4 * hidden * hidden, "lstm w_hh shape");
        ensure!(b_ih.len() == 4 * hidden, "lstm b_ih shape");
        ensure!(b_hh.len() == 4 * hidden, "lstm b_hh shape");
        Ok(Self {
            hidden,
            input,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// One LSTM step: `(h', c') = cell(x, h, c)`.
    pub fn step(&self, x: &[f32], h: &[f32], c: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let hh = self.hidden;
        // pre-activation gates = W_ih x + b_ih + W_hh h + b_hh
        let mut g = vec![0.0f32; 4 * hh];
        for row in 0..4 * hh {
            let mut acc = self.b_ih[row] + self.b_hh[row];
            let ih = &self.w_ih[row * self.input..(row + 1) * self.input];
            for k in 0..self.input {
                acc += ih[k] * x[k];
            }
            let hhw = &self.w_hh[row * hh..(row + 1) * hh];
            for k in 0..hh {
                acc += hhw[k] * h[k];
            }
            g[row] = acc;
        }
        let mut h_new = vec![0.0f32; hh];
        let mut c_new = vec![0.0f32; hh];
        for j in 0..hh {
            let i = sigmoid(g[j]);
            let f = sigmoid(g[hh + j]);
            let gg = g[2 * hh + j].tanh();
            let o = sigmoid(g[3 * hh + j]);
            let cj = f * c[j] + i * gg;
            c_new[j] = cj;
            h_new[j] = o * cj.tanh();
        }
        (h_new, c_new)
    }
}

/// RNN-T prediction network: embed previous label, run a stack of LSTMs.
pub struct PredictionNet {
    /// `[vocab, embed_dim]` embedding (blank uses a zero input vector).
    embed: Vec<f32>,
    embed_dim: usize,
    vocab: usize,
    /// Stacked LSTM layers (Nemotron uses 2).
    lstms: Vec<LstmCell>,
}

/// Carried prediction-network state between symbols — one `(h, c)` per layer.
#[derive(Clone)]
pub struct PredState {
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl PredictionNet {
    pub fn new(
        embed: Vec<f32>,
        embed_dim: usize,
        vocab: usize,
        lstms: Vec<LstmCell>,
    ) -> Result<Self> {
        ensure!(embed.len() == vocab * embed_dim, "prediction embed shape");
        ensure!(!lstms.is_empty(), "prediction net needs ≥1 LSTM layer");
        ensure!(
            lstms[0].input == embed_dim,
            "lstm0 input must equal embed dim"
        );
        for w in lstms.windows(2) {
            ensure!(w[1].input == w[0].hidden, "stacked lstm dim mismatch");
        }
        Ok(Self {
            embed,
            embed_dim,
            vocab,
            lstms,
        })
    }

    pub fn zero_state(&self) -> PredState {
        PredState {
            layers: self
                .lstms
                .iter()
                .map(|l| (vec![0.0; l.hidden], vec![0.0; l.hidden]))
                .collect(),
        }
    }

    /// Advance the prediction net by one label. `label == None` denotes the
    /// start-of-sequence (zero input, as NeMo does for the initial step).
    /// Returns the top layer's hidden state and the new stacked state.
    pub fn step(&self, label: Option<usize>, state: &PredState) -> (Vec<f32>, PredState) {
        let mut x = vec![0.0f32; self.embed_dim];
        if let Some(l) = label {
            if l < self.vocab {
                x.copy_from_slice(&self.embed[l * self.embed_dim..(l + 1) * self.embed_dim]);
            }
        }
        let mut new_layers = Vec::with_capacity(self.lstms.len());
        let mut cur = x;
        for (i, lstm) in self.lstms.iter().enumerate() {
            let (h, c) = &state.layers[i];
            let (hn, cn) = lstm.step(&cur, h, c);
            cur = hn.clone();
            new_layers.push((hn, cn));
        }
        (cur, PredState { layers: new_layers })
    }

    pub fn hidden(&self) -> usize {
        self.lstms.last().map(|l| l.hidden).unwrap_or(0)
    }
}

/// RNN-T joint network: project encoder + prediction features, fuse the
/// language vector, sum, activate, classify.
pub struct Joint {
    enc_w: Vec<f32>, // [joint_hidden, enc_in]
    enc_b: Vec<f32>,
    enc_in: usize,
    pred_w: Vec<f32>, // [joint_hidden, pred_in]
    pred_b: Vec<f32>,
    pred_in: usize,
    net_w: Vec<f32>, // [n_classes, joint_hidden]
    net_b: Vec<f32>,
    joint_hidden: usize,
    pub n_classes: usize,
    /// Width of the language vector folded into the encoder input
    /// (`enc_in == d_model + lang_dim`).
    lang_dim: usize,
}

impl Joint {
    /// `enc_in` is the joint encoder-projection input width; if it exceeds
    /// `d_model` the surplus is the fused language one-hot width.
    pub fn from_weights(w: &mut dyn WeightSource, d_model: usize) -> Result<Self> {
        let (enc_w, enc_sh) = w.take(keys::JOINT_ENC_W, false)?;
        let (enc_b, _) = w.take(keys::JOINT_ENC_B, false)?;
        let (pred_w, pred_sh) = w.take(keys::JOINT_PRED_W, false)?;
        let (pred_b, _) = w.take(keys::JOINT_PRED_B, false)?;
        let (net_w, net_sh) = w.take(keys::JOINT_NET_W, false)?;
        let (net_b, _) = w.take(keys::JOINT_NET_B, false)?;

        let joint_hidden = enc_sh[0];
        let enc_in = enc_sh[1];
        let pred_in = pred_sh[1];
        let n_classes = net_sh[0];
        let lang_dim = enc_in.saturating_sub(d_model);
        ensure!(pred_sh[0] == joint_hidden, "joint pred/enc hidden mismatch");
        ensure!(net_sh[1] == joint_hidden, "joint_net input width mismatch");
        Ok(Self {
            enc_w,
            enc_b,
            enc_in,
            pred_w,
            pred_b,
            pred_in,
            net_w,
            net_b,
            joint_hidden,
            n_classes,
            lang_dim,
        })
    }

    pub(crate) fn enc_proj(&self, enc_frame: &[f32], lang: &[f32]) -> Vec<f32> {
        // Fuse: [enc_frame ; lang] of width enc_in.
        let mut x = vec![0.0f32; self.enc_in];
        let d = self.enc_in - self.lang_dim;
        x[..d].copy_from_slice(&enc_frame[..d.min(enc_frame.len())]);
        if self.lang_dim > 0 {
            let take = self.lang_dim.min(lang.len());
            x[d..d + take].copy_from_slice(&lang[..take]);
        }
        matvec(&self.enc_w, &self.enc_b, &x, self.joint_hidden, self.enc_in)
    }

    fn pred_proj(&self, pred_out: &[f32]) -> Vec<f32> {
        matvec(
            &self.pred_w,
            &self.pred_b,
            pred_out,
            self.joint_hidden,
            self.pred_in,
        )
    }

    /// Logits over `n_classes` (last index is the blank).
    pub fn logits(&self, enc_proj: &[f32], pred_out: &[f32]) -> Vec<f32> {
        let pp = self.pred_proj(pred_out);
        let mut h = vec![0.0f32; self.joint_hidden];
        for j in 0..self.joint_hidden {
            h[j] = (enc_proj[j] + pp[j]).max(0.0); // sum + ReLU
        }
        matvec(
            &self.net_w,
            &self.net_b,
            &h,
            self.n_classes,
            self.joint_hidden,
        )
    }
}

fn matvec(w: &[f32], b: &[f32], x: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out];
    for o in 0..out {
        let row = &w[o * inp..(o + 1) * inp];
        let mut acc = b.get(o).copied().unwrap_or(0.0);
        for k in 0..inp.min(x.len()) {
            acc += row[k] * x[k];
        }
        y[o] = acc;
    }
    y
}

/// The Nemotron 3.5 language-conditioning `prompt_kernel`: a 2-layer MLP
/// over `concat(encoder_frame[d_model], language_one_hot[num_languages])`
/// that re-projects the fused features to `d_model` before the joint.
pub struct PromptKernel {
    w0: Vec<f32>,
    b0: Vec<f32>,
    in0: usize,
    hidden: usize,
    w2: Vec<f32>,
    b2: Vec<f32>,
    out: usize,
    d_model: usize,
    lang_dim: usize,
}

impl PromptKernel {
    /// Load `prompt_kernel.{0,2}.{weight,bias}` if present (probe via `take`).
    pub fn from_weights(w: &mut dyn WeightSource, d_model: usize) -> Result<Option<Self>> {
        let Ok((w0, w0_sh)) = w.take(keys::PROMPT_K0_W, false) else {
            return Ok(None);
        };
        let (b0, _) = w.take(keys::PROMPT_K0_B, false)?;
        let (w2, w2_sh) = w.take(keys::PROMPT_K2_W, false)?;
        let (b2, _) = w.take(keys::PROMPT_K2_B, false)?;
        let hidden = w0_sh[0];
        let in0 = w0_sh[1];
        let out = w2_sh[0];
        let lang_dim = in0.saturating_sub(d_model);
        Ok(Some(Self {
            w0,
            b0,
            in0,
            hidden,
            w2,
            b2,
            out,
            d_model,
            lang_dim,
        }))
    }

    pub fn out_dim(&self) -> usize {
        self.out
    }

    /// Fuse one encoder frame with the language vector → `[out]` (== d_model).
    pub fn fuse(&self, enc_frame: &[f32], lang: &[f32]) -> Vec<f32> {
        let mut x = vec![0.0f32; self.in0];
        let d = self.d_model.min(enc_frame.len());
        x[..d].copy_from_slice(&enc_frame[..d]);
        if self.lang_dim > 0 {
            let take = self.lang_dim.min(lang.len());
            x[self.d_model..self.d_model + take].copy_from_slice(&lang[..take]);
        }
        let mut h = matvec(&self.w0, &self.b0, &x, self.hidden, self.in0);
        for v in h.iter_mut() {
            *v = v.max(0.0); // ReLU
        }
        matvec(&self.w2, &self.b2, &h, self.out, self.hidden)
    }
}

/// Greedy RNN-T decode over encoder frames `enc` (`[t_frames, d_model]`).
/// `lang` is the one-hot (or empty) language vector. Returns emitted token
/// ids (blank excluded).
pub fn greedy_decode(
    cfg: &AsrConfig,
    pred: &PredictionNet,
    joint: &Joint,
    prompt: Option<&PromptKernel>,
    enc: &[f32],
    d_model: usize,
    lang: &[f32],
) -> Vec<u32> {
    let blank = joint.n_classes - 1;
    let t_frames = enc.len().checked_div(d_model).unwrap_or(0);

    let mut tokens = Vec::new();
    let mut state = pred.zero_state();
    let (mut pred_out, _) = pred.step(None, &state); // SOS

    for t in 0..t_frames {
        let frame = &enc[t * d_model..(t + 1) * d_model];
        // Language fusion (Nemotron prompt_kernel) re-projects the frame;
        // the joint then sees a plain d_model vector (no lang concat).
        let fused = match prompt {
            Some(pk) => pk.fuse(frame, lang),
            None => frame.to_vec(),
        };
        let enc_proj = joint.enc_proj(&fused, &[]);
        let mut emitted = 0;
        loop {
            let logits = joint.logits(&enc_proj, &pred_out);
            let best = argmax(&logits);
            if best == blank {
                break;
            }
            tokens.push(best as u32);
            let (po, st) = pred.step(Some(best), &state);
            pred_out = po;
            state = st;
            emitted += 1;
            if emitted >= cfg.max_symbols_per_step {
                break;
            }
        }
    }
    tokens
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lstm_step_matches_reference() {
        // hidden=1, input=1; gates i,f,g,o each weight 0, bias chosen so
        // outputs are analytically checkable.
        // w_ih, w_hh all zero -> gates == biases.
        let h = 1;
        let inp = 1;
        let w_ih = vec![0.0; 4];
        let w_hh = vec![0.0; 4];
        // biases split across ih/hh: i=0, f=0, g=0, o=0 -> sig(0)=.5, tanh(0)=0
        let b_ih = vec![0.0; 4];
        let b_hh = vec![0.0; 4];
        let cell = LstmCell::new(h, inp, w_ih, w_hh, b_ih, b_hh).unwrap();
        let (hn, cn) = cell.step(&[1.0], &[0.0], &[0.0]);
        // i=.5, f=.5, g=0, o=.5 -> c' = .5*0 + .5*0 = 0; h' = .5*tanh(0)=0.
        assert!((cn[0] - 0.0).abs() < 1e-6);
        assert!((hn[0] - 0.0).abs() < 1e-6);

        // Now drive g positive: set g-gate bias to a big value via b_ih.
        let cell2 = LstmCell::new(
            1,
            1,
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0, 0.0, 10.0, 0.0], // g bias huge -> tanh ~ 1
            vec![0.0; 4],
        )
        .unwrap();
        let (hn2, cn2) = cell2.step(&[0.0], &[0.0], &[0.0]);
        // i=.5, g~1 -> c' ~ .5; h' = .5 * tanh(.5)
        assert!((cn2[0] - 0.5).abs() < 1e-3, "c'={}", cn2[0]);
        assert!((hn2[0] - 0.5 * 0.5f32.tanh()).abs() < 1e-3);
    }

    fn tiny_pred() -> PredictionNet {
        // vocab=2, embed_dim=hidden=1, identity-ish LSTM.
        let lstm =
            LstmCell::new(1, 1, vec![0.0; 4], vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]).unwrap();
        PredictionNet::new(vec![0.5, -0.5], 1, 2, vec![lstm]).unwrap()
    }

    #[test]
    fn greedy_emits_then_blanks() {
        let pred = tiny_pred();
        // Joint with 3 classes (2 tokens + blank). Construct so frame 0
        // favors token 0, frame 1 favors blank.
        // enc_in = pred_in = joint_hidden = 1; n_classes = 3.
        let joint = Joint {
            enc_w: vec![1.0],
            enc_b: vec![0.0],
            enc_in: 1,
            pred_w: vec![0.0],
            pred_b: vec![0.0],
            pred_in: 1,
            net_w: vec![1.0, 0.0, -1.0], // class0 ~ +h, blank ~ -h
            net_b: vec![0.0, 0.0, 0.5],  // blank wins when h == 0 (frame 1)
            joint_hidden: 1,
            n_classes: 3,
            lang_dim: 0,
        };
        let cfg = mk_cfg();
        // frame0 = +1 (-> class0 wins), frame1 = -1 (-> blank wins).
        let enc = vec![1.0, -1.0];
        let toks = greedy_decode(&cfg, &pred, &joint, None, &enc, 1, &[]);
        assert!(toks.contains(&0));
        // It must terminate (blank caps), not loop forever.
        assert!(toks.len() <= cfg.max_symbols_per_step + 1);
    }

    fn mk_cfg() -> AsrConfig {
        use rlx_nemo::NemoConfig;
        let yaml = b"encoder:\n  d_model: 8\n  n_layers: 1\n  n_heads: 2\n";
        AsrConfig::from_nemo(&NemoConfig::from_yaml_bytes(yaml).unwrap()).unwrap()
    }
}
