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

//! Page image → transcription (with DAN's layout tokens, or without).

use crate::graph::{self, IMAGE, MEMORY_MASK, STEP_MASK, STEP_X, cross_k_name, cross_v_name};
use crate::model::{DIM, DanCheckpoint, H_MAX, L_MAX, LAYERS, W_MAX, WINDOW, pe_1d};
use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::{attach_built_params, graph_from_hir};
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::Path;

/// `max_char_prediction` of the released configs.
pub const DEFAULT_MAX_CHARS: usize = 3000;

/// The layout tokens DAN's RIMES / READ charsets use for page structure
/// (`predict_example.py` strips them for plain text).
pub const LAYOUT_TOKENS: &str = "ⒷⓄⓅⓇⓈⓌⓎⒶⓃⓑⓞⓟⓡⓢⓦⓨⓐⓝ";

/// A recognized page.
#[derive(Debug, Clone)]
pub struct Recognition {
    /// Transcription with layout tokens.
    pub raw: String,
    /// Generated token ids (start token excluded, end token kept if reached).
    pub ids: Vec<usize>,
}

impl Recognition {
    /// The text with layout tokens removed.
    pub fn text(&self) -> String {
        self.raw
            .chars()
            .filter(|c| !LAYOUT_TOKENS.contains(*c))
            .collect()
    }
}

/// A loaded DAN model; encoder and step graphs are compiled per page size.
pub struct DanRunner {
    ck: DanCheckpoint,
    device: Device,
    /// Per-channel `(mean, std)` applied to the 0–255 input, if any.
    normalize: Option<([f32; 3], [f32; 3])>,
    encoders: HashMap<(usize, usize), CompiledGraph>,
    steps: HashMap<usize, CompiledGraph>,
    /// Feature positions of the page last encoded (selects the step graph).
    memory: Option<usize>,
    /// Ring of the last `WINDOW − 1` self-attention K/V rows per layer.
    kv: Vec<(Vec<f32>, Vec<f32>)>,
}

fn compile(g: graph::BuiltGraph, device: Device) -> Result<CompiledGraph> {
    let (graph, params) = graph_from_hir(g.hir, g.params)?;
    let opts = compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    attach_built_params(&mut compiled, params, &[]);
    Ok(compiled)
}

impl DanRunner {
    /// Open a DAN training checkpoint (`dan_rimes_page.pt`, …).
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        validate_standard_device("dan", device)?;
        let ck = DanCheckpoint::open(path)?;
        let past = (WINDOW - 1) * DIM;
        Ok(Self {
            ck,
            device,
            normalize: None,
            encoders: HashMap::new(),
            steps: HashMap::new(),
            memory: None,
            kv: vec![(vec![0f32; past], vec![0f32; past]); LAYERS],
        })
    }

    /// Normalize the 0–255 input with the training set's per-channel
    /// statistics. The released checkpoints do not store them and
    /// `predict_example.py` feeds raw values, which the encoder's
    /// InstanceNorm layers absorb; this is for checkpoints that do.
    pub fn with_normalization(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.normalize = Some((mean, std));
        self
    }

    pub fn charset(&self) -> &[String] {
        &self.ck.charset
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// `[3, H, W]` network input for an RGB page (gray pages repeat their
    /// channel, as `predict_example.py` does).
    pub fn input(&self, img: &image::RgbImage) -> Vec<f32> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut out = vec![0f32; 3 * h * w];
        for (i, p) in img.pixels().enumerate() {
            for c in 0..3 {
                let v = f32::from(p.0[c]);
                out[c * h * w + i] = match self.normalize {
                    Some((m, s)) => (v - m[c]) / s[c],
                    None => v,
                };
            }
        }
        out
    }

    /// Encode a page `[3, h, w]`; binds the cross K/V for decoding and
    /// returns the encoder features `[256, h/32, w/8]`.
    pub fn encode(&mut self, input: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        ensure!(input.len() == 3 * h * w, "input is not [3, {h}, {w}]");
        let [fh, fw] = graph::feature_size(h, w);
        ensure!(
            fh <= H_MAX && fw <= W_MAX,
            "a {h}x{w} page gives {fh}x{fw} features; the 2-D positions stop at {H_MAX}x{W_MAX}"
        );
        if !self.encoders.contains_key(&(h, w)) {
            let g = graph::build_encoder(&self.ck.encoder, &self.ck.decoder, h, w)?;
            self.encoders.insert((h, w), compile(g, self.device)?);
        }
        let mut outs = self
            .encoders
            .get_mut(&(h, w))
            .expect("inserted")
            .run(&[(IMAGE, input)])
            .into_iter();
        let features = outs
            .next()
            .ok_or_else(|| anyhow!("encoder returned nothing"))?;
        let memory = fh * fw;
        if !self.steps.contains_key(&memory) {
            let g = graph::build_step(&self.ck.decoder, memory, self.ck.vocab())?;
            self.steps.insert(memory, compile(g, self.device)?);
        }
        let step = self.steps.get_mut(&memory).expect("inserted");
        for i in 0..LAYERS {
            let k = outs.next().ok_or_else(|| anyhow!("cross K {i} missing"))?;
            let v = outs.next().ok_or_else(|| anyhow!("cross V {i} missing"))?;
            step.set_param(&cross_k_name(i), &k);
            step.set_param(&cross_v_name(i), &v);
        }
        // `generate_enc_mask`: positions past (h / 32, w / 8) are padding.
        let (rh, rw) = (h / 32, w / 8);
        let mask: Vec<f32> = (0..fh)
            .flat_map(|y| (0..fw).map(move |x| f32::from(y < rh && x < rw)))
            .collect();
        step.set_param(MEMORY_MASK, &mask);
        self.memory = Some(memory);
        for (k, v) in &mut self.kv {
            k.fill(0.0);
            v.fill(0.0);
        }
        Ok(features)
    }

    /// Feed `token` at position `pos` (all earlier positions fed since the
    /// last [`Self::encode`]); returns the decision-layer logits
    /// `[vocab + 1]`.
    pub fn step(&mut self, token: usize, pos: usize) -> Result<Vec<f32>> {
        ensure!(pos < L_MAX, "position {pos} beyond the 1-D table ({L_MAX})");
        let memory = self.memory.ok_or_else(|| anyhow!("call encode first"))?;
        let (emb, _) = self.ck.decoder.get("emb.weight").expect("checked at open");
        ensure!(
            token < emb.len() / DIM,
            "token {token} outside the embedding"
        );
        let x: Vec<f32> = emb[token * DIM..(token + 1) * DIM]
            .iter()
            .zip(pe_1d(pos))
            .map(|(e, p)| e + p)
            .collect();
        let past = WINDOW - 1;
        let mask = bucket_decode_mask(pos.min(past), past);
        let names = rlx_core::autoregressive::past_kv_input_names(LAYERS);
        let mut inputs: Vec<(&str, &[f32])> = vec![(STEP_X, &x), (STEP_MASK, &mask)];
        for (i, (k, v)) in self.kv.iter().enumerate() {
            inputs.push((&names[2 * i], k));
            inputs.push((&names[2 * i + 1], v));
        }
        let step = self.steps.get_mut(&memory).expect("compiled in encode");
        let mut outs = step.run(&inputs).into_iter();
        let logits = outs
            .next()
            .ok_or_else(|| anyhow!("step returned nothing"))?;
        // The window keeps the last `past` tokens: the new one replaces the
        // oldest, which sits at `pos % past`.
        let row = pos % past;
        for (i, (k, v)) in self.kv.iter_mut().enumerate() {
            let kn = outs.next().ok_or_else(|| anyhow!("K row {i} missing"))?;
            let vn = outs.next().ok_or_else(|| anyhow!("V row {i} missing"))?;
            k[row * DIM..(row + 1) * DIM].copy_from_slice(&kn);
            v[row * DIM..(row + 1) * DIM].copy_from_slice(&vn);
        }
        Ok(logits)
    }

    /// Greedy decoding after [`Self::encode`], up to `max_chars` tokens.
    pub fn generate(&mut self, max_chars: usize) -> Result<Vec<usize>> {
        let end = self.ck.end_token();
        let mut tokens = vec![self.ck.start_token()];
        for pos in 0..max_chars {
            let logits = self.step(tokens[pos], pos)?;
            let next = rlx_core::ctc::argmax_frames(&logits, logits.len())[0];
            tokens.push(next);
            if next == end {
                break;
            }
        }
        Ok(tokens[1..].to_vec())
    }

    /// Transcribe a page image.
    pub fn recognize(&mut self, img: &image::RgbImage, max_chars: usize) -> Result<Recognition> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let x = self.input(img);
        self.encode(&x, h, w)?;
        let ids = self.generate(max_chars)?;
        let raw = ids
            .iter()
            .filter_map(|&i| self.ck.charset.get(i))
            .map(String::as_str)
            .collect();
        Ok(Recognition { raw, ids })
    }

    pub fn recognize_path(&mut self, path: &Path, max_chars: usize) -> Result<Recognition> {
        let img = image::open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .to_rgb8();
        self.recognize(&img, max_chars)
    }
}
