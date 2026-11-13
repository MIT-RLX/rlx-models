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

//! Line image → text: preprocessing, the encoder, and greedy decoding.

use crate::config::TrOcrConfig;
use crate::graph::{self, DEC, PIXELS, STEP_MASK, STEP_X, cross_k_name, cross_v_name};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::{attach_built_params, graph_from_hir};
use rlx_core::image_preprocess::pil_resize_rgb8_exact;
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::path::Path;

/// Decoder positions the step graph holds unless told otherwise. A text
/// line rarely needs more than a few dozen tokens.
pub const DEFAULT_MAX_TOKENS: usize = 128;

/// HF's `TrOCRLearnedPositionalEmbedding` offset.
const POSITION_OFFSET: usize = 2;

/// A recognized line.
#[derive(Debug, Clone)]
pub struct Recognition {
    pub text: String,
    /// Generated ids, starting with the decoder start token.
    pub ids: Vec<u32>,
}

/// A loaded TrOCR model.
pub struct TrOcrRunner {
    cfg: TrOcrConfig,
    tokenizer: Option<Tokenizer>,
    device: Device,
    encoder: CompiledGraph,
    step: CompiledGraph,
    /// Self-attention positions the step graph was built for.
    capacity: usize,
    embed_tokens: Vec<f32>,
    embed_positions: Vec<f32>,
    embed_scale: f32,
    /// Self-attention K/V per layer, `[capacity, d]`, rows past the
    /// current length zero.
    kv: Vec<(Vec<f32>, Vec<f32>)>,
}

fn compile(g: graph::BuiltGraph, device: Device) -> Result<CompiledGraph> {
    let (graph, params) = graph_from_hir(g.hir, g.params)?;
    let opts = compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    attach_built_params(&mut compiled, params, &[]);
    Ok(compiled)
}

impl TrOcrRunner {
    /// Open a TrOCR model directory (`config.json`, the checkpoint, the
    /// processor config and the tokenizer files).
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        Self::open_with(dir, device, DEFAULT_MAX_TOKENS)
    }

    /// [`Self::open`] with room for `max_tokens` decoder positions.
    pub fn open_with(dir: &Path, device: Device, max_tokens: usize) -> Result<Self> {
        let cfg = TrOcrConfig::from_dir(dir)?;
        let mut w = crate::weights::load(dir)?;
        let tokenizer = Tokenizer::from_dir(dir).ok();
        Self::from_weights(cfg, &mut w, tokenizer, device, max_tokens)
    }

    /// Build from an already-loaded checkpoint, consuming its tensors.
    pub fn from_weights(
        cfg: TrOcrConfig,
        w: &mut rlx_core::weight_map::WeightMap,
        tokenizer: Option<Tokenizer>,
        device: Device,
        max_tokens: usize,
    ) -> Result<Self> {
        validate_standard_device("trocr", device)?;
        let dc = &cfg.decoder;
        ensure!(
            max_tokens >= 1 && max_tokens <= dc.max_positions,
            "max_tokens {max_tokens} outside 1..={}",
            dc.max_positions
        );
        let (embed_tokens, es) = w.take(&format!("{DEC}.embed_tokens.weight"))?;
        ensure!(
            es == [dc.vocab, dc.d_model],
            "embed_tokens {es:?}, expected [{}, {}]",
            dc.vocab,
            dc.d_model
        );
        if !w.has("decoder.output_projection.weight") {
            // Tied (base / large): the projection is the embedding matrix.
            w.insert(
                "decoder.output_projection.weight",
                embed_tokens.clone(),
                es.clone(),
            );
        }
        let (embed_positions, ps) = w.take(&format!("{DEC}.embed_positions.weight"))?;
        ensure!(
            ps == [dc.max_positions + POSITION_OFFSET, dc.d_model],
            "embed_positions {ps:?}"
        );
        let embed_scale = if dc.scale_embedding {
            (dc.d_model as f32).sqrt()
        } else {
            1.0
        };
        let encoder = compile(graph::build_encoder(&cfg, w)?, device)?;
        let step = compile(graph::build_step(&cfg, w, max_tokens)?, device)?;
        let left: Vec<&str> = w.keys().collect();
        ensure!(left.is_empty(), "checkpoint tensors nothing read: {left:?}");
        let kv = vec![
            (
                vec![0f32; max_tokens * dc.d_model],
                vec![0f32; max_tokens * dc.d_model]
            );
            dc.layers
        ];
        Ok(Self {
            cfg,
            tokenizer,
            device,
            encoder,
            step,
            capacity: max_tokens,
            embed_tokens,
            embed_positions,
            embed_scale,
            kv,
        })
    }

    pub fn config(&self) -> &TrOcrConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Most decoder positions one image can use (start token included).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// `pixel_values [3, S, S]` for an RGB image: the processor's resize
    /// (Pillow's arithmetic), then `(v / 255 − mean) / std`.
    pub fn pixels(&self, img: &image::RgbImage) -> Vec<f32> {
        let p = &self.cfg.preprocess;
        let (w, h) = (img.width() as usize, img.height() as usize);
        let s = p.size;
        let rgb = pil_resize_rgb8_exact(img.as_raw(), w, h, s, s, p.filter);
        let mut out = vec![0f32; 3 * s * s];
        for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                out[c * s * s + i] = (f32::from(px[c]) / 255.0 - p.mean[c]) / p.std[c];
            }
        }
        out
    }

    /// Run the encoder on `pixels` and bind its cross-attention K/V to the
    /// step graph. Returns the encoder states `[S_enc, d_enc]`.
    pub fn encode(&mut self, pixels: &[f32]) -> Result<Vec<f32>> {
        let e = &self.cfg.encoder;
        ensure!(
            pixels.len() == e.channels * e.image_size * e.image_size,
            "pixels: {} values, expected {}x{s}x{s}",
            pixels.len(),
            e.channels,
            s = e.image_size
        );
        let mut outs = self.encoder.run(&[(PIXELS, pixels)]).into_iter();
        let enc = outs
            .next()
            .ok_or_else(|| anyhow!("encoder returned nothing"))?;
        for i in 0..self.cfg.decoder.layers {
            let k = outs.next().ok_or_else(|| anyhow!("cross K {i} missing"))?;
            let v = outs.next().ok_or_else(|| anyhow!("cross V {i} missing"))?;
            self.step.set_param(&cross_k_name(i), &k);
            self.step.set_param(&cross_v_name(i), &v);
        }
        for (k, v) in &mut self.kv {
            k.fill(0.0);
            v.fill(0.0);
        }
        Ok(enc)
    }

    /// One decoder step: feed `token` at position `pos` (all earlier
    /// positions already fed since the last [`Self::encode`]); returns the
    /// next-token logits `[vocab]`.
    pub fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let dc = &self.cfg.decoder;
        let d = dc.d_model;
        ensure!(
            pos < self.capacity,
            "decoder position {pos} beyond the {} the step graph holds",
            self.capacity
        );
        let tok = token as usize;
        ensure!(tok < dc.vocab, "token {token} outside the vocabulary");
        let prow = &self.embed_positions[(pos + POSITION_OFFSET) * d..][..d];
        let x: Vec<f32> = self.embed_tokens[tok * d..(tok + 1) * d]
            .iter()
            .zip(prow)
            .map(|(e, p)| e * self.embed_scale + p)
            .collect();
        let mask = bucket_decode_mask(pos, self.capacity);
        let names = rlx_core::autoregressive::past_kv_input_names(dc.layers);
        let mut inputs: Vec<(&str, &[f32])> = vec![(STEP_X, &x), (STEP_MASK, &mask)];
        for (i, (k, v)) in self.kv.iter().enumerate() {
            inputs.push((&names[2 * i], k));
            inputs.push((&names[2 * i + 1], v));
        }
        let mut outs = self.step.run(&inputs).into_iter();
        let logits = outs
            .next()
            .ok_or_else(|| anyhow!("step returned nothing"))?;
        for (i, (k, v)) in self.kv.iter_mut().enumerate() {
            let kn = outs.next().ok_or_else(|| anyhow!("K row {i} missing"))?;
            let vn = outs.next().ok_or_else(|| anyhow!("V row {i} missing"))?;
            k[pos * d..(pos + 1) * d].copy_from_slice(&kn);
            v[pos * d..(pos + 1) * d].copy_from_slice(&vn);
        }
        Ok(logits)
    }

    /// Greedy decoding after [`Self::encode`]: up to `max_new_tokens` ids
    /// after the start token, stopping at end-of-sequence.
    pub fn generate(&mut self, max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut ids = vec![self.cfg.decoder_start];
        let limit = max_new_tokens.min(self.capacity);
        for pos in 0..limit {
            let logits = self.step(ids[pos], pos)?;
            let next = rlx_core::ctc::argmax_frames(&logits, logits.len())[0] as u32;
            ids.push(next);
            if next == self.cfg.eos {
                break;
            }
        }
        Ok(ids)
    }

    /// Recognize one text-line image.
    pub fn recognize(
        &mut self,
        img: &image::RgbImage,
        max_new_tokens: usize,
    ) -> Result<Recognition> {
        let px = self.pixels(img);
        self.encode(&px)?;
        let ids = self.generate(max_new_tokens)?;
        let text = self
            .tokenizer
            .as_ref()
            .map(|t| t.decode(&ids))
            .ok_or_else(|| anyhow!("no tokenizer files beside the model"))?;
        Ok(Recognition { text, ids })
    }

    pub fn recognize_path(&mut self, path: &Path, max_new_tokens: usize) -> Result<Recognition> {
        let img = image::open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .to_rgb8();
        self.recognize(&img, max_new_tokens)
    }
}
