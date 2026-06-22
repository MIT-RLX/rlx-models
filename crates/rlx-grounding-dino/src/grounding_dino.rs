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

//! Top-level Grounding DINO model: loads a checkpoint and runs the full
//! open-vocabulary detection pipeline (Swin + BERT → enhancer → query
//! selection → decoder → detections).

use crate::config::GroundingDinoConfig;
use crate::decoder::Decoder;
use crate::deform_attn::LevelShape;
use crate::enhancer_ir::EncoderIr;
use crate::neck::Neck;
use crate::postprocess::{Detection, post_process};
use crate::preprocess::preprocess_rgb;
use crate::query_select::QuerySelect;
use crate::swin::SwinBackbone;
use crate::text_encoder_ir::TextEncoderIr;
use crate::tokenizer::TextTokens;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;
use std::path::Path;

/// A loaded Grounding DINO model. Heavy compute (Swin backbone, text encoder,
/// feature enhancer, decoder) runs on-device as compiled IR graphs for the
/// configured [`Device`]; scalar glue and the fused deformable op run on the
/// host. A CPU-native path is retained as the parity anchor.
pub struct GroundingDino {
    cfg: GroundingDinoConfig,
    device: Device,
    swin: SwinBackbone,
    neck: Neck,
    text_encoder: TextEncoderIr,
    encoder: EncoderIr,
    query_select: QuerySelect,
    decoder: Decoder,
}

impl GroundingDino {
    /// Build from an in-memory weight map (IR components compiled for CPU).
    pub fn from_weights(wm: &WeightMap, cfg: GroundingDinoConfig) -> Result<Self> {
        Self::from_weights_on(wm, cfg, Device::Cpu)
    }

    /// Build from an in-memory weight map with the IR (graph) components
    /// compiled for `device`. On `Device::Cpu` this is numerically equivalent
    /// to the former native path.
    pub fn from_weights_on(
        wm: &WeightMap,
        cfg: GroundingDinoConfig,
        device: Device,
    ) -> Result<Self> {
        let swin = SwinBackbone::from_weights_on(wm, cfg.backbone_config.clone(), device)?;
        let neck = Neck::from_weights(wm, &cfg)?;
        let text_encoder =
            TextEncoderIr::from_weights(wm, cfg.text_config.clone(), cfg.d_model, device)?;
        let encoder = EncoderIr::from_weights(wm, &cfg, device)?;
        let query_select = QuerySelect::from_weights(wm, &cfg)?;
        let decoder = Decoder::from_weights(wm, &cfg)?;
        Ok(Self {
            cfg,
            device,
            swin,
            neck,
            text_encoder,
            encoder,
            query_select,
            decoder,
        })
    }

    /// Load a safetensors/GGUF checkpoint (IR components compiled for CPU).
    pub fn from_checkpoint(path: &Path, cfg: GroundingDinoConfig) -> Result<Self> {
        Self::from_checkpoint_on(path, cfg, Device::Cpu)
    }

    /// Load a checkpoint with the IR (graph) components compiled for `device`.
    pub fn from_checkpoint_on(
        path: &Path,
        cfg: GroundingDinoConfig,
        device: Device,
    ) -> Result<Self> {
        let wm = WeightMap::from_resolved_path(path)?;
        Self::from_weights_on(&wm, cfg, device)
    }

    pub fn config(&self) -> &GroundingDinoConfig {
        &self.cfg
    }

    /// Device the IR (graph) components are compiled for.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Detect on a raw interleaved-RGB `u8` image (`[h*w*3]`).
    pub fn detect(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
        tokens: &TextTokens,
        box_threshold: f32,
        text_threshold: f32,
    ) -> Vec<Detection> {
        let pre = preprocess_rgb(rgb, height, width);
        self.detect_preprocessed(
            &pre.pixel_values,
            pre.height,
            pre.width,
            height,
            width,
            tokens,
            box_threshold,
            text_threshold,
        )
    }

    /// Detect on already-normalized pixel values (`[3, h, w]`, NCHW). `orig_*`
    /// are the original image dims used to rescale boxes.
    #[allow(clippy::too_many_arguments)]
    pub fn detect_preprocessed(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
        orig_height: usize,
        orig_width: usize,
        tokens: &TextTokens,
        box_threshold: f32,
        text_threshold: f32,
    ) -> Vec<Detection> {
        self.detect_inner(
            pixel_values,
            height,
            width,
            orig_height,
            orig_width,
            tokens,
            box_threshold,
            text_threshold,
            None,
        )
        .expect("graph decode on the model device should not fail")
    }

    /// Detect with the decoder's attention/FFN stack compiled and run on
    /// `device` (the rest of the pipeline stays on the host). Equivalent to
    /// [`Self::detect_preprocessed`] on `Device::Cpu`.
    #[allow(clippy::too_many_arguments)]
    pub fn detect_preprocessed_on(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
        orig_height: usize,
        orig_width: usize,
        tokens: &TextTokens,
        box_threshold: f32,
        text_threshold: f32,
        device: Device,
    ) -> Result<Vec<Detection>> {
        self.detect_inner(
            pixel_values,
            height,
            width,
            orig_height,
            orig_width,
            tokens,
            box_threshold,
            text_threshold,
            Some(device),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_inner(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
        orig_height: usize,
        orig_width: usize,
        tokens: &TextTokens,
        box_threshold: f32,
        text_threshold: f32,
        device: Option<Device>,
    ) -> Result<Vec<Detection>> {
        let d = self.cfg.d_model;

        // Env-gated per-stage timing (`RLX_GDINO_PROFILE=1`). Each stage syncs to
        // a host `Vec<f32>`, so the elapsed time is a true wall-clock segment.
        let profile = std::env::var("RLX_GDINO_PROFILE").is_ok();
        let mark = |label: &str, t: &std::time::Instant| {
            if profile {
                eprintln!("[gdino-profile] {label}: {:.3}s", t.elapsed().as_secs_f64());
            }
        };

        // 1. Vision backbone + neck.
        let t = std::time::Instant::now();
        let maps = self.swin.forward(pixel_values, height, width);
        mark("1.swin", &t);
        let t = std::time::Instant::now();
        let levels = self.neck.forward(&maps);
        mark("1.neck", &t);
        let shapes: Vec<LevelShape> = levels
            .iter()
            .map(|l| LevelShape { h: l.h, w: l.w })
            .collect();

        // 2. Flatten multi-scale sources + position embeddings to [seq, d].
        let mut vision = Vec::new();
        let mut vision_pos = Vec::new();
        for l in &levels {
            append_chw_tokens(&l.source, d, l.h, l.w, &mut vision);
            append_chw_tokens(&l.pos, d, l.h, l.w, &mut vision_pos);
        }

        // 3. Text backbone (IR graph on `self.device`).
        let t = std::time::Instant::now();
        let text = self.text_encoder.forward(tokens)?;
        mark("3.text", &t);
        // Text position embedding for the enhancer's text self-attention:
        // HF `get_sine_pos_embed(position_ids)` (was incorrectly zeroed).
        let text_pos = text_sine_pos_embed(&tokens.position_ids, d);

        // 4. Feature enhancer.
        let t = std::time::Instant::now();
        let enc = self.encoder.forward(
            &vision,
            &vision_pos,
            &text.features,
            &text_pos,
            &tokens.self_attn_mask,
            &shapes,
        )?;
        mark("4.enhancer", &t);

        // 5. Language-guided query selection.
        let t = std::time::Instant::now();
        let sel =
            self.query_select
                .forward(&enc.vision, &enc.text, &tokens.attention_mask, &shapes);
        mark("5.query_select", &t);

        // 6. Cross-modality decoder (IR graph; per-call `device` overrides the
        // model's compiled device, else falls back to `self.device`).
        let t = std::time::Instant::now();
        let dec = self.decoder.forward_on_device(
            &sel.target,
            &sel.reference_points,
            &enc.vision,
            &enc.text,
            &tokens.attention_mask,
            &shapes,
            device.unwrap_or(self.device),
        )?;
        mark("6.decoder", &t);

        // 7. Postprocess → detections.
        Ok(post_process(
            &dec.hidden,
            &dec.boxes,
            &enc.text,
            &tokens.attention_mask,
            d,
            orig_height,
            orig_width,
            box_threshold,
            text_threshold,
        ))
    }
}

/// Sine position embedding for text tokens, matching HF
/// `get_sine_pos_embed(position_ids[..., None], num_pos_feats=d, exchange_xy=False)`.
/// For position `p` and dim `2k`/`2k+1`: `sin`/`cos(p · 2π / 10000^(2k/d))`.
fn text_sine_pos_embed(position_ids: &[u32], d: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = 2.0 * PI;
    let half = d / 2;
    let mut out = vec![0f32; position_ids.len() * d];
    for (t, &p) in position_ids.iter().enumerate() {
        let p = p as f32;
        for k in 0..half {
            let freq = 10000f32.powf(2.0 * k as f32 / d as f32);
            let angle = p * scale / freq;
            out[t * d + 2 * k] = angle.sin();
            out[t * d + 2 * k + 1] = angle.cos();
        }
    }
    out
}

/// Append a `[c, h, w]` map flattened to channel-last tokens `[h*w, c]`.
fn append_chw_tokens(src: &[f32], c: usize, h: usize, w: usize, out: &mut Vec<f32>) {
    let hw = h * w;
    out.reserve(hw * c);
    for p in 0..hw {
        for ch in 0..c {
            out.push(src[ch * hw + p]);
        }
    }
}
