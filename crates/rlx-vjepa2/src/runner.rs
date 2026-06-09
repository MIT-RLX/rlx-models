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

use anyhow::{Result, anyhow};
use rlx_core::validate_standard_device;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;
use std::path::PathBuf;

/// Encoder token output from [`Vjepa2Runner::encode_video`].
#[derive(Debug, Clone)]
pub struct Vjepa2Output {
    pub per_batch: Vec<Vec<f32>>,
    pub seq: usize,
    pub hidden: usize,
}

/// Predictor output (projected target tokens).
#[derive(Debug, Clone)]
pub struct Vjepa2PredictOutput {
    pub per_batch: Vec<Vec<f32>>,
    pub num_target: usize,
    pub hidden: usize,
}

/// Attentive pooler output (+ optional classifier logits).
#[derive(Debug, Clone)]
pub struct Vjepa2PoolOutput {
    pub embedding: Vec<f32>,
    pub logits: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
pub struct Vjepa2RunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<crate::Vjepa2Config>,
    config_path: Option<PathBuf>,
    batch: Option<usize>,
    device: Option<Device>,
    /// Fixed context/target masks for compiled predictor graphs.
    predictor_masks: Option<crate::Vjepa2Masks>,
}

impl Vjepa2RunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn config(mut self, cfg: crate::Vjepa2Config) -> Self {
        self.config = Some(cfg);
        self
    }
    pub fn config_path<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.config_path = Some(p.into());
        self
    }
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    /// When set, the encoder trunk runs via compiled IR (CPU / Metal / …).
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Context/target masks baked into the compiled predictor graph.
    pub fn predictor_masks(mut self, masks: crate::Vjepa2Masks) -> Self {
        self.predictor_masks = Some(masks);
        self
    }

    pub fn build(self) -> Result<Vjepa2Runner> {
        use crate::{
            Vjepa2Config, Vjepa2GraphParams, build_vjepa2_encoder_graph_sized,
            build_vjepa2_pooler_graph_sized, build_vjepa2_predictor_graph_sized,
            extract_model_weights, predictor_mask_rows, prepare_predictor_layout,
        };
        use rlx_runtime::Session;

        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let cfg = match (self.config, self.config_path) {
            (Some(c), _) => c,
            (_, Some(p)) => Vjepa2Config::from_file(&p)?,
            _ => Vjepa2Config::vit_g_384(),
        };
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("vjepa2", device)?;
        let batch = self.batch.unwrap_or(1);

        let mut wm = rlx_core::load_weight_map(&weights_path, rlx_core::VJEPA2_GGUF_ARCHES)?;
        let model = extract_model_weights(&mut wm, &cfg)?;

        let compiled = if self.device.is_some() {
            let (graph, params, _pre) =
                build_vjepa2_encoder_graph_sized(&cfg, &model.encoder, batch)?;
            let opts = rlx_core::flow_bridge::compile_options_for_profile(
                &CompileProfile::encoder(),
                device,
            );
            let mut compiled = Session::new(device).compile_with(graph, &opts);
            Vjepa2GraphParams::from_f32(params).load(&mut compiled);
            Some(compiled)
        } else {
            None
        };

        let compiled_predictor = if self.device.is_some() {
            if let (Some(pred), Some(masks)) = (&model.predictor, &self.predictor_masks) {
                let layout = prepare_predictor_layout(&cfg, masks, batch)?;
                let mask_rows = predictor_mask_rows(pred, &cfg, masks, batch);
                let (graph, params) =
                    build_vjepa2_predictor_graph_sized(&cfg, pred, &layout, &mask_rows, batch)?;
                let opts = rlx_core::flow_bridge::compile_options_for_profile(
                    &CompileProfile::encoder(),
                    device,
                );
                let mut compiled = Session::new(device).compile_with(graph, &opts);
                params.load(&mut compiled);
                Some((compiled, masks.clone()))
            } else {
                None
            }
        } else {
            None
        };

        let compiled_pooler = if self.device.is_some() {
            if let Some(pooler) = &model.pooler {
                let (graph, params) = build_vjepa2_pooler_graph_sized(&cfg, pooler, batch)?;
                let opts = rlx_core::flow_bridge::compile_options_for_profile(
                    &CompileProfile::encoder(),
                    device,
                );
                let mut compiled = Session::new(device).compile_with(graph, &opts);
                params.load(&mut compiled);
                Some(compiled)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Vjepa2Runner {
            model,
            cfg,
            batch,
            device,
            compiled,
            compiled_predictor,
            compiled_pooler,
        })
    }
}

/// V-JEPA2 runner — encoder (+ optional predictor / pooler).
pub struct Vjepa2Runner {
    model: crate::Vjepa2ModelWeights,
    cfg: crate::Vjepa2Config,
    batch: usize,
    device: Device,
    compiled: Option<rlx_runtime::CompiledGraph>,
    compiled_predictor: Option<(rlx_runtime::CompiledGraph, crate::Vjepa2Masks)>,
    compiled_pooler: Option<rlx_runtime::CompiledGraph>,
}

impl Vjepa2Runner {
    pub fn builder() -> Vjepa2RunnerBuilder {
        Vjepa2RunnerBuilder::default()
    }
    pub fn config(&self) -> &crate::Vjepa2Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn has_predictor(&self) -> bool {
        self.model.predictor.is_some()
    }
    pub fn has_pooler(&self) -> bool {
        self.model.pooler.is_some()
    }

    fn encode_tokens_inner(&mut self, video_ncthw: &[f32]) -> Result<Vjepa2Output> {
        use crate::{conv3d_patch_embed, encode_video_native};

        let crop = self.cfg.crop_size;
        let frames = self.cfg.frames_per_clip;
        let expected = 3 * frames * crop * crop;
        anyhow::ensure!(
            video_ncthw.len() == expected,
            "expected {expected} f32 values for NCTHW video, got {}",
            video_ncthw.len()
        );

        let out = if let Some(compiled) = self.compiled.as_mut() {
            let patch = &self.model.encoder.patch;
            let mut hidden = conv3d_patch_embed(patch, video_ncthw, frames, crop, crop)?;
            if self.batch > 1 {
                let per = hidden.len();
                let mut batched = Vec::with_capacity(per * self.batch);
                for _ in 0..self.batch {
                    batched.extend_from_slice(&hidden);
                }
                hidden = batched;
            }
            let flat = compiled
                .run(&[("hidden", hidden.as_slice())])
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("vjepa2 graph forward returned no output"))?;
            crate::Vjepa2EncoderOutput {
                tokens: flat,
                seq: self.cfg.num_patches(),
                hidden: self.cfg.hidden_size,
            }
        } else {
            encode_video_native(&self.model.encoder, &self.cfg, video_ncthw, self.batch)?
        };

        let per = out.seq * out.hidden;
        let mut per_batch = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            per_batch.push(out.tokens[b * per..(b + 1) * per].to_vec());
        }
        Ok(Vjepa2Output {
            per_batch,
            seq: out.seq,
            hidden: out.hidden,
        })
    }

    /// Encode a pre-normalized video tensor `[C, T, H, W]` (NCTHW f32).
    pub fn encode_video(&mut self, video_ncthw: &[f32]) -> Result<Vjepa2Output> {
        self.encode_tokens_inner(video_ncthw)
    }

    /// Convenience: u8 HWC frames `[num_frames, crop, crop, 3]` → encode.
    pub fn encode_video_hwc(&mut self, frames: &[u8]) -> Result<Vjepa2Output> {
        use crate::normalize_video_hwc;

        let crop = self.cfg.crop_size;
        let nframes = self.cfg.frames_per_clip;
        let expected = nframes * crop * crop * 3;
        anyhow::ensure!(
            frames.len() == expected,
            "expected {expected} u8 pixels HWC, got {}",
            frames.len()
        );
        let ncthw = normalize_video_hwc(frames, nframes, crop);
        self.encode_video(&ncthw)
    }

    /// Run the JEPA predictor on encoder outputs with context/target masks.
    pub fn predict(
        &mut self,
        enc: &Vjepa2Output,
        masks: &crate::Vjepa2Masks,
    ) -> Result<Vjepa2PredictOutput> {
        use crate::predict_native;

        let pred = self
            .model
            .predictor
            .as_ref()
            .ok_or_else(|| anyhow!("checkpoint has no predictor weights"))?;
        let mut flat = Vec::with_capacity(enc.per_batch.len() * enc.seq * enc.hidden);
        for batch in &enc.per_batch {
            flat.extend_from_slice(batch);
        }

        let out = if let Some((compiled, cached_masks)) = self.compiled_predictor.as_mut() {
            if cached_masks == masks {
                let mut outputs = compiled.run(&[("encoder", flat.as_slice())]);
                let tokens = outputs
                    .pop()
                    .ok_or_else(|| anyhow!("vjepa2 predictor graph returned no output"))?;
                let num_target = masks.target.len();
                crate::Vjepa2PredictorOutput {
                    tokens,
                    num_target,
                    hidden: enc.hidden,
                }
            } else {
                predict_native(&flat, pred, &self.cfg, self.batch, enc.seq, masks)?
            }
        } else {
            predict_native(&flat, pred, &self.cfg, self.batch, enc.seq, masks)?
        };
        let per = out.num_target * out.hidden;
        let mut per_batch = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            per_batch.push(out.tokens[b * per..(b + 1) * per].to_vec());
        }
        Ok(Vjepa2PredictOutput {
            per_batch,
            num_target: out.num_target,
            hidden: out.hidden,
        })
    }

    /// Attentive pooler (+ classifier when present) on encoder tokens.
    pub fn pool(&self, enc: &Vjepa2Output) -> Result<Vjepa2PoolOutput> {
        use crate::pool_native;

        let pooler = self
            .model
            .pooler
            .as_ref()
            .ok_or_else(|| anyhow!("checkpoint has no pooler weights"))?;
        let mut flat = Vec::with_capacity(enc.per_batch.len() * enc.seq * enc.hidden);
        for batch in &enc.per_batch {
            flat.extend_from_slice(batch);
        }

        let out = if let Some(compiled) = &self.compiled_pooler {
            let mut compiled = compiled.clone();
            let mut outputs = compiled.run(&[("encoder", flat.as_slice())]);
            anyhow::ensure!(
                !outputs.is_empty(),
                "vjepa2 pooler graph returned no embedding"
            );
            let embedding = outputs.remove(0);
            let logits = outputs.pop();
            crate::Vjepa2PoolerOutput { embedding, logits }
        } else {
            pool_native(&flat, pooler, &self.cfg, self.batch, enc.seq)?
        };
        Ok(Vjepa2PoolOutput {
            embedding: out.embedding,
            logits: out.logits,
        })
    }
}
