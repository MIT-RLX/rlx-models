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

//! Llama-3.2-Vision runner: vision tower + cross-attention Llama-3.2 decoder.
//!
//! v1 uses a full-sequence (no KV-cache) decode: the text graph is compiled once
//! at `max_len = prompt + max_new`, and each step re-runs the padded sequence and
//! reads logits at the true last position (self-attention is causal, so trailing
//! pad positions never affect it). Correct but O(L²); optimizing to a KV cache
//! (self layers cached, cross K/V cached once) is future work.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, Shape};
use rlx_llama32::{Llama32Config, Llama32Flow, Llama32RopeScaling, Llama32RopeType};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::MllamaConfig;
use crate::cross_attn::{CROSS_STATES_INPUT, CrossAttnDims, cross_attn_stage};
use crate::preprocess::{VisionEmbedWeights, VisionInputs, extract_vision_embed_weights};
use crate::vision::build_vision_flow;

type Raw = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// A loaded, compiled Llama-3.2-Vision model.
pub struct MllamaRunner {
    cfg: MllamaConfig,
    text_cfg: Llama32Config,
    device: Device,

    vis_embed: VisionEmbedWeights,
    vis_raw: Raw,               // vision transformer + projector weights
    text_wm: Option<WeightMap>, // model.* text weights, consumed on first text build
    lm_head: Vec<f32>,          // [vocab, hidden]

    vision_cache: HashMap<usize, CompiledGraph>, // by num_tiles
    text_cache: HashMap<(usize, usize), CompiledGraph>, // by (kv_seq, max_len)
}

impl MllamaRunner {
    /// Load the checkpoint directory (HF safetensors + `config.json`).
    pub fn from_checkpoint(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = MllamaConfig::from_file(dir.join("config.json"))?;

        let mut full = WeightMap::from_safetensors_dir(dir)
            .with_context(|| format!("loading mllama safetensors from {}", dir.display()))?;

        // Partition: vision (vision_model.* + multi_modal_projector.*) vs text
        // (language_model.model.* -> model.*, language_model.lm_head.* -> lm_head.*).
        let mut vis_raw: Raw = HashMap::new();
        let mut text_t: Raw = HashMap::new();
        for k in full.remaining_keys() {
            let (data, shape) = full.take(&k)?;
            if let Some(r) = k.strip_prefix("language_model.model.") {
                text_t.insert(format!("model.{r}"), (data, shape));
            } else if let Some(r) = k.strip_prefix("language_model.lm_head.") {
                text_t.insert(format!("lm_head.{r}"), (data, shape));
            } else if k.starts_with("vision_model.") || k.starts_with("multi_modal_projector.") {
                vis_raw.insert(k, (data, shape));
            } else {
                // Some checkpoints already flatten language_model.model -> model.
                text_t.insert(k, (data, shape));
            }
        }

        // Host-side vision stem (patch embed + tile/position tables).
        let mut vwm = WeightMap::from_tensors(vis_raw);
        let vis_embed = extract_vision_embed_weights(&mut vwm, &cfg.vision_config)?;
        // Keep the remaining vision weights (transformer + projector) for per-tiling builds.
        let mut vis_after: Raw = HashMap::new();
        for k in vwm.remaining_keys() {
            let (d, s) = vwm.take(&k)?;
            vis_after.insert(k, (d, s));
        }

        // Separate the (untied) LM head; the text graph itself is hidden-only.
        let (lm_head, _) = text_t
            .remove("lm_head.weight")
            .ok_or_else(|| anyhow!("mllama checkpoint missing language_model.lm_head.weight"))?;
        let text_wm = WeightMap::from_tensors(text_t);
        let text_cfg = to_llama32_config(&cfg);

        Ok(Self {
            cfg,
            text_cfg,
            device,
            vis_embed,
            vis_raw: vis_after,
            text_wm: Some(text_wm),
            lm_head,
            vision_cache: HashMap::new(),
            text_cache: HashMap::new(),
        })
    }

    pub fn config(&self) -> &MllamaConfig {
        &self.cfg
    }

    /// Preprocess + run the vision tower only (no text decode). Returns the
    /// projected cross-attention states and `(num_tiles, num_patches, hidden)`.
    /// For parity against HF `MllamaVisionModel` + `multi_modal_projector`.
    pub fn vision_features(
        &mut self,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
    ) -> Result<(Vec<f32>, usize, usize, usize)> {
        let vi = self.vis_embed.preprocess(rgb, img_w, img_h)?;
        let nt = vi.num_tiles;
        let cs = self.encode_vision(&vi)?;
        Ok((
            cs,
            nt,
            self.cfg.vision_config.num_patches(),
            self.cfg.text_config.hidden_size,
        ))
    }

    /// Compile (if needed) and run the vision tower for the image's tiling,
    /// returning the projected cross-attention states `[seq * hidden]`.
    fn encode_vision(&mut self, vi: &VisionInputs) -> Result<Vec<f32>> {
        let nt = vi.num_tiles;
        if !self.vision_cache.contains_key(&nt) {
            let mut wm = WeightMap::from_tensors(self.vis_raw.clone());
            let built = build_vision_flow(
                &self.cfg.vision_config,
                &mut wm,
                self.cfg.text_config.hidden_size,
                nt,
            )?;
            let compiled = compile_built(built, self.device)?;
            self.vision_cache.insert(nt, compiled);
        }
        let compiled = self.vision_cache.get_mut(&nt).unwrap();
        compiled
            .run(&[
                ("hidden", vi.hidden.as_slice()),
                ("post_tile", vi.post_tile.as_slice()),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("vision forward returned no output"))
    }

    /// Compile (if needed) the cross-attention text graph for `(kv_seq, max_len)`.
    fn ensure_text_graph(&mut self, kv_seq: usize, max_len: usize) -> Result<()> {
        let key = (kv_seq, max_len);
        if self.text_cache.contains_key(&key) {
            return Ok(());
        }
        let mut wm = self.text_wm.take().ok_or_else(|| {
            anyhow!("mllama text weights already consumed (one tiling/len per session in v1)")
        })?;

        let tcfg = &self.text_cfg;
        let cross_layers = self.cfg.text_config.cross_attention_layers.clone();
        let d = CrossAttnDims {
            hidden: tcfg.hidden_size,
            num_heads: tcfg.num_attention_heads,
            num_kv_heads: tcfg.num_key_value_heads,
            head_dim: tcfg.head_dim(),
            eps: tcfg.rms_norm_eps as f32,
            text_seq: max_len,
            kv_seq,
        };
        let hidden = tcfg.hidden_size;
        let kv_shape = Shape::new(&[1, kv_seq, hidden], DType::F32);

        let built = Llama32Flow::new(tcfg)
            .prefill()
            .batch(1)
            .seq(max_len)
            .hidden_only()
            .layer(move |ctx| {
                if cross_layers.contains(&ctx.index()) {
                    cross_attn_stage(ctx.weight_index(), d)
                } else {
                    ctx.default_stage()
                }
            })
            .patch_flow(move |flow| flow.input(CROSS_STATES_INPUT, kv_shape.clone()))
            .build(&mut wm)?;
        let compiled = compile_built(built, self.device)?;
        self.text_cache.insert(key, compiled);
        Ok(())
    }

    /// Generate from a tokenized prompt + one RGB image (HWC `u8`).
    /// Returns the newly generated token ids.
    pub fn generate_multimodal_ids(
        &mut self,
        prompt_ids: &[u32],
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        max_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let vi = self.vis_embed.preprocess(rgb, img_w, img_h)?;
        let cross_states = self.encode_vision(&vi)?;
        let kv_seq = vi.seq;

        let prompt_len = prompt_ids.len();
        let max_len = prompt_len + max_new;
        self.ensure_text_graph(kv_seq, max_len)?;

        let hidden = self.text_cfg.hidden_size;
        let vocab = self.text_cfg.vocab_size;
        let eos = self.cfg.text_config.eos_token_id;
        let compiled = self.text_cache.get_mut(&(kv_seq, max_len)).unwrap();
        let lm_head = self.lm_head.as_slice();

        let mut ids: Vec<u32> = prompt_ids.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let real_len = ids.len();
            if real_len > max_len {
                break;
            }
            let mut input_ids = vec![0f32; max_len];
            for (i, &t) in ids.iter().enumerate() {
                input_ids[i] = t as f32;
            }
            let out = compiled
                .run(&[
                    ("input_ids", input_ids.as_slice()),
                    (CROSS_STATES_INPUT, cross_states.as_slice()),
                ])
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("text forward returned no output"))?;

            let row = &out[(real_len - 1) * hidden..real_len * hidden];
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for v in 0..vocab {
                let w = &lm_head[v * hidden..(v + 1) * hidden];
                let mut acc = 0.0f32;
                for h in 0..hidden {
                    acc += row[h] * w[h];
                }
                if acc > best_v {
                    best_v = acc;
                    best = v;
                }
            }
            let next = best as u32;
            generated.push(next);
            if !on_token(next) || next == eos || ids.len() + 1 >= max_len {
                break;
            }
            ids.push(next);
        }
        Ok(generated)
    }
}

/// Map the mllama text config onto the shared `Llama32Config`.
fn to_llama32_config(cfg: &MllamaConfig) -> Llama32Config {
    let t = &cfg.text_config;
    let rope_scaling = t.rope_scaling.as_ref().map(|s| Llama32RopeScaling {
        factor: s.factor,
        low_freq_factor: s.low_freq_factor,
        high_freq_factor: s.high_freq_factor,
        original_max_position_embeddings: s.original_max_position_embeddings,
        rope_type: Llama32RopeType::Llama3,
    });
    Llama32Config {
        vocab_size: t.vocab_size,
        hidden_size: t.hidden_size,
        intermediate_size: t.intermediate_size,
        num_hidden_layers: t.num_hidden_layers,
        num_attention_heads: t.num_attention_heads,
        num_key_value_heads: t.num_key_value_heads,
        max_position_embeddings: t.max_position_embeddings,
        rms_norm_eps: t.rms_norm_eps as f64,
        rope_theta: t.rope_theta as f64,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        head_dim: None,
        rope_scaling,
        embedding_scale: None,
        residual_scale: None,
        attention_scale: None,
        logit_scale: None,
        num_loops: 1,
        skip_loop_final_norm: false,
        rope_style: rlx_ir::RopeStyle::NeoX,
        gguf_arch: None,
        rope_dim: None,
    }
}
