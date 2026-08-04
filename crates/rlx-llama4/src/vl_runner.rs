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

//! Llama-4 vision-language runner (early fusion). Vision features are prepended
//! to the text token embeddings and the decoder runs on `inputs_embeds`.
//!
//! v1 places the image at the start of the sequence (image feature rows, then
//! prompt token embeddings) rather than matching `<|image|>` placeholders, and
//! decodes with no KV-cache (pad-to-max, host lm_head via the graph's own head).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};

use crate::config::{Llama4TextConfig, Llama4VisionConfig};
use crate::flow::build_llama4_text_flow;
use crate::preprocess::{Llama4VisionStem, extract_vision_stem};
use crate::rope::{build_rope_tables, build_vision_rope_tables};
use crate::vision::build_llama4_vision_flow;

type Raw = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub struct Llama4VlRunner {
    text_cfg: Llama4TextConfig,
    vision_cfg: Llama4VisionConfig,
    device: Device,

    text_wm: Option<WeightMap>, // model.layers.* + model.norm + lm_head (no embed_tokens)
    embed_tokens: Vec<f32>,     // [vocab, hidden]
    vision_stem: Llama4VisionStem,
    vision_raw: Raw, // vision transformer + adapter + projector
    n_img_tokens: usize,

    vision_compiled: Option<CompiledGraph>,
    text_compiled: Option<(usize, CompiledGraph)>,
    rope_text: HashMap<usize, (Vec<f32>, Vec<f32>)>,
    rope_vision: (Vec<f32>, Vec<f32>),
}

impl Llama4VlRunner {
    pub fn from_checkpoint(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg_text = Llama4TextConfig::from_file(dir.join("config.json"))?;
        let cfg_vision = load_vision_config(&dir.join("config.json"))?;

        let mut full = WeightMap::from_safetensors_dir(dir)
            .with_context(|| format!("loading llama4 safetensors from {}", dir.display()))?;

        let mut text_t: Raw = HashMap::new();
        let mut vision_raw: Raw = HashMap::new();
        let mut embed: Option<Vec<f32>> = None;
        for k in full.remaining_keys() {
            let (d, s) = full.take(&k)?;
            let tk = k
                .strip_prefix("language_model.")
                .map(String::from)
                .unwrap_or_else(|| k.clone());
            if tk == "model.embed_tokens.weight" {
                embed = Some(d);
            } else if tk.starts_with("model.") || tk == "lm_head.weight" {
                text_t.insert(tk, (d, s));
            } else if k.starts_with("vision_model.") || k.starts_with("multi_modal_projector.") {
                vision_raw.insert(k, (d, s));
            }
        }
        let embed_tokens =
            embed.ok_or_else(|| anyhow!("checkpoint missing model.embed_tokens.weight"))?;

        // Vision stem (host) — consumes patch_embedding/class/pos from vision_raw.
        let mut vwm = WeightMap::from_tensors(vision_raw);
        let vision_stem = extract_vision_stem(&mut vwm, &cfg_vision)?;
        let mut vision_after: Raw = HashMap::new();
        for k in vwm.remaining_keys() {
            let (d, s) = vwm.take(&k)?;
            vision_after.insert(k, (d, s));
        }

        let grid = cfg_vision.image_size / cfg_vision.patch_size;
        let ps_out = (grid as f64 * cfg_vision.pixel_shuffle_ratio as f64) as usize;
        let n_img_tokens = ps_out * ps_out;
        let rope_vision = build_vision_rope_tables(
            cfg_vision.image_size,
            cfg_vision.patch_size,
            cfg_vision.hidden_size,
            cfg_vision.num_attention_heads,
            cfg_vision.rope_theta(),
        );

        Ok(Self {
            text_cfg: cfg_text,
            vision_cfg: cfg_vision,
            device,
            text_wm: Some(WeightMap::from_tensors(text_t)),
            embed_tokens,
            vision_stem,
            vision_raw: vision_after,
            n_img_tokens,
            vision_compiled: None,
            text_compiled: None,
            rope_text: HashMap::new(),
            rope_vision,
        })
    }

    fn encode_image(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        let hidden = self.vision_stem.preprocess(rgb, w, h)?;
        if self.vision_compiled.is_none() {
            let mut wm = WeightMap::from_tensors(self.vision_raw.clone());
            let built =
                build_llama4_vision_flow(&self.vision_cfg, &mut wm, self.text_cfg.hidden_size)?;
            self.vision_compiled = Some(compile_built(built, self.device)?);
        }
        let (cos, sin) = &self.rope_vision;
        let compiled = self.vision_compiled.as_mut().unwrap();
        compiled
            .run(&[
                ("hidden", hidden.as_slice()),
                ("v_rope_cos", cos.as_slice()),
                ("v_rope_sin", sin.as_slice()),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("vision forward returned no output"))
    }

    fn ensure_text_graph(&mut self, max_len: usize) -> Result<()> {
        if self.text_compiled.as_ref().map(|(l, _)| *l) == Some(max_len) {
            return Ok(());
        }
        let mut wm = self.text_wm.take().ok_or_else(|| {
            anyhow!("llama4 text weights already consumed (one max_len per session)")
        })?;
        let built = build_llama4_text_flow(&self.text_cfg, &mut wm, max_len, true, true)?;
        self.text_compiled = Some((max_len, compile_built(built, self.device)?));
        self.rope_text.entry(max_len).or_insert_with(|| {
            build_rope_tables(
                self.text_cfg.head_dim(),
                self.text_cfg.rope_theta(),
                max_len,
            )
        });
        Ok(())
    }

    fn embed_row(&self, token: u32, out: &mut [f32]) {
        let hidden = self.text_cfg.hidden_size;
        let base = token as usize * hidden;
        out.copy_from_slice(&self.embed_tokens[base..base + hidden]);
    }

    /// Generate from a text prompt + one image (prepended). Returns new tokens.
    pub fn generate_multimodal(
        &mut self,
        prompt_ids: &[u32],
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        max_new: usize,
        eos: Option<u32>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let hidden = self.text_cfg.hidden_size;
        let vocab = self.text_cfg.vocab_size;
        let img_feats = self.encode_image(rgb, img_w, img_h)?; // [n_img * hidden]
        let n_img = self.n_img_tokens;

        let prompt_len = n_img + prompt_ids.len();
        let max_len = prompt_len + max_new;
        self.ensure_text_graph(max_len)?;

        // Assemble inputs_embeds: [image features | prompt token embeddings | pad].
        let mut embeds = vec![0f32; max_len * hidden];
        embeds[..n_img * hidden].copy_from_slice(&img_feats[..n_img * hidden]);
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let dst = (n_img + i) * hidden;
            self.embed_row(tok, &mut embeds[dst..dst + hidden]);
        }

        let (cos, sin) = self.rope_text.get(&max_len).unwrap();
        let compiled = &mut self.text_compiled.as_mut().unwrap().1;

        let mut generated = Vec::new();
        for i in 0..max_new {
            let real = prompt_len + i;
            let out = compiled
                .run(&[
                    ("inputs_embeds", embeds.as_slice()),
                    ("rope_cos", cos.as_slice()),
                    ("rope_sin", sin.as_slice()),
                ])
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("text forward returned no output"))?;
            let row = &out[(real - 1) * vocab..real * vocab];
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (v, &x) in row.iter().enumerate() {
                if x > best_v {
                    best_v = x;
                    best = v;
                }
            }
            let next = best as u32;
            generated.push(next);
            if !on_token(next) || Some(next) == eos || real >= max_len {
                break;
            }
            // Append the next token's embedding and advance.
            let dst = real * hidden;
            let base = next as usize * hidden;
            embeds[dst..dst + hidden].copy_from_slice(&self.embed_tokens[base..base + hidden]);
        }
        Ok(generated)
    }
}

fn load_vision_config(config_json: &Path) -> Result<Llama4VisionConfig> {
    #[derive(serde::Deserialize)]
    struct Wrap {
        vision_config: Option<Llama4VisionConfig>,
    }
    let text = std::fs::read_to_string(config_json)
        .with_context(|| format!("reading {}", config_json.display()))?;
    if let Ok(w) = serde_json::from_str::<Wrap>(&text) {
        if let Some(v) = w.vision_config {
            return Ok(v);
        }
    }
    serde_json::from_str(&text).context("parsing llama4 vision config")
}
