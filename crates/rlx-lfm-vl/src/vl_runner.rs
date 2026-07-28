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

//! Combined LFM2.5-VL runner — LFM LM + SigLIP2 mmproj vision.

use crate::runner::LfmVlVisionRunner;
use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_cli::LmRunner;
use rlx_lfm::LfmRunner;
use rlx_qwen35::encode_prompt_auto;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

pub const MEDIA_MARKER: &str = "<__media__>";
pub const IMAGE_MARKER: &str = "<image>";

#[derive(Debug, Clone, Default)]
pub struct LfmVlRunnerBuilder {
    weights: Option<PathBuf>,
    mmproj: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    device: Option<Device>,
}

impl LfmVlRunnerBuilder {
    pub fn weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn mmproj(mut self, p: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(p.into());
        self
    }
    pub fn hf_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<LfmVlRunner> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow!("LfmVlRunner: .weights(...) required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let mut lm_b = LfmRunner::builder().weights(&weights).device(device);
        if let Some(hf) = &self.hf_config {
            lm_b = lm_b.hf_config(hf);
        }
        let lm = lm_b
            .build()
            .with_context(|| format!("LfmVlRunner: load LM {weights:?}"))?;

        let vision = match self.mmproj {
            Some(mp) => {
                let mut vb = LfmVlVisionRunner::builder().mmproj(&mp).device(device);
                let hf = self.hf_config.clone().or_else(|| {
                    weights
                        .parent()
                        .map(|p| p.join("config.json"))
                        .filter(|p| p.is_file())
                        .or_else(|| {
                            mp.parent()
                                .map(|p| p.join("config.json"))
                                .filter(|p| p.is_file())
                        })
                });
                if let Some(cfg) = hf {
                    vb = vb.hf_config(cfg);
                } else {
                    vb = vb.config(crate::config::LfmVlVisionConfig {
                        hidden_size: 1152,
                        num_hidden_layers: 27,
                        num_attention_heads: 16,
                        intermediate_size: 4304,
                        image_size: 384,
                        patch_size: 14,
                        num_channels: 3,
                        layer_norm_eps: 1e-6,
                        projector_output_dim: lm.config().hidden_size,
                    });
                }
                Some(
                    vb.build()
                        .with_context(|| format!("LfmVlRunner: load mmproj {mp:?}"))?,
                )
            }
            None => None,
        };

        Ok(LfmVlRunner {
            lm,
            vision,
            weights_path: weights,
        })
    }
}

pub struct LfmVlRunner {
    lm: LfmRunner,
    vision: Option<LfmVlVisionRunner>,
    weights_path: PathBuf,
}

impl LfmVlRunner {
    pub fn builder() -> LfmVlRunnerBuilder {
        LfmVlRunnerBuilder::default()
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    fn encode_rgb(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<(Vec<f32>, usize)> {
        let enc = self
            .vision
            .as_mut()
            .ok_or_else(|| anyhow!("LfmVlRunner: encode requires .mmproj(...)"))?;
        let n_embd = enc.config().projector_output_dim;
        let tmp = write_rgb_temp_png(rgb, w, h)?;
        let flat = enc.embed_image_path(&tmp)?;
        let _ = std::fs::remove_file(&tmp);
        ensure!(
            flat.len() % n_embd == 0,
            "vision embeds len {} not divisible by {n_embd}",
            flat.len()
        );
        let n_tokens = flat.len() / n_embd;
        Ok((flat, n_tokens))
    }

    fn assemble_hidden(
        &mut self,
        prompt: &str,
        vision_embd: &[f32],
        n_vision: usize,
        tokenizer: Option<&Path>,
    ) -> Result<Vec<f32>> {
        let n_embd = self.lm.config().hidden_size;
        ensure!(
            vision_embd.len() == n_vision * n_embd,
            "vision dim mismatch: {} vs {}×{}",
            vision_embd.len(),
            n_vision,
            n_embd
        );
        let prompt = normalize_prompt(prompt);
        let parts: Vec<&str> = prompt.split(MEDIA_MARKER).collect();
        ensure!(
            parts.len() == 2,
            "prompt must contain exactly one `{MEDIA_MARKER}`"
        );
        let weights = self.weights_path.clone();
        let tok = |s: &str| -> Result<Vec<u32>> { encode_prompt_auto(&weights, tokenizer, s) };
        let before = tok(parts[0])?;
        let after = tok(parts[1])?;
        // Optional image marker tokens around vision span.
        let start = tok(IMAGE_MARKER).unwrap_or_default();

        let emb = self.lm.token_embd_table()?.to_vec();
        let vocab = emb.len() / n_embd;

        let mut seq = Vec::new();
        seq.extend_from_slice(&before);
        seq.extend_from_slice(&start);
        let vision_start = seq.len();
        seq.extend(std::iter::repeat(0u32).take(n_vision));
        seq.extend_from_slice(&after);

        let mut hidden = Vec::with_capacity(seq.len() * n_embd);
        for &tok_id in &seq {
            let row = tok_id as usize;
            ensure!(row < vocab, "token {tok_id} out of vocab {vocab}");
            let off = row * n_embd;
            hidden.extend_from_slice(&emb[off..off + n_embd]);
        }
        for t in 0..n_vision {
            let dst = (vision_start + t) * n_embd;
            let src = t * n_embd;
            hidden[dst..dst + n_embd].copy_from_slice(&vision_embd[src..src + n_embd]);
        }
        Ok(hidden)
    }

    pub fn generate_multimodal_rgb(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        if self.vision.is_none() {
            bail!("LfmVlRunner: generate_multimodal requires .mmproj(...)");
        }
        let (vision_embd, n_vision) = self.encode_rgb(rgb, img_w, img_h)?;
        let n_embd = self.lm.config().hidden_size;
        let proj = self.vision.as_ref().unwrap().config().projector_output_dim;
        ensure!(
            proj == n_embd,
            "LfmVlRunner: projector_output_dim {proj} != LM hidden {n_embd}"
        );
        let hidden = self.assemble_hidden(prompt, &vision_embd, n_vision, tokenizer)?;
        let seq = hidden.len() / n_embd;
        let mut logits = self.lm.prefill_from_embeds(&hidden, seq)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax_u32(&logits);
            out.push(next);
            if !on_token(next) {
                break;
            }
            logits = self.lm.step(next);
        }
        Ok(out)
    }
}

impl LmRunner for LfmVlRunner {
    fn family(&self) -> &'static str {
        "lfm-vl"
    }
    fn vocab_size(&self) -> usize {
        self.lm.config().vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.lm.predict_logits(prompt_ids)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        // Use LmRunner impl on LfmRunner (honours stop signal).
        LmRunner::generate(&mut self.lm, prompt_ids, n_new, on_token)
    }
    fn supports_multimodal(&self) -> bool {
        self.has_vision()
    }
    fn generate_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_multimodal_rgb(prompt, rgb, img_w, img_h, tokenizer, n_new, on_token)
    }
}

fn normalize_prompt(prompt: &str) -> String {
    if prompt.contains(MEDIA_MARKER) {
        return prompt.to_string();
    }
    // Also accept <image> as the media marker.
    if prompt.contains(IMAGE_MARKER) {
        return prompt.replacen(IMAGE_MARKER, MEDIA_MARKER, 1);
    }
    let mut p = prompt.to_string();
    if !p.is_empty() && !p.ends_with(|c: char| c.is_whitespace()) {
        p.push(' ');
    }
    p.push_str(MEDIA_MARKER);
    p
}

fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

fn write_rgb_temp_png(rgb: &[u8], w: usize, h: usize) -> Result<PathBuf> {
    use image::{ImageBuffer, Rgb};
    ensure!(
        rgb.len() == w.saturating_mul(h).saturating_mul(3),
        "rgb len {} != {w}×{h}×3",
        rgb.len()
    );
    let img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(w as u32, h as u32, rgb.to_vec())
        .ok_or_else(|| anyhow!("failed to wrap rgb"))?;
    let path = std::env::temp_dir().join(format!(
        "rlx-lfmvl-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    img.save(&path)?;
    Ok(path)
}
