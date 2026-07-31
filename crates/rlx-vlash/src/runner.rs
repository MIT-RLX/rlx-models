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

//! High-level VLASH runner: load a checkpoint, compile the vision + denoise
//! graphs (for a fixed image count and prompt length), and predict action
//! chunks from images + robot state + a language instruction.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use rlx_runtime::{CompiledGraph, Device};
use rlx_siglip2::{VisionEmbedWeights, assemble_vision_hidden};

use crate::config::{VlashConfig, VlashVariant};
use crate::normalize::{Normalization, pad_to};
use crate::prefix::{assemble_prefix, build_attn_inputs};
use crate::sample::sample_actions;
use crate::tokenizer::PaligemmaTokenizer;
use crate::vision::{build_vision_flow, extract_vision_embed};

/// A compiled VLASH policy ready to predict action chunks.
pub struct VlashRunner {
    cfg: VlashConfig,
    device: Device,
    num_images: usize,
    prompt_tokens: usize,
    vision: CompiledGraph,
    denoise: CompiledGraph,
    vision_embed: VisionEmbedWeights,
    embed_tokens: Vec<f32>,
    vocab: usize,
    norm: Normalization,
    tokenizer: PaligemmaTokenizer,
}

/// Builder for [`VlashRunner`].
pub struct VlashRunnerBuilder {
    variant: VlashVariant,
    device: Device,
    num_images: usize,
    prompt_tokens: usize,
    model_dir: Option<PathBuf>,
}

impl VlashRunnerBuilder {
    pub fn new(variant: VlashVariant) -> Self {
        Self {
            variant,
            device: Device::Cpu,
            num_images: 1,
            prompt_tokens: 200,
            model_dir: None,
        }
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }
    /// Number of camera views concatenated into the prefix (fixed at compile).
    pub fn num_images(mut self, n: usize) -> Self {
        self.num_images = n;
        self
    }
    /// Fixed prompt token length (prompts are right-padded/truncated to this).
    pub fn prompt_tokens(mut self, n: usize) -> Self {
        self.prompt_tokens = n;
        self
    }
    pub fn model_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(p.into());
        self
    }

    pub fn build(self) -> Result<VlashRunner> {
        let dir = self
            .model_dir
            .ok_or_else(|| anyhow!("VlashRunnerBuilder requires model_dir"))?;
        VlashRunner::load(
            &dir,
            self.variant,
            self.device,
            self.num_images,
            self.prompt_tokens,
        )
    }
}

fn find_safetensors(dir: &Path) -> Result<PathBuf> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(single);
    }
    // Fall back to the first *.safetensors in the directory.
    for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            return Ok(p);
        }
    }
    Err(anyhow!("no .safetensors found in {}", dir.display()))
}

impl VlashRunner {
    pub fn builder(variant: VlashVariant) -> VlashRunnerBuilder {
        VlashRunnerBuilder::new(variant)
    }

    /// Download a checkpoint from the Hugging Face Hub (e.g. `lerobot/pi05_base`)
    /// and load it. Requires the `pipeline` feature.
    #[cfg(feature = "pipeline")]
    pub fn from_pretrained(
        repo: &str,
        variant: VlashVariant,
        device: Device,
        num_images: usize,
        prompt_tokens: usize,
    ) -> Result<Self> {
        use hf_hub::api::sync::Api;
        let api = Api::new()?.model(repo.to_string());
        // Resolve into the shared HF cache; both files land in the same dir.
        let st = api.get("model.safetensors").context("download model.safetensors")?;
        let _ = api.get("tokenizer.json").context("download tokenizer.json")?;
        let dir = st
            .parent()
            .ok_or_else(|| anyhow!("no parent dir for {}", st.display()))?;
        Self::load(dir, variant, device, num_images, prompt_tokens)
    }

    /// Load + compile from a checkpoint directory (containing `model.safetensors`
    /// and `tokenizer.json`).
    pub fn load(
        model_dir: &Path,
        variant: VlashVariant,
        device: Device,
        num_images: usize,
        prompt_tokens: usize,
    ) -> Result<Self> {
        let cfg = VlashConfig::for_variant(variant);
        let st = find_safetensors(model_dir)?;
        let mut wm = crate::weights::load_remapped(st.to_str().unwrap())?;

        // Host-consumed pieces (before the graph builders drain their keys).
        let vision_embed = extract_vision_embed(&mut wm, &cfg.vision)?;
        let (embed_tokens, embed_shape) = wm
            .take("vlm.embed_tokens.weight")
            .context("vlm.embed_tokens.weight missing")?;
        let vocab = embed_shape[0];
        let norm = Normalization::from_weight_map(&wm);

        // Compile the two graphs.
        let vision_built = build_vision_flow(&cfg.vision, &mut wm, num_images)?;
        let vision = rlx_core::flow_util::compile_built(vision_built, device)?;

        let prefix_len = num_images * cfg.vision.num_patches() + prompt_tokens;
        let denoise_built = crate::flow::build_denoise_flow(&cfg, &mut wm, prefix_len)?;
        let denoise = rlx_core::flow_util::compile_built(denoise_built, device)?;

        let tokenizer = PaligemmaTokenizer::from_dir(model_dir)?;

        Ok(Self {
            cfg,
            device,
            num_images,
            prompt_tokens,
            vision,
            denoise,
            vision_embed,
            embed_tokens,
            vocab,
            norm,
            tokenizer,
        })
    }

    pub fn config(&self) -> &VlashConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run the SigLIP tower over `num_images` NCHW-normalized images (each
    /// `[3·224·224]`, in `[-1,1]`), returning `[num_images·256·projection_dim]`.
    pub fn encode_images(&mut self, images_nchw: &[&[f32]]) -> Result<Vec<f32>> {
        if images_nchw.len() != self.num_images {
            return Err(anyhow!(
                "expected {} images, got {}",
                self.num_images,
                images_nchw.len()
            ));
        }
        let img = self.cfg.image_size;
        let ps = self.cfg.vision.patch_size;
        let mut concat = Vec::with_capacity(self.num_images * 3 * img * img);
        for im in images_nchw {
            concat.extend_from_slice(im);
        }
        let hidden = assemble_vision_hidden(&self.vision_embed, &concat, self.num_images, ps, img)?;
        Ok(self
            .vision
            .run(&[("hidden", hidden.as_slice())])
            .into_iter()
            .next()
            .expect("vision → image_features"))
    }

    /// Predict a full action chunk (`[chunk · raw_action_dim]`, unnormalized).
    ///
    /// `state_raw` is the raw (un-padded) robot state; `noise` (`[chunk ·
    /// max_action_dim]`) is injected when provided (parity) else sampled from a
    /// fixed-seed Gaussian.
    pub fn predict_action_chunk(
        &mut self,
        images_nchw: &[&[f32]],
        state_raw: &[f32],
        prompt: &str,
        noise: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let image_features = self.encode_images(images_nchw)?;
        let tok = self.tokenizer.encode(prompt, Some(self.prompt_tokens))?;

        let prefix = assemble_prefix(
            &image_features,
            self.num_images,
            self.cfg.vision.num_patches(),
            self.cfg.vlm.hidden,
            &self.embed_tokens,
            self.vocab,
            &tok.ids,
            &tok.mask,
        );
        let attn = build_attn_inputs(&self.cfg, &prefix.pad);

        // Normalize + pad state.
        let raw_dim = state_raw.len();
        let state_norm = self.norm.normalize_state(state_raw);
        let state = pad_to(&state_norm, raw_dim, self.cfg.max_state_dim)?;

        // Noise.
        let n = self.cfg.chunk_size * self.cfg.max_action_dim;
        let noise_vec = match noise {
            Some(x) => x.to_vec(),
            None => gaussian_noise(n, 0x5eed_1234),
        };

        let x_t = sample_actions(
            &mut self.denoise,
            &self.cfg,
            &prefix.emb,
            &state,
            &attn,
            &noise_vec,
        );

        // Truncate to raw action dim, unnormalize.
        let raw_action_dim = self.norm.action_dim().unwrap_or(self.cfg.max_action_dim);
        let mut out = Vec::with_capacity(self.cfg.chunk_size * raw_action_dim);
        for c in 0..self.cfg.chunk_size {
            let base = c * self.cfg.max_action_dim;
            out.extend_from_slice(&x_t[base..base + raw_action_dim]);
        }
        Ok(self.norm.unnormalize_action(&out))
    }
}

/// Deterministic xorshift + Box-Muller Gaussian noise (used when the caller
/// injects none). Production callers should pass their own RNG stream.
fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        if out.len() < n {
            out.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
        }
    }
    out
}
