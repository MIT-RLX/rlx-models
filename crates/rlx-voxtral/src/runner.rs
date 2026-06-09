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

//! Voxtral end-to-end runner — mel → audio encoder → projector → fused Llama decode.

use crate::audio::{MelSpectrogram, pcm_to_mel_and_prompt};
use crate::config::VoxtralConfig;
use crate::embed::{argmax_token, fuse_inputs_embeds, validate_prompt_audio_match};
use crate::encoder::build_voxtral_encoder_built;
use crate::lm_flow::{build_voxtral_decode_built, build_voxtral_prefill_built};
use crate::load::{VoxtralWeightStore, resolve_model_dir};
use crate::projector::build_voxtral_projector_built;
use crate::weights::VoxtralWeightPrefix;
use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::compile_built;
use rlx_core::validate_standard_device;
use rlx_llama32::rope::{resolve_inv_freq, rope_slice};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct VoxtralRunnerBuilder {
    weights: Option<PathBuf>,
    config_path: Option<PathBuf>,
    config: Option<VoxtralConfig>,
    device: Option<Device>,
    max_new_tokens: usize,
}

impl VoxtralRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn config(mut self, cfg: VoxtralConfig) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    pub fn build(self) -> Result<VoxtralRunner> {
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?;
        let model_dir = resolve_model_dir(&weights_path)?;
        let cfg_path = self
            .config_path
            .clone()
            .unwrap_or_else(|| model_dir.join("config.json"));
        let cfg = match self.config {
            Some(c) => c,
            None => VoxtralConfig::from_file(&cfg_path)
                .with_context(|| format!("reading {cfg_path:?}"))?,
        };
        cfg.validate()?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("voxtral", device)?;
        let max_new_tokens = if self.max_new_tokens == 0 {
            256
        } else {
            self.max_new_tokens
        };
        let weight_store = VoxtralWeightStore::open(&weights_path)?;

        Ok(VoxtralRunner {
            cfg,
            device,
            max_new_tokens,
            weight_store,
        })
    }
}

pub struct VoxtralRunner {
    cfg: VoxtralConfig,
    device: Device,
    max_new_tokens: usize,
    weight_store: VoxtralWeightStore,
}

impl VoxtralRunner {
    pub fn builder() -> VoxtralRunnerBuilder {
        VoxtralRunnerBuilder::default()
    }

    pub fn config(&self) -> &VoxtralConfig {
        &self.cfg
    }

    pub fn model_dir(&self) -> &Path {
        self.weight_store.model_dir()
    }

    pub fn encode_audio(&self, mel: &MelSpectrogram) -> Result<Vec<f32>> {
        let batch = 1;
        let mel_frames = mel.n_frames;
        let enc_seq = self.cfg.audio_config.encoder_seq_len(mel_frames);
        ensure!(
            enc_seq.is_multiple_of(4),
            "encoder seq {enc_seq} not divisible by 4 — pad mel to a compatible length"
        );

        let mut wm = self.weight_store.load_audio_weights()?;
        let enc_built =
            build_voxtral_encoder_built(&self.cfg.audio_config, &mut wm, batch, mel_frames)?;
        let enc_params = enc_built.params().clone();
        let mut enc = compile_built(enc_built, self.device)?;
        for (n, d) in &enc_params {
            enc.set_param(n, d);
        }
        let enc_out = enc
            .run(&[("mel", mel.data.as_slice())])
            .into_iter()
            .next()
            .context("encoder output")?;
        drop(wm);

        let mut wm2 = self.weight_store.load_projector_weights()?;
        let proj_built = build_voxtral_projector_built(&self.cfg, &mut wm2, batch, enc_seq)?;
        let proj_params = proj_built.params().clone();
        let mut proj = compile_built(proj_built, self.device)?;
        for (n, d) in &proj_params {
            proj.set_param(n, d);
        }
        let audio_embeds = proj
            .run(&[("encoder_hidden", &enc_out)])
            .into_iter()
            .next()
            .context("projector output")?;
        Ok(audio_embeds)
    }

    pub fn generate(&self, prompt_ids: &[u32], mel: &MelSpectrogram) -> Result<Vec<u32>> {
        let batch = 1;
        let audio_embeds = self.encode_audio(mel)?;
        let h = self.cfg.text_config.hidden_size;
        let n_audio = audio_embeds.len() / h;
        validate_prompt_audio_match(&self.cfg, prompt_ids, n_audio)?;

        let embed_wm = self
            .weight_store
            .load_keys(&[VoxtralWeightPrefix::lm_embed_tokens()])?;
        let inputs_embeds = fuse_inputs_embeds(&self.cfg, &embed_wm, prompt_ids, &audio_embeds)?;
        drop(embed_wm);
        let seq = prompt_ids.len();

        let mut wm = self.weight_store.load_language_model_weights()?;
        let prefill_built =
            build_voxtral_prefill_built(&self.cfg, &mut wm, batch, seq, true, true)?;
        let prefill_params = prefill_built.params().clone();
        let mut prefill = compile_built(prefill_built, self.device)?;
        for (n, d) in &prefill_params {
            prefill.set_param(n, d);
        }

        let pre_in = [("inputs_embeds", inputs_embeds.as_slice())];
        let outs = prefill.run(&pre_in);
        let logits = &outs[0];
        let vocab = self.cfg.text_config.vocab_size;
        let mut tokens: Vec<u32> = prompt_ids.to_vec();
        ensure!(
            logits.len() == batch * vocab,
            "expected last-token logits [{batch}, {vocab}], got {}",
            logits.len()
        );
        let mut next = argmax_token(logits);

        let kv_start = 1usize;
        let mut kv_caches: Vec<Vec<f32>> = outs[kv_start..].to_vec();
        drop(prefill);

        let layers = self.cfg.text_config.num_hidden_layers;
        let key_past: Vec<String> = (0..layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();

        let llama = self.cfg.llama_config();
        let inv_freq = resolve_inv_freq(llama, None);

        for past_len in seq..seq.saturating_add(self.max_new_tokens) {
            if next == 0 {
                break;
            }
            tokens.push(next);

            let mut wm_dec = self.weight_store.load_language_model_weights()?;
            let dec_built =
                build_voxtral_decode_built(&self.cfg, &mut wm_dec, batch, past_len, false)?;
            let dec_params = dec_built.params().clone();
            let mut dec = compile_built(dec_built, self.device)?;
            for (n, d) in &dec_params {
                dec.set_param(n, d);
            }
            drop(wm_dec);

            let token_f = [next as f32];
            let (cos, sin) = rope_slice(&inv_freq, past_len);
            let mut dec_in: Vec<(&str, &[f32])> = vec![
                ("input_ids", &token_f),
                ("rope_cos", cos.as_slice()),
                ("rope_sin", sin.as_slice()),
            ];
            for i in 0..layers {
                dec_in.push((key_past[2 * i].as_str(), kv_caches[2 * i].as_slice()));
                dec_in.push((
                    key_past[2 * i + 1].as_str(),
                    kv_caches[2 * i + 1].as_slice(),
                ));
            }
            let step_out = dec.run(&dec_in);
            next = argmax_token(&step_out[0]);
            kv_caches = step_out[1..].to_vec();
        }

        Ok(tokens)
    }

    pub fn transcribe_wav(&self, wav: &Path, language: Option<&str>) -> Result<Vec<u32>> {
        let (mel, prompt) = pcm_to_mel_and_prompt(self.model_dir(), Some(wav), language)?;
        self.generate(&prompt, &mel)
    }
}
