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

use crate::audio::{MelSpectrogram, load_wav_mono_f32, pcm_to_mel};
use crate::config::VoxtralConfig;
use crate::embed::{
    argmax_token, fuse_inputs_embeds, transcription_prompt_ids, validate_prompt_audio_match,
};
use crate::encoder::build_voxtral_encoder_built;
use crate::lm_flow::{build_voxtral_decode_built, build_voxtral_prefill_built};
use crate::load::{VoxtralWeightStore, resolve_model_dir};
use crate::projector::build_voxtral_projector_built;
use crate::weights::VoxtralWeightPrefix;
use anyhow::{Context, Result, ensure};
use rlx_core::compact_bucketed_kv_buffer;
use rlx_core::flow_util::compile_built;
use rlx_core::validate_standard_device;
use rlx_runtime::Device;
use rlx_runtime::attn_mask::bucket_decode_mask;
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
        let dbg = std::env::var("RLX_VOXTRAL_DEBUG").is_ok();
        let ck = |label: &str, v: &[f32]| {
            if dbg {
                let sum: f64 = v.iter().map(|&x| x as f64).sum();
                let sumsq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
                let finite = v.iter().filter(|x| x.is_finite()).count();
                eprintln!(
                    "[dbg] {label}: n={} sum={sum:.5} sumsq={sumsq:.5} finite={finite} first={:?}",
                    v.len(),
                    &v[..v.len().min(4)]
                );
            }
        };
        let audio_embeds = self.encode_audio(mel)?;
        let h = self.cfg.text_config.hidden_size;
        let n_audio = audio_embeds.len() / h;
        ck("audio_embeds", &audio_embeds);
        validate_prompt_audio_match(&self.cfg, prompt_ids, n_audio)?;

        let embed_wm = self
            .weight_store
            .load_keys(&[VoxtralWeightPrefix::lm_embed_tokens()])?;
        let inputs_embeds = fuse_inputs_embeds(&self.cfg, &embed_wm, prompt_ids, &audio_embeds)?;
        drop(embed_wm);
        let seq = prompt_ids.len();

        // Debug bisection: skip the `gather_last_token` tail so the prefill emits
        // logits for ALL positions — lets us compare cpu↔metal↔mlx and tell whether
        // a backend diverges in the layers vs the gather/head tail.
        let no_gather = std::env::var("RLX_VOXTRAL_NOGATHER").is_ok();
        let last_logits_only = !no_gather;
        // Debug: drop the per-layer KvTap side outputs (only valid with NOGATHER,
        // which bails before decode needs the KV cache) to test if the tap stage
        // is what corrupts the Metal layer output.
        let with_kv = std::env::var("RLX_VOXTRAL_NOKV").is_err();
        let mut wm = self.weight_store.load_language_model_weights()?;
        let prefill_built =
            build_voxtral_prefill_built(&self.cfg, &mut wm, batch, seq, with_kv, last_logits_only)?;
        // `build` copied the weights into the built graph; release the ~12 GB f32
        // weight map now. `compile_built` already moves the params out of the built
        // graph and streams them into the compiled model (freeing each tensor as it
        // uploads), so an explicit `.params().clone()` + `set_param` loop here would
        // just stack a redundant full-size copy — keep the unified-memory peak low.
        drop(wm);
        let mut prefill = compile_built(prefill_built, self.device)?;

        ck("inputs_embeds", &inputs_embeds);
        let pre_in = [("inputs_embeds", inputs_embeds.as_slice())];
        let outs = prefill.run(&pre_in);
        let logits = &outs[0];
        if no_gather {
            ck("prefill_all_logits", logits);
            anyhow::bail!("RLX_VOXTRAL_NOGATHER: stopped after full-sequence prefill checksum");
        }
        ck("prefill_logits", logits);
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
        let kv_dim = self.cfg.llama_config().kv_proj_dim();

        // Compile the decode graph ONCE at a fixed bucket cap and upload the
        // ~9 GB language-model weights ONCE. Each step pads the KV cache to
        // `upper` and supplies a per-step mask that zeros the padded slots, so
        // a single compiled graph serves every `past_len` — no per-token
        // reload/recompile (previously the dominant cost of generation).
        let upper = seq.saturating_add(self.max_new_tokens);
        let mut wm_dec = self.weight_store.load_language_model_weights()?;
        let dec_built =
            build_voxtral_decode_built(&self.cfg, &mut wm_dec, batch, upper, false, true)?;
        // Release the f32 weight map before compiling; `compile_built` moves the
        // params out of the built graph and streams them into the compiled model.
        drop(wm_dec);
        let mut dec = compile_built(dec_built, self.device)?;

        let mut padded_k = vec![vec![0f32; upper * kv_dim]; layers];
        let mut padded_v = vec![vec![0f32; upper * kv_dim]; layers];

        let eos = self.cfg.eos_token_id;
        for past_len in seq..upper {
            // Stop at end-of-stream so the transcript doesn't ramble past the
            // utterance (token 0 = <unk> guard; eos = tekken </s>).
            if next == 0 || next == eos {
                break;
            }
            tokens.push(next);

            let token_f = [next as f32];
            let position = [past_len as f32];
            let mask = bucket_decode_mask(past_len, upper);

            // Place the compact past KV into the fixed-size padded buffers.
            for i in 0..layers {
                let kf = &kv_caches[2 * i];
                let vf = &kv_caches[2 * i + 1];
                padded_k[i][..kf.len()].copy_from_slice(kf);
                padded_v[i][..vf.len()].copy_from_slice(vf);
            }

            let mut dec_in: Vec<(&str, &[f32])> = vec![
                ("input_ids", &token_f),
                ("position", &position),
                ("mask", mask.as_slice()),
            ];
            for i in 0..layers {
                dec_in.push((key_past[2 * i].as_str(), padded_k[i].as_slice()));
                dec_in.push((key_past[2 * i + 1].as_str(), padded_v[i].as_slice()));
            }
            let step_out = dec.run(&dec_in);
            next = argmax_token(&step_out[0]);

            // Graph emits K/V at length `upper + 1` (padded past + new row at
            // index `upper`); compact back to `past_len + 1` dense rows.
            let new_len = past_len + 1;
            kv_caches = step_out[1..]
                .iter()
                .map(|buf| compact_bucketed_kv_buffer(buf, new_len, kv_dim, batch))
                .collect();
        }

        Ok(tokens)
    }

    /// Transcribe a 16 kHz mono WAV fully natively: Rust log-mel + Mistral
    /// transcription prompt → RLX encoder/projector/decoder. No Python.
    pub fn transcribe_wav(&self, wav: &Path, language: Option<&str>) -> Result<Vec<u32>> {
        let pcm = load_wav_mono_f32(wav)?;
        let mel = pcm_to_mel(&self.cfg.audio_config, &pcm)?;
        let n_audio = self.cfg.audio_config.audio_token_count(mel.n_frames);
        let prompt =
            transcription_prompt_ids(&self.cfg, n_audio, language, Some(self.model_dir()))?;
        self.generate(&prompt, &mel)
    }
}
