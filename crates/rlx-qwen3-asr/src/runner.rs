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

//! Qwen3-ASR end-to-end runner — mel → audio encoder → fuse → Qwen3 decode.

use crate::audio::{AudioGeometry, MelSpectrogram, pcm_to_log_mel};
use crate::config::Qwen3AsrConfig;
use crate::embed::{argmax_token, count_audio_placeholders, fuse_inputs_embeds};
use crate::encoder::build_encoder_built;
use crate::lm_flow::{build_asr_decode_built_opts, build_asr_prefill_built, rope_slice};
use crate::load::{AsrWeightStore, resolve_model_dir};
use crate::weights::{
    KEY_EMBED_TOKENS, KEY_LM_HEAD, LanguageModelPrefixLoader, PREFIX_LANGUAGE_MODEL,
};
use anyhow::{Context, Result, ensure};
use rlx_core::autoregressive::{KvCacheState, kv_from_prefill_outputs, run_bucketed_kv_decode};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_flow::CompileProfile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::CompiledGraph;
use rlx_runtime::Device;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Persistent compiled-graph cache so repeated [`AsrRunner::transcribe_pcm`]
/// calls reuse identical-shape graphs (and their resident weights) instead of
/// recompiling + reloading every call. Keyed by shape: encoder by audio
/// geometry, prefill by prompt length, decode by its bucket ladder. This is
/// what turns a one-shot transcriber into a fast real-time/streaming loop.
#[derive(Default)]
struct GraphCaches {
    /// Audio encoder keyed by `(num_chunks, max_chunk_len)`.
    encoder: HashMap<(usize, usize), CompiledGraph>,
    /// Decoder prefill keyed by prompt `seq` length.
    prefill: HashMap<usize, CompiledGraph>,
    /// Bucketed decode cache, kept alive across calls; rebuilt only if the
    /// `seq + max_new_tokens` horizon (the ladder shape) changes.
    decode: Option<(u64, BucketedCompileCache)>,
}

/// Stop tokens: `<|endoftext|>` and `<|im_end|>` (generation_config eos ids).
const EOS_IDS: [u32; 2] = [151643, 151645];

#[derive(Debug, Clone, Default)]
pub struct AsrRunnerBuilder {
    weights: Option<PathBuf>,
    config_path: Option<PathBuf>,
    config: Option<Qwen3AsrConfig>,
    device: Option<Device>,
    max_new_tokens: usize,
}

impl AsrRunnerBuilder {
    pub fn weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn config_path(mut self, p: impl Into<PathBuf>) -> Self {
        self.config_path = Some(p.into());
        self
    }
    pub fn config(mut self, c: Qwen3AsrConfig) -> Self {
        self.config = Some(c);
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

    pub fn build(self) -> Result<AsrRunner> {
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?;
        let model_dir = resolve_model_dir(&weights_path)?;
        let cfg = match self.config {
            Some(c) => c,
            None => {
                let p = self
                    .config_path
                    .clone()
                    .unwrap_or_else(|| model_dir.join("config.json"));
                Qwen3AsrConfig::from_file(&p)?
            }
        };
        cfg.validate()?;
        let device = self.device.unwrap_or(Device::Cpu);
        let max_new_tokens = if self.max_new_tokens == 0 {
            440
        } else {
            self.max_new_tokens
        };
        let store = AsrWeightStore::open(&weights_path)?;
        Ok(AsrRunner {
            cfg,
            device,
            max_new_tokens,
            store,
            caches: RefCell::new(GraphCaches::default()),
        })
    }
}

pub struct AsrRunner {
    cfg: Qwen3AsrConfig,
    device: Device,
    max_new_tokens: usize,
    store: AsrWeightStore,
    /// Compiled graphs reused across calls (interior mutability keeps the
    /// `&self` API for one-shot callers like `examples/jfk.rs`).
    caches: RefCell<GraphCaches>,
}

impl AsrRunner {
    pub fn builder() -> AsrRunnerBuilder {
        AsrRunnerBuilder::default()
    }

    pub fn config(&self) -> &Qwen3AsrConfig {
        &self.cfg
    }

    pub fn model_dir(&self) -> &Path {
        self.store.model_dir()
    }

    /// Number of `<|audio_pad|>` placeholders this mel produces.
    pub fn num_audio_tokens(&self, mel: &MelSpectrogram) -> Result<usize> {
        Ok(AudioGeometry::new(&self.cfg.audio, mel.n_frames)?.num_audio_tokens)
    }

    /// Run the audio tower → `[num_audio_tokens * output_dim]` row-major.
    pub fn encode_audio(&self, mel: &MelSpectrogram) -> Result<(Vec<f32>, usize)> {
        ensure!(
            mel.n_mels == self.cfg.audio.num_mel_bins,
            "mel bins {} != configured {}",
            mel.n_mels,
            self.cfg.audio.num_mel_bins
        );
        let geom = AudioGeometry::new(&self.cfg.audio, mel.n_frames)?;
        let padded = pad_mel_time(mel, geom.num_chunks * geom.max_chunk_len);

        let key = (geom.num_chunks, geom.max_chunk_len);
        let mut caches = self.caches.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(e) = caches.encoder.entry(key) {
            let mut wm = self.store.load_audio_weights()?;
            let built = build_encoder_built(&self.cfg.audio, &mut wm, &geom)?;
            let params = built.params().clone();
            let mut compiled = compile_built(built, self.device)?;
            for (n, d) in &params {
                compiled.set_param(n, d);
            }
            e.insert(compiled);
        }
        let compiled = caches.encoder.get_mut(&key).expect("encoder cached");
        let out = compiled
            .run(&[("mel", padded.as_slice())])
            .into_iter()
            .next()
            .context("encoder output")?;
        Ok((out, geom.num_audio_tokens))
    }

    /// Generate transcription token ids for a prompt + audio.
    pub fn generate(&self, prompt_ids: &[u32], mel: &MelSpectrogram) -> Result<Vec<u32>> {
        let batch = 1;
        // Metal's fused decoder kernels drift over depth; use the unfused path
        // there (CPU/MLX keep fusion). The audio encoder is unfused everywhere.
        let skip_fusion = matches!(self.device, Device::Metal);
        let timing = rlx_ir::env::var("RLX_ASR_TIMING").is_some();
        let t_enc = std::time::Instant::now();
        let (audio_embeds, n_audio) = self.encode_audio(mel)?;
        if timing {
            eprintln!("[asr-timing] encoder {:.2}s", t_enc.elapsed().as_secs_f64());
        }
        let t_prefill = std::time::Instant::now();
        ensure!(
            count_audio_placeholders(&self.cfg, prompt_ids) == n_audio,
            "prompt has {} audio placeholders, audio produced {n_audio}",
            count_audio_placeholders(&self.cfg, prompt_ids)
        );

        let mut embed_wm = self.store.load_keys(&[KEY_EMBED_TOKENS])?;
        let (embed, _) = embed_wm.take(KEY_EMBED_TOKENS)?;
        let inputs_embeds = fuse_inputs_embeds(&self.cfg, &embed, prompt_ids, &audio_embeds)?;
        drop(embed_wm);

        let seq = prompt_ids.len();
        let vocab = self.cfg.text.vocab_size;
        let layers = self.cfg.text.num_hidden_layers;

        // Prefill — compile once per `seq` bucket, reuse the graph (with its
        // resident weights) on every later call of the same prompt length.
        let outs = {
            let mut caches = self.caches.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(e) = caches.prefill.entry(seq) {
                let mut wm = self
                    .store
                    .load_prefixes(&[PREFIX_LANGUAGE_MODEL, KEY_LM_HEAD])?;
                let built = {
                    let mut loader = LanguageModelPrefixLoader::new(&mut wm);
                    build_asr_prefill_built(&self.cfg.text, &mut loader, batch, seq, skip_fusion)?
                };
                let params = built.params().clone();
                let mut prefill = compile_built(built, self.device)?;
                for (n, d) in &params {
                    prefill.set_param(n, d);
                }
                e.insert(prefill);
            }
            let prefill = caches.prefill.get_mut(&seq).expect("prefill cached");
            prefill.run(&[("inputs_embeds", inputs_embeds.as_slice())])
        };
        ensure!(
            outs[0].len() == batch * vocab,
            "prefill last-token logits len {} != {}",
            outs[0].len(),
            batch * vocab
        );
        let kv_dim = self.cfg.text.kv_proj_dim();
        let (logits0, mut kv) = kv_from_prefill_outputs(outs, batch, seq, kv_dim, layers)?;
        let mut next = argmax_token(&logits0);
        if timing {
            eprintln!(
                "[asr-timing] prefill compile+run {:.2}s (seq={seq})",
                t_prefill.elapsed().as_secs_f64()
            );
        }
        let t_decode = std::time::Instant::now();

        let mut tokens: Vec<u32> = prompt_ids.to_vec();

        // Bucketed decode: compile O(log N) graphs at power-of-two `past_seq`
        // buckets and reuse them across steps (a fixed-size graph + per-step
        // custom mask serves every actual past length). Weights load only at
        // (re)compile, never per token — fixes the per-token reload slowness and
        // the Metal pipeline-cache growth that OOMs on shared RAM.
        let max_total = seq.saturating_add(self.max_new_tokens).max(1) as u64;
        // Persistent decode cache: keep compiled bucket graphs (and their LM
        // weights) resident across calls; rebuild only if the ladder changes.
        let mut caches = self.caches.borrow_mut();
        if !matches!(&caches.decode, Some((mt, _)) if *mt == max_total) {
            caches.decode = Some((
                max_total,
                BucketedCompileCache::power_of_two_ladder(self.device, 1, max_total),
            ));
        }
        let decode_cache = &mut caches.decode.as_mut().expect("decode cache").1;
        let mut decode_profile = CompileProfile::llama32_decode();
        if skip_fusion {
            decode_profile.fusion.skip = true;
        }
        let options = compile_options_from_profile(
            &decode_profile,
            self.device,
            KernelDispatchConfig::default(),
        );

        for past_seq in seq..seq.saturating_add(self.max_new_tokens) {
            if EOS_IDS.contains(&next) {
                break;
            }
            tokens.push(next);

            let upper = decode_cache
                .bucket_for(past_seq as u64)
                .and_then(|idx| {
                    decode_cache
                        .buckets()
                        .nth(idx)
                        .map(|r| (r.end - 1) as usize)
                })
                .unwrap_or(past_seq);
            let (cos, sin) = rope_slice(&self.cfg.text, past_seq);
            let token_f = [next as f32];
            let mask = bucket_decode_mask(past_seq, upper);
            let fixed = [
                CacheRunInput {
                    name: "input_ids",
                    data: &token_f,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "rope_cos",
                    data: &cos,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "rope_sin",
                    data: &sin,
                    row_inner: None,
                },
                CacheRunInput {
                    name: "mask",
                    data: &mask,
                    row_inner: None,
                },
            ];
            let cfg = self.cfg.text.clone();
            let store = self.store.clone();
            let (logits, new_k, new_v) = run_bucketed_kv_decode(
                decode_cache,
                past_seq,
                &kv,
                kv_dim,
                layers,
                &fixed,
                |upper_u64| {
                    let mut wm = store
                        .load_language_model_weights()
                        .expect("load decode weights");
                    let mut loader = LanguageModelPrefixLoader::new(&mut wm);
                    let built = build_asr_decode_built_opts(
                        &cfg,
                        &mut loader,
                        batch,
                        upper_u64 as usize,
                        skip_fusion,
                        true,
                    )
                    .expect("build decode graph");
                    graph_from_built(built).expect("lower decode graph")
                },
                &options,
            )?;
            next = argmax_token(&logits);
            if rlx_ir::env::var("RLX_ASR_DEBUG_STEPS").is_some() {
                eprintln!("[step] past_seq={past_seq} upper={upper} -> token={next}");
            }
            kv = KvCacheState {
                past_len: past_seq + 1,
                layers_k: new_k,
                layers_v: new_v,
            };
        }

        if timing {
            eprintln!(
                "[asr-timing] decode {} tok {:.2}s",
                tokens.len().saturating_sub(seq),
                t_decode.elapsed().as_secs_f64()
            );
        }

        Ok(tokens)
    }
}

/// Right-pad a mel `[n_mels, n_frames]` to `target` frames with zeros.
fn pad_mel_time(mel: &MelSpectrogram, target: usize) -> Vec<f32> {
    if target == mel.n_frames {
        return mel.data.clone();
    }
    let mut out = vec![0f32; mel.n_mels * target];
    for m in 0..mel.n_mels {
        let src = &mel.data[m * mel.n_frames..(m + 1) * mel.n_frames];
        out[m * target..m * target + mel.n_frames].copy_from_slice(src);
    }
    out
}

#[cfg(feature = "tokenizer")]
impl AsrRunner {
    /// Transcribe a 16 kHz mono WAV file to text.
    pub fn transcribe_wav(&self, wav: &Path, system_text: &str) -> Result<String> {
        let pcm = crate::audio::load_wav_mono_f32(wav)?;
        self.transcribe_pcm(&pcm, system_text)
    }

    /// Transcribe 16 kHz mono PCM samples to text.
    pub fn transcribe_pcm(&self, pcm: &[f32], system_text: &str) -> Result<String> {
        let tok = crate::tokenizer::AsrTokenizer::from_model_dir(self.model_dir())?;
        let mel = pcm_to_log_mel(pcm, self.cfg.audio.num_mel_bins)?;
        let n_audio = self.num_audio_tokens(&mel)?;
        let prompt = tok.build_prompt(&self.cfg, system_text, n_audio)?;
        let out = self.generate(&prompt, &mel)?;
        // Drop the prompt prefix; keep only generated text tokens.
        let generated = &out[prompt.len().min(out.len())..];
        tok.decode(generated)
    }

    /// Transcribe to the raw generated **token ids** (Qwen3 BPE vocab), skipping
    /// detokenization. Because Qwen3-ASR / Qwen3 LM / Qwen3-TTS share one vocab,
    /// these ids can be spliced straight into a downstream LM prompt without a
    /// detokenize→retokenize round-trip. The leading `language <lang>` tag is
    /// still present in the ids (strip it downstream if unwanted).
    pub fn transcribe_pcm_ids(&self, pcm: &[f32], system_text: &str) -> Result<Vec<u32>> {
        let tok = crate::tokenizer::AsrTokenizer::from_model_dir(self.model_dir())?;
        let mel = pcm_to_log_mel(pcm, self.cfg.audio.num_mel_bins)?;
        let n_audio = self.num_audio_tokens(&mel)?;
        let prompt = tok.build_prompt(&self.cfg, system_text, n_audio)?;
        let out = self.generate(&prompt, &mel)?;
        Ok(out[prompt.len().min(out.len())..].to_vec())
    }

    /// Chunked streaming transcription: split PCM into `chunk_s`-second windows
    /// (transcribed independently, in order) and return one [`StreamChunk`] per
    /// window with its partial text and processing latency. Models true offline
    /// chunk-by-chunk emission; segments are concatenated by the caller.
    pub fn transcribe_pcm_streaming(
        &self,
        pcm: &[f32],
        system_text: &str,
        chunk_s: f32,
    ) -> Result<Vec<StreamChunk>> {
        use crate::audio::SAMPLE_RATE;
        let tok = crate::tokenizer::AsrTokenizer::from_model_dir(self.model_dir())?;
        let win = ((chunk_s * SAMPLE_RATE as f32) as usize).max(SAMPLE_RATE / 2);
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < pcm.len() {
            let end = (start + win).min(pcm.len());
            let seg = &pcm[start..end];
            if seg.len() <= 400 {
                break;
            }
            let t = std::time::Instant::now();
            let mel = pcm_to_log_mel(seg, self.cfg.audio.num_mel_bins)?;
            let n_audio = self.num_audio_tokens(&mel)?;
            let prompt = tok.build_prompt(&self.cfg, system_text, n_audio)?;
            let ids = self.generate(&prompt, &mel)?;
            let text = tok.decode(&ids[prompt.len().min(ids.len())..])?;
            out.push(StreamChunk {
                start_s: start as f32 / SAMPLE_RATE as f32,
                end_s: end as f32 / SAMPLE_RATE as f32,
                text,
                latency_ms: t.elapsed().as_secs_f64() * 1e3,
            });
            start = end;
        }
        Ok(out)
    }
}

/// One streaming window's result.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub start_s: f32,
    pub end_s: f32,
    pub text: String,
    pub latency_ms: f64,
}
