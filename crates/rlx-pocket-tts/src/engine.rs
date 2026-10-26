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

//! [`PocketTts`]: text → 24 kHz speech on any RLX device.
//!
//! Mirrors `TTSModel.generate_audio_stream`: the text is prepared and split
//! into chunks; for each chunk the voice state is copied, the chunk's tokens are
//! fed through the backbone, latents are sampled one 80 ms frame at a time until
//! the EOS head fires (plus a few frames), and the latents are decoded by Mimi
//! from a fresh decoder state with a 5 ms fade-in.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rlx_runtime::{CompiledGraph, Device};

use crate::checkpoint::Checkpoint;
use crate::graph::flow_lm::{
    FlowLmDims, OUT_EOS, OUT_HIDDEN, OUT_KV, OUT_LATENT, build_step, past_k_name, past_v_name,
};
use crate::graph::mimi::{MimiDecoder, MimiEncoder};
use crate::graph::{Tensor, compile, rope_tables};
use crate::model::Audio;
use crate::spec::ModelSpec;
use crate::text::{MAX_TOKENS_PER_CHUNK, TextOptions, prepare_text_prompt, split_into_chunks};
use crate::tokenizer::PocketTokenizer;
use crate::voice_state::{VoiceFile, VoiceState};

/// KV capacity granularity: one step graph is compiled per multiple of this.
const KV_BUCKET: usize = 64;
/// Upstream `_MIN_FRAMES_BEFORE_EOS`.
const MIN_FRAMES_BEFORE_EOS: usize = 6;
/// Upstream `_TOKENS_PER_SECOND_ESTIMATE` / `_GEN_SECONDS_PADDING`.
const TOKENS_PER_SECOND: f32 = 3.0;
const GEN_SECONDS_PADDING: f32 = 2.0;
/// Default Mimi decoder block (frames per graph call; 8 frames = 640 ms).
pub const DEFAULT_DECODER_BLOCK: usize = 8;

/// Sampling settings; [`SynthOptions::for_spec`] gives the model's defaults.
#[derive(Debug, Clone)]
pub struct SynthOptions {
    /// Noise variance of the sampler head (`std = sqrt(temperature)`).
    pub temperature: f32,
    /// Generation stops once the EOS logit exceeds this.
    pub eos_threshold: f32,
    /// Sampler-head steps (LSD / flow matching; 1 for the released models).
    pub sampler_steps: usize,
    /// Truncate the noise to `[-c, c]` when set.
    pub noise_clamp: Option<f32>,
    /// Frames generated after EOS; derived per chunk when `None`.
    pub frames_after_eos: Option<usize>,
    pub max_tokens_per_chunk: usize,
    pub seed: u64,
}

impl SynthOptions {
    pub fn for_spec(spec: &ModelSpec) -> Self {
        Self {
            temperature: spec.default_temperature,
            eos_threshold: spec.default_eos_threshold,
            sampler_steps: 1,
            noise_clamp: None,
            frames_after_eos: spec.model_recommended_frames_after_eos,
            max_tokens_per_chunk: MAX_TOKENS_PER_CHUNK,
            seed: 0,
        }
    }
}

/// Where a model's files are.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
    /// Directory holding `<voice>.safetensors`, if any.
    pub voices: Option<PathBuf>,
}

impl ModelFiles {
    /// Resolve `language`'s files under `root`, which may be the language
    /// directory itself (`…/languages/english`), a checkout of the HF repo
    /// (`kyutai/pocket-tts[-without-voice-cloning]`), or a flat directory with
    /// one `*.safetensors` model and `tokenizer.model` (the January mirror).
    pub fn resolve(root: &Path, language: &str, spec: &ModelSpec) -> Result<Self> {
        let lang_dir = [root.join("languages").join(language), root.to_path_buf()]
            .into_iter()
            .find(|d| d.join("model.safetensors").is_file());
        let (weights, dir) = match lang_dir {
            Some(d) => (d.join("model.safetensors"), d),
            None => (find_flat_model(root)?, root.to_path_buf()),
        };
        // The repository root: `<root>/languages/<lang>` → `<root>`.
        let repo = dir
            .parent()
            .filter(|p| p.file_name().is_some_and(|n| n == "languages"))
            .and_then(Path::parent)
            .unwrap_or(&dir)
            .to_path_buf();
        let mut candidates = vec![dir.join("tokenizer.model")];
        if let Some(rel) = spec.tokenizer_repo_path() {
            let p = repo.join(&rel);
            candidates.push(p.with_extension("model"));
            candidates.push(p);
        }
        candidates.push(dir.join("tokenizer.json"));
        candidates.push(repo.join("tokenizer.model"));
        let tokenizer = candidates
            .iter()
            .find(|p| p.is_file())
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "no tokenizer for {language} (looked for {})",
                    candidates
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        let voices = Some(dir.join("embeddings")).filter(|p| p.is_dir());
        Ok(Self {
            weights,
            tokenizer,
            voices,
        })
    }
}

fn find_flat_model(dir: &Path) -> Result<PathBuf> {
    let mut found = Vec::new();
    for e in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "safetensors") {
            found.push(p);
        }
    }
    found.sort();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => bail!("no model.safetensors under {}", dir.display()),
        many => bail!("several *.safetensors in {}: {many:?}", dir.display()),
    }
}

/// Host-held KV cache padded to a bucketed capacity.
struct Kv {
    cap: usize,
    len: usize,
    kv_dim: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Kv {
    fn from_state(state: &VoiceState, cap: usize) -> Self {
        let kv_dim = state.kv_dim;
        let pad = |rows: &Vec<f32>| {
            let mut b = vec![0.0f32; cap * kv_dim];
            b[..rows.len()].copy_from_slice(rows);
            b
        };
        Self {
            cap,
            len: state.offset,
            kv_dim,
            k: state.k.iter().map(pad).collect(),
            v: state.v.iter().map(pad).collect(),
        }
    }

    fn to_state(&self) -> VoiceState {
        let n = self.len * self.kv_dim;
        VoiceState {
            offset: self.len,
            kv_dim: self.kv_dim,
            k: self.k.iter().map(|b| b[..n].to_vec()).collect(),
            v: self.v.iter().map(|b| b[..n].to_vec()).collect(),
        }
    }
}

/// One step's head outputs.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub latent: Vec<f32>,
    pub eos_logit: f32,
    /// Backbone output fed to the heads (`[cond_dim]`).
    pub hidden: Vec<f32>,
}

/// Everything one chunk produced.
#[derive(Debug, Clone, Default)]
pub struct ChunkOutput {
    /// Generated latents before de-normalization, row-major `[frames, 32]`.
    pub latents: Vec<f32>,
    /// EOS logit of every sampled frame (including the one that stopped).
    pub eos_logits: Vec<f32>,
    /// Backbone output of every sampled frame.
    pub hidden: Vec<Vec<f32>>,
    /// Decoded audio before the fade-in.
    pub audio_raw: Vec<f32>,
    /// Decoded audio, faded in over the first 5 ms.
    pub audio: Vec<f32>,
}

/// Pocket TTS on one RLX device.
pub struct PocketTts {
    spec: ModelSpec,
    ck: Checkpoint,
    device: Device,
    files: ModelFiles,
    tokenizer: PocketTokenizer,
    text: TextOptions,
    dims: FlowLmDims,
    embed: Tensor,
    embed_proj: Option<Tensor>,
    input_linear: Tensor,
    bos_emb: Vec<f32>,
    bos_before_voice: Option<Vec<f32>>,
    emb_mean: Vec<f32>,
    emb_std: Vec<f32>,
    steps: HashMap<usize, CompiledGraph>,
    step_names: Vec<(String, String)>,
    mimi: MimiDecoder,
    encoder: Option<MimiEncoder>,
    pub options: SynthOptions,
}

impl PocketTts {
    /// Open the bundled config `language` with files under `root`
    /// (see [`ModelFiles::resolve`]).
    pub fn open(root: impl AsRef<Path>, language: &str, device: Device) -> Result<Self> {
        let spec = ModelSpec::builtin(language)?;
        let files = ModelFiles::resolve(root.as_ref(), language, &spec)?;
        Self::open_with(spec, files, device)
    }

    /// Open with an explicit config and file set.
    pub fn open_with(spec: ModelSpec, files: ModelFiles, device: Device) -> Result<Self> {
        let ck = Checkpoint::open(&files.weights)
            .with_context(|| format!("weights {}", files.weights.display()))?;
        let tokenizer = PocketTokenizer::open(&files.tokenizer)?;
        let dims = FlowLmDims::new(&spec);
        let lut = &spec.flow_lm.lookup_table;
        let embed = ck.tensor("flow_lm.conditioner.embed.weight")?;
        let width = lut.rank.unwrap_or(lut.dim);
        if embed.shape != [lut.n_bins + 1, width] {
            bail!(
                "conditioner.embed.weight is {:?}, config wants [{}, {width}]",
                embed.shape,
                lut.n_bins + 1
            );
        }
        if tokenizer.vocab_size() != lut.n_bins {
            bail!(
                "tokenizer {} has {} pieces, config says {}",
                files.tokenizer.display(),
                tokenizer.vocab_size(),
                lut.n_bins
            );
        }
        let embed_proj = ck.opt("flow_lm.conditioner.proj.weight")?;
        if lut.rank.is_some() != embed_proj.is_some() {
            bail!("config rank {:?} vs checkpoint conditioner.proj", lut.rank);
        }
        let ldim = dims.latent_dim;
        let input_linear = ck.tensor("flow_lm.input_linear.weight")?;
        if input_linear.shape != [dims.d_model, ldim] {
            bail!("input_linear.weight is {:?}", input_linear.shape);
        }
        let bos_before_voice = if spec.flow_lm.insert_bos_before_voice {
            Some(ck.vec("flow_lm.bos_before_voice", dims.d_model)?)
        } else {
            None
        };
        let mimi = MimiDecoder::new(&ck, &spec, DEFAULT_DECODER_BLOCK, device)?;
        let step_names = (0..dims.num_layers)
            .map(|l| (past_k_name(l), past_v_name(l)))
            .collect();
        Ok(Self {
            options: SynthOptions::for_spec(&spec),
            text: TextOptions::from_spec(&spec),
            bos_emb: ck.vec("flow_lm.bos_emb", ldim)?,
            emb_mean: ck.vec("flow_lm.emb_mean", ldim)?,
            emb_std: ck.vec("flow_lm.emb_std", ldim)?,
            embed,
            embed_proj,
            input_linear,
            bos_before_voice,
            dims,
            tokenizer,
            files,
            ck,
            device,
            spec,
            steps: HashMap::new(),
            step_names,
            mimi,
            encoder: None,
        })
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn files(&self) -> &ModelFiles {
        &self.files
    }
    pub fn tokenizer(&self) -> &PocketTokenizer {
        &self.tokenizer
    }
    pub fn sample_rate(&self) -> u32 {
        self.spec.mimi.sample_rate as u32
    }

    /// Recompile the Mimi decoder for blocks of `frames` frames.
    pub fn set_decoder_block(&mut self, frames: usize) -> Result<()> {
        self.mimi = MimiDecoder::new(&self.ck, &self.spec, frames, self.device)?;
        Ok(())
    }

    /// Load a voice: a name from the model's `embeddings/` directory, or a path
    /// to a voice-state / `audio_prompt` safetensors file.
    pub fn load_voice(&mut self, voice: &str) -> Result<VoiceState> {
        let path = Path::new(voice);
        let path = if path.is_file() {
            path.to_path_buf()
        } else {
            let dir = self.files.voices.as_ref().ok_or_else(|| {
                anyhow!("{voice:?} is not a file and the model has no embeddings/ directory")
            })?;
            let p = dir.join(format!("{voice}.safetensors"));
            if !p.is_file() {
                bail!("voice {voice:?} not found in {}", dir.display());
            }
            p
        };
        match VoiceFile::open(&path)? {
            VoiceFile::State(s) => {
                if s.num_layers() != self.dims.num_layers || s.kv_dim != self.dims.kv_dim() {
                    bail!(
                        "voice {} has {} layers × {} kv, the model {} × {}",
                        path.display(),
                        s.num_layers(),
                        s.kv_dim,
                        self.dims.num_layers,
                        self.dims.kv_dim()
                    );
                }
                Ok(s)
            }
            VoiceFile::Prompt { rows, dim, data } => {
                if dim != self.dims.d_model {
                    bail!("audio_prompt width {dim} vs d_model {}", self.dims.d_model);
                }
                self.voice_from_prompt(&data, rows)
            }
        }
    }

    /// Voice cloning: encode a WAV prompt into a voice state, as upstream's
    /// `get_state_for_audio_prompt` does (mono, resampled to 24 kHz, cut to end
    /// on a pause, Mimi-encoded, speaker-projected, run through the backbone).
    /// `truncate` keeps the first 30 s. Needs the gated checkpoint: the
    /// `-without-voice-cloning` release ships the Mimi encoder zeroed.
    pub fn voice_from_wav(&mut self, wav: impl AsRef<Path>, truncate: bool) -> Result<VoiceState> {
        let (mut audio, rate) = crate::audio::read_wav_mono(wav.as_ref())?;
        if truncate {
            audio.truncate(30 * rate as usize);
        }
        self.voice_from_audio(&audio, rate)
    }

    /// [`Self::voice_from_wav`] on mono samples at `rate` Hz.
    pub fn voice_from_audio(&mut self, audio: &[f32], rate: u32) -> Result<VoiceState> {
        let sr = self.spec.mimi.sample_rate;
        let audio = rlx_core::audio::resample_sinc(audio, rate as usize, sr);
        let audio = crate::audio::end_on_pause(&audio, sr);
        let conditioning = self.encode_prompt(&audio)?;
        let rows = conditioning.len() / self.dims.d_model;
        self.voice_from_prompt(&conditioning, rows)
    }

    /// Mimi-encode 24 kHz mono audio and apply the speaker projection:
    /// `[frames, d_model]` conditioning rows.
    pub fn encode_prompt(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let proj = self.ck.tensor("flow_lm.speaker_proj_weight")?;
        let ldim = self.dims.latent_dim;
        if proj.shape != [self.dims.d_model, ldim] {
            bail!("speaker_proj_weight is {:?}", proj.shape);
        }
        if self.encoder.is_none() {
            let w = self.ck.tensor("mimi.encoder.model.0.conv.weight")?;
            if w.data.iter().all(|&x| x == 0.0) {
                bail!(
                    "{} has no voice-cloning weights (the Mimi encoder is zeroed, as in \
                     kyutai/pocket-tts-without-voice-cloning); use the gated kyutai/pocket-tts \
                     checkpoint or a released voice",
                    self.files.weights.display()
                );
            }
            self.encoder = Some(MimiEncoder::new(
                &self.ck,
                &self.spec,
                DEFAULT_DECODER_BLOCK,
                self.device,
            )?);
        }
        let latents = self.encoder.as_mut().expect("set above").encode(audio)?;
        Ok(latents
            .chunks(ldim)
            .flat_map(|z| matvec(&proj.data, self.dims.d_model, ldim, z))
            .collect())
    }

    /// Save a voice state in upstream's `export-voice` format.
    pub fn save_voice(&self, state: &VoiceState, path: impl AsRef<Path>) -> Result<()> {
        crate::voice_state::save_state(state, self.dims.head_dim, path)
    }

    /// Run conditioning rows `[rows, d_model]` (speaker-projected audio) through
    /// the backbone, preceded by `bos_before_voice` when the config asks for it.
    pub fn voice_from_prompt(&mut self, prompt: &[f32], rows: usize) -> Result<VoiceState> {
        let d = self.dims.d_model;
        if prompt.len() != rows * d {
            bail!("prompt has {} values, expected {rows} × {d}", prompt.len());
        }
        let mut seq: Vec<&[f32]> = Vec::with_capacity(rows + 1);
        let bos = self.bos_before_voice.clone();
        if let Some(b) = bos.as_deref() {
            seq.push(b);
        }
        seq.extend(prompt.chunks(d));
        let empty = VoiceState::empty(self.dims.num_layers, self.dims.kv_dim());
        let mut kv = Kv::from_state(&empty, bucket(seq.len()));
        let zeros = vec![0.0; self.dims.latent_dim];
        for row in seq {
            self.step(row, &zeros, &mut kv)?;
        }
        Ok(kv.to_state())
    }

    /// The text embedding of `token` (`[d_model]`).
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let w = self.embed.shape[1];
        let row = &self.embed.data[token as usize * w..(token as usize + 1) * w];
        match &self.embed_proj {
            Some(p) => matvec(&p.data, p.shape[0], p.shape[1], row),
            None => row.to_vec(),
        }
    }

    /// `input_linear` of a latent; `None` is the BOS position.
    fn embed_latent(&self, latent: Option<&[f32]>) -> Vec<f32> {
        let x = latent.unwrap_or(&self.bos_emb);
        matvec(
            &self.input_linear.data,
            self.dims.d_model,
            self.dims.latent_dim,
            x,
        )
    }

    fn step_graph(&mut self, cap: usize) -> Result<&mut CompiledGraph> {
        if !self.steps.contains_key(&cap) {
            let b = build_step(&self.ck, &self.spec, cap, self.options.sampler_steps)?;
            let compiled = compile(b, self.device).context("compile FlowLM step")?;
            self.steps.insert(cap, compiled);
        }
        Ok(self.steps.get_mut(&cap).expect("inserted"))
    }

    /// One backbone row: run the graph, append the row's K/V.
    fn step(&mut self, x: &[f32], noise: &[f32], kv: &mut Kv) -> Result<StepOutput> {
        if kv.len >= kv.cap {
            bail!("KV cache full ({} rows)", kv.cap);
        }
        let hd = self.dims.head_dim;
        let (cos, sin) = rope_tables(std::iter::once(kv.len), hd, self.dims.max_period);
        let mut mask = vec![0.0f32; kv.cap + 1];
        mask[..kv.len].fill(1.0);
        mask[kv.cap] = 1.0;
        let names = std::mem::take(&mut self.step_names);
        let outs = {
            let graph = self.step_graph(kv.cap)?;
            let mut inputs: Vec<(&str, &[f32])> = vec![
                ("x", x),
                ("noise", noise),
                ("mask", &mask),
                ("rope_cos", &cos),
                ("rope_sin", &sin),
            ];
            for (l, (kn, vn)) in names.iter().enumerate() {
                inputs.push((kn.as_str(), &kv.k[l]));
                inputs.push((vn.as_str(), &kv.v[l]));
            }
            graph.run(&inputs)
        };
        self.step_names = names;
        let n_out = OUT_KV + 2 * self.dims.num_layers;
        if outs.len() != n_out {
            bail!(
                "step graph returned {} outputs, expected {n_out}",
                outs.len()
            );
        }
        let row = kv.len * kv.kv_dim;
        for l in 0..self.dims.num_layers {
            let (k, v) = (&outs[OUT_KV + 2 * l], &outs[OUT_KV + 2 * l + 1]);
            if k.len() != kv.kv_dim || v.len() != kv.kv_dim {
                bail!(
                    "layer {l}: new K/V has {} values, expected {}",
                    k.len(),
                    kv.kv_dim
                );
            }
            kv.k[l][row..row + kv.kv_dim].copy_from_slice(k);
            kv.v[l][row..row + kv.kv_dim].copy_from_slice(v);
        }
        kv.len += 1;
        Ok(StepOutput {
            latent: outs[OUT_LATENT].clone(),
            eos_logit: outs[OUT_EOS][0],
            hidden: outs[OUT_HIDDEN].clone(),
        })
    }

    /// `TTSModel._estimate_max_gen_len`.
    pub fn max_gen_len(&self, tokens: usize) -> usize {
        let secs = tokens as f32 / TOKENS_PER_SECOND + GEN_SECONDS_PADDING;
        (secs * self.spec.mimi.frame_rate).ceil() as usize
    }

    /// Generate one prepared chunk (`text` already through
    /// [`prepare_text_prompt`]). `noise` yields each frame's sampler noise.
    pub fn generate_chunk(
        &mut self,
        text: &str,
        voice: &VoiceState,
        frames_after_eos: usize,
        noise: &mut dyn FnMut() -> Vec<f32>,
    ) -> Result<ChunkOutput> {
        self.chunk(text, voice, frames_after_eos, noise, None)
    }

    /// [`Self::generate_chunk`] with teacher forcing: frame `i > 0` is fed
    /// `forced[i-1]` (row-major `[frames, 32]`) instead of the latent sampled
    /// at `i - 1`. Every frame then sees the same input as the run that
    /// produced `forced`, so a comparison isolates per-step error from the
    /// way an autoregressive sampler amplifies it.
    pub fn generate_chunk_forced(
        &mut self,
        text: &str,
        voice: &VoiceState,
        frames_after_eos: usize,
        noise: &mut dyn FnMut() -> Vec<f32>,
        forced: &[f32],
    ) -> Result<ChunkOutput> {
        self.chunk(text, voice, frames_after_eos, noise, Some(forced))
    }

    fn chunk(
        &mut self,
        text: &str,
        voice: &VoiceState,
        frames_after_eos: usize,
        noise: &mut dyn FnMut() -> Vec<f32>,
        forced: Option<&[f32]>,
    ) -> Result<ChunkOutput> {
        let tokens = self.tokenizer.encode(text)?;
        if tokens.is_empty() {
            bail!("chunk {text:?} has no tokens");
        }
        let timing = rlx_ir::env::flag("RLX_POCKET_TTS_TIMING");
        let t0 = std::time::Instant::now();
        let max_gen = self.max_gen_len(tokens.len());
        let mut kv = Kv::from_state(voice, bucket(voice.offset + tokens.len() + max_gen));
        self.step_graph(kv.cap)?;
        let t_compile = t0.elapsed();
        let zeros = vec![0.0; self.dims.latent_dim];
        for &t in &tokens {
            let x = self.embed_token(t);
            self.step(&x, &zeros, &mut kv)?;
        }
        let t_prompt = t0.elapsed();

        let ldim = self.dims.latent_dim;
        let mut out = ChunkOutput::default();
        let mut prev: Option<Vec<f32>> = None;
        let mut eos_step: Option<usize> = None;
        for step in 0..max_gen {
            let x = self.embed_latent(prev.as_deref());
            let z = noise();
            if z.len() != ldim {
                bail!("noise has {} values, expected {ldim}", z.len());
            }
            let s = self.step(&x, &z, &mut kv)?;
            out.eos_logits.push(s.eos_logit);
            out.hidden.push(s.hidden);
            if s.eos_logit > self.options.eos_threshold
                && eos_step.is_none()
                && step >= MIN_FRAMES_BEFORE_EOS
            {
                eos_step = Some(step);
            }
            if eos_step.is_some_and(|e| step >= e + frames_after_eos) {
                break;
            }
            out.latents.extend_from_slice(&s.latent);
            prev = Some(match forced {
                Some(f) => match f.get(step * ldim..(step + 1) * ldim) {
                    Some(row) => row.to_vec(),
                    None => break,
                },
                None => s.latent,
            });
        }

        let t_gen = t0.elapsed();
        out.audio_raw = self.decode_latents(&out.latents)?;
        if timing {
            let frames = out.eos_logits.len();
            eprintln!(
                "pocket-tts: kv cap {} | step graph {:.0} ms | {} prompt tokens {:.0} ms | \
                 {frames} frames {:.0} ms ({:.1} ms/frame) | decode {:.0} ms",
                kv.cap,
                t_compile.as_secs_f64() * 1e3,
                tokens.len(),
                (t_prompt - t_compile).as_secs_f64() * 1e3,
                (t_gen - t_prompt).as_secs_f64() * 1e3,
                (t_gen - t_prompt).as_secs_f64() * 1e3 / frames.max(1) as f64,
                (t0.elapsed() - t_gen).as_secs_f64() * 1e3,
            );
        }
        out.audio = out.audio_raw.clone();
        let fade = self.spec.mimi.sample_rate / 200;
        let n = fade.min(out.audio.len());
        for (i, s) in out.audio[..n].iter_mut().enumerate() {
            // torch.linspace(0, 1, fade)
            *s *= i as f32 / (fade - 1) as f32;
        }
        Ok(out)
    }

    /// De-normalize generated latents (`[frames, 32]`, row-major) and decode
    /// them with Mimi from a fresh decoder state (no fade-in).
    pub fn decode_latents(&mut self, latents: &[f32]) -> Result<Vec<f32>> {
        let ldim = self.dims.latent_dim;
        let denorm: Vec<f32> = latents
            .chunks(ldim)
            .flat_map(|f| {
                f.iter()
                    .zip(&self.emb_std)
                    .zip(&self.emb_mean)
                    .map(|((x, s), m)| x * s + m)
            })
            .collect();
        if denorm.is_empty() {
            return Ok(Vec::new());
        }
        self.mimi.decode_all(&denorm)
    }

    /// Text → speech with `voice`, using [`Self::options`].
    pub fn generate(&mut self, text: &str, voice: &VoiceState) -> Result<Audio> {
        let opts = self.options.clone();
        let chunks =
            split_into_chunks(&self.tokenizer, text, opts.max_tokens_per_chunk, &self.text)?;
        let mut rng = StdRng::seed_from_u64(opts.seed);
        let std = opts.temperature.max(0.0).sqrt();
        let normal = Normal::new(0.0f32, std.max(f32::MIN_POSITIVE)).map_err(|e| anyhow!("{e}"))?;
        let ldim = self.dims.latent_dim;
        let clamp = opts.noise_clamp;
        let mut noise = move || -> Vec<f32> {
            (0..ldim)
                .map(|_| {
                    if std == 0.0 {
                        return 0.0;
                    }
                    loop {
                        let v = normal.sample(&mut rng);
                        match clamp {
                            Some(c) if v.abs() > c => continue,
                            _ => return v,
                        }
                    }
                })
                .collect()
        };
        let mut samples = Vec::new();
        for chunk in chunks {
            let (prepared, guess) = prepare_text_prompt(&chunk, &self.text)?;
            let fae = opts.frames_after_eos.unwrap_or(guess + 2);
            let out = self.generate_chunk(&prepared, voice, fae, &mut noise)?;
            samples.extend(out.audio);
        }
        Ok(Audio {
            samples,
            sample_rate: self.sample_rate(),
        })
    }
}

fn bucket(rows: usize) -> usize {
    rows.max(1).div_ceil(KV_BUCKET) * KV_BUCKET
}

/// `W x` for row-major `W [rows, cols]`.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}
