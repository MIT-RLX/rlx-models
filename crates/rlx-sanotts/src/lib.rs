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

//! [sanoTTS](https://github.com/ampixa/sanoTTS) on RLX — a native Rust port of
//! the Root-A student stack, a ~1.5 M-parameter distillation of Piper VITS.
//!
//! Four stages: `text → duration → acoustic latent → waveform`.
//!
//! - **duration** (`duration_conv`): phoneme ids → per-token frame counts.
//! - **acoustic** (`token_context`): a token-rate context stack, expanded to
//!   frame rate by the predicted durations, then a frame-rate stack producing a
//!   192-channel latent.
//! - **decoder** (`piperlite`): three transposed-conv upsampling stages
//!   (8 × 8 × 4 = 256 samples/frame) each followed by a 3-branch residual bank.
//!
//! Two execution paths, both fed by the same weights:
//!
//! - [`model`] is the host-eager CPU reference, transcribed from upstream's
//!   pure-numpy forward passes and gated against them bit-for-bit-ish (see
//!   `tests/reference_parity.rs`).
//! - [`graph`] compiles the frame-rate acoustic stage and the whole decoder —
//!   the >95 % of the arithmetic that scales with audio length — into one
//!   rlx-ir graph, so they run on every RLX backend. The duration model and the
//!   token-rate stage stay on the host: they are microseconds of work at 64
//!   channels, and the data-dependent duration expansion sits between them.
//!   Graphs are built at bucketed lengths and masked, so varied utterances
//!   share a few compiled graphs rather than one apiece.

pub mod audio;
pub mod config;
pub mod frontend;
#[cfg(feature = "rlx-graph")]
pub mod graph;
pub mod model;
pub mod ops;
pub mod voicepack;

use std::path::Path;

use anyhow::{Context, Result};

pub use config::{AcousticConfig, DecoderConfig, DurationConfig, PhonemeTable};
pub use ops::Mat;
pub use voicepack::{TensorStore, VoicePack};

/// Samples emitted per acoustic frame (the decoder's total upsampling factor).
pub const HOP: usize = 256;

/// A synthesized waveform.
#[derive(Debug, Clone)]
pub struct Wav {
    /// Mono float samples in [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Wav {
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        audio::write_wav(path, &self.samples, self.sample_rate)
    }
}

/// Where the frame-rate stages run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Host-eager reference kernels (always available).
    #[default]
    Host,
    /// Compiled rlx-ir graph on the given device.
    #[cfg(feature = "rlx-graph")]
    Graph(rlx_runtime::Device),
}

/// A loaded voice, ready to render text repeatedly without re-reading disk.
pub struct Synthesizer {
    pack: VoicePack,
    duration_cfg: DurationConfig,
    acoustic_cfg: AcousticConfig,
    decoder_cfg: DecoderConfig,
    duration_ts: TensorStore,
    acoustic_ts: TensorStore,
    decoder_ts: TensorStore,
    table: Option<PhonemeTable>,
    backend: Backend,
    /// Compiled frame-stage graphs keyed by `(device, bucketed capacity)`.
    /// Shapes are static in rlx-ir, so each graph covers one capacity and every
    /// shorter length that fits in it; the AOT cache makes a cold capacity a
    /// load rather than a compile.
    #[cfg(feature = "rlx-graph")]
    graphs: std::sync::Mutex<
        std::collections::HashMap<(rlx_runtime::Device, usize), graph::FrameGraph>,
    >,
}

impl Synthesizer {
    /// Load a voice package directory (`manifest.json` + weights + phoneme config).
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let pack = VoicePack::load_from_dir(directory)?;
        Self::from_pack(pack)
    }

    /// Build a synthesizer around an already-loaded pack.
    pub fn from_pack(pack: VoicePack) -> Result<Self> {
        let duration_cfg = pack.component_config::<DurationConfig>("duration")?;
        let acoustic_cfg = pack.component_config::<AcousticConfig>("acoustic")?;
        let decoder_cfg = pack.component_config::<DecoderConfig>("decoder")?;
        model::check_decoder(&decoder_cfg)?;
        let duration_ts = pack.component_tensors("duration")?;
        let acoustic_ts = pack.component_tensors("acoustic")?;
        let decoder_ts = pack.component_tensors("decoder")?;
        // The phoneme table is optional: id-only callers do not need it, and a
        // pack shipped without one should still synthesize from ids.
        let table = pack.phoneme_table().ok();
        Ok(Self {
            pack,
            duration_cfg,
            acoustic_cfg,
            decoder_cfg,
            duration_ts,
            acoustic_ts,
            decoder_ts,
            table,
            backend: Backend::default(),
            #[cfg(feature = "rlx-graph")]
            graphs: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Select where the frame-rate stages run. Defaults to [`Backend::Host`].
    pub fn set_backend(&mut self, backend: Backend) {
        self.backend = backend;
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn pack(&self) -> &VoicePack {
        &self.pack
    }

    pub fn sample_rate(&self) -> u32 {
        self.pack.sample_rate()
    }

    /// The voice's default speaking-rate scale (larger = slower).
    pub fn default_length_scale(&self) -> f32 {
        self.pack.duration_length_scale()
    }

    /// The voice's phoneme table, if the pack shipped one.
    pub fn phoneme_table(&self) -> Option<&PhonemeTable> {
        self.table.as_ref()
    }

    /// Text → Piper phoneme ids (needs the `espeak` feature).
    pub fn phoneme_ids(&self, text: &str) -> Result<Vec<i64>> {
        let table = self.table()?;
        frontend::text_to_phoneme_ids(text, table)
    }

    /// An espeak IPA string → Piper phoneme ids, skipping G2P entirely.
    ///
    /// The bundled pure-Rust espeak-ng renders a few phonemes differently from
    /// the C library the voices were distilled against (see the crate README),
    /// so this is the way in for a caller who already has the reference
    /// phoneme stream — from `espeak-ng --ipa`, from Piper, or from a
    /// hand-written pronunciation.
    pub fn ids_from_phonemes(&self, phonemes: &str) -> Result<Vec<i64>> {
        let table = self.table()?;
        let ids = frontend::phonemes_to_ids(phonemes, table);
        if ids.len() <= frontend::FRAMING_IDS {
            anyhow::bail!("no codepoint of {phonemes:?} is in this voice's phoneme table");
        }
        Ok(ids)
    }

    /// An espeak IPA string → waveform.
    pub fn synthesize_phonemes(&self, phonemes: &str, length_scale: f32) -> Result<Wav> {
        let ids = self.ids_from_phonemes(phonemes)?;
        self.synthesize_ids(&ids, length_scale)
    }

    fn table(&self) -> Result<&PhonemeTable> {
        self.table
            .as_ref()
            .context("voice pack has no phoneme config; synthesize from ids instead")
    }

    /// Predicted per-token frame counts.
    pub fn durations(&self, ids: &[i64], length_scale: f32) -> Result<Vec<usize>> {
        if length_scale.is_nan() || length_scale <= 0.0 {
            anyhow::bail!("length_scale must be positive, got {length_scale}");
        }
        model::duration_forward(&self.duration_ts, &self.duration_cfg, ids, length_scale)
    }

    /// Acoustic latent `[out_channels, frames]` for `ids`/`durations`.
    pub fn latent(&self, ids: &[i64], durations: &[usize]) -> Result<Mat> {
        match self.backend {
            Backend::Host => {
                model::acoustic_forward(&self.acoustic_ts, &self.acoustic_cfg, ids, durations)
            }
            #[cfg(feature = "rlx-graph")]
            Backend::Graph(_) => {
                let input = model::acoustic_frame_input(
                    &self.acoustic_ts,
                    &self.acoustic_cfg,
                    ids,
                    durations,
                )?;
                Ok(self.run_graph(&input)?.0)
            }
        }
    }

    /// Latent → mono waveform.
    pub fn decode(&self, latent: &Mat) -> Result<Vec<f32>> {
        model::decoder_forward(&self.decoder_ts, &self.decoder_cfg, latent)
    }

    /// Phoneme ids → waveform.
    pub fn synthesize_ids(&self, ids: &[i64], length_scale: f32) -> Result<Wav> {
        let durations = self.durations(ids, length_scale)?;
        let samples = match self.backend {
            Backend::Host => {
                let latent = model::acoustic_forward(
                    &self.acoustic_ts,
                    &self.acoustic_cfg,
                    ids,
                    &durations,
                )?;
                self.decode(&latent)?
            }
            #[cfg(feature = "rlx-graph")]
            Backend::Graph(_) => {
                let input = model::acoustic_frame_input(
                    &self.acoustic_ts,
                    &self.acoustic_cfg,
                    ids,
                    &durations,
                )?;
                self.run_graph(&input)?.1
            }
        };
        Ok(Wav {
            samples: samples.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect(),
            sample_rate: self.sample_rate(),
        })
    }

    /// Text → waveform, using the voice's default speaking rate.
    pub fn synthesize(&self, text: &str) -> Result<Wav> {
        self.synthesize_with(text, self.default_length_scale())
    }

    /// Text → waveform at an explicit speaking-rate scale.
    pub fn synthesize_with(&self, text: &str, length_scale: f32) -> Result<Wav> {
        let ids = self.phoneme_ids(text)?;
        self.synthesize_ids(&ids, length_scale)
    }

    /// Run (or compile and cache) the frame-stage graph. Returns `(latent, audio)`.
    #[cfg(feature = "rlx-graph")]
    fn run_graph(&self, input: &Mat) -> Result<(Mat, Vec<f32>)> {
        let Backend::Graph(device) = self.backend else {
            anyhow::bail!("run_graph called with a non-graph backend");
        };
        let mut graphs = self
            .graphs
            .lock()
            .map_err(|_| anyhow::anyhow!("frame-graph cache mutex poisoned"))?;
        // Graphs are built at a bucketed capacity and accept any shorter
        // length, so a run of varied utterances reuses a handful of graphs
        // instead of compiling one apiece.
        let capacity = graph::bucket_capacity(input.cols);
        let g = match graphs.entry((device, capacity)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(graph::FrameGraph::compile(
                &self.acoustic_ts,
                &self.acoustic_cfg,
                &self.decoder_ts,
                &self.decoder_cfg,
                capacity,
                device,
            )?),
        };
        g.forward(input)
    }
}
