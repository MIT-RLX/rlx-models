//! Inflect-Nano-v1 English text-to-speech for RLX.
//!
//! Two deterministic stages: a FastSpeech-style `MicroFastSpeech` acoustic model
//! (token ids → 80-bin log-mel) and a Snake HiFi-GAN vocoder (mel → 24 kHz wav).
//! A full standalone text frontend (clean → normalize → g2p → ids) feeds it.

pub mod acoustic;
pub mod audio;
pub mod config;
pub mod frontend;
#[cfg(feature = "rlx-graph")]
pub mod graph;
#[cfg(feature = "onnx")]
pub mod onnx_vocoder;
pub mod ops;
pub mod vocoder;
pub mod weights;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::Array2;

pub use config::{BundleConfig, ExecutionMode, InferOpts};

/// A loaded Inflect-Nano model (acoustic + vocoder weights + config).
pub struct InflectNano {
    pub cfg: BundleConfig,
    acoustic_w: weights::Weights,
    vocoder_w: weights::Weights,
    dir: PathBuf,
    frontend: once_cell::sync::OnceCell<frontend::English>,
    #[cfg(feature = "onnx")]
    coreml_vocoder: once_cell::sync::OnceCell<std::sync::Mutex<onnx_vocoder::OnnxVocoder>>,
}

/// Synthesized waveform.
pub struct Wav {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Real-time stats for a streaming synthesis run.
#[derive(Debug, Clone)]
pub struct RealtimeReport {
    /// Total audio produced (seconds).
    pub audio_secs: f32,
    /// Total compute time (seconds).
    pub compute_secs: f32,
    /// Number of chunks emitted.
    pub chunks: usize,
    /// Worst (smallest) per-chunk real-time factor seen.
    pub worst_chunk_rtf: f32,
}

impl RealtimeReport {
    /// Overall real-time factor (audio / compute); ≥ 1.0 means faster than real time.
    pub fn rtf(&self) -> f32 {
        if self.compute_secs > 0.0 {
            self.audio_secs / self.compute_secs
        } else {
            f32::INFINITY
        }
    }

    /// Whether every chunk was produced in less than its own playback duration
    /// (i.e. the stream can sustain real-time playback without underrun).
    pub fn sustains_realtime(&self) -> bool {
        self.worst_chunk_rtf >= 1.0
    }
}

/// Mel context (in frames) kept on each side of a streaming chunk so the
/// vocoder's convolutions see the same inputs as a full-utterance run. The
/// vocoder's mel-frame receptive field is well under this; validated against
/// the full output in `tests/streaming_parity.rs`.
const STREAM_OVERLAP_FRAMES: usize = 32;

impl InflectNano {
    /// Load the RLX asset bundle (see `scripts/export_inflect_nano.py`).
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let cfg_s = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", dir.display()))?;
        let cfg = BundleConfig::from_json(&cfg_s)?;
        let acoustic_w = weights::Weights::load(&dir.join("acoustic.safetensors"))?;
        let vocoder_w = weights::Weights::load(&dir.join("vocoder.safetensors"))?;
        Ok(Self {
            cfg,
            acoustic_w,
            vocoder_w,
            dir: dir.to_path_buf(),
            frontend: once_cell::sync::OnceCell::new(),
            #[cfg(feature = "onnx")]
            coreml_vocoder: once_cell::sync::OnceCell::new(),
        })
    }

    pub fn asset_dir(&self) -> &Path {
        &self.dir
    }

    /// Lazily load the text frontend (CMUdict + g2p_en + tagger + bert).
    pub fn frontend(&self) -> Result<&frontend::English> {
        self.frontend
            .get_or_try_init(|| frontend::English::load(&self.dir.join("frontend")))
    }

    /// Raw text → (phone_ids, tone_ids, lang_ids) with blank insertion.
    pub fn text_to_ids(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        self.frontend()?.text_to_ids(text, self.cfg.add_blank)
    }

    /// Full pipeline: raw text → normalized 24 kHz waveform (host-eager CPU path).
    pub fn synthesize(&self, text: &str, opts: &InferOpts) -> Result<Wav> {
        let (phone, tone, lang) = self.text_to_ids(text)?;
        let speaker = self.cfg.default_speaker();
        self.synthesize_from_ids(&phone, &tone, &lang, speaker, opts)
    }

    /// The fastest available accelerator device for the vocoder graph
    /// (Metal → MLX → wgpu), or `None` if only the CPU is compiled in/available.
    #[cfg(feature = "rlx-graph")]
    pub fn preferred_accelerator() -> Option<rlx_runtime::Device> {
        use rlx_runtime::Device;
        [Device::Metal, Device::Mlx, Device::Gpu]
            .into_iter()
            .find(|&d| rlx_runtime::is_available(d))
    }

    /// Synthesize using an [`ExecutionMode`] policy (latency / precision /
    /// memory footprint / iOS-style CPU+GPU hybrid). See [`ExecutionMode`].
    pub fn synthesize_mode(
        &self,
        text: &str,
        opts: &InferOpts,
        mode: ExecutionMode,
    ) -> Result<Wav> {
        match mode {
            // Pure host-eager f32: deterministic reference / lowest memory.
            ExecutionMode::Precision | ExecutionMode::MemoryFootprint => {
                self.synthesize(text, opts)
            }
            // Vocoder on the best accelerator; CPU acoustic. CPU fallback otherwise.
            ExecutionMode::Latency | ExecutionMode::Hybrid => {
                #[cfg(feature = "rlx-graph")]
                if let Some(dev) = Self::preferred_accelerator() {
                    return self.synthesize_on(text, opts, dev);
                }
                self.synthesize(text, opts)
            }
        }
    }

    /// Full pipeline with the vocoder run as an rlx-ir graph on `device`
    /// (CPU/Metal/MLX/CUDA/ROCm/wgpu). The acoustic stage runs on the host.
    #[cfg(feature = "rlx-graph")]
    pub fn synthesize_on(
        &self,
        text: &str,
        opts: &InferOpts,
        device: rlx_runtime::Device,
    ) -> Result<Wav> {
        let (phone, tone, lang) = self.text_to_ids(text)?;
        let mel = self.mel_from_ids(&phone, &tone, &lang, self.cfg.default_speaker(), opts)?;
        let mut g = self.compile_vocoder_graph(mel.dim().1, device)?;
        let raw = g.forward(&mel)?;
        Ok(Wav {
            samples: audio::normalize_audio(&raw),
            sample_rate: self.cfg.sample_rate,
        })
    }

    /// Acoustic forward: token ids → mel `[80, T]`.
    pub fn mel_from_ids(
        &self,
        phone: &[i64],
        tone: &[i64],
        lang: &[i64],
        speaker: i64,
        opts: &InferOpts,
    ) -> Result<Array2<f32>> {
        acoustic::Acoustic::new(&self.acoustic_w, &self.cfg.acoustic)
            .infer(phone, tone, lang, speaker, opts)
    }

    /// Vocoder forward: mel `[80, T]` → raw waveform (pre-normalization).
    pub fn wav_from_mel(&self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        vocoder::Vocoder::new(&self.vocoder_w, &self.cfg.vocoder).forward(mel)
    }

    /// Streaming synthesis: emit raw 24 kHz audio in ~`chunk_secs`-long chunks via
    /// `on_chunk`, so playback can start after the first chunk and a long utterance
    /// sustains faster-than-real-time throughput with bounded latency.
    ///
    /// The acoustic model runs once (it is cheap); the vocoder runs per chunk over
    /// `chunk + STREAM_OVERLAP_FRAMES` context on each side, trimmed so the
    /// concatenation is identical to a full-utterance vocode. Chunks are the raw
    /// (tanh-bounded) vocoder output — the whole-clip RMS normalization in
    /// [`Self::synthesize`] needs the full signal and is intentionally skipped here.
    pub fn synthesize_stream(
        &self,
        text: &str,
        opts: &InferOpts,
        chunk_secs: f32,
        mut on_chunk: impl FnMut(&[f32]),
    ) -> Result<RealtimeReport> {
        let (phone, tone, lang) = self.text_to_ids(text)?;
        let mel = self.mel_from_ids(&phone, &tone, &lang, self.cfg.default_speaker(), opts)?;
        let voc = vocoder::Vocoder::new(&self.vocoder_w, &self.cfg.vocoder);

        let hop = self.cfg.vocoder.hop_size;
        let total_frames = mel.dim().1;
        let chunk_frames =
            ((chunk_secs * self.cfg.sample_rate as f32 / hop as f32).round() as usize).max(1);

        let mut compute_secs = 0.0f32;
        let mut audio_samples = 0usize;
        let mut chunks = 0usize;
        let mut worst_chunk_rtf = f32::INFINITY;

        let mut start = 0usize;
        while start < total_frames {
            let end = (start + chunk_frames).min(total_frames);
            let ctx_start = start.saturating_sub(STREAM_OVERLAP_FRAMES);
            let ctx_end = (end + STREAM_OVERLAP_FRAMES).min(total_frames);
            let slice = mel.slice(ndarray::s![.., ctx_start..ctx_end]).to_owned();

            let t0 = std::time::Instant::now();
            let wav = voc.forward(&slice)?;
            let dt = t0.elapsed().as_secs_f32();

            // trim the context samples → exactly the [start, end) frames
            let left = (start - ctx_start) * hop;
            let len = (end - start) * hop;
            let chunk = &wav[left..(left + len).min(wav.len())];

            let chunk_audio = chunk.len() as f32 / self.cfg.sample_rate as f32;
            if dt > 0.0 {
                worst_chunk_rtf = worst_chunk_rtf.min(chunk_audio / dt);
            }
            compute_secs += dt;
            audio_samples += chunk.len();
            chunks += 1;
            on_chunk(chunk);
            start = end;
        }

        Ok(RealtimeReport {
            audio_secs: audio_samples as f32 / self.cfg.sample_rate as f32,
            compute_secs,
            chunks,
            worst_chunk_rtf: if worst_chunk_rtf.is_finite() {
                worst_chunk_rtf
            } else {
                0.0
            },
        })
    }

    /// Vocoder via ONNX Runtime with the CoreML execution provider (Apple).
    /// The acoustic stage runs host-eager; the vocoder runs through `vocoder.onnx`.
    #[cfg(feature = "onnx")]
    pub fn vocode_onnx(&self, mel: &Array2<f32>, coreml: bool) -> Result<Vec<f32>> {
        let mut voc =
            onnx_vocoder::OnnxVocoder::load(&self.dir, self.cfg.vocoder.hop_size, coreml)?;
        voc.forward(mel)
    }

    /// Cached CoreML vocoder session — the ORT session (and the one-time CoreML
    /// model compilation) is built on first use and reused across calls.
    #[cfg(feature = "onnx")]
    fn coreml_vocoder(&self) -> Result<&std::sync::Mutex<onnx_vocoder::OnnxVocoder>> {
        self.coreml_vocoder.get_or_try_init(|| {
            onnx_vocoder::OnnxVocoder::load(&self.dir, self.cfg.vocoder.hop_size, true)
                .map(std::sync::Mutex::new)
        })
    }

    /// Full pipeline with the vocoder on ONNX Runtime (CoreML EP). The CoreML
    /// session is cached, so only the first call pays the model-compile cost.
    #[cfg(feature = "onnx")]
    pub fn synthesize_coreml(&self, text: &str, opts: &InferOpts) -> Result<Wav> {
        let (phone, tone, lang) = self.text_to_ids(text)?;
        let mel = self.mel_from_ids(&phone, &tone, &lang, self.cfg.default_speaker(), opts)?;
        let raw = {
            let mut voc = self.coreml_vocoder()?.lock().expect("coreml vocoder mutex");
            voc.forward(&mel)?
        };
        Ok(Wav {
            samples: audio::normalize_audio(&raw),
            sample_rate: self.cfg.sample_rate,
        })
    }

    /// Compile the vocoder as an rlx-ir graph for `device` at a fixed frame count
    /// (runs on every RLX backend). Mel→wav via the compiled graph.
    #[cfg(feature = "rlx-graph")]
    pub fn compile_vocoder_graph(
        &self,
        t_frames: usize,
        device: rlx_runtime::Device,
    ) -> Result<graph::VocoderGraph> {
        graph::VocoderGraph::compile(&self.vocoder_w, &self.cfg.vocoder, t_frames, device)
    }

    /// Full acoustic+vocoder path from token ids, with output normalization.
    pub fn synthesize_from_ids(
        &self,
        phone: &[i64],
        tone: &[i64],
        lang: &[i64],
        speaker: i64,
        opts: &InferOpts,
    ) -> Result<Wav> {
        let mel = self.mel_from_ids(phone, tone, lang, speaker, opts)?;
        let raw = self.wav_from_mel(&mel)?;
        Ok(Wav {
            samples: audio::normalize_audio(&raw),
            sample_rate: self.cfg.sample_rate,
        })
    }
}
