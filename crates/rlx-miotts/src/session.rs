//! High-level MioTTS session: Qwen3 LM (CPU) + MioCodec decode (any device).

use std::path::{Path, PathBuf};

use anyhow::Result;
use rlx_runtime::Device;
use tokenizers::Tokenizer;

use crate::codec::{MioCodec, SAMPLE_RATE, load_preset_embedding};
use crate::lm::{MioLm, MioLmConfig};
use crate::tokens;

#[derive(Debug, Clone)]
pub struct GenerateOpts {
    pub seed: u64,
    pub max_new_tokens: usize,
    pub preset: String,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        Self {
            seed: 42,
            max_new_tokens: 400,
            preset: "en_female".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesisResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub content_codes: Vec<u32>,
}

pub struct MioSession {
    lm: MioLm,
    tokenizer: Tokenizer,
    codec: MioCodec,
    presets_dir: PathBuf,
    model_dir: PathBuf,
    codec_dir: PathBuf,
    device: Device,
}

impl MioSession {
    pub fn open(
        model_dir: impl AsRef<Path>,
        codec_dir: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let codec_dir = codec_dir.as_ref().to_path_buf();
        anyhow::ensure!(
            model_dir.join("model.safetensors").is_file(),
            "missing {} — run `just fetch-miotts`",
            model_dir.join("model.safetensors").display()
        );
        let cfg = MioLmConfig::load(&model_dir)?;
        // LM stays on CPU (eager); `--device` selects codec backend.
        let lm = MioLm::load(&model_dir, &cfg, Device::Cpu)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let codec = MioCodec::load(&codec_dir, device)?;
        let presets_dir = model_dir.join("presets");
        Ok(Self {
            lm,
            tokenizer,
            codec,
            presets_dir,
            model_dir,
            codec_dir,
            device,
        })
    }

    pub fn open_defaults(device: Device) -> Result<Self> {
        Self::open(default_model_dir(), default_codec_dir(), device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn codec_dir(&self) -> &Path {
        &self.codec_dir
    }

    /// Chat-style user prompt → speech content codes.
    pub fn generate_codes(&mut self, text: &str, opts: &GenerateOpts) -> Result<Vec<u32>> {
        let prompt_ids = self.encode_chat_prompt(text)?;
        self.lm
            .generate_speech_codes(&prompt_ids, opts.max_new_tokens, opts.seed)
    }

    pub fn synthesize(&mut self, text: &str, opts: &GenerateOpts) -> Result<SynthesisResult> {
        let codes = self.generate_codes(text, opts)?;
        let emb = load_preset_embedding(&self.presets_dir, &opts.preset)
            .or_else(|_| load_preset_embedding(&self.codec_dir.join("fixtures"), &opts.preset))?;
        let samples = self.codec.decode(&codes, &emb)?;
        Ok(SynthesisResult {
            samples,
            sample_rate: SAMPLE_RATE,
            content_codes: codes,
        })
    }

    /// Decode precomputed codes with a preset on this session's codec device.
    pub fn decode_codes(&self, codes: &[u32], preset: &str) -> Result<Vec<f32>> {
        let emb = load_preset_embedding(&self.presets_dir, preset)
            .or_else(|_| load_preset_embedding(&self.codec_dir.join("fixtures"), preset))?;
        self.codec.decode(codes, &emb)
    }

    /// Open codec-only on `device` and decode (for backend matrix).
    pub fn decode_codes_on(
        codec_dir: &Path,
        presets_dir: &Path,
        device: Device,
        codes: &[u32],
        preset: &str,
    ) -> Result<Vec<f32>> {
        let codec = MioCodec::load(codec_dir, device)?;
        let emb = load_preset_embedding(presets_dir, preset)
            .or_else(|_| load_preset_embedding(&codec_dir.join("fixtures"), preset))?;
        codec.decode(codes, &emb)
    }

    fn encode_chat_prompt(&self, text: &str) -> Result<Vec<u32>> {
        // Match HF apply_chat_template for Qwen: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
        let rendered = format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let enc = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}

pub fn default_model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/miotts")
}

pub fn default_codec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/miocodec")
}

/// Re-export for callers that only need the pad length.
pub use tokens::SPEECH_LEN;
