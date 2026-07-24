use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_voxtral_tts::{
    GenerationConfig, VoxtralTtsRunnerBuilder, speech_tokenizer::SpeechTokenizer,
};

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};
use crate::wav::read_wav_mono;

const DEFAULT_DIR: &str = "weights/tts/voxtral-tts";

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "voxtral-tts",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_DIR),
            env_keys: vec!["RLX_VOXTRAL_TTS_DIR"],
            marker_files: vec!["consolidated.safetensors", "tekken.json"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = VoxtralTtsRunnerBuilder::default()
        .model_dir(&dir)
        .device(device)
        .build()
        .context("load voxtral-tts")?;
    let voice = std::env::var("RLX_VOXTRAL_VOICE").unwrap_or_else(|_| "neutral_female".into());
    Ok(Box::new(VoxtralAdapter { inner, dir, voice }))
}

struct VoxtralAdapter {
    inner: rlx_voxtral_tts::VoxtralTtsRunner,
    dir: PathBuf,
    voice: String,
}

impl TtsAdapter for VoxtralAdapter {
    fn id(&self) -> &'static str {
        "voxtral-tts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let tok = SpeechTokenizer::from_model_dir(&self.dir).context("load tekken tokenizer")?;
        let prompt_tokens = tok
            .encode_speech(req.text, &self.voice)
            .with_context(|| format!("tokenize for voice {:?}", self.voice))?;
        let mut gen_cfg = GenerationConfig::default();
        gen_cfg.seed = req.seed;
        let tmp = std::env::temp_dir().join(format!(
            "rlx-tts-bench-voxtral-{}.wav",
            std::process::id()
        ));
        let t0 = Instant::now();
        self.inner
            .synthesize_native(&prompt_tokens, &self.voice, &tmp, &gen_cfg)?;
        let (pcm, sr) = read_wav_mono(&tmp)?;
        let _ = std::fs::remove_file(&tmp);
        Ok(SynthResult {
            pcm,
            sample_rate: sr,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
