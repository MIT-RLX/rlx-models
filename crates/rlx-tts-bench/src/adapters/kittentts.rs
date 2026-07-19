use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_kittentts::{DEFAULT_LOCAL_DIR, KittenTTS, SAMPLE_RATE};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "kittentts",
        supports_clone: false,
        feature: "lm-tts",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_KITTENTTS_DIR", "KITTEN_RLX_WEIGHTS"],
            marker_files: vec![],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = KittenTTS::load_native_from_dir(&dir, device, 256, 200_000)
        .context("load kittentts native")?;
    let voice = inner
        .available_voices
        .first()
        .cloned()
        .unwrap_or_else(|| "expr-voice-2-f".into());
    Ok(Box::new(KittenAdapter { inner, voice }))
}

struct KittenAdapter {
    inner: KittenTTS,
    voice: String,
}

impl TtsAdapter for KittenAdapter {
    fn id(&self) -> &'static str {
        "kittentts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let t0 = Instant::now();
        let pcm = self
            .inner
            .generate_from_text(req.text, &self.voice, 1.0, "en")?;
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}
