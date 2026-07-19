use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_qwen3_tts::VoiceClone;
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "qwen3-tts",
        supports_clone: true,
        feature: "lm-tts",
        hints: WeightHints {
            default_dir: PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base"),
            env_keys: vec!["RLX_QWEN3_TTS_DIR"],
            marker_files: vec!["config.json"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = VoiceClone::open(&dir, device).context("load qwen3-tts VoiceClone")?;
    Ok(Box::new(QwenAdapter { inner, dir }))
}

struct QwenAdapter {
    inner: VoiceClone,
    dir: PathBuf,
}

impl TtsAdapter for QwenAdapter {
    fn id(&self) -> &'static str {
        "qwen3-tts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        true
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let ref_path = req
            .clone
            .map(|c| c.ref_wav.to_path_buf())
            .or_else(default_ref)
            .ok_or_else(|| anyhow::anyhow!("qwen3-tts needs --clone ref wav or assets/jfk"))?;
        let reference = self
            .inner
            .extract_reference(&ref_path)
            .context("extract_reference")?;
        let t0 = Instant::now();
        let pcm = self.inner.generate(&reference, req.text)?;
        Ok(SynthResult {
            pcm,
            sample_rate: 24_000,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?} ({})", req.device, self.dir.display()),
        })
    }
}

fn default_ref() -> Option<PathBuf> {
    let jfk = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    jfk.is_file().then_some(jfk)
}
