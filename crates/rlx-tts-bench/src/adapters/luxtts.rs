use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_luxtts::{DEFAULT_LOCAL_DIR, InferOpts, LuxTts};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};
use crate::wav::read_wav_mono;

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "luxtts",
        supports_clone: true,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_LUXTTS_DIR"],
            marker_files: vec!["encoder_body.onnx", "fm_decoder.onnx"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = LuxTts::load_on(&dir, device).context("load luxtts")?;
    Ok(Box::new(LuxAdapter { inner, dir }))
}

struct LuxAdapter {
    inner: LuxTts,
    dir: PathBuf,
}

impl TtsAdapter for LuxAdapter {
    fn id(&self) -> &'static str {
        "luxtts"
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
            .as_ref()
            .map(|c| c.ref_wav.to_path_buf())
            .or_else(|| find_prompt_wav(&self.dir))
            .ok_or_else(|| {
                anyhow::anyhow!("luxtts needs --clone ref wav or prompt under weights")
            })?;
        let (prompt, _) = read_wav_mono(&ref_path)?;
        let prompt_text = req
            .clone
            .as_ref()
            .and_then(|c| c.ref_text)
            .unwrap_or("The quick brown fox jumps over the lazy dog.");
        let opts = InferOpts::default();
        let t0 = Instant::now();
        let pcm = self
            .inner
            .synthesize(req.text, &prompt, prompt_text, &opts)?;
        Ok(SynthResult {
            pcm,
            sample_rate: self.inner.sample_rate(),
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: self.inner.ort_ep(),
        })
    }
}

fn find_prompt_wav(dir: &std::path::Path) -> Option<PathBuf> {
    for name in ["prompt.wav", "default_voice.wav", "ref.wav"] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let jfk = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    jfk.is_file().then_some(jfk)
}
