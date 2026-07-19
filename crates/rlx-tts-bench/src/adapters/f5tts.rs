use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_f5tts::{DEFAULT_LOCAL_DIR, F5Native, InferOpts, SAMPLE_RATE};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};
use crate::wav::read_wav_mono;

/// Transcript of the default reference clip — `assets/jfk/jfk_voice_clone.wav`
/// (~5.2 s), which both `find_ref` (plain) and the bench's default clone ref
/// resolve to. F5-TTS is an infilling model: `ref_text` MUST transcribe
/// `ref_audio`, or the duration estimate `ref_audio_len/ref_text_len * gen_len`
/// runs long and the tail degenerates into repeated filler. The old placeholder
/// ("the quick brown fox …") was unrelated to the JFK audio, so long-form
/// outputs bloated ~2× and collapsed into "point point point …".
const DEFAULT_REF_TEXT: &str =
    "And so my fellow Americans, ask not what your country can do for you.";

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "f5tts",
        supports_clone: true,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_F5TTS_DIR"],
            marker_files: vec!["vocab.txt"],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = F5Native::load_on(&dir, device).context("load f5tts")?;
    Ok(Box::new(F5Adapter { inner, dir }))
}

struct F5Adapter {
    inner: F5Native,
    dir: PathBuf,
}

impl TtsAdapter for F5Adapter {
    fn id(&self) -> &'static str {
        "f5tts"
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
            .or_else(|| find_ref(&self.dir))
            .ok_or_else(|| anyhow::anyhow!("f5tts needs reference wav"))?;
        let (ref_audio, _) = read_wav_mono(&ref_path)?;
        let ref_text = req
            .clone
            .as_ref()
            .and_then(|c| c.ref_text)
            .unwrap_or(DEFAULT_REF_TEXT);
        let opts = InferOpts::default();
        let t0 = Instant::now();
        let pcm = self
            .inner
            .synthesize(req.text, &ref_audio, ref_text, &opts)?;
        Ok(SynthResult {
            pcm,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}

fn find_ref(dir: &std::path::Path) -> Option<PathBuf> {
    for name in ["ref.wav", "prompt.wav", "default_voice.wav"] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let jfk = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    jfk.is_file().then_some(jfk)
}
