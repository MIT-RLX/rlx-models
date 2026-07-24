use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_pocket_tts::{GenerationOptions, SAMPLE_RATE, TtsModel, Voice};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

const DEFAULT_DIR: &str = "weights/tts/pocket-tts";
const WEIGHTS_FILE: &str = "tts_b6369a24.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.model";

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "pocket-tts",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_DIR),
            env_keys: vec!["RLX_POCKET_TTS_DIR"],
            marker_files: vec![TOKENIZER_FILE],
        },
    }
}

pub fn make(_device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let weights = find_weights(&dir)?;
    let tokenizer = dir.join(TOKENIZER_FILE);
    let model = TtsModel::open(&weights, &tokenizer).context("load pocket-tts")?;
    let voice_path = resolve_voice_path(&dir)?;
    let voice = Voice::open(&voice_path)
        .with_context(|| format!("load voice {}", voice_path.display()))?;
    Ok(Box::new(PocketAdapter { model, voice }))
}

struct PocketAdapter {
    model: TtsModel,
    voice: Voice,
}

impl TtsAdapter for PocketAdapter {
    fn id(&self) -> &'static str {
        "pocket-tts"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        let mut opts = GenerationOptions::default();
        opts.seed = req.seed;
        let t0 = Instant::now();
        let audio = self.model.generate(req.text, &self.voice, opts)?;
        Ok(SynthResult {
            pcm: audio.samples,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}

fn find_weights(dir: &Path) -> Result<PathBuf> {
    let known = dir.join(WEIGHTS_FILE);
    if known.is_file() {
        return Ok(known);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "safetensors")
            && path.file_name().is_some_and(|n| n != "tokenizer.model")
            && !path.components().any(|c| c.as_os_str() == "embeddings")
        {
            return Ok(path);
        }
    }
    bail!(
        "pocket-tts: no weights safetensors in {} (expected {WEIGHTS_FILE})",
        dir.display()
    )
}

fn resolve_voice_path(dir: &Path) -> Result<PathBuf> {
    let name = std::env::var("RLX_POCKET_TTS_VOICE").unwrap_or_else(|_| "alba".into());
    for candidate in [
        dir.join("embeddings").join(format!("{name}.safetensors")),
        dir.join(format!("{name}.safetensors")),
        dir.join("voices").join(format!("{name}.safetensors")),
    ] {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "pocket-tts: voice {name:?} not found under {} (set RLX_POCKET_TTS_VOICE)",
        dir.display()
    )
}
