use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_pocket_tts::{ModelSpec, PocketTts, SAMPLE_RATE, VoiceState};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

/// The January mirror; point `RLX_POCKET_TTS_DIR` at a
/// `kyutai/pocket-tts-without-voice-cloning` `languages/<name>` directory for
/// the current models.
const DEFAULT_DIR: &str = "weights/tts/pocket-tts";
const TOKENIZER_FILE: &str = "tokenizer.model";
const JANUARY_WEIGHTS: &str = "tts_b6369a24.safetensors";

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

/// `RLX_POCKET_TTS_LANGUAGE`, else the January config for the mirror, else the
/// directory's own name when it is a bundled config, else `english`.
fn language_for(dir: &Path) -> String {
    if let Ok(l) = rlx_ir::env::var_result("RLX_POCKET_TTS_LANGUAGE") {
        return l;
    }
    if dir.join(JANUARY_WEIGHTS).is_file() {
        return "english_2026-01".into();
    }
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    if ModelSpec::builtin_languages().contains(&name) {
        name.to_string()
    } else {
        "english".into()
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let language = language_for(&dir);
    let mut model = PocketTts::open(&dir, &language, device)
        .with_context(|| format!("load pocket-tts {language} from {}", dir.display()))?;
    let name = rlx_ir::env::var_result("RLX_POCKET_TTS_VOICE").unwrap_or_else(|_| "alba".into());
    let voice = model
        .load_voice(&name)
        .with_context(|| format!("load voice {name} (set RLX_POCKET_TTS_VOICE)"))?;
    Ok(Box::new(PocketAdapter { model, voice }))
}

struct PocketAdapter {
    model: PocketTts,
    voice: VoiceState,
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
        self.model.options.seed = req.seed;
        if req.deterministic {
            // Zero sampler noise: the same latent path on every backend.
            self.model.options.temperature = 0.0;
        }
        let t0 = Instant::now();
        let audio = self.model.generate(req.text, &self.voice)?;
        Ok(SynthResult {
            pcm: audio.samples,
            sample_rate: SAMPLE_RATE,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", self.model.device()),
        })
    }
}
