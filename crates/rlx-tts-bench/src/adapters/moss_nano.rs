use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_moss_nano::{DEFAULT_LOCAL_DIR, MossNative, NativeOpts};
use rlx_runtime::Device;

use crate::adapter::{AdapterMeta, SynthRequest, SynthResult, TtsAdapter, WeightHints};

pub fn meta() -> AdapterMeta {
    AdapterMeta {
        id: "moss-nano",
        supports_clone: false,
        feature: "matrix-onnx",
        hints: WeightHints {
            default_dir: PathBuf::from(DEFAULT_LOCAL_DIR),
            env_keys: vec!["RLX_MOSS_NANO_DIR"],
            marker_files: vec![
                "moss-nano.rlxp",
                "moss-nano.rlx",
                "moss-nano.gguf",
                "browser_poc_manifest.json",
                "tokenizer.json",
            ],
        },
    }
}

pub fn make(device: Device) -> Result<Box<dyn TtsAdapter>> {
    let dir = meta()
        .hints
        .resolve_dir()
        .ok_or_else(|| anyhow::anyhow!(meta().hints.missing_reason()))?;
    let inner = MossNative::load_on(&dir, device)
        .with_context(|| format!("load moss-nano from {}", dir.display()))?;
    // The FIRST builtin voice is Chinese ("Junhao"), which garbles English text
    // on every backend. Prefer an English voice ("Trump" is the validated one),
    // overridable via RLX_MOSS_VOICE.
    let voices = inner.voice_names();
    let voice = std::env::var("RLX_MOSS_VOICE")
        .ok()
        .filter(|v| voices.iter().any(|n| n == v))
        .or_else(|| {
            ["Trump", "Alice", "Bob", "en", "English"].iter().find_map(|w| {
                voices
                    .iter()
                    .find(|n| n.eq_ignore_ascii_case(w) || n.to_lowercase().contains(&w.to_lowercase()))
                    .cloned()
            })
        })
        .or_else(|| voices.first().cloned())
        .unwrap_or_else(|| "default".into());
    eprintln!("[moss-nano] voice={voice}");
    Ok(Box::new(MossAdapter { inner, voice }))
}

struct MossAdapter {
    inner: MossNative,
    voice: String,
}

impl TtsAdapter for MossAdapter {
    fn id(&self) -> &'static str {
        "moss-nano"
    }
    fn weight_hints(&self) -> WeightHints {
        meta().hints
    }
    fn supports_clone(&self) -> bool {
        false
    }

    fn synthesize(&mut self, req: SynthRequest<'_>) -> Result<SynthResult> {
        // Fox pangram validates at ~64–96 frames; longer prompts need more or the
        // model stops early and Whisper hears filler ("..."). Scale with words.
        // Also sentence-chunk long text — a single long English utterance often
        // collapses to unintelligible audio with the builtin Trump codes.
        let t0 = Instant::now();
        let chunks = split_moss_chunks(req.text);
        let mut pcm_acc: Vec<f32> = Vec::new();
        let ch = self.inner.channels().max(1) as usize;
        let sr = self.inner.sample_rate();
        let gap = (sr as usize) / 20 * ch; // ~50 ms
        for chunk in &chunks {
            let words = chunk.split_whitespace().count().max(1);
            let base = std::env::var("RLX_MOSS_FRAMES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(64);
            let max_frames = (base + words.saturating_mul(4)).clamp(64, 256);
            let opts = NativeOpts {
                max_frames,
                ..NativeOpts::default()
            };
            let stereo = self
                .inner
                .synthesize(chunk, &self.voice, &opts)
                .with_context(|| format!("moss-nano synthesize chunk: {chunk:?}"))?;
            let mono: Vec<f32> = if ch <= 1 {
                stereo
            } else {
                stereo
                    .chunks(ch)
                    .map(|c| c.iter().sum::<f32>() / ch as f32)
                    .collect()
            };
            if !pcm_acc.is_empty() {
                pcm_acc.extend(std::iter::repeat_n(0.0f32, gap / ch.max(1)));
            }
            pcm_acc.extend(mono);
        }
        Ok(SynthResult {
            pcm: pcm_acc,
            sample_rate: sr,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
            exec_label: format!("{:?}", req.device),
        })
    }
}

fn split_moss_chunks(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '.' | '?' | '!') {
            let t = buf.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            buf.clear();
        }
    }
    let t = buf.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    if out.is_empty() {
        out.push(text.to_string());
    }
    // Keep chunks short enough for the codec-LM (~18 words works for fox).
    const MAX_WORDS: usize = 18;
    let mut packed = Vec::new();
    for chunk in out {
        let words: Vec<&str> = chunk.split_whitespace().collect();
        if words.len() <= MAX_WORDS {
            packed.push(chunk);
            continue;
        }
        for part in words.chunks(MAX_WORDS) {
            packed.push(part.join(" "));
        }
    }
    packed
}
