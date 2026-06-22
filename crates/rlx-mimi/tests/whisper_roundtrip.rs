//! Mimi encode → decode → Whisper ASR on clips from the Whisper bench and Qwen3-TTS examples.
//!
//! ```sh
//! just fetch-mimi fetch-whisper-base
//! just test-mimi-whisper
//! ```

use anyhow::{Context, Result, ensure};
use rlx_mimi::{MimiCodec, SAMPLE_RATE as MIMI_RATE};
use rlx_runtime::Device;
use rlx_whisper::{
    JFK_REFERENCE, SAMPLE_RATE as WHISPER_RATE, WhisperRunner, ensure_jfk_fixture,
    load_wav_mono_f32, normalize_transcript,
};
use std::path::{Path, PathBuf};

const MIN_PEAK: f32 = 1e-4;
const TARGET_PEAK: f32 = 0.95;
const MIN_PCM_CORR: f32 = 0.75;
const MIN_WORD_OVERLAP: f32 = 0.85;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn mimi_dir() -> Option<PathBuf> {
    std::env::var("RLX_MIMI_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
        .or_else(|| {
            let d = repo_root().join(".cache/mimi");
            d.join("model.safetensors").is_file().then_some(d)
        })
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    let cache = repo_root().join(".cache");
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        let p = cache.join(name);
        if p.join("model.safetensors").is_file() && p.join("tokenizer.json").is_file() {
            return Some(p);
        }
    }
    None
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * from_hz as f64 / to_hz as f64;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

fn scale_to_peak(pcm: &[f32], target: f32) -> Vec<f32> {
    let p = peak(pcm);
    if p < MIN_PEAK {
        return pcm.to_vec();
    }
    pcm.iter().map(|v| v * (target / p)).collect()
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let n = n as f32;
    let ma = a.iter().take(n as usize).sum::<f32>() / n;
    let mb = b.iter().take(n as usize).sum::<f32>() / n;
    let mut num = 0f32;
    let mut da = 0f32;
    let mut db = 0f32;
    for i in 0..n as usize {
        let dx = a[i] - ma;
        let dy = b[i] - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-8)
}

fn normalize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

fn word_overlap(reference: &str, transcript: &str) -> f32 {
    let reference_words = normalize_words(reference);
    if reference_words.is_empty() {
        return 0.0;
    }
    let heard = normalize_words(transcript);
    let hits = reference_words
        .iter()
        .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
        .count();
    hits as f32 / reference_words.len() as f32
}

fn whisper_runner(dir: &Path) -> Result<WhisperRunner> {
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
}

fn transcribe_for_whisper(
    pcm: &[f32],
    sample_rate: u32,
    whisper: &mut WhisperRunner,
) -> Result<String> {
    let scaled = scale_to_peak(pcm, TARGET_PEAK);
    let pcm_16k = resample_linear(&scaled, sample_rate, WHISPER_RATE as u32);
    ensure!(
        pcm_16k.len() >= WHISPER_RATE / 2,
        "audio too short for Whisper after resample"
    );
    whisper.transcribe_greedy(&pcm_16k)
}

struct Clip {
    id: &'static str,
    wav: PathBuf,
    /// When set, use this text instead of Whisper on the source clip.
    reference: Option<&'static str>,
    source_rate: u32,
}

fn qwen3_audio(name: &str) -> PathBuf {
    repo_root()
        .join("crates/rlx-qwen3-tts/examples/audio")
        .join(name)
}

fn collect_clips() -> Result<Vec<Clip>> {
    let mut clips = Vec::new();

    let (jfk_wav, _jfk_ref) = ensure_jfk_fixture().context("JFK bench wav")?;
    clips.push(Clip {
        id: "jfk",
        wav: jfk_wav,
        reference: Some(JFK_REFERENCE),
        source_rate: WHISPER_RATE as u32,
    });

    for (id, file) in [
        ("ask_not", "ask_not.wav"),
        ("moon", "moon.wav"),
        ("voice_chat_question", "voice_chat_question.wav"),
        ("voice_chat_reply", "voice_chat_reply.wav"),
    ] {
        let path = qwen3_audio(file);
        ensure!(path.is_file(), "missing qwen3-tts example wav {path:?}");
        clips.push(Clip {
            id,
            wav: path,
            reference: None,
            source_rate: 0, // filled from loader
        });
    }
    Ok(clips)
}

fn load_pcm_any(path: &Path) -> Result<(Vec<f32>, u32)> {
    use hound::WavReader;
    let reader = WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let rate = reader.spec().sample_rate;
    let pcm = if rate == MIMI_RATE {
        rlx_mimi::audio::load_wav_mono(path, MIMI_RATE)?
    } else {
        load_wav_mono_f32(path)?
    };
    Ok((pcm, rate))
}

#[test]
fn mimi_whisper_roundtrip_on_tts_and_bench_clips() -> Result<()> {
    let Some(mimi_dir) = mimi_dir() else {
        eprintln!("skip: run `just fetch-mimi`");
        return Ok(());
    };
    let Some(whisper_dir) = whisper_dir() else {
        eprintln!("skip: run `just fetch-whisper-base` or `just fetch-whisper`");
        return Ok(());
    };

    let codec = MimiCodec::open(&mimi_dir)?;
    let clips = collect_clips()?;
    let mut whisper = whisper_runner(&whisper_dir)?;

    eprintln!("mimi: {}", mimi_dir.display());
    eprintln!("whisper: {}", whisper_dir.display());

    for clip in clips {
        let (pcm, rate) = load_pcm_any(&clip.wav)?;
        let rate = if clip.source_rate > 0 {
            clip.source_rate
        } else {
            rate
        };
        let pcm_24k = if rate == MIMI_RATE {
            pcm.clone()
        } else {
            resample_linear(&pcm, rate, MIMI_RATE)
        };
        ensure!(!pcm_24k.is_empty(), "{}: empty pcm", clip.id);

        let frames = codec.encode_pcm(&pcm_24k, None)?;
        ensure!(!frames.frames.is_empty(), "{}: no codec frames", clip.id);
        let mut recon = codec.decode_codes(&frames)?;
        recon.truncate(pcm_24k.len().min(recon.len()));

        let corr = pearson(&pcm_24k, &recon);
        eprintln!(
            "{:>22}: {} frames, pcm corr={corr:.3}, len {} → {}",
            clip.id,
            frames.num_frames(),
            pcm_24k.len(),
            recon.len()
        );
        assert!(
            corr >= MIN_PCM_CORR,
            "{}: pcm correlation {corr:.3} < {MIN_PCM_CORR}",
            clip.id
        );

        let reference_text = if let Some(fixed) = clip.reference {
            fixed.to_string()
        } else {
            transcribe_for_whisper(&pcm, rate, &mut whisper)?
        };
        let recon_text = transcribe_for_whisper(&recon, MIMI_RATE, &mut whisper)?;
        let overlap = word_overlap(&reference_text, &recon_text);

        eprintln!("  ref whisper: {}", normalize_transcript(&reference_text));
        eprintln!("  recon ASR:   {}", normalize_transcript(&recon_text));
        eprintln!("  word overlap: {overlap:.2}");

        assert!(
            overlap >= MIN_WORD_OVERLAP,
            "{}: Whisper word overlap {overlap:.2} < {MIN_WORD_OVERLAP}\n\
             ref: {}\nrecon: {}",
            clip.id,
            normalize_transcript(&reference_text),
            normalize_transcript(&recon_text),
        );
    }
    Ok(())
}
