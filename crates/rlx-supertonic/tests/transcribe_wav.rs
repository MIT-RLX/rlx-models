// Ad-hoc: transcribe an arbitrary WAV via rlx-whisper. Used to sanity-check
// vocoder/copy-synthesis output. Set WAV_PATH + RLX_WHISPER_DIR. Skips otherwise.
#![cfg(feature = "onnx")]

use std::path::PathBuf;

use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    None
}

#[test]
fn transcribe_env_wav() {
    let Ok(wav_path) = std::env::var("WAV_PATH") else {
        eprintln!("skip: set WAV_PATH");
        return;
    };
    let Some(wd) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR");
        return;
    };
    let mut r = hound::WavReader::open(&wav_path).expect("open wav");
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    // resample to 16k
    let from = spec.sample_rate;
    let to = WR as u32;
    let n = (samples.len() as u64 * to as u64 / from as u64).max(1) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * f
        })
        .collect();
    let mut w = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper");
    let t = w.transcribe_greedy(&pcm).expect("transcribe");
    eprintln!("WAV_PATH={wav_path}\nTRANSCRIPT: {t}");
}
