use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let wav = std::env::args()
        .nth(1)
        .unwrap_or("/tmp/gepard_hf_codes.wav".into());
    let reader = hound::WavReader::open(&wav)?;
    let sr = reader.spec().sample_rate;
    let pcm: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let pcm16 = if sr == 16000 {
        pcm
    } else {
        let n_out = (pcm.len() as u64 * 16000 / sr as u64) as usize;
        let scale = sr as f64 / 16000.0;
        (0..n_out)
            .map(|i| {
                let src = i as f64 * scale;
                let i0 = src.floor() as usize;
                let i1 = (i0 + 1).min(pcm.len() - 1);
                let t = (src - i0 as f64) as f32;
                pcm[i0] * (1.0 - t) + pcm[i1] * t
            })
            .collect()
    };
    let wd = PathBuf::from(".cache/whisper-tiny");
    let mut session = rlx_whisper::WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .build()?;
    let hyp = session.transcribe_greedy(&pcm16)?;
    println!("transcript: {hyp:?}");
    Ok(())
}
