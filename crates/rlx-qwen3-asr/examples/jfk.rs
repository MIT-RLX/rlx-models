// JFK transcription test: offline + streaming, with time / RTF / WER.
//   cargo run --release -p rlx-qwen3-asr --features metal --example jfk -- <model_dir> <wav> <device>
use anyhow::Result;
use rlx_qwen3_asr::{AsrRunner, audio::SAMPLE_RATE, audio::load_wav_mono_f32};
use std::time::Instant;

// JFK inaugural address (1961) — public-domain US government work; reference for WER.
const REF: &str = "and so my fellow americans ask not what your country can do for you \
ask what you can do for your country";

fn norm(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(|w| w.to_string())
        .collect()
}

fn wer(reference: &str, hyp: &str) -> f64 {
    let r = norm(reference);
    let h = norm(hyp);
    let (n, m) = (r.len(), h.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    if n == 0 {
        0.0
    } else {
        dp[n][m] as f64 / n as f64
    }
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let model_dir = a.next().unwrap_or("/Users/Shared/qwen3-asr-0.6b".into());
    let wav = a
        .next()
        .unwrap_or("/Users/Shared/rlx-models/.cache/whisper-bench/jfk_16k.wav".into());
    let dev_str = a.next().unwrap_or("cpu".into());
    let device = rlx_cli::parse_standard_device("jfk", &dev_str)?;

    let pcm = load_wav_mono_f32(std::path::Path::new(&wav))?;
    let dur_s = pcm.len() as f64 / SAMPLE_RATE as f64;
    let runner = AsrRunner::builder()
        .weights(&model_dir)
        .device(device)
        .max_new_tokens(80)
        .build()?;

    println!("══ JFK test · device={dev_str} · {dur_s:.2}s audio ══");

    // ── offline ──
    let t = Instant::now();
    let text = runner.transcribe_pcm(&pcm, "")?;
    let off_s = t.elapsed().as_secs_f64();
    println!(
        "\n[offline]  {off_s:.2}s  RTF={:.1}×  WER={:.1}%",
        dur_s / off_s,
        wer(REF, &text) * 100.0
    );
    println!("  text: {text:?}");

    // ── streaming (6 s chunks) ──
    let t = Instant::now();
    let chunks = runner.transcribe_pcm_streaming(&pcm, "", 6.0)?;
    let str_s = t.elapsed().as_secs_f64();
    let joined = chunks
        .iter()
        .map(|c| c.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "\n[streaming] {str_s:.2}s  RTF={:.1}×  WER={:.1}%  ({} chunks)",
        dur_s / str_s,
        wer(REF, &joined) * 100.0,
        chunks.len()
    );
    for c in &chunks {
        println!(
            "  [{:.1}-{:.1}s] {:.0}ms: {:?}",
            c.start_s,
            c.end_s,
            c.latency_ms,
            c.text.trim()
        );
    }
    println!(
        "\nRESULT device={dev_str} offline_s={off_s:.3} offline_wer={:.3} stream_s={str_s:.3} stream_wer={:.3}",
        wer(REF, &text),
        wer(REF, &joined)
    );
    Ok(())
}
