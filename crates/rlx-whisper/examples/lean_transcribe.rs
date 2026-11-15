//! Whisper on a small-memory CPU, such as a Raspberry Pi CM4 with 4 GB (Reachy Mini Wireless).
//!
//! The CLI's greedy decoder compiles one decoder graph per power-of-two length bucket, each
//! holding its own copy of the decoder's weights: whisper-small then needs well over 4 GB. Here
//! the compiled encoder produces the cross-attention cache, is dropped with its weights, and the
//! host [`FastDecoder`](rlx_whisper::fast_decoder::FastDecoder) (weights loaded once, nothing
//! compiled per length) decodes, so the two never share memory.
//!
//! usage: lean_transcribe MODEL_DIR WAV [--lang en] [--beam N] [--pad]
//!   MODEL_DIR: model.safetensors, config.json, tokenizer.json, generation_config.json
//!   (Hugging Face `openai/whisper-*`); WAV: 16 kHz. Without `--pad`, the encoder takes the
//!   clip's own length rather than OpenAI's 30 s (the CLI's `--no-pad`).
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use rlx_whisper::fast_decoder::FastDecoder;
use rlx_whisper::whisperx::{ByteLevelDecoder, Ct2Suppression, ct2_beam_search_fast};

/// Resident memory now, MB (Linux; 0 elsewhere).
fn rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("VmRSS:").map(|v| v.trim().trim_end_matches(" kB").parse::<u64>().ok())))
        .flatten()
        .map_or(0, |kb| kb / 1024)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(dir), Some(wav)) = (args.get(1), args.get(2)) else {
        anyhow::bail!("usage: lean_transcribe MODEL_DIR WAV [--lang en] [--beam N] [--pad]");
    };
    let lang = flag(&args, "--lang").unwrap_or("en");
    let beam: usize = flag(&args, "--beam").map_or(Ok(1), str::parse).context("--beam")?;
    let pad = args.iter().any(|a| a == "--pad");
    let file = |name: &str| Path::new(dir).join(name);
    let start = Instant::now();

    let gen_json = std::fs::read_to_string(file("generation_config.json")).context("generation_config.json")?;
    let generation: serde_json::Value = serde_json::from_str(&gen_json)?;
    let id = |v: &serde_json::Value, what: &str| v.as_u64().map(|x| x as u32).with_context(|| format!("generation_config: {what}"));
    let eot = id(&generation["eos_token_id"], "eos_token_id")?;
    let mut prompt = vec![id(&generation["decoder_start_token_id"], "decoder_start_token_id")?];
    // English-only checkpoints (`*.en`) have no language or task tokens.
    if generation["is_multilingual"].as_bool().unwrap_or(false) {
        prompt.push(id(&generation["lang_to_id"][format!("<|{lang}|>")], "lang_to_id")?);
        prompt.push(id(&generation["task_to_id"]["transcribe"], "task_to_id")?);
    }
    prompt.push(id(&generation["no_timestamps_token_id"], "no_timestamps_token_id")?);
    let max_length = generation["max_length"].as_u64().unwrap_or(448) as usize;
    let suppress = Ct2Suppression::from_generation_config(&gen_json)?;
    let pcm = rlx_whisper::load_wav_mono_f32(Path::new(wav))?;
    eprintln!("[lean] {:.1} s of audio, prompt {prompt:?}", pcm.len() as f64 / 16_000.0);

    // Encoder, then the cross-attention cache; the runner and its weights go before the decoder
    // loads.
    let t = Instant::now();
    let (cross_k, cross_v, heads) = {
        let mut runner = rlx_whisper::WhisperRunner::builder()
            .weights(file("model.safetensors"))
            .config_path(file("config.json"))
            .device(rlx_runtime::Device::Cpu)
            .no_pad(!pad)
            .build()?;
        eprintln!("[lean] runner built in {:.1} s (RSS {} MB)", t.elapsed().as_secs_f64(), rss_mb());
        let te = Instant::now();
        let enc = runner.encode_pcm(&pcm)?;
        eprintln!("[lean] encode_pcm {:.1} s (RSS {} MB)", te.elapsed().as_secs_f64(), rss_mb());
        let cross = runner.cross_cache_batch(&enc, 1)?;
        eprintln!("[lean] encoded in {:.1} s (RSS {} MB)", t.elapsed().as_secs_f64(), rss_mb());
        (cross.layers_k, cross.layers_v, runner.config().decoder_attention_heads)
    };
    eprintln!("[lean] encoder dropped (RSS {} MB)", rss_mb());

    let t = Instant::now();
    let decoder = FastDecoder::load(&file("model.safetensors"), heads)?;
    eprintln!("[lean] decoder loaded in {:.1} s (RSS {} MB)", t.elapsed().as_secs_f64(), rss_mb());
    let t = Instant::now();
    let result = ct2_beam_search_fast(&decoder, &cross_k, &cross_v, &prompt, beam, 1.0, 1.0, max_length, &suppress, eot)?;
    let tokens = result.ids.len();
    eprintln!(
        "[lean] decoded {tokens} tokens in {:.1} s ({:.0} ms/token, RSS {} MB)",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() * 1000.0 / tokens.max(1) as f64,
        rss_mb()
    );
    let text = ByteLevelDecoder::from_tokenizer_json(&std::fs::read_to_string(file("tokenizer.json"))?)?
        .decode(&result.ids.iter().copied().filter(|&t| t < eot).collect::<Vec<_>>());
    eprintln!("[lean] total {:.1} s", start.elapsed().as_secs_f64());
    println!("{}", text.trim());
    Ok(())
}
