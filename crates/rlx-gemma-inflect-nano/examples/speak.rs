//! Gemma 3 270M chat → Inflect-Nano TTS: prompt in, WAV out.
//!
//! ```sh
//! just fetch-gemma3-270m
//! # export Inflect bundle once (see crates/rlx-inflect-nano/README.md)
//! just gemma-inflect-speak -- --user "What is two plus two?"
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_cli::{parse_gemma_device, req};
use rlx_gemma::{GemmaRunner, encode_chat_prompt_auto};
use rlx_inflect_nano::{InferOpts, InflectNano};
use rlx_qwen3::SampleOpts;
use rlx_qwen35::decode_ids_auto;
use rlx_runtime::{Device, is_available};

fn usage() {
    eprintln!(
        "speak — Gemma 3 270M + Inflect-Nano TTS\n\
         \n\
         Flags:\n\
           --gemma-gguf PATH     Gemma GGUF (default: RLX_GEMMA3_GGUF or /tmp/rlx-weights/gemma-3-270m.gguf)\n\
           --tokenizer PATH      tokenizer.json (default: sibling of GGUF or RLX_GEMMA3_TOKENIZER)\n\
           --inflect-data PATH   Inflect RLX bundle (default: RLX_INFLECT_NANO_DATA or weights/inflect-nano-rlx)\n\
           --user TEXT           User message (chat template)\n\
           --system TEXT         Optional system prompt\n\
           --device DEVICE       Gemma backend (cpu, metal, mlx, …)\n\
           --tts-device DEVICE   Inflect vocoder backend (auto, cpu, metal, mlx, …)\n\
           --max-tokens N        New tokens to generate (default: 64)\n\
           --max-seq N           Compile context length (default: 256)\n\
           --packed              Packed GGUF decode (default on)\n\
           --no-packed           Disable packed GGUF\n\
           --out PATH            Output WAV (default: /tmp/gemma-inflect-speak.wav)\n\
           --full-reply          Speak full reply instead of first sentence\n"
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
        return Ok(());
    }

    let mut gemma_gguf = rlx_gemma_inflect_nano::default_gemma_gguf();
    let mut tokenizer: Option<PathBuf> = rlx_gemma_inflect_nano::default_gemma_tokenizer();
    let mut inflect_data = rlx_gemma_inflect_nano::default_inflect_data_dir();
    let mut user: Option<String> = None;
    let mut system: Option<String> = None;
    let mut device = "metal".to_string();
    let mut tts_device = "auto".to_string();
    let mut max_tokens = 64usize;
    let mut max_seq = 256usize;
    let mut packed = true;
    let mut out = PathBuf::from("/tmp/gemma-inflect-speak.wav");
    let mut first_sentence_only = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gemma-gguf" => gemma_gguf = req(&args, &mut i)?.into(),
            "--tokenizer" => tokenizer = Some(req(&args, &mut i)?.into()),
            "--inflect-data" => inflect_data = req(&args, &mut i)?.into(),
            "--user" | "--prompt" => user = Some(req(&args, &mut i)?),
            "--system" => system = Some(req(&args, &mut i)?),
            "--device" => device = req(&args, &mut i)?,
            "--tts-device" => tts_device = req(&args, &mut i)?,
            "--max-tokens" => {
                max_tokens = req(&args, &mut i)?.parse().context("--max-tokens: usize")?;
            }
            "--max-seq" => max_seq = req(&args, &mut i)?.parse().context("--max-seq: usize")?,
            "--packed" => {
                packed = true;
                i += 1;
            }
            "--no-packed" => {
                packed = false;
                i += 1;
            }
            "--out" => out = req(&args, &mut i)?.into(),
            "--full-reply" => {
                first_sentence_only = false;
                i += 1;
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let user = user.unwrap_or_else(|| {
        "Say one short friendly sentence about running small language models on device.".to_string()
    });

    let tok = tokenizer.as_deref();
    rlx_gemma_inflect_nano::ensure_paths_exist(&gemma_gguf, tok, &inflect_data)?;

    let lm_device = parse_gemma_device(&device)?;
    let tts_dev = resolve_tts_device(&tts_device, lm_device)?;

    eprintln!(
        "[speak] gemma={gemma_gguf:?} inflect={inflect_data:?} lm={lm_device:?} tts={tts_dev:?} packed={packed}"
    );

    let prompt_ids = encode_chat_prompt_auto(
        &gemma_gguf,
        tok,
        system.as_deref(),
        &user,
        true,
    )?;
    eprintln!("[speak] user: {user}");
    eprintln!("[speak] prompt tokens: {}", prompt_ids.len());

    let sample = SampleOpts::greedy();
    let mut runner = GemmaRunner::builder()
        .weights(&gemma_gguf)
        .device(lm_device)
        .max_seq(max_seq)
        .stream(true)
        .sample(sample)
        .packed_weights(packed)
        .build()?;

    let t_lm = Instant::now();
    print!("[gemma] ");
    let tokens = runner.generate(&prompt_ids, max_tokens, |tok_id| {
        match rlx_gemma::decode_token_auto(&gemma_gguf, tok, tok_id) {
            Ok(piece) => print!("{piece}"),
            Err(_) => print!("{tok_id} "),
        }
        std::io::stdout().flush().ok();
    })?;
    let lm_secs = t_lm.elapsed().as_secs_f32();
    println!();
    eprintln!(
        "[gemma] {} new tokens in {lm_secs:.2}s ({:.1} tok/s)",
        tokens.len(),
        tokens.len() as f32 / lm_secs.max(1e-6)
    );

    let reply = decode_ids_auto(&gemma_gguf, tok, &tokens, true)?;
    let speech = if first_sentence_only {
        rlx_gemma_inflect_nano::speech_text_from_reply(&reply)
    } else {
        reply.trim().replace('\n', " ")
    };
    if speech.is_empty() {
        bail!("empty reply from Gemma — try a longer --max-tokens");
    }
    eprintln!("[tts] text: {speech}");

    let inflect = InflectNano::load_from_dir(&inflect_data)?;
    let opts = InferOpts::default();
    let t_tts = Instant::now();
    let wav = inflect.synthesize_on(&speech, &opts, tts_dev)?;
    let tts_secs = t_tts.elapsed().as_secs_f32();
    let audio_secs = wav.samples.len() as f32 / wav.sample_rate as f32;

    rlx_inflect_nano::audio::write_wav(&out, &wav.samples, wav.sample_rate)?;
    eprintln!(
        "[tts] wrote {} ({audio_secs:.2}s audio in {tts_secs:.3}s, RTF {:.2}x)",
        out.display(),
        audio_secs / tts_secs.max(1e-6)
    );
    Ok(())
}

fn resolve_tts_device(spec: &str, lm_device: Device) -> Result<Device> {
    if spec.eq_ignore_ascii_case("auto") {
        if let Some(d) = InflectNano::preferred_accelerator() {
            return Ok(d);
        }
        return Ok(if is_available(lm_device) {
            lm_device
        } else {
            Device::Cpu
        });
    }
    parse_gemma_device(spec)
}
