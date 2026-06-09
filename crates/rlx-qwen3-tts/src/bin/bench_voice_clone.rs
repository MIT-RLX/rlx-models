// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Bench ICL vs XVectorOnly voice-clone modes on a Base checkpoint.
//!
//! Reads a fixture JSON (baked via `scripts/qwen3_tts_bake_clone_fixtures.py`)
//! that holds `ref_text`, per-frame `ref_code` (T × 16), and the 1024-d
//! `ref_spk_embedding`. Builds the two clone prompts, runs talker prefill +
//! the AR codec loop for each mode, reports prefill_ms, talker s, CP s, and
//! audio RTF.
//!
//! Speech-decode is skipped (we measure the talker+CP work, not the vocoder).

use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::parse_device;
use rlx_qwen3_tts::config::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::megakernel::Qwen3TtsMegakernel;
use rlx_qwen3_tts::prompt::load_text_tokenizer;
use rlx_qwen3_tts::text_embed::TextEmbedder;
use rlx_qwen3_tts::voice_clone::{VoiceClonePrompt, build_icl_prompt, build_x_vector_prompt};
use rlx_runtime::Device;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct CloneFixture {
    ref_text: String,
    n_frames: usize,
    n_groups: usize,
    ref_code: Vec<Vec<u32>>,
    spk_dim: usize,
    ref_spk_embedding: Vec<f32>,
}

struct Args {
    model_dir: PathBuf,
    fixture: PathBuf,
    target_text: String,
    device: Device,
    max_frames: usize,
    warmup_iters: usize,
    bench_iters: usize,
    modes: Vec<&'static str>,
}

fn parse_args() -> Result<Args> {
    let mut model_dir: Option<PathBuf> = None;
    let mut fixture: Option<PathBuf> = None;
    let mut target_text = "Hello world. This is a synthetic clone target sentence.".to_string();
    let mut device = Device::Cpu;
    let mut max_frames = 128usize;
    let mut warmup_iters = 1usize;
    let mut bench_iters = 3usize;
    let mut modes: Vec<&'static str> = vec!["icl", "xvec"];
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            raw.get(*i)
                .cloned()
                .ok_or_else(|| anyhow!("missing value for {}", raw[*i - 1]))
        };
        match raw[i].as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(take(&mut i)?)),
            "--fixture" => fixture = Some(PathBuf::from(take(&mut i)?)),
            "--target-text" => target_text = take(&mut i)?,
            "--device" => device = parse_device(&take(&mut i)?)?,
            "--max-frames" => max_frames = take(&mut i)?.parse()?,
            "--warmup-iters" => warmup_iters = take(&mut i)?.parse()?,
            "--bench-iters" => bench_iters = take(&mut i)?.parse()?,
            "--mode" => {
                let v = take(&mut i)?;
                modes = match v.as_str() {
                    "icl" => vec!["icl"],
                    "xvec" | "x-vector" | "xvector" => vec!["xvec"],
                    "both" => vec!["icl", "xvec"],
                    other => bail!("unknown --mode {other:?} (icl|xvec|both)"),
                };
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown arg {other:?}"),
        }
        i += 1;
    }
    Ok(Args {
        model_dir: model_dir.context("--model-dir required")?,
        fixture: fixture.context("--fixture required")?,
        target_text,
        device,
        max_frames,
        warmup_iters,
        bench_iters,
        modes,
    })
}

fn print_help() {
    eprintln!(
        "Usage: bench_voice_clone --model-dir <Base> --fixture <baked.json> [opts]\n\n\
         Options:\n  \
           --target-text <str>      (default: 'Hello world. ...')\n  \
           --device <cpu|metal|mlx|cuda|rocm|auto>  (default: cpu)\n  \
           --max-frames <N>         (default: 128 codec frames)\n  \
           --warmup-iters <N>       (default: 1)\n  \
           --bench-iters <N>        (default: 3)\n  \
           --mode <icl|xvec|both>   (default: both)\n"
    );
}

struct IterTiming {
    prefill_secs: f64,
    talker_secs: f64,
    cp_secs: f64,
    total_secs: f64,
    n_frames: usize,
}

fn run_one(
    mk: &mut Qwen3TtsMegakernel,
    cfg: &Qwen3TtsConfig,
    prompt: &VoiceClonePrompt,
    max_frames: usize,
) -> Result<IterTiming> {
    let t = Instant::now();
    let (frames, ts) = mk.synthesize_codec_ar(
        prompt.embeds.view(),
        cfg.talker(),
        max_frames,
        1,
        1.0,
        &prompt.tts_pad_embed,
        None,
    )?;
    let total = t.elapsed().as_secs_f64();
    Ok(IterTiming {
        prefill_secs: ts.prefill_secs,
        talker_secs: ts.talker_secs,
        cp_secs: ts.cp_secs,
        total_secs: total,
        n_frames: frames.len(),
    })
}

fn fmt_ms(secs: f64) -> String {
    format!("{:>8.1} ms", secs * 1000.0)
}

fn summarize(label: &str, prefill_rows: usize, results: &[IterTiming]) {
    let n = results.len() as f64;
    let mean = |sel: fn(&IterTiming) -> f64| results.iter().map(sel).sum::<f64>() / n;
    let median = |sel: fn(&IterTiming) -> f64| {
        let mut xs: Vec<f64> = results.iter().map(sel).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };
    let prefill = mean(|t| t.prefill_secs);
    let talker = mean(|t| t.talker_secs);
    let cp = mean(|t| t.cp_secs);
    let total = mean(|t| t.total_secs);
    let frames = results.iter().map(|t| t.n_frames as f64).sum::<f64>() / n;
    let audio_secs = frames / 12.0;
    let rtf = if audio_secs > 0.0 {
        total / audio_secs
    } else {
        f64::NAN
    };
    println!("\n=== {label} (prefill_rows={prefill_rows}) ===");
    println!(
        "  prefill: {} (median {})",
        fmt_ms(prefill),
        fmt_ms(median(|t| t.prefill_secs))
    );
    println!(
        "  talker : {} ({:.1} ms/frame)",
        fmt_ms(talker),
        if frames > 0.0 {
            talker * 1000.0 / frames
        } else {
            0.0
        }
    );
    println!(
        "  CP     : {} ({:.1} ms/frame)",
        fmt_ms(cp),
        if frames > 0.0 {
            cp * 1000.0 / frames
        } else {
            0.0
        }
    );
    println!(
        "  total  : {} (median {})",
        fmt_ms(total),
        fmt_ms(median(|t| t.total_secs))
    );
    println!(
        "  frames : {:.1}  →  audio {:.2} s  →  rtf {:.2}x",
        frames, audio_secs, rtf
    );
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let fixture_bytes = std::fs::read(&args.fixture)
        .with_context(|| format!("read fixture {}", args.fixture.display()))?;
    let fx: CloneFixture = serde_json::from_slice(&fixture_bytes)?;
    if fx.ref_code.len() != fx.n_frames {
        bail!(
            "fixture: ref_code rows {} != n_frames {}",
            fx.ref_code.len(),
            fx.n_frames
        );
    }
    if fx.ref_spk_embedding.len() != fx.spk_dim {
        bail!(
            "fixture: spk len {} != spk_dim {}",
            fx.ref_spk_embedding.len(),
            fx.spk_dim
        );
    }

    println!(
        "[bench] model={} device={:?} max_frames={} warmup={} iters={}",
        args.model_dir.display(),
        args.device,
        args.max_frames,
        args.warmup_iters,
        args.bench_iters
    );
    println!(
        "[bench] fixture: ref_code {}x{}  spk_dim={}",
        fx.n_frames, fx.n_groups, fx.spk_dim
    );

    let cfg = Qwen3TtsConfig::from_model_dir(&args.model_dir)?;
    let talker = cfg.talker();
    if fx.spk_dim != talker.hidden_size {
        bail!(
            "fixture spk_dim {} != talker hidden_size {}",
            fx.spk_dim,
            talker.hidden_size
        );
    }
    if fx.n_groups != talker.num_code_groups {
        bail!(
            "fixture n_groups {} != talker num_code_groups {}",
            fx.n_groups,
            talker.num_code_groups
        );
    }

    let store = Qwen3TtsWeightStore::open(&args.model_dir)?;
    let text_embedder = TextEmbedder::open(&store)?;
    let tokenizer = load_text_tokenizer(&args.model_dir)?;

    for mode in args.modes.iter() {
        let prompt = match *mode {
            "icl" => build_icl_prompt(
                &cfg,
                &store,
                &text_embedder,
                &tokenizer,
                &args.target_text,
                &fx.ref_text,
                &fx.ref_code,
                &fx.ref_spk_embedding,
            )?,
            "xvec" => build_x_vector_prompt(
                &cfg,
                &store,
                &text_embedder,
                &tokenizer,
                &args.target_text,
                &fx.ref_spk_embedding,
            )?,
            _ => unreachable!(),
        };
        let prefill_rows = prompt.embeds.nrows();
        println!(
            "\n[mode={}] prompt prefill rows = {} (hidden={})",
            mode,
            prefill_rows,
            prompt.embeds.ncols()
        );

        let mut mk =
            Qwen3TtsMegakernel::open(&store, cfg.talker(), cfg.code_predictor(), args.device)?;
        mk.warmup(prompt.embeds.view(), args.max_frames, None)?;

        for w in 0..args.warmup_iters {
            let t = run_one(&mut mk, &cfg, &prompt, args.max_frames)?;
            println!(
                "  [warmup {}] frames={} total={:.1} ms",
                w,
                t.n_frames,
                t.total_secs * 1000.0
            );
        }

        let mut results = Vec::with_capacity(args.bench_iters);
        for k in 0..args.bench_iters {
            let t = run_one(&mut mk, &cfg, &prompt, args.max_frames)?;
            println!(
                "  [iter {}] frames={} prefill={:.1} talker={:.1} cp={:.1} total={:.1} ms",
                k,
                t.n_frames,
                t.prefill_secs * 1000.0,
                t.talker_secs * 1000.0,
                t.cp_secs * 1000.0,
                t.total_secs * 1000.0,
            );
            results.push(t);
        }

        let label = match *mode {
            "icl" => "ICL",
            "xvec" => "XVectorOnly",
            _ => unreachable!(),
        };
        summarize(label, prefill_rows, &results);
    }

    Ok(())
}
