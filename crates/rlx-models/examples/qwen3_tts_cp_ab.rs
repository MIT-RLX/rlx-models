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

// A/B: code predictor CPU eager vs CPU compiled (`predict_groups` loop).
//
// ```bash
// export RLX_QWEN3_TTS_DIR=.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice
// cargo run -p rlx-models --example qwen3_tts_cp_ab --release -- --frames 22
// ```

use anyhow::Context;
use rlx_qwen3_tts::{
    Qwen3TtsConfig, bench_cp_ab,
    load::Qwen3TtsWeightStore,
    prompt::{build_custom_voice_prompt, load_text_tokenizer},
    talker::TalkerEngine,
    text_embed::TextEmbedder,
};
use rlx_runtime::Device;
use std::env;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut frames = 22usize;
    let mut warmup = 2usize;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--frames" => frames = it.next().context("--frames")?.parse()?,
            "--warmup" => warmup = it.next().context("--warmup")?.parse()?,
            "--help" | "-h" => {
                eprintln!("qwen3_tts_cp_ab [--frames N] [--warmup N]");
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }

    let model_dir = env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
    anyhow::ensure!(model_dir.is_dir(), "model dir: {}", model_dir.display());

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir)?;
    let store = Qwen3TtsWeightStore::open(&model_dir)?;
    let tokenizer = load_text_tokenizer(&model_dir)?;
    let text_embedder = TextEmbedder::open(&store)?;
    let prompt = build_custom_voice_prompt(
        &cfg,
        &store,
        &text_embedder,
        &tokenizer,
        "Hi.",
        "vivian",
        "english",
    )?;

    let mut talker = TalkerEngine::open(&store, cfg.talker(), Device::Cpu)?;
    talker.warmup(prompt.embeds.nrows().max(8))?;
    let hidden = talker.prefill(prompt.embeds.view())?;
    let h_last = hidden.row(hidden.nrows() - 1);

    let (eager, compiled) =
        bench_cp_ab(&store, cfg.code_predictor(), h_last.view(), frames, warmup)?;
    eager.print_line();
    compiled.print_line();
    let winner = if compiled.ms_per_frame < eager.ms_per_frame {
        "CPU compiled"
    } else {
        "CPU eager"
    };
    let delta = (eager.ms_per_frame - compiled.ms_per_frame).abs();
    eprintln!("[qwen3-tts cp-bench] winner: {winner} (Δ {delta:.2}ms/frame, {frames} frames)");
    Ok(())
}
