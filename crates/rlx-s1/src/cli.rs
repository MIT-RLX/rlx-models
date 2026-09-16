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

//! `rlx-s1` command line — normalize a transcript from a flag, a file, or stdin.

use crate::prompt::{Context, Controls, Structure, Styling, render_prompt};
use crate::runner::S1Runner;
use anyhow::{Context as _, Result, anyhow, bail};
use rlx_qwen3::Precision;
use std::io::Write;
use std::path::PathBuf;

const HELP: &str = "\
rlx-s1 — S1-mini by Superwhisper: text normalization for speech-to-text transcripts

USAGE:
    rlx-s1 --weights <PATH> (--transcript <TEXT> | --file <PATH> | --stdin) [OPTIONS]

MODEL:
    --weights <PATH>        s1-mini HF directory, .safetensors, or .gguf  (required)
    --tokenizer <PATH>      explicit tokenizer.json (default: next to the weights)
    --config <PATH>         explicit config.json (safetensors only)
    --device <NAME>         cpu|metal|mlx|cuda|rocm|gpu|vulkan|coreml (default cpu)
    --precision <P>         f32|f16-lm (default f32)
    --packed | --no-packed  keep K-quant GGUF weights packed in the arena
    --prefer-quant <SUB>    when --weights is a dir of .gguf, pick one by name (default q4_k_m)
    --strict-shape          fail instead of warn when the checkpoint isn't S1-mini's topology

INPUT (pick one):
    -t, --transcript <TEXT> raw ASR transcript
    -f, --file <PATH>       read the transcript from a file
        --stdin             read the transcript from stdin

CONTROL LINE (the model's only steering mechanism):
    --styling <V>           casual|semi-casual|semi-formal|formal   (default semi-formal)
    --structure <V>         prose|lists                             (default prose)
    --context <V>           general|email                           (default general)

DECODING (greedy always — normalization is deterministic):
    -n, --max-new-tokens <N>  default: 1.3 x prompt_tokens + 32
    --max-seq <N>             decode-bucket ceiling
    --chunk-tokens <N>        transcript budget per pass (default 896; 0 disables chunking)
    --prefill-bucket <N>      round prompt length up to a multiple of N before compiling the
                              prefill graph (default 64; 0 disables). Varying transcript lengths
                              otherwise recompile the prefill graph on nearly every call.
    --stream                  print tokens as they decode

OUTPUT:
    --show-prompt           print the exact prompt string and exit (no weights needed)
    --json                  emit a JSON object instead of bare text
    -h, --help              show this help

An empty result is a valid answer: filler-only input normalizes to the empty string.
";

/// CLI entry point (`rlx-s1 …`).
pub fn cli_run(args: &[String]) -> Result<()> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        return Ok(());
    }

    let mut weights: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut precision = "f32".to_string();
    let mut packed: Option<bool> = None;
    let mut prefer_quant: Option<String> = None;
    let mut strict_shape = false;

    let mut transcript: Option<String> = None;
    let mut file: Option<PathBuf> = None;
    let mut use_stdin = false;

    let mut styling = Styling::default();
    let mut structure = Structure::default();
    let mut context = Context::default();

    let mut max_new_tokens: Option<usize> = None;
    let mut max_seq: Option<usize> = None;
    let mut chunk_tokens: Option<usize> = None;
    let mut prefill_bucket: Option<usize> = None;
    let mut stream = false;
    let mut show_prompt = false;
    let mut json = false;

    let mut i = 0usize;
    let need = |args: &[String], i: &mut usize, flag: &str| -> Result<String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| anyhow!("{flag} needs a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(need(args, &mut i, "--weights")?.into()),
            "--tokenizer" => tokenizer = Some(need(args, &mut i, "--tokenizer")?.into()),
            "--config" => config = Some(need(args, &mut i, "--config")?.into()),
            "--device" => device = need(args, &mut i, "--device")?,
            "--precision" => precision = need(args, &mut i, "--precision")?,
            "--packed" => packed = Some(true),
            "--no-packed" => packed = Some(false),
            "--prefer-quant" => prefer_quant = Some(need(args, &mut i, "--prefer-quant")?),
            "--strict-shape" => strict_shape = true,
            "-t" | "--transcript" => transcript = Some(need(args, &mut i, "--transcript")?),
            "-f" | "--file" => file = Some(need(args, &mut i, "--file")?.into()),
            "--stdin" => use_stdin = true,
            "--styling" => styling = need(args, &mut i, "--styling")?.parse()?,
            "--structure" => structure = need(args, &mut i, "--structure")?.parse()?,
            "--context" => context = need(args, &mut i, "--context")?.parse()?,
            "-n" | "--max-new-tokens" => {
                max_new_tokens = Some(
                    need(args, &mut i, "--max-new-tokens")?
                        .parse()
                        .context("--max-new-tokens: expected integer")?,
                );
            }
            "--max-seq" => {
                max_seq = Some(
                    need(args, &mut i, "--max-seq")?
                        .parse()
                        .context("--max-seq: expected integer")?,
                );
            }
            "--chunk-tokens" => {
                chunk_tokens = Some(
                    need(args, &mut i, "--chunk-tokens")?
                        .parse()
                        .context("--chunk-tokens: expected integer")?,
                );
            }
            "--prefill-bucket" => {
                prefill_bucket = Some(
                    need(args, &mut i, "--prefill-bucket")?
                        .parse()
                        .context("--prefill-bucket: expected integer")?,
                );
            }
            "--stream" => stream = true,
            "--show-prompt" => show_prompt = true,
            "--json" => json = true,
            other => bail!("unknown arg {other:?} (see --help)"),
        }
        i += 1;
    }

    let controls = Controls {
        styling,
        structure,
        context,
    };

    let transcript =
        match (transcript, file, use_stdin) {
            (Some(t), None, false) => t,
            (None, Some(p), false) => std::fs::read_to_string(&p)
                .with_context(|| format!("reading transcript from {p:?}"))?,
            (None, None, true) => std::io::read_to_string(std::io::stdin())
                .context("reading transcript from stdin")?,
            (None, None, false) => {
                bail!("provide --transcript \"…\", --file <PATH>, or --stdin (see --help)")
            }
            _ => bail!("--transcript, --file and --stdin are mutually exclusive"),
        };

    // Pure string work — useful for checking an integration's wire format
    // without paying for a model load.
    if show_prompt {
        print!("{}", render_prompt(controls, &transcript));
        return Ok(());
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights <PATH> is required"))?;
    let device = rlx_cli::parse_standard_device("s1", &device)?;
    let precision = match precision.as_str() {
        "f32" => Precision::F32,
        "f16-lm" | "f16_lm" => Precision::F16LmHead,
        other => bail!("--precision: expected f32|f16-lm, got {other:?}"),
    };

    let mut b = S1Runner::builder()
        .weights(&weights)
        .device(device)
        .precision(precision)
        .strict_shape(strict_shape);
    if let Some(p) = tokenizer {
        b = b.tokenizer(p);
    }
    if let Some(p) = config {
        b = b.config(p);
    }
    if let Some(n) = max_new_tokens {
        b = b.max_new_tokens(n);
    }
    if let Some(n) = max_seq {
        b = b.max_seq(n);
    }
    if let Some(n) = chunk_tokens {
        b = b.chunk_tokens(n);
    }
    if let Some(n) = prefill_bucket {
        b = b.prefill_bucket(n);
    }
    if let Some(on) = packed {
        b = b.packed_weights(on);
    }
    if let Some(q) = prefer_quant {
        b = b.prefer_gguf_quant(q);
    }

    eprintln!("[rlx-s1] weights={weights:?} device={device:?} controls={controls}");
    let mut runner = b.build()?;
    eprintln!(
        "[rlx-s1] loaded — {} layers, hidden {}, {} Q / {} KV heads, vocab {}",
        runner.shape().num_hidden_layers,
        runner.shape().hidden_size,
        runner.shape().num_attention_heads,
        runner.shape().num_key_value_heads,
        runner.shape().vocab_size,
    );

    let chunks = runner.chunk(&transcript)?;
    if chunks.len() > 1 {
        eprintln!(
            "[rlx-s1] transcript exceeds the {}-token per-pass budget — {} chunks",
            runner.chunk_tokens(),
            chunks.len()
        );
    }

    // Copied out so the streaming callback can detokenize while
    // `normalize_pass` holds the runner mutably.
    let tok_weights = runner.weights_path().to_path_buf();
    let tok_explicit = runner.tokenizer_path().map(|p| p.to_path_buf());

    let t0 = std::time::Instant::now();
    let mut n_tokens = 0usize;
    let mut parts: Vec<String> = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        if stream && parts.iter().any(|p: &String| !p.trim().is_empty()) {
            print!("{}", controls.chunk_separator());
        }
        let out = if stream {
            // Detokenize the running id list each step rather than each id on
            // its own, so multi-byte characters never split across a token
            // boundary. At dictation lengths that costs nothing next to a
            // forward pass.
            let mut live: Vec<u32> = Vec::new();
            let mut shown = String::new();
            let mut emit = |tok: u32| {
                live.push(tok);
                n_tokens += 1;
                let Ok(text) = rlx_s1_decode(&tok_weights, tok_explicit.as_deref(), &live) else {
                    return;
                };
                let at = common_prefix_len(&text, &shown);
                if at < text.len() {
                    print!("{}", &text[at..]);
                    let _ = std::io::stdout().flush();
                }
                shown = text;
            };
            runner.normalize_pass(chunk, controls, &mut emit)?
        } else {
            let mut count = 0usize;
            let out = runner.normalize_pass(chunk, controls, |_| count += 1)?;
            n_tokens += count;
            out
        };
        parts.push(out);
    }
    if stream {
        println!();
    }
    let dt = t0.elapsed().as_secs_f64();

    let output = crate::runner::join_chunk_outputs(&parts, controls);

    if json {
        let payload = serde_json::json!({
            "input": transcript,
            "output": output,
            "controls": {
                "styling": controls.styling.as_str(),
                "structure": controls.structure.as_str(),
                "context": controls.context.as_str(),
            },
            "chunks": chunks.len(),
            "generated_tokens": n_tokens,
            "seconds": dt,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if !stream {
        println!("{output}");
    }

    eprintln!(
        "[rlx-s1] {n_tokens} tokens in {dt:.2}s ({:.1} tok/s)",
        n_tokens as f64 / dt.max(1e-9)
    );
    Ok(())
}

use crate::runner::decode_ids as rlx_s1_decode;

/// Length of the longest shared prefix of `a` and `b`, floored to a `char`
/// boundary in `a` — so a streamed suffix never slices a multi-byte character
/// in half when an incremental detokenization changes its tail.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes())
        .take_while(|(x, y)| x == y)
        .count();
    while n > 0 && !a.is_char_boundary(n) {
        n -= 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_prefix_is_char_aligned() {
        assert_eq!(common_prefix_len("hello", "hell"), 4);
        assert_eq!(common_prefix_len("hello", ""), 0);
        assert_eq!(common_prefix_len("", "hello"), 0);
        // The shared bytes land mid-`é`; back off to the boundary before it.
        let a = "caf\u{e9} au lait";
        let b = "caf";
        assert_eq!(common_prefix_len(a, b), 3);
        assert!(a.is_char_boundary(common_prefix_len(a, b)));
        // A replacement char resolving into a real one must not panic.
        let partial = "the \u{fffd}";
        let full = "the \u{e9}clair";
        let n = common_prefix_len(full, partial);
        assert!(full.is_char_boundary(n));
        assert_eq!(&full[..n], "the ");
    }

    #[test]
    fn control_axes_parse_from_cli_spellings() {
        assert_eq!("formal".parse::<Styling>().unwrap(), Styling::Formal);
        assert_eq!("lists".parse::<Structure>().unwrap(), Structure::Lists);
        assert_eq!("email".parse::<Context>().unwrap(), Context::Email);
    }

    #[test]
    fn help_needs_no_weights() {
        assert!(cli_run(&[]).is_ok());
        assert!(cli_run(&["--help".to_string()]).is_ok());
    }

    #[test]
    fn show_prompt_runs_without_weights() {
        let args: Vec<String> = [
            "--transcript",
            "so um hi",
            "--show-prompt",
            "--styling",
            "formal",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert!(cli_run(&args).is_ok());
    }

    #[test]
    fn untrained_axis_value_is_rejected() {
        let args: Vec<String> = [
            "--transcript",
            "hi",
            "--styling",
            "business",
            "--show-prompt",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert!(cli_run(&args).is_err());
    }

    #[test]
    fn missing_input_is_an_error() {
        let args: Vec<String> = ["--weights", "/nonexistent"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(cli_run(&args).is_err());
    }
}
