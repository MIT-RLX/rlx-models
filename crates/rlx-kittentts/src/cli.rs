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

//! CLI for IPA → WAV synthesis (ONNX Runtime or native RLX graph).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_runtime::Device;

use crate::{
    KittenTTS,
    assets::{self, ModelLayout},
    config::DEFAULT_HF_REPO,
    download,
    tokenize::{ipa_content_len, ipa_style_index, ipa_to_ids},
};

pub fn run(args: &[String]) -> Result<()> {
    let parsed = parse_cli(args)?;
    if parsed.help {
        print_help();
        return Ok(());
    }
    #[cfg(not(feature = "espeak"))]
    let _ = &parsed.lang;

    if parsed.download {
        let dest = parsed
            .model_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(assets::DEFAULT_LOCAL_DIR));
        let repo = parsed.repo.as_deref().unwrap_or(DEFAULT_HF_REPO);
        download::fetch_to_local_dir(repo, &dest)?;
        return Ok(());
    }

    let model_dir = parsed
        .model_dir
        .clone()
        .map(Ok)
        .unwrap_or_else(assets::default_model_dir)?;
    let layout = ModelLayout::resolve(&model_dir)?;

    if parsed.list_voices {
        for name in layout.voice_names()? {
            println!("{name}");
        }
        return Ok(());
    }

    let dev = parsed.device;
    let (sequence_length, max_waveform) = native_compile_dims(
        parsed.serve,
        parsed.native,
        &layout,
        parsed.sequence_length,
        parsed.max_waveform,
        parsed.ipa.as_deref(),
        parsed.text.as_deref(),
        &parsed.lang,
    )?;
    if parsed.serve {
        return run_serve(parsed, &layout, sequence_length, max_waveform);
    }

    let tts = load_tts(&layout, parsed.native, dev, sequence_length, max_waveform)?;

    if !tts.has_voice(tts.resolve_voice(&parsed.voice)) {
        let names = layout.voice_names()?;
        bail!(
            "unknown voice '{}'. Available: {}",
            parsed.voice,
            names.join(", ")
        );
    }

    let voice = parsed.voice.clone();
    let speed = parsed.speed;
    let style_idx = parsed.style_idx;
    let out = parsed
        .out_wav
        .clone()
        .unwrap_or_else(|| PathBuf::from("kittentts_out.wav"));

    let audio = if let Some(text) = parsed.text.as_deref() {
        if parsed.ipa.is_some() {
            bail!("use only one of --text or --ipa");
        }
        #[cfg(not(feature = "espeak"))]
        {
            let _ = (text, tts, out);
            bail!(
                "--text requires rlx-kittentts built with `--features espeak` \
                 (pure-Rust espeak-ng phonemization)"
            );
        }
        #[cfg(feature = "espeak")]
        {
            if text.trim().is_empty() {
                bail!("--text must not be empty");
            }
            eprintln!("[kittentts] phonemizing (lang={})", parsed.lang);
            tts.generate_from_text(text, &voice, speed, &parsed.lang)?
        }
    } else {
        let ipa = parsed.ipa.as_deref().context(
            "--ipa or --text is required (or pass IPA as the first positional argument)",
        )?;

        if ipa_content_len(ipa) == 0 {
            bail!(
                "IPA input has no tokenizable phoneme characters. \
                 Use --text for plain English (needs `--features espeak`) or IPA symbols (e.g. həˈloʊ)."
            );
        }

        let style = style_idx.unwrap_or_else(|| ipa_style_index(ipa));
        tts.generate_from_ipa(ipa, &voice, speed, style)?
    };

    tts.write_wav(&audio, &out)?;
    Ok(())
}

fn run_serve(
    parsed: Cli,
    layout: &ModelLayout,
    sequence_length: usize,
    max_waveform: usize,
) -> Result<()> {
    let t0 = Instant::now();
    let tts = load_tts(
        layout,
        parsed.native,
        parsed.device,
        sequence_length,
        max_waveform,
    )?;
    eprintln!(
        "[kittentts] serve ready in {:.3}s (one line per request: IPA or IPA<TAB>out.wav; quit/exit to stop)",
        t0.elapsed().as_secs_f64()
    );
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit") {
            break;
        }
        let (ipa, out_path) = if let Some((ipa, path)) = line.split_once('\t') {
            (ipa.trim().to_string(), PathBuf::from(path.trim()))
        } else {
            (line.to_string(), PathBuf::from("kittentts_serve.wav"))
        };
        if ipa_content_len(&ipa) == 0 {
            writeln!(stdout, "err\tempty or invalid IPA")?;
            stdout.flush()?;
            continue;
        }
        let t1 = Instant::now();
        match synthesize_ipa(&tts, &parsed, &ipa) {
            Ok(audio) => {
                if let Err(e) = tts.write_wav(&audio, &out_path) {
                    writeln!(stdout, "err\t{e}")?;
                } else {
                    writeln!(
                        stdout,
                        "ok\t{}\t{}\t{:.3}",
                        out_path.display(),
                        audio.len(),
                        t1.elapsed().as_secs_f64()
                    )?;
                }
            }
            Err(e) => writeln!(stdout, "err\t{e}")?,
        }
        stdout.flush()?;
    }
    Ok(())
}

fn synthesize_ipa(tts: &KittenTTS, parsed: &Cli, ipa: &str) -> Result<Vec<f32>> {
    if !tts.has_voice(tts.resolve_voice(&parsed.voice)) {
        bail!("unknown voice '{}'", parsed.voice);
    }
    let style = parsed.style_idx.unwrap_or_else(|| ipa_style_index(ipa));
    tts.generate_from_ipa(ipa, &parsed.voice, parsed.speed, style)
}

struct Cli {
    help: bool,
    download: bool,
    list_voices: bool,
    serve: bool,
    native: bool,
    model_dir: Option<PathBuf>,
    repo: Option<String>,
    ipa: Option<String>,
    text: Option<String>,
    lang: String,
    voice: String,
    speed: f32,
    style_idx: Option<usize>,
    out_wav: Option<PathBuf>,
    device: Device,
    sequence_length: usize,
    max_waveform: usize,
}

fn parse_cli(args: &[String]) -> Result<Cli> {
    let mut model_dir: Option<PathBuf> = None;
    let mut model: Option<PathBuf> = None;
    let mut voices: Option<PathBuf> = None;
    let mut weights_dir: Option<PathBuf> = None;
    let mut ipa: Option<String> = None;
    let mut text: Option<String> = None;
    let mut lang = String::from("en");
    let mut voice = String::from("Jasper");
    let mut speed = 1.0f32;
    let mut style_idx = None;
    let mut out_wav = None;
    let mut device = None;
    let mut native = false;
    let mut download = false;
    let mut list_voices = false;
    let mut serve = false;
    let mut help = false;
    let mut repo = None;
    let mut sequence_length = 128usize;
    let mut max_waveform = 48_000usize;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if let Some(v) = arg.strip_prefix("--device=") {
            device = Some(v.to_string());
            i += 1;
            continue;
        }
        match arg {
            "--model-dir" | "--dir" => model_dir = Some(req(args, &mut i)?.into()),
            "--model" | "--onnx" => model = Some(req(args, &mut i)?.into()),
            "--voices" => voices = Some(req(args, &mut i)?.into()),
            "--weights-dir" => weights_dir = Some(req(args, &mut i)?.into()),
            "--ipa" => ipa = Some(req(args, &mut i)?),
            "--text" => text = Some(req(args, &mut i)?),
            "--lang" => lang = req(args, &mut i)?,
            "--voice" => voice = req(args, &mut i)?,
            "--speed" => {
                speed = req(args, &mut i)?
                    .parse()
                    .context("--speed expects a float")?;
            }
            "--style-idx" => {
                style_idx = Some(
                    req(args, &mut i)?
                        .parse()
                        .context("--style-idx expects usize")?,
                );
            }
            "--out-wav" | "--output" => out_wav = Some(req(args, &mut i)?.into()),
            "--device" => device = Some(req(args, &mut i)?),
            "--native" => {
                native = true;
                i += 1;
            }
            "--serve" => {
                serve = true;
                i += 1;
            }
            "--download" | "--fetch" => {
                download = true;
                i += 1;
            }
            "--repo" => repo = Some(req(args, &mut i)?),
            "--list-voices" => {
                list_voices = true;
                i += 1;
            }
            "--seq-len" | "--sequence-length" => {
                sequence_length = req(args, &mut i)?
                    .parse()
                    .context("--seq-len expects usize")?;
            }
            "--max-waveform-samples" => {
                max_waveform = req(args, &mut i)?
                    .parse()
                    .context("--max-waveform-samples expects usize")?;
            }
            "--help" | "-h" => {
                help = true;
                i += 1;
            }
            other if other.starts_with('-') => bail!("unknown flag: {other}"),
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    if ipa.is_none() && text.is_none() && positional.len() == 1 {
        ipa = Some(positional.remove(0));
    } else if ipa.is_none() && text.is_none() && !positional.is_empty() {
        bail!(
            "unexpected positional arguments: {} (expected a single IPA string, or use --text)",
            positional.join(" ")
        );
    }

    if ipa.is_some() && text.is_some() {
        bail!("use only one of --ipa or --text");
    }

    if model_dir.is_none() {
        if let (Some(onnx), Some(v)) = (&model, &voices) {
            model_dir = onnx.parent().map(|p| p.to_path_buf());
            let _ = v;
        } else if let Some(w) = &weights_dir {
            model_dir = w.parent().map(|p| p.to_path_buf());
        }
    }

    let device = device
        .map(|d| parse_standard_device("kittentts", &d))
        .transpose()?
        .unwrap_or_else(|| {
            std::env::var("KITTENTTS_DEVICE")
                .ok()
                .and_then(|s| parse_standard_device("kittentts", &s).ok())
                .unwrap_or(Device::Cpu)
        });

    Ok(Cli {
        help,
        download,
        list_voices,
        serve,
        native,
        model_dir,
        repo,
        ipa,
        text,
        lang,
        voice,
        speed,
        style_idx,
        out_wav,
        device,
        sequence_length,
        max_waveform,
    })
}

fn native_compile_dims(
    serve: bool,
    native: bool,
    layout: &ModelLayout,
    sequence_length: usize,
    max_waveform: usize,
    ipa: Option<&str>,
    text: Option<&str>,
    lang: &str,
) -> Result<(usize, usize)> {
    let use_native = will_use_native(native, layout);
    if !use_native {
        return Ok((sequence_length, max_waveform));
    }
    if std::env::var("KITTENTTS_AUTO_NATIVE_OPTS").ok().as_deref() == Some("0") {
        return Ok((sequence_length, max_waveform));
    }
    let token_len = if let Some(ipa) = ipa {
        ipa_to_ids(ipa).len()
    } else if let Some(text) = text {
        estimate_text_token_len(text, lang)
    } else if serve {
        8
    } else {
        return Ok((sequence_length, max_waveform));
    };
    let (seq, wave) = crate::infer_opts::recommended_native_compile_opts(token_len);
    if sequence_length != 128 || max_waveform != 48_000 {
        return Ok((sequence_length, max_waveform));
    }
    eprintln!(
        "[kittentts] native compile opts: --seq-len {seq} --max-waveform-samples {wave} \
         (from token_len={token_len}; set KITTENTTS_AUTO_NATIVE_OPTS=0 to disable)"
    );
    Ok((seq, wave))
}

#[cfg(feature = "espeak")]
fn estimate_text_token_len(text: &str, lang: &str) -> usize {
    crate::phonemize::phonemize_lang(lang, text)
        .ok()
        .map(|ipa| ipa_to_ids(&ipa).len())
        .unwrap_or_else(|| text.len().max(8))
}

#[cfg(not(feature = "espeak"))]
fn estimate_text_token_len(text: &str, _lang: &str) -> usize {
    text.len().max(8)
}

fn will_use_native(native: bool, layout: &ModelLayout) -> bool {
    native
        || (cfg!(feature = "native")
            && layout.native_weights.is_some()
            && std::env::var("KITTENTTS_FORCE_ONNX").is_err())
}

fn load_tts(
    layout: &ModelLayout,
    native: bool,
    device: Device,
    sequence_length: usize,
    max_waveform: usize,
) -> Result<KittenTTS> {
    let use_native = will_use_native(native, layout);

    if use_native {
        #[cfg(not(feature = "native"))]
        {
            let _ = (layout, device, sequence_length, max_waveform);
            bail!(
                "native backend requested but rlx-kittentts was built without `--features native`"
            );
        }
        #[cfg(feature = "native")]
        {
            return KittenTTS::load_native_from_dir(
                &layout.dir,
                device,
                sequence_length,
                max_waveform,
            );
        }
    }

    #[cfg(not(feature = "onnx"))]
    {
        let _ = (layout, device);
        bail!("ONNX backend unavailable; rebuild with `onnx` feature or pass --native");
    }
    #[cfg(feature = "onnx")]
    {
        KittenTTS::load_on(
            &layout.onnx,
            &layout.voices,
            layout.config.speed_priors.clone(),
            layout.config.voice_aliases.clone(),
            device,
        )
    }
}

fn print_help() {
    eprintln!(
        "rlx-kittentts — KittenTTS IPA text-to-speech\n\
         \n\
         Quick start:\n\
           just fetch-kittentts\n\
           just kittentts-demo\n\
         \n\
         Synthesize:\n\
           rlx-kittentts --ipa \"həˈloʊ\" [--voice Jasper] [--out-wav out.wav]\n\
           rlx-kittentts --text \"Hello world\"   # needs --features espeak\n\
           rlx-kittentts \"həˈloʊ\"   # positional IPA\n\
         \n\
         Phonemization (feature espeak):\n\
           --text STR   plain text → espeak-ng → KittenTTS\n\
           --lang TAG   espeak voice (default en)\n\
         \n\
         Paths (auto if omitted):\n\
           --model-dir DIR     RLX_KITTENTTS_DIR / .cache/kittentts-mini-0.8 / HF cache\n\
           --model PATH --voices PATH   legacy explicit paths (dir inferred)\n\
         \n\
         Weights:\n\
           --download [--repo ID] [--model-dir DIR]   fetch HF checkpoint\n\
           --list-voices                                print voice names\n\
         \n\
         Backends:\n\
           --native            decomposed RLX graph (needs model.safetensors)\n\
           --serve             load once, read IPA lines from stdin (tab-separated out path)\n\
           --device NAME       cpu | metal | mlx | cuda | … (env: KITTENTTS_DEVICE)\n\
           KITTENTTS_FORCE_ONNX=1   skip auto-native when safetensors present\n\
         \n\
         Tuning:\n\
           --speed F  --style-idx N  --seq-len N  --max-waveform-samples N"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_ipa() {
        let cli = parse_cli(&["həˈloʊ".into()]).unwrap();
        assert_eq!(cli.ipa.as_deref(), Some("həˈloʊ"));
    }
}
