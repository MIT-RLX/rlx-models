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

//! `rlx-locateanything` CLI — grounding on a bundled sample or your own image.

use crate::config::LocateAnythingConfig;
use crate::device::resolve_device;
use crate::fixtures::{probe_image_path, resolve_image_path};
use crate::generation::GenerationMode;
use crate::hub::{default_model_dir, resolve_weights_path};
use crate::infer::{InferenceOptions, LocateAnythingSession, PromptStyle};
use crate::load::LocateAnythingWeightStore;
use crate::output::print_grounding;
use crate::prompts;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::req;
use std::path::PathBuf;

/// Parsed CLI flags.
struct CliArgs {
    weights: Option<PathBuf>,
    config: Option<PathBuf>,
    image: Option<PathBuf>,
    prompt: Option<String>,
    prompt_ids: Option<Vec<u32>>,
    task: Option<String>,
    phrase: Option<String>,
    categories: Vec<String>,
    device: Option<String>,
    dry: bool,
    list_keys: bool,
    max_tokens: usize,
    temperature: Option<f32>,
    repetition_penalty: Option<f32>,
    generation_mode: GenerationMode,
    prompt_style: PromptStyle,
    max_image_side: Option<u32>,
    warmup: bool,
    preload_lm: bool,
    download_only: bool,
}

pub fn run(args: &[String]) -> Result<()> {
    let Some(cli) = parse_args(args)? else {
        return Ok(());
    };
    run_parsed(cli)
}

/// `Ok(None)` after `--help`.
fn parse_args(args: &[String]) -> Result<Option<CliArgs>> {
    let mut weights = None;
    let mut config = None;
    let mut image = None;
    let mut prompt = None;
    let mut prompt_ids = None;
    let mut task = None;
    let mut phrase = None;
    let mut categories = Vec::new();
    let mut device = None;
    let mut dry = false;
    let mut list_keys = false;
    let mut max_tokens = 64;
    let mut temperature = None;
    let mut repetition_penalty = None;
    let mut generation_mode = GenerationMode::Hybrid;
    let mut prompt_style = PromptStyle::Processor;
    let mut max_image_side = None;
    let mut warmup = false;
    let mut preload_lm = false;
    let mut download_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model-dir" => weights = Some(req(args, &mut i)?.into()),
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--prompt-ids" => {
                let s = req(args, &mut i)?;
                prompt_ids = Some(
                    s.split(',')
                        .map(|p| p.trim().parse::<u32>().context("--prompt-ids"))
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            "--task" => task = Some(req(args, &mut i)?),
            "--phrase" => phrase = Some(req(args, &mut i)?),
            "--categories" => {
                let s = req(args, &mut i)?;
                categories = s.split(',').map(|c| c.trim().to_string()).collect();
            }
            "--device" => device = Some(req(args, &mut i)?),
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?.parse().context("--max-tokens")?;
            }
            "--temperature" => {
                temperature = Some(req(args, &mut i)?.parse().context("--temperature")?);
            }
            "--repetition-penalty" => {
                repetition_penalty =
                    Some(req(args, &mut i)?.parse().context("--repetition-penalty")?);
            }
            "--max-image-side" => {
                max_image_side = Some(req(args, &mut i)?.parse().context("--max-image-side")?);
            }
            "--generation-mode" => {
                let s = req(args, &mut i)?;
                generation_mode = GenerationMode::parse(&s)
                    .ok_or_else(|| anyhow!("unknown --generation-mode {s} (fast|slow|hybrid)"))?;
            }
            "--processor-prompt" => {
                prompt_style = PromptStyle::Processor;
                i += 1;
            }
            "--rlx-prompt" => {
                prompt_style = PromptStyle::Rlx;
                i += 1;
            }
            "--warmup" => {
                warmup = true;
                i += 1;
            }
            "--preload-lm" => {
                preload_lm = true;
                i += 1;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--list-keys" => {
                list_keys = true;
                i += 1;
            }
            "--download" => {
                download_only = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    Ok(Some(CliArgs {
        weights,
        config,
        image,
        prompt,
        prompt_ids,
        task,
        phrase,
        categories,
        device,
        dry,
        list_keys,
        max_tokens,
        temperature,
        repetition_penalty,
        generation_mode,
        prompt_style,
        max_image_side,
        warmup,
        preload_lm,
        download_only,
    }))
}

fn run_parsed(cli: CliArgs) -> Result<()> {
    if cli.download_only {
        #[cfg(feature = "hf-download")]
        {
            let dir = crate::download::fetch_default()?;
            eprintln!("\nDone. Snapshot:\n  {}", dir.display());
            eprintln!(
                "\nProcessor prompt needs tokenizer.json:\n  just fetch-locateanything-tokenizer"
            );
            return Ok(());
        }
        #[cfg(not(feature = "hf-download"))]
        {
            anyhow::bail!(
                "rebuild with --features hf-download, or run:\n  \
                 huggingface-cli download nvidia/LocateAnything-3B"
            );
        }
    }
    let model_dir = match &cli.weights {
        Some(w) => resolve_weights_path(w)?,
        None => default_model_dir()?,
    };
    if cli.weights.is_none() {
        eprintln!("[rlx-locateanything] weights {}", model_dir.display());
    }
    let store = LocateAnythingWeightStore::open(&model_dir)?;
    let cfg_path = cli
        .config
        .clone()
        .unwrap_or_else(|| store.model_dir().join("config.json"));
    let cfg = LocateAnythingConfig::from_file(&cfg_path)
        .with_context(|| format!("load config {cfg_path:?}"))?;
    cfg.validate()?;
    store.validate_tensor_layout()?;

    if cli.list_keys {
        let mut keys: Vec<_> = store.keys().iter().cloned().collect();
        keys.sort();
        for k in keys {
            println!("{k}");
        }
    }

    if cli.dry {
        let dev = resolve_device(cli.device.as_deref())?;
        eprintln!(
            "[rlx-locateanything] dry ok — tensors={} projector_in={} vocab={} device={dev:?}",
            store.keys().len(),
            cfg.projector_input_dim(),
            cfg.text_config.vocab_size,
        );
        return Ok(());
    }

    let image_path = resolve_image_path(cli.image.as_deref());
    if cli.image.is_none() {
        eprintln!(
            "[rlx-locateanything] using bundled sample {}",
            probe_image_path().display()
        );
    }
    if !image_path.is_file() {
        bail!("image not found: {}", image_path.display());
    }

    let user_text = user_text_from_cli(&cli)?;
    let options = inference_options_from_cli(&cli)?;
    let mut session = LocateAnythingSession::open_with_options(&model_dir, options)?;

    let prep = session.preprocess_file(&image_path)?;
    log_preprocess(&cfg, &prep, session.device());

    if cli.warmup {
        session.warmup(&prep, &user_text)?;
        eprintln!("[rlx-locateanything] warmup ok");
    }

    if let Some(ids) = cli.prompt_ids {
        let tokens = session.runner_mut().generate(&ids, &prep)?;
        print_token_output(
            session.runner().model_dir(),
            &tokens,
            ids.len(),
            prep.pixel_w,
            prep.pixel_h,
        )?;
    } else {
        #[cfg(feature = "tokenizer")]
        {
            let result = session.ground(&prep, &user_text)?;
            print_grounding(&result);
            eprintln!(
                "[rlx-locateanything] done — {} prompt + {} new tokens",
                result.prompt_len, result.new_tokens
            );
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = (&session, &user_text, &prep);
            bail!("rebuild with --features tokenizer, or pass --prompt-ids");
        }
    }

    Ok(())
}

fn user_text_from_cli(cli: &CliArgs) -> Result<String> {
    if let Some(p) = &cli.prompt {
        return Ok(p.clone());
    }
    let task = cli.task.as_deref().unwrap_or("ground-single");
    let phrase = cli.phrase.as_deref().unwrap_or("object");
    Ok(match task {
        "detect" => {
            let cats: Vec<&str> = if cli.categories.is_empty() {
                vec!["object"]
            } else {
                cli.categories.iter().map(String::as_str).collect()
            };
            prompts::detect(&cats)
        }
        "ground-single" => prompts::ground_single(phrase),
        "ground-multi" => prompts::ground_multi(phrase),
        "ground-text" => prompts::ground_text(phrase),
        "detect-text" => prompts::detect_text(),
        "point" => prompts::point(phrase),
        "ground-gui" => prompts::ground_gui_box(phrase),
        other => bail!("unknown --task {other}"),
    })
}

fn inference_options_from_cli(cli: &CliArgs) -> Result<InferenceOptions> {
    let mut options = InferenceOptions::for_grounding();
    options.device = resolve_device(cli.device.as_deref())?;
    options.max_new_tokens = cli.max_tokens;
    options.generation_mode = cli.generation_mode;
    if let Some(t) = cli.temperature {
        options.temperature = t;
    }
    if let Some(r) = cli.repetition_penalty {
        options.repetition_penalty = r;
    }
    options.max_image_side = cli.max_image_side;
    options.preload_language_model = cli.preload_lm;
    options.prompt_style = cli.prompt_style;
    Ok(options)
}

fn log_preprocess(
    cfg: &LocateAnythingConfig,
    prep: &crate::preprocess::PreprocessedImage,
    device: rlx_runtime::Device,
) {
    let kh = cfg.vision_config.merge_kernel_size[0];
    let kw = cfg.vision_config.merge_kernel_size[1];
    let n_vision = (prep.grid_h / kh) * (prep.grid_w / kw);
    eprintln!(
        "[rlx-locateanything] image {}x{} px → patch grid {}x{} ({} vision tokens) device={device:?}",
        prep.pixel_w, prep.pixel_h, prep.grid_w, prep.grid_h, n_vision,
    );
}

fn print_help() {
    eprintln!(
        "rlx-locateanything — NVIDIA LocateAnything-3B VLM\n\
         \n\
         Weights (optional — HF Hub cache by default):\n\
           [--model-dir PATH]        Dir, `hf`, or Hub id `nvidia/LocateAnything-3B`\n\
         \n\
         Inference (default image: bundled fixtures/sample.jpg):\n\
           [--image PATH]            JPEG/PNG (omit to use sample)\n\
           [--task TASK]             detect | ground-single | ground-multi | …\n\
           [--phrase TEXT]           For ground-* / point tasks\n\
           [--prompt TEXT]           Raw user message (overrides --task)\n\
           [--processor-prompt]      HF processor layout (default)\n\
           [--rlx-prompt]            RLX Qwen chat layout\n\
         \n\
         Device & speed:\n\
           [--device auto|cpu|metal|cuda|…]  default: auto (RLX_DEVICE)\n\
           [--max-image-side N]      Resize before patchify (e.g. 640)\n\
           [--warmup]                Compile vision + LM prefill first\n\
           [--preload-lm]            Load LM weights at open\n\
         \n\
         Generation:\n\
           [--max-tokens N]          default: 64\n\
           [--temperature F]         default: 0 (greedy)\n\
           [--repetition-penalty F]  default: 1\n\
           [--generation-mode fast|slow|hybrid]  default: hybrid\n\
         \n\
         Other:\n\
           [--download]            Fetch weights into Hugging Face cache\n\
           [--dry] [--list-keys] [--config PATH] [--prompt-ids …]\n\
         \n\
         Env: RLX_LOCATEANYTHING_DIR, RLX_LOCATEANYTHING_IMAGE, RLX_DEVICE\n\
         \n\
         Quick start:\n\
           just fetch-locateanything\n\
           just locateanything-demo"
    );
}

#[cfg(feature = "tokenizer")]
fn print_token_output(
    model_dir: &std::path::Path,
    tokens: &[u32],
    prompt_len: usize,
    w: u32,
    h: u32,
) -> Result<()> {
    use crate::parse::parse_grounding;
    use crate::tokenizer::{decode, load_tokenizer};

    let new_tokens = &tokens[prompt_len..];
    let tok = load_tokenizer(model_dir)?;
    let text = decode(&tok, new_tokens)?;
    let mut parsed = parse_grounding(&text, w, h);
    parsed.raw = tok
        .decode(new_tokens, false)
        .unwrap_or_else(|_| text.clone());
    parsed.prompt_len = prompt_len;
    parsed.new_tokens = new_tokens.len();
    print_grounding(&parsed);
    eprintln!(
        "[rlx-locateanything] done — {prompt_len} prompt + {} new tokens",
        new_tokens.len()
    );
    Ok(())
}
