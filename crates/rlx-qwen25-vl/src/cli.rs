// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use crate::aif::AifDynamicsMode;
use anyhow::{Context, Result, bail};
use rlx_cli::{WeightsResolveCli, parse_device, resolve_weights_cli};
use rlx_qwen3::SampleOpts;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut mmproj: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt_ids: Vec<u32> = vec![1, 2, 3];
    let mut max_tokens = 0usize;
    let mut max_seq = 0usize;
    let mut prompt_text: Option<String> = None;
    let mut image: Option<PathBuf> = None;
    let mut aif = false;
    let mut aif_native = false;
    let mut aif_ratio: Option<f32> = None;
    let mut aif_dynamics = AifDynamicsMode::from_env();
    let mut vlmevalkit_prompt = false;
    let mut resolve_cli = WeightsResolveCli::default();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => weights = Some(PathBuf::from(it.next().context("--weights")?)),
            "--mmproj" => mmproj = Some(PathBuf::from(it.next().context("--mmproj")?)),
            "--prefer-quant" | "--prefer" | "-p" => {
                resolve_cli.prefer_gguf = Some(it.next().context("--prefer-quant")?.clone());
            }
            "--device" => device = it.next().context("--device")?.clone(),
            "--prompt-ids" => {
                prompt_ids = it
                    .next()
                    .context("--prompt-ids")?
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()
                    .context("u32")?;
            }
            "--prompt" => prompt_text = Some(it.next().context("--prompt")?.clone()),
            "--max-tokens" => max_tokens = it.next().context("--max-tokens")?.parse()?,
            "--max-seq" => max_seq = it.next().context("--max-seq")?.parse()?,
            "--image" => image = Some(PathBuf::from(it.next().context("--image")?)),
            "--aif" => aif = true,
            "--aif-native" => aif_native = true,
            "--aif-dynamics" => {
                let s = it.next().context("--aif-dynamics")?;
                aif_dynamics = AifDynamicsMode::parse(s)
                    .ok_or_else(|| anyhow::anyhow!("unknown --aif-dynamics {s}"))?;
            }
            "--aif-ratio" => aif_ratio = Some(it.next().context("--aif-ratio")?.parse()?),
            "--vlmevalkit-prompt" => vlmevalkit_prompt = true,
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow::anyhow!("--weights required"))?;
    let weights = resolve_weights_cli(&weights, &resolve_cli)?;
    let dev = parse_device(&device)?;

    let mut builder = crate::Qwen25VlRunner::builder()
        .weights(&weights)
        .device(dev)
        .sample(SampleOpts::greedy());
    if let Some(m) = mmproj {
        builder = builder.mmproj(m);
    }
    if max_seq > 0 {
        builder = builder.max_seq(max_seq);
    }
    builder = builder.aif_dynamics_mode(aif_dynamics);
    let mut runner = builder.build()?;

    #[cfg(feature = "qwen25-vl-vision")]
    if let Some(image_path) = image {
        use crate::MEDIA_MARKER;
        #[cfg(feature = "tokenizer")]
        {
            let user_q = prompt_text.as_deref().ok_or_else(|| {
                anyhow::anyhow!("--image requires --prompt containing `{MEDIA_MARKER}` or plain question with --vlmevalkit-prompt")
            })?;
            let prompt = if vlmevalkit_prompt {
                crate::vlmevalkit_chat_prompt(user_q, None)
            } else if user_q.contains(MEDIA_MARKER) {
                user_q.to_string()
            } else {
                crate::user_turn_with_media(user_q)
            };
            if !vlmevalkit_prompt && !prompt.contains(MEDIA_MARKER) {
                bail!(
                    "--prompt must contain `{MEDIA_MARKER}` for multimodal runs (or use --vlmevalkit-prompt)"
                );
            }
            let (rgb, w, h) = crate::vision::load_rgb_image(
                image_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("non-utf8 --image path"))?,
            )?;
            let tok_path = std::env::var("RLX_QWEN25_VL_TOKENIZER")
                .ok()
                .map(PathBuf::from)
                .or_else(|| crate::resolve_tokenizer_path(&weights));
            let tok_path = tok_path.ok_or_else(|| {
                anyhow::anyhow!(
                    "multimodal run needs tokenizer.json beside weights or RLX_QWEN25_VL_TOKENIZER"
                )
            })?;
            let tokenizer = crate::load_tokenizer(&tok_path)?;
            let mut tokenize = |text: &str| crate::encode_prompt(&tokenizer, text);
            if max_tokens == 0 {
                let logits = runner.prefill_multimodal(&prompt, &rgb, w, h, &mut tokenize)?;
                eprintln!(
                    "[rlx-qwen25-vl] multimodal prefill ok: logits={} top={}",
                    logits.len(),
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                );
                return Ok(());
            }
            let stop = tokenizer
                .token_to_id("")
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"));
            let ids = if aif_native || (aif && aif_ratio.is_none()) {
                runner.generate_multimodal_aif_native(
                    &prompt,
                    &rgb,
                    w,
                    h,
                    max_tokens,
                    &mut tokenize,
                    stop,
                )?
            } else if aif {
                use crate::{AifConfig, AifProbe};
                let ratio = aif_ratio.unwrap_or(0.5);
                let n_vis = 8usize;
                let n_layers = runner.lm_config().lm.num_hidden_layers;
                let mut probe = AifProbe::build(vec![vec![0.1; n_layers]; n_vis]);
                probe.mask_ratio = ratio;
                let aif_cfg = AifConfig::from_probe(probe);
                runner.generate_multimodal_aif(
                    &prompt,
                    &rgb,
                    w,
                    h,
                    max_tokens,
                    &mut tokenize,
                    stop,
                    &aif_cfg,
                )?
            } else {
                runner.generate_multimodal(&prompt, &rgb, w, h, max_tokens, &mut tokenize, stop)?
            };
            eprintln!("[rlx-qwen25-vl] generated ids: {ids:?} (aif={aif} native={aif_native})");
            return Ok(());
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = image_path;
            let _ = prompt_text;
            bail!("rebuild with feature tokenizer for --image multimodal runs");
        }
    }

    if let Some(_text) = prompt_text {
        bail!("text prompt tokenization not wired yet; use --prompt-ids for text-only runs");
    }

    if max_tokens == 0 {
        let logits = runner.predict_logits(&prompt_ids).context("prefill")?;
        eprintln!(
            "[rlx-qwen25-vl] prefill ok: prompt_len={} logits={}",
            prompt_ids.len(),
            logits.len()
        );
        return Ok(());
    }

    let new_ids = runner.generate_text(&prompt_ids, max_tokens)?;
    eprintln!("[rlx-qwen25-vl] generated: {new_ids:?}");
    Ok(())
}
