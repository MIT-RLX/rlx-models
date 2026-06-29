// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// VLMEvalKit-style JSONL eval: baseline vs AIF on Qwen2.5-VL.
//
// ```bash
// # 1) Export HF probes (once per dataset)
// RLX_QWEN25_VL_HF_DIR=/path/to/Qwen2.5-VL-7B-Instruct \
// python3 scripts/aif_export_probes.py \
//   --jsonl /path/realworldqa.jsonl --image-root /path/images \
//   --out-dir /tmp/aif-probes --vlmevalkit-prompt --limit 100
//
// # 2) RLX eval with cached probes
// cargo run -p rlx-qwen25-vl --example aif_eval --release -- \
//   --weights /path/lm.gguf --mmproj /path/mmproj.gguf \
//   --jsonl /path/realworldqa.jsonl --image-root /path/images \
//   --tokenizer /path/tokenizer.json --aif-probe-dir /tmp/aif-probes \
//   --vlmevalkit-prompt --limit 100
//
// # Or native RLX probe (no Python / cached probes):
// cargo run … --aif-native-probe --vlmevalkit-prompt --limit 100
// ```

use anyhow::{Context, Result, bail};
use rlx_qwen3::SampleOpts;
use rlx_qwen25_vl::{
    AifConfig, AifDynamicsMode, AifProbe, Qwen25VlRunner, encode_prompt, load_tokenizer,
    load_vqa_jsonl, normalized_exact_match, run_hf_python_probe, sanitize_sample_id,
    user_turn_with_media, vlmevalkit_chat_prompt,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights = None;
    let mut mmproj = None;
    let mut jsonl = None;
    let mut image_root = None;
    let mut tokenizer_path = None;
    let mut device = "cpu".to_string();
    let mut limit = usize::MAX;
    let mut max_tokens = 32usize;
    let mut aif_probe_dir: Option<PathBuf> = None;
    let mut aif_probe_cache: Option<PathBuf> = None;
    let mut aif_hf_probe = false;
    let mut aif_native_probe = false;
    let mut aif_dynamics = AifDynamicsMode::from_env();
    let mut vlmevalkit = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => weights = Some(PathBuf::from(it.next().context("--weights")?)),
            "--mmproj" => mmproj = Some(PathBuf::from(it.next().context("--mmproj")?)),
            "--jsonl" => jsonl = Some(PathBuf::from(it.next().context("--jsonl")?)),
            "--image-root" => image_root = Some(PathBuf::from(it.next().context("--image-root")?)),
            "--tokenizer" => {
                tokenizer_path = Some(PathBuf::from(it.next().context("--tokenizer")?))
            }
            "--device" => device = it.next().context("--device")?.clone(),
            "--limit" => limit = it.next().context("--limit")?.parse()?,
            "--max-tokens" => max_tokens = it.next().context("--max-tokens")?.parse()?,
            "--aif-probe-dir" => {
                aif_probe_dir = Some(PathBuf::from(it.next().context("--aif-probe-dir")?))
            }
            "--aif-probe-cache" => {
                aif_probe_cache = Some(PathBuf::from(it.next().context("--aif-probe-cache")?))
            }
            "--aif-hf-probe" => aif_hf_probe = true,
            "--aif-native-probe" => aif_native_probe = true,
            "--aif-dynamics" => {
                let s = it.next().context("--aif-dynamics")?;
                aif_dynamics = AifDynamicsMode::parse(s)
                    .ok_or_else(|| anyhow::anyhow!("unknown --aif-dynamics {s}"))?;
            }
            "--vlmevalkit-prompt" => vlmevalkit = true,
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = weights.context("--weights required")?;
    let mmproj = mmproj.context("--mmproj required")?;
    let jsonl = jsonl.context("--jsonl required")?;
    let image_root = image_root.context("--image-root required")?;
    let tokenizer_path = tokenizer_path.context("--tokenizer required")?;

    let dev = rlx_cli::parse_device(&device)?;
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let samples = load_vqa_jsonl(&jsonl, &image_root)?;

    let mut runner = Qwen25VlRunner::builder()
        .weights(&weights)
        .mmproj(&mmproj)
        .device(dev)
        .sample(SampleOpts::greedy())
        .aif_dynamics_mode(aif_dynamics)
        .build()?;

    let stop = tokenizer.token_to_id("").or(Some(151645));

    let mut base_ok = 0usize;
    let mut aif_ok = 0usize;
    let mut n = 0usize;

    for sample in samples.into_iter().take(limit) {
        if !sample.image_path.is_file() {
            eprintln!("skip missing image {}", sample.image_path.display());
            continue;
        }
        let (rgb, w, h) = rlx_qwen25_vl::vision::load_rgb_image(
            sample.image_path.to_str().context("image path utf8")?,
        )?;

        let prompt = if vlmevalkit {
            vlmevalkit_chat_prompt(&sample.question, None)
        } else {
            user_turn_with_media(&sample.question)
        };

        let mut tokenize = |text: &str| encode_prompt(&tokenizer, text);

        let base_ids =
            runner.generate_multimodal(&prompt, &rgb, w, h, max_tokens, &mut tokenize, stop)?;
        let base_text = detokenize(&tokenizer, &base_ids)?;
        if normalized_exact_match(&base_text, &sample.answer) {
            base_ok += 1;
        }

        let sid = sanitize_sample_id(&sample.id);
        let aif_ids = if let Some(ref dir) = aif_probe_dir {
            runner.generate_multimodal_aif_sample(
                &prompt,
                &rgb,
                w,
                h,
                max_tokens,
                &mut tokenize,
                stop,
                dir,
                &sid,
            )?
        } else if aif_hf_probe {
            let cache = aif_probe_cache
                .as_ref()
                .map(|p| p.join(&sid))
                .unwrap_or_else(|| std::env::temp_dir().join(format!("aif-{sid}")));
            if !cache.join(format!("{sid}_vision_dynamics.npy")).exists() {
                std::fs::create_dir_all(&cache)?;
                run_hf_python_probe(
                    &sample.image_path,
                    &sample.question,
                    &sid,
                    &cache,
                    vlmevalkit,
                )?;
            }
            runner.generate_multimodal_aif_sample(
                &prompt,
                &rgb,
                w,
                h,
                max_tokens,
                &mut tokenize,
                stop,
                &cache,
                &sid,
            )?
        } else if aif_native_probe {
            runner.generate_multimodal_aif_native(
                &prompt,
                &rgb,
                w,
                h,
                max_tokens,
                &mut tokenize,
                stop,
            )?
        } else {
            let mut probe = AifProbe::build(vec![vec![0.5; 2]; 16]);
            probe.mask_ratio = 0.5;
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
        };
        let aif_text = detokenize(&tokenizer, &aif_ids)?;
        if normalized_exact_match(&aif_text, &sample.answer) {
            aif_ok += 1;
        }

        n += 1;
        eprintln!(
            "[{n}] id={} base={base_ok}/{n} aif={aif_ok}/{n} gt={:?} pred={base_text:?} aif={aif_text:?}",
            sample.id, sample.answer
        );
    }

    eprintln!(
        "aif_eval: n={n} baseline={:.1}% aif={:.1}%",
        100.0 * base_ok as f64 / n.max(1) as f64,
        100.0 * aif_ok as f64 / n.max(1) as f64,
    );
    Ok(())
}

fn detokenize(tokenizer: &tokenizers::Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
}
