// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Native VLMEvalKit eval on Qwen2.5-VL (baseline vs AIF).
//
// ```bash
// cargo run -p rlx-qwen25-vl --example vlmevalkit_eval --release -- \
//   --weights /path/lm.gguf --mmproj /path/mmproj.gguf \
//   --data /path/RealWorldQA.tsv --image-root /path/images \
//   --tokenizer /path/tokenizer.json --dataset realworldqa \
//   --aif-native --aif-dynamics prefill_v2t --limit 100 --out /tmp/vlmevalkit_report.json
// ```

use anyhow::{Context, Result, bail};
use rlx_qwen3::SampleOpts;
use rlx_qwen25_vl::{
    AifConfig, AifDynamicsMode, AifProbe, Qwen25VlRunner, VlmevalkitDataset, VlmevalkitMetric,
    VlmevalkitRecord, VlmevalkitReport, encode_prompt, infer_dataset, load_tokenizer,
    load_vlmevalkit_dataset, sample_question_text, sanitize_sample_id, score_prediction,
    vlmevalkit_chat_prompt,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights = None;
    let mut mmproj = None;
    let mut data_path = None;
    let mut image_root = None;
    let mut tokenizer_path = None;
    let mut device = "cpu".to_string();
    let mut limit = usize::MAX;
    let mut max_tokens = 32usize;
    let mut dataset: Option<VlmevalkitDataset> = None;
    let mut metric: Option<VlmevalkitMetric> = None;
    let mut out_json: Option<PathBuf> = None;
    let mut aif_probe_dir: Option<PathBuf> = None;
    let mut aif_native = false;
    let mut aif_dynamics = AifDynamicsMode::default();
    let mut baseline_only = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => weights = Some(PathBuf::from(it.next().context("--weights")?)),
            "--mmproj" => mmproj = Some(PathBuf::from(it.next().context("--mmproj")?)),
            "--data" | "--tsv" | "--jsonl" => {
                data_path = Some(PathBuf::from(it.next().context("--data")?))
            }
            "--image-root" => image_root = Some(PathBuf::from(it.next().context("--image-root")?)),
            "--tokenizer" => {
                tokenizer_path = Some(PathBuf::from(it.next().context("--tokenizer")?))
            }
            "--device" => device = it.next().context("--device")?.clone(),
            "--limit" => limit = it.next().context("--limit")?.parse()?,
            "--max-tokens" => max_tokens = it.next().context("--max-tokens")?.parse()?,
            "--dataset" => {
                let s = it.next().context("--dataset")?;
                dataset = Some(
                    VlmevalkitDataset::parse(s)
                        .ok_or_else(|| anyhow::anyhow!("unknown --dataset {s}"))?,
                );
            }
            "--metric" => {
                let s = it.next().context("--metric")?;
                metric = Some(match s.as_str() {
                    "exact" | "em" => VlmevalkitMetric::ExactMatch,
                    "mcq" => VlmevalkitMetric::McqLetter,
                    "textvqa" | "soft" => VlmevalkitMetric::TextVqaSoft,
                    _ => bail!("unknown --metric {s}"),
                });
            }
            "--out" => out_json = Some(PathBuf::from(it.next().context("--out")?)),
            "--aif-probe-dir" => {
                aif_probe_dir = Some(PathBuf::from(it.next().context("--aif-probe-dir")?))
            }
            "--aif-native" => aif_native = true,
            "--aif-dynamics" => {
                let s = it.next().context("--aif-dynamics")?;
                aif_dynamics = AifDynamicsMode::parse(s)
                    .ok_or_else(|| anyhow::anyhow!("unknown --aif-dynamics {s}"))?;
            }
            "--baseline-only" => baseline_only = true,
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = weights.context("--weights required")?;
    let mmproj = mmproj.context("--mmproj required")?;
    let data_path = data_path.context("--data required")?;
    let image_root = image_root.context("--image-root required")?;
    let tokenizer_path = tokenizer_path.context("--tokenizer required")?;

    let dataset = dataset.unwrap_or_else(|| infer_dataset(&data_path).expect("infer dataset"));
    let metric = metric.unwrap_or_else(|| VlmevalkitMetric::for_dataset(dataset));

    let dev = rlx_cli::parse_device(&device)?;
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let samples = load_vlmevalkit_dataset(dataset, &data_path, &image_root)?;

    let mut runner = Qwen25VlRunner::builder()
        .weights(&weights)
        .mmproj(&mmproj)
        .device(dev)
        .sample(SampleOpts::greedy())
        .aif_dynamics_mode(aif_dynamics)
        .build()?;

    let stop = tokenizer.token_to_id("").or(Some(151645));

    let mut records = Vec::new();
    for sample in samples.into_iter().take(limit) {
        if !sample.image_path.is_file() {
            eprintln!("skip missing image {}", sample.image_path.display());
            continue;
        }
        let (rgb, w, h) = rlx_qwen25_vl::vision::load_rgb_image(
            sample.image_path.to_str().context("image path utf8")?,
        )?;

        let qtext = sample_question_text(&sample);
        let prompt = vlmevalkit_chat_prompt(&qtext, None);
        let mut tokenize = |text: &str| encode_prompt(&tokenizer, text);

        let base_ids =
            runner.generate_multimodal(&prompt, &rgb, w, h, max_tokens, &mut tokenize, stop)?;
        let base_text = detokenize(&tokenizer, &base_ids)?;
        let base_ok = score_prediction(&base_text, &sample, metric);

        let (aif_text, aif_ok) = if baseline_only {
            (String::new(), false)
        } else if let Some(ref dir) = aif_probe_dir {
            let sid = sanitize_sample_id(&sample.id);
            let aif_ids = runner.generate_multimodal_aif_sample(
                &prompt,
                &rgb,
                w,
                h,
                max_tokens,
                &mut tokenize,
                stop,
                dir,
                &sid,
            )?;
            let text = detokenize(&tokenizer, &aif_ids)?;
            let ok = score_prediction(&text, &sample, metric);
            (text, ok)
        } else if aif_native {
            let aif_ids = runner.generate_multimodal_aif_native(
                &prompt,
                &rgb,
                w,
                h,
                max_tokens,
                &mut tokenize,
                stop,
            )?;
            let text = detokenize(&tokenizer, &aif_ids)?;
            let ok = score_prediction(&text, &sample, metric);
            (text, ok)
        } else {
            let mut probe = AifProbe::build(vec![vec![0.5; 2]; 16]);
            probe.mask_ratio = 0.5;
            let aif_cfg = AifConfig::from_probe(probe);
            let aif_ids = runner.generate_multimodal_aif(
                &prompt,
                &rgb,
                w,
                h,
                max_tokens,
                &mut tokenize,
                stop,
                &aif_cfg,
            )?;
            let text = detokenize(&tokenizer, &aif_ids)?;
            let ok = score_prediction(&text, &sample, metric);
            (text, ok)
        };

        records.push(VlmevalkitRecord {
            id: sample.id.clone(),
            question: sample.question.clone(),
            ground_truth: sample.answer.clone(),
            baseline_pred: base_text.clone(),
            aif_pred: aif_text.clone(),
            baseline_correct: base_ok,
            aif_correct: aif_ok,
        });

        eprintln!(
            "[{}] base={base_ok} aif={aif_ok} gt={:?} pred={base_text:?}",
            sample.id, sample.answer
        );
    }

    let report = VlmevalkitReport::from_records(dataset, metric, records);
    eprintln!(
        "vlmevalkit_eval: n={} baseline={:.1}% aif={:.1}% metric={:?} dynamics={}",
        report.total,
        report.baseline_acc * 100.0,
        report.aif_acc * 100.0,
        metric,
        aif_dynamics.as_str(),
    );

    if let Some(path) = out_json {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&path, json)?;
        eprintln!("wrote {}", path.display());
    }

    Ok(())
}

fn detokenize(tokenizer: &tokenizers::Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
}
