//! Large-corpus stress bench: synthetic prompts → optional ref TTS → target validate.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_runtime::Device;
use serde::{Deserialize, Serialize};

use crate::adapter::SynthRequest;
use crate::adapters::make_adapter;
use crate::corpus::{CorpusItem, generate_corpus, load_corpus_file};
use crate::devices::device_label;
use crate::metrics::{WhisperState, spectral_vs_ref, try_load_whisper, whisper_coverage};
use crate::phrases::content_words;
use crate::wav::{peak_normalize, write_wav_mono};

#[derive(Debug, Clone)]
pub struct StressConfig {
    pub n: usize,
    pub seed: u64,
    pub target_model: String,
    pub ref_model: Option<String>,
    pub device: Device,
    pub whisper: bool,
    pub spectral: bool,
    pub save_wav: bool,
    pub save_wav_every: usize,
    pub out_dir: PathBuf,
    pub resume: bool,
    pub offset: usize,
    pub fail_under_coverage: Option<f64>,
    pub fail_under_ok_rate: Option<f64>,
    pub corpus_file: Option<PathBuf>,
    /// Skip MatchPrompt / keep product WR; for stress we still use product defaults.
    pub write_corpus: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressRow {
    pub id: String,
    pub category: String,
    pub text: String,
    pub status: String,
    pub error: Option<String>,
    pub target_ms: Option<f64>,
    pub target_rtf: Option<f64>,
    pub target_secs: Option<f64>,
    pub target_peak: Option<f32>,
    pub ref_ms: Option<f64>,
    pub stft_cosine: Option<f64>,
    pub logmel_cosine: Option<f64>,
    pub whisper_coverage: Option<f64>,
    pub whisper_cer: Option<f64>,
    pub whisper_transcript: Option<String>,
    pub content_hits: Option<usize>,
    pub content_total: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressSummary {
    pub n_planned: usize,
    pub n_run: usize,
    pub n_ok: usize,
    pub n_err: usize,
    pub n_skipped: usize,
    pub ok_rate: f64,
    pub median_coverage: Option<f64>,
    pub mean_coverage: Option<f64>,
    pub median_cer: Option<f64>,
    pub median_rtf: Option<f64>,
    pub median_stft: Option<f64>,
    pub median_logmel: Option<f64>,
    pub silent_or_tiny: usize,
    pub target_model: String,
    pub ref_model: Option<String>,
    pub whisper: bool,
}

pub fn run_stress(cfg: &StressConfig) -> Result<Vec<StressRow>> {
    std::fs::create_dir_all(&cfg.out_dir)?;
    std::fs::create_dir_all(cfg.out_dir.join("wav"))?;

    let mut items = if let Some(path) = &cfg.corpus_file {
        load_corpus_file(path)?
    } else {
        generate_corpus(cfg.n + cfg.offset, cfg.seed)
    };
    if cfg.corpus_file.is_none() {
        // generate_corpus already sized; apply offset window of length n
        if cfg.offset > 0 {
            items = items.into_iter().skip(cfg.offset).take(cfg.n).collect();
        } else {
            items.truncate(cfg.n);
        }
    } else {
        items = items.into_iter().skip(cfg.offset).take(cfg.n).collect();
    }

    if cfg.write_corpus {
        let path = cfg.out_dir.join("corpus.jsonl");
        let mut f = std::fs::File::create(&path)?;
        for it in &items {
            writeln!(
                f,
                "{}",
                serde_json::json!({ "id": it.id, "category": it.category, "text": it.text })
            )?;
        }
        eprintln!("wrote corpus {} ({} lines)", path.display(), items.len());
    }

    let json_path = cfg.out_dir.join("stress_results.jsonl");
    let mut done: HashSet<String> = HashSet::new();
    if cfg.resume && json_path.is_file() {
        for line in std::fs::read_to_string(&json_path)?.lines() {
            if let Ok(v) = serde_json::from_str::<StressRow>(line) {
                done.insert(v.id);
            }
        }
        eprintln!(
            "resume: {} completed id(s) in {}",
            done.len(),
            json_path.display()
        );
    } else if !cfg.resume {
        let _ = std::fs::remove_file(&json_path);
    }

    let mut whisper = if cfg.whisper {
        try_load_whisper()
    } else {
        None
    };
    if cfg.whisper && whisper.is_none() {
        eprintln!(
            "warning: --whisper requested but no Whisper weights found (RLX_WHISPER_DIR / .cache/whisper-*)"
        );
    }

    eprintln!(
        "stress: {} phrase(s) target={} ref={:?} device={} whisper={}",
        items.len(),
        cfg.target_model,
        cfg.ref_model,
        device_label(cfg.device),
        whisper.is_some()
    );

    let mut target = make_adapter(&cfg.target_model, cfg.device)
        .with_context(|| format!("load target model {}", cfg.target_model))?;
    let mut reference = match &cfg.ref_model {
        Some(ref_id) => Some(
            make_adapter(ref_id, cfg.device).with_context(|| format!("load ref model {ref_id}"))?,
        ),
        None => None,
    };

    let mut rows = Vec::new();
    let mut skipped = 0usize;

    for (i, item) in items.iter().enumerate() {
        if done.contains(&item.id) {
            skipped += 1;
            continue;
        }
        let row = {
            let ref_slot = reference
                .as_mut()
                .map(|b| b.as_mut() as &mut dyn crate::adapter::TtsAdapter);
            run_one(
                cfg,
                item,
                i,
                items.len(),
                target.as_mut(),
                ref_slot,
                &mut whisper,
            )?
        };
        if i < 5 || (i + 1) % 25 == 0 || i + 1 == items.len() {
            eprintln!(
                "  [{}/{}] {} status={} cov={:?} cer={:?} rtf={:?}",
                i + 1,
                items.len(),
                row.id,
                row.status,
                row.whisper_coverage,
                row.whisper_cer,
                row.target_rtf
            );
        }
        append_row(&json_path, &row)?;
        rows.push(row);
    }

    // Include resumed rows for summary.
    let all_rows = if cfg.resume && json_path.is_file() {
        load_rows(&json_path)?
    } else {
        rows
    };
    let summary = summarize(&all_rows, cfg, skipped);
    let summary_path = cfg.out_dir.join("stress_summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!(
        "stress done: ok={} err={} skipped={} ok_rate={:.3} median_cov={:?} → {}",
        summary.n_ok,
        summary.n_err,
        summary.n_skipped,
        summary.ok_rate,
        summary.median_coverage,
        summary_path.display()
    );

    if let Some(min_cov) = cfg.fail_under_coverage {
        if let Some(med) = summary.median_coverage {
            if med < min_cov {
                anyhow::bail!("median Whisper coverage {med:.3} < fail-under-coverage {min_cov}");
            }
        }
    }
    if let Some(min_ok) = cfg.fail_under_ok_rate {
        if summary.ok_rate < min_ok {
            anyhow::bail!(
                "ok_rate {:.3} < fail-under-ok-rate {min_ok}",
                summary.ok_rate
            );
        }
    }

    Ok(all_rows)
}

fn run_one(
    cfg: &StressConfig,
    item: &CorpusItem,
    idx: usize,
    total: usize,
    target: &mut dyn crate::adapter::TtsAdapter,
    reference: Option<&mut dyn crate::adapter::TtsAdapter>,
    whisper: &mut Option<WhisperState>,
) -> Result<StressRow> {
    let _ = (idx, total);
    let mut row = StressRow {
        id: item.id.clone(),
        category: item.category.to_string(),
        text: item.text.clone(),
        status: "ok".into(),
        error: None,
        target_ms: None,
        target_rtf: None,
        target_secs: None,
        target_peak: None,
        ref_ms: None,
        stft_cosine: None,
        logmel_cosine: None,
        whisper_coverage: None,
        whisper_cer: None,
        whisper_transcript: None,
        content_hits: None,
        content_total: None,
    };

    let mut ref_pcm: Option<(Vec<f32>, u32)> = None;
    if let Some(r) = reference {
        match r.synthesize(SynthRequest {
            text: &item.text,
            phrase_id: &item.id,
            device: cfg.device,
            clone: None,
            seed: cfg.seed,
        }) {
            Ok(s) => {
                row.ref_ms = Some(s.wall_ms);
                ref_pcm = Some((s.pcm, s.sample_rate));
            }
            Err(e) => {
                // Ref failure is non-fatal; still validate target.
                row.error = Some(format!("ref: {e:#}"));
            }
        }
    }

    let synth = match target.synthesize(SynthRequest {
        text: &item.text,
        phrase_id: &item.id,
        device: cfg.device,
        clone: None,
        seed: cfg.seed,
    }) {
        Ok(s) => s,
        Err(e) => {
            row.status = "error".into();
            row.error = Some(format!("target: {e:#}"));
            return Ok(row);
        }
    };

    let dur = synth.pcm.len() as f64 / synth.sample_rate.max(1) as f64;
    let peak = synth
        .pcm
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    row.target_ms = Some(synth.wall_ms);
    row.target_secs = Some(dur);
    row.target_peak = Some(peak);
    row.target_rtf = if dur > 1e-6 {
        Some((synth.wall_ms / 1000.0) / dur)
    } else {
        None
    };

    if peak < 1e-4 || dur < 0.05 {
        row.status = "error".into();
        row.error = Some(format!("silent_or_tiny peak={peak:.3e} dur={dur:.3}"));
        return Ok(row);
    }

    if cfg.save_wav || (cfg.save_wav_every > 0 && idx.is_multiple_of(cfg.save_wav_every)) {
        let path = cfg.out_dir.join("wav").join(format!("{}.wav", item.id));
        let _ = write_wav_mono(&path, &synth.pcm, synth.sample_rate);
    }

    if cfg.spectral {
        if let Some((rpcm, rsr)) = ref_pcm.as_ref() {
            let m = spectral_vs_ref(&synth.pcm, synth.sample_rate, rpcm, *rsr);
            row.stft_cosine = Some(m.stft_cosine);
            row.logmel_cosine = Some(m.logmel_cosine);
        }
    }

    if let Some(w) = whisper.as_mut() {
        match whisper_coverage(w, &synth.pcm, synth.sample_rate, &item.text) {
            Ok(m) => {
                row.whisper_coverage = Some(m.coverage);
                row.content_hits = Some(m.content_hits);
                row.content_total = Some(m.content_total);
                row.whisper_transcript = Some(m.transcript.clone());
                row.whisper_cer = Some(char_error_rate(&item.text, &m.transcript));
            }
            Err(e) => {
                row.error = Some(format!("whisper: {e:#}"));
            }
        }
    }

    // Peak-normalize path used only to keep metrics stable if we later add dumps.
    let _ = peak_normalize(&synth.pcm, 0.95);

    Ok(row)
}

fn append_row(path: &Path, row: &StressRow) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", serde_json::to_string(row)?)?;
    Ok(())
}

fn load_rows(path: &Path) -> Result<Vec<StressRow>> {
    let mut out = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line)?);
    }
    Ok(out)
}

fn summarize(rows: &[StressRow], cfg: &StressConfig, skipped: usize) -> StressSummary {
    let n_ok = rows.iter().filter(|r| r.status == "ok").count();
    let n_err = rows.iter().filter(|r| r.status != "ok").count();
    let silent = rows
        .iter()
        .filter(|r| {
            r.error
                .as_deref()
                .is_some_and(|e| e.starts_with("silent_or_tiny"))
        })
        .count();
    let covs: Vec<f64> = rows.iter().filter_map(|r| r.whisper_coverage).collect();
    let cers: Vec<f64> = rows.iter().filter_map(|r| r.whisper_cer).collect();
    let rtfs: Vec<f64> = rows.iter().filter_map(|r| r.target_rtf).collect();
    let stfts: Vec<f64> = rows.iter().filter_map(|r| r.stft_cosine).collect();
    let logmels: Vec<f64> = rows.iter().filter_map(|r| r.logmel_cosine).collect();
    let n_run = rows.len();
    StressSummary {
        n_planned: cfg.n,
        n_run,
        n_ok,
        n_err,
        n_skipped: skipped,
        ok_rate: if n_run == 0 {
            0.0
        } else {
            n_ok as f64 / n_run as f64
        },
        median_coverage: median(&covs),
        mean_coverage: if covs.is_empty() {
            None
        } else {
            Some(covs.iter().sum::<f64>() / covs.len() as f64)
        },
        median_cer: median(&cers),
        median_rtf: median(&rtfs),
        median_stft: median(&stfts),
        median_logmel: median(&logmels),
        silent_or_tiny: silent,
        target_model: cfg.target_model.clone(),
        ref_model: cfg.ref_model.clone(),
        whisper: cfg.whisper,
    }
}

fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

/// Character error rate over lowercased alphanumeric+space strings.
pub fn char_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let a: Vec<char> = normalize_chars(reference).chars().collect();
    let b: Vec<char> = normalize_chars(hypothesis).chars().collect();
    if a.is_empty() {
        return if b.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = levenshtein(&a, &b);
    dist as f64 / a.len() as f64
}

fn normalize_chars(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let n = a.len();
    let m = b.len();
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

#[allow(dead_code)]
fn _content_preview(text: &str) -> usize {
    content_words(text).len()
}
