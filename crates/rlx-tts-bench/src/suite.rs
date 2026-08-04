//! Suite planner: models × devices × phrases × scenarios → metrics → artifacts.

use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use anyhow::Result;
use rlx_runtime::Device;

use crate::adapter::{CloneRequest, SynthRequest};
use crate::adapters::{catalog, make_adapter};
use crate::devices::device_label;
use crate::metrics::{
    WhisperState, noise_metrics, spectral_vs_ref, try_load_whisper, whisper_coverage,
};
use crate::report::{BenchRow, append_results_jsonl};
use crate::wav::{add_gaussian_noise, cosine, median, read_wav_mono, write_wav_mono};

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub models: Vec<String>,
    pub devices: Vec<Device>,
    pub phrases: Vec<(String, String)>,
    pub whisper: bool,
    pub spectral: bool,
    pub noise: bool,
    pub clone: bool,
    pub iters: usize,
    pub warmup: usize,
    pub seed: u64,
    pub out_dir: PathBuf,
    pub clone_ref: Option<PathBuf>,
    pub clone_ref_text: Option<String>,
    pub fail_under_fox: Option<usize>,
    /// When set, append each finished row immediately (worker / crash resilience).
    pub incremental_json: Option<PathBuf>,
}

pub fn run_suite(cfg: &RunConfig) -> Result<Vec<BenchRow>> {
    std::fs::create_dir_all(cfg.out_dir.join("wav"))?;
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

    let mut rows = Vec::new();
    // CPU PCM refs keyed by (model, phrase, scenario)
    let mut cpu_refs: HashMap<(String, String, String), (Vec<f32>, u32)> = HashMap::new();

    for model in &cfg.models {
        let meta = catalog().into_iter().find(|m| m.id == model.as_str());
        let Some(meta) = meta else {
            let row = skipped_row(model, "—", "—", "plain", format!("unknown model '{model}'"));
            push_row(cfg, &mut rows, row);
            continue;
        };
        if model != "fake" && !meta.hints.available() {
            for device in &cfg.devices {
                for (phrase_id, _) in &cfg.phrases {
                    let row = skipped_row(
                        model,
                        device_label(*device),
                        phrase_id,
                        "plain",
                        meta.hints.missing_reason(),
                    );
                    push_row(cfg, &mut rows, row);
                }
            }
            continue;
        }

        for &device in &cfg.devices {
            let adapter = match make_adapter(model, device) {
                Ok(a) => a,
                Err(e) => {
                    let msg = format!("{e:#}");
                    let skip = msg.contains("host CPU only")
                        || msg.contains("not supported")
                        || msg.contains("unavailable");
                    for (phrase_id, _) in &cfg.phrases {
                        let row = if skip {
                            skipped_row(
                                model,
                                device_label(device),
                                phrase_id,
                                "plain",
                                msg.clone(),
                            )
                        } else {
                            failed_row(model, device_label(device), phrase_id, "plain", msg.clone())
                        };
                        push_row(cfg, &mut rows, row);
                    }
                    continue;
                }
            };
            let mut adapter = adapter;
            let supports_clone = adapter.supports_clone();

            for (phrase_id, text) in &cfg.phrases {
                let scenarios = scenarios_for(cfg, supports_clone);
                for scenario in scenarios {
                    let row = match panic::catch_unwind(AssertUnwindSafe(|| {
                        run_one(
                            &mut *adapter,
                            model,
                            device,
                            phrase_id,
                            text,
                            scenario,
                            cfg,
                            &mut whisper,
                            &mut cpu_refs,
                        )
                    })) {
                        Ok(Ok(row)) => row,
                        Ok(Err(e)) => failed_row(
                            model,
                            device_label(device),
                            phrase_id,
                            scenario,
                            format!("{e:#}"),
                        ),
                        Err(payload) => failed_row(
                            model,
                            device_label(device),
                            phrase_id,
                            scenario,
                            format!("panic: {}", panic_message(&payload)),
                        ),
                    };
                    eprintln!(
                        "{:<12} {:<6} {:<6} {:<12} {:>8} {:>7} {}",
                        model,
                        device_label(device),
                        phrase_id,
                        scenario,
                        row.status,
                        row.wall_ms
                            .map(|m| format!("{m:.0}ms"))
                            .unwrap_or_else(|| "—".into()),
                        row.whisper
                            .as_ref()
                            .map(|w| format!("fox {}/{}", w.fox_hits, w.fox_total))
                            .unwrap_or_default(),
                    );
                    push_row(cfg, &mut rows, row);
                }
            }
        }
    }
    Ok(rows)
}

fn push_row(cfg: &RunConfig, rows: &mut Vec<BenchRow>, row: BenchRow) {
    if let Some(path) = &cfg.incremental_json {
        let _ = append_results_jsonl(path, std::slice::from_ref(&row));
    }
    rows.push(row);
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

fn scenarios_for(cfg: &RunConfig, supports_clone: bool) -> Vec<&'static str> {
    scenarios_for_flags(cfg.clone, supports_clone)
}

pub fn scenarios_for_flags(clone: bool, supports_clone: bool) -> Vec<&'static str> {
    let mut v = vec!["plain"];
    if clone && supports_clone {
        v.push("clone");
        v.push("clone_noisy_ref");
    }
    v
}

fn run_one(
    adapter: &mut dyn crate::adapter::TtsAdapter,
    model: &str,
    device: Device,
    phrase_id: &str,
    text: &str,
    scenario: &str,
    cfg: &RunConfig,
    whisper: &mut Option<WhisperState>,
    cpu_refs: &mut HashMap<(String, String, String), (Vec<f32>, u32)>,
) -> Result<BenchRow> {
    let clone_path_owned: Option<PathBuf> = match scenario {
        "clone" => Some(resolve_clone_ref(cfg)?),
        "clone_noisy_ref" => {
            let clean = resolve_clone_ref(cfg)?;
            let (pcm, sr) = read_wav_mono(&clean)?;
            let noisy = add_gaussian_noise(&pcm, 15.0, cfg.seed);
            let noisy_path = cfg.out_dir.join("wav").join(format!(
                "_noisy_ref_{}_{}.wav",
                model,
                device_label(device)
            ));
            write_wav_mono(&noisy_path, &noisy, sr)?;
            Some(noisy_path)
        }
        _ => None,
    };
    let ref_text = cfg.clone_ref_text.as_deref();

    let mk_clone = || {
        clone_path_owned.as_ref().map(|p| CloneRequest {
            ref_wav: p.as_path(),
            ref_text,
        })
    };

    for _ in 0..cfg.warmup {
        let req = SynthRequest {
            text,
            phrase_id,
            device,
            clone: mk_clone(),
            seed: cfg.seed,
            deterministic: true,
        };
        let _ = adapter.synthesize(req);
    }

    let mut walls = Vec::new();
    let mut last = None;
    for _ in 0..cfg.iters.max(1) {
        let req = SynthRequest {
            text,
            phrase_id,
            device,
            clone: mk_clone(),
            seed: cfg.seed,
            deterministic: true,
        };
        match adapter.synthesize(req) {
            Ok(r) => {
                walls.push(r.wall_ms);
                last = Some(r);
            }
            Err(e) => {
                return Ok(failed_row(
                    model,
                    device_label(device),
                    phrase_id,
                    scenario,
                    format!("{e:#}"),
                ));
            }
        }
    }
    let result = last.unwrap();
    let wall_ms = median(walls);
    let audio_sec = result.pcm.len() as f64 / result.sample_rate.max(1) as f64;
    let rtf = if wall_ms > 0.0 {
        audio_sec / (wall_ms / 1000.0)
    } else {
        f64::NAN
    };

    let wav_name = format!(
        "{}_{}_{}_{}.wav",
        model,
        device_label(device),
        phrase_id,
        scenario
    );
    let wav_path = cfg.out_dir.join("wav").join(&wav_name);
    write_wav_mono(&wav_path, &result.pcm, result.sample_rate)?;

    let key = (
        model.to_string(),
        phrase_id.to_string(),
        scenario.to_string(),
    );
    let cosine_vs_cpu = if device == Device::Cpu {
        cpu_refs.insert(key, (result.pcm.clone(), result.sample_rate));
        Some(1.0)
    } else if let Some((ref_pcm, ref_sr)) = cpu_refs.get(&key) {
        let a = crate::wav::resample_linear(&result.pcm, result.sample_rate, *ref_sr);
        Some(cosine(&a, ref_pcm))
    } else if let Some((ref_pcm, ref_sr)) = load_cpu_ref_wav(cfg, model, phrase_id, scenario) {
        cpu_refs.insert(key, (ref_pcm.clone(), ref_sr));
        let a = crate::wav::resample_linear(&result.pcm, result.sample_rate, ref_sr);
        Some(cosine(&a, &ref_pcm))
    } else {
        None
    };

    let whisper_m = if let Some(w) = whisper.as_mut() {
        whisper_coverage(w, &result.pcm, result.sample_rate, text).ok()
    } else {
        None
    };

    if let Some(min_fox) = cfg.fail_under_fox {
        if phrase_id == "short" && scenario == "plain" {
            if let Some(w) = whisper_m.as_ref() {
                if w.fox_hits < min_fox {
                    let fox_hits = w.fox_hits;
                    let mut row = ok_row(
                        model,
                        device_label(device),
                        phrase_id,
                        scenario,
                        wall_ms,
                        rtf,
                        audio_sec,
                        &result,
                        cosine_vs_cpu,
                        whisper_m,
                        None,
                        None,
                        Some(wav_name),
                    );
                    row.status = "failed".into();
                    row.error = Some(format!("fox hits {fox_hits} < --fail-under-fox {min_fox}"));
                    return Ok(row);
                }
            }
        }
    }

    let spectral = if cfg.spectral {
        if scenario == "clone_noisy_ref" {
            cpu_refs
                .get(&(
                    model.to_string(),
                    phrase_id.to_string(),
                    "clone".to_string(),
                ))
                .cloned()
                .or_else(|| load_cpu_ref_wav(cfg, model, phrase_id, "clone"))
                .map(|(ref_pcm, ref_sr)| {
                    spectral_vs_ref(&result.pcm, result.sample_rate, &ref_pcm, ref_sr)
                })
        } else if let Some((ref_pcm, ref_sr)) = cpu_refs
            .get(&(
                model.to_string(),
                phrase_id.to_string(),
                scenario.to_string(),
            ))
            .cloned()
            .or_else(|| load_cpu_ref_wav(cfg, model, phrase_id, scenario))
        {
            Some(spectral_vs_ref(
                &result.pcm,
                result.sample_rate,
                &ref_pcm,
                ref_sr,
            ))
        } else {
            None
        }
    } else {
        None
    };

    // Keep clean clone PCM for noisy-ref spectral compare (any device).
    if scenario == "clone" {
        cpu_refs.insert(
            (model.to_string(), phrase_id.to_string(), "clone".into()),
            (result.pcm.clone(), result.sample_rate),
        );
    }

    let noise = if cfg.noise {
        Some(noise_metrics(&result.pcm))
    } else {
        None
    };

    Ok(ok_row(
        model,
        device_label(device),
        phrase_id,
        scenario,
        wall_ms,
        rtf,
        audio_sec,
        &result,
        cosine_vs_cpu,
        whisper_m,
        spectral,
        noise,
        Some(wav_name),
    ))
}

fn load_cpu_ref_wav(
    cfg: &RunConfig,
    model: &str,
    phrase_id: &str,
    scenario: &str,
) -> Option<(Vec<f32>, u32)> {
    let path = cfg
        .out_dir
        .join("wav")
        .join(format!("{model}_cpu_{phrase_id}_{scenario}.wav"));
    read_wav_mono(&path).ok()
}

fn resolve_clone_ref(cfg: &RunConfig) -> Result<PathBuf> {
    if let Some(p) = &cfg.clone_ref {
        if p.is_file() {
            return Ok(p.clone());
        }
        anyhow::bail!("--clone-ref not found: {}", p.display());
    }
    for p in [
        "assets/jfk/jfk_voice_clone.wav",
        "weights/tts/chatterbox/default_voice.wav",
    ] {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Ok(pb);
        }
    }
    anyhow::bail!("--clone set but no reference wav (pass --clone-ref)");
}

fn skipped_row(
    model: &str,
    device: &str,
    phrase: &str,
    scenario: &str,
    reason: String,
) -> BenchRow {
    BenchRow {
        model: model.into(),
        device: device.into(),
        phrase: phrase.into(),
        scenario: scenario.into(),
        status: "skipped".into(),
        skip_reason: Some(reason),
        error: None,
        wall_ms: None,
        rtf: None,
        audio_sec: None,
        sample_rate: None,
        exec_label: None,
        cosine_vs_cpu: None,
        whisper: None,
        spectral: None,
        noise: None,
        wav_rel: None,
    }
}

pub fn failed_row(
    model: &str,
    device: &str,
    phrase: &str,
    scenario: &str,
    err: String,
) -> BenchRow {
    BenchRow {
        model: model.into(),
        device: device.into(),
        phrase: phrase.into(),
        scenario: scenario.into(),
        status: "failed".into(),
        skip_reason: None,
        error: Some(err),
        wall_ms: None,
        rtf: None,
        audio_sec: None,
        sample_rate: None,
        exec_label: None,
        cosine_vs_cpu: None,
        whisper: None,
        spectral: None,
        noise: None,
        wav_rel: None,
    }
}

fn ok_row(
    model: &str,
    device: &str,
    phrase: &str,
    scenario: &str,
    wall_ms: f64,
    rtf: f64,
    audio_sec: f64,
    result: &crate::adapter::SynthResult,
    cosine_vs_cpu: Option<f64>,
    whisper: Option<crate::metrics::WhisperMetrics>,
    spectral: Option<crate::metrics::SpectralMetrics>,
    noise: Option<crate::metrics::NoiseMetrics>,
    wav_name: Option<String>,
) -> BenchRow {
    BenchRow {
        model: model.into(),
        device: device.into(),
        phrase: phrase.into(),
        scenario: scenario.into(),
        status: "ok".into(),
        skip_reason: None,
        error: None,
        wall_ms: Some(wall_ms),
        rtf: Some(rtf),
        audio_sec: Some(audio_sec),
        sample_rate: Some(result.sample_rate),
        exec_label: Some(result.exec_label.clone()),
        cosine_vs_cpu,
        whisper,
        spectral,
        noise,
        wav_rel: wav_name.map(|n| format!("wav/{n}")),
    }
}

pub fn list_adapters() {
    println!("{:<14} {:<8} {:<12} weights", "model", "clone", "feature");
    for m in catalog() {
        let status = if m.id == "fake" || m.hints.available() {
            format!("OK {}", m.hints.resolve_dir().unwrap_or_default().display())
        } else {
            format!("MISSING {}", m.hints.missing_reason())
        };
        println!(
            "{:<14} {:<8} {:<12} {}",
            m.id,
            if m.supports_clone { "yes" } else { "no" },
            m.feature,
            status
        );
    }
}

pub fn select_models(spec: &str) -> Vec<String> {
    let all: Vec<_> = catalog().into_iter().map(|m| m.id.to_string()).collect();
    if spec.trim().eq_ignore_ascii_case("all") {
        return all.into_iter().filter(|id| id != "fake").collect();
    }
    spec.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn gate_failed(rows: &[BenchRow], fail_under_fox: Option<usize>) -> bool {
    if fail_under_fox.is_none() {
        return false;
    }
    rows.iter().any(|r| {
        r.status == "failed"
            && r.error
                .as_ref()
                .is_some_and(|e| e.contains("fail-under-fox"))
    })
}

#[allow(dead_code)]
fn _path_str(p: &Path) -> String {
    p.display().to_string()
}
