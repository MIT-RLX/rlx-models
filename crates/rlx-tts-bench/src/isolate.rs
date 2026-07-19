//! Per-(model, device) subprocess isolation so abort/OOM/hang cannot kill the matrix.

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rlx_runtime::Device;

use crate::adapters::catalog;
use crate::cli::RunArgs;
use crate::devices::device_label;
use crate::report::{
    BenchRow, append_results_jsonl, read_results_jsonl, write_html, write_markdown,
    write_results_jsonl, write_summary_json,
};
use crate::suite::{failed_row, scenarios_for_flags};

const WORKER_ENV: &str = "RLX_TTS_BENCH_WORKER";

pub fn is_worker() -> bool {
    env::var_os(WORKER_ENV).is_some()
}

pub fn run_isolated(
    args: &RunArgs,
    models: &[String],
    devices: &[Device],
) -> Result<Vec<BenchRow>> {
    let json_path = resolve_out_path(&args.out_dir, &args.json);
    let html_path = resolve_out_path(&args.out_dir, &args.html);
    let md_path = resolve_out_path(&args.out_dir, &args.md);
    let summary_path = args.out_dir.join("summary.json");
    std::fs::create_dir_all(args.out_dir.join("wav"))?;
    std::fs::create_dir_all(args.out_dir.join("_workers"))?;

    let mut rows = if args.resume && json_path.is_file() {
        read_results_jsonl(&json_path)?
    } else {
        Vec::new()
    };
    if !args.resume {
        write_results_jsonl(&json_path, &[])?;
        rows.clear();
    }

    let mut done: HashSet<(String, String, String, String)> = rows
        .iter()
        .map(|r| {
            (
                r.model.clone(),
                r.device.clone(),
                r.phrase.clone(),
                r.scenario.clone(),
            )
        })
        .collect();

    let timeout = Duration::from_secs(args.timeout_secs);
    let exe = env::current_exe().context("current_exe")?;
    let phrase_ids: Vec<&str> = args
        .phrases
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    // CPU first so later devices can load reference WAVs from disk.
    let mut ordered = devices.to_vec();
    ordered.sort_by_key(|d| if *d == Device::Cpu { 0 } else { 1 });

    eprintln!(
        "rlx-tts-bench: isolate=on timeout={}s resume={} → {}",
        args.timeout_secs,
        args.resume,
        args.out_dir.display()
    );

    for model in models {
        let meta = catalog().into_iter().find(|m| m.id == model.as_str());
        let supports_clone = meta.as_ref().map(|m| m.supports_clone).unwrap_or(false);
        let scenarios = scenarios_for_flags(args.clone, supports_clone);

        for &device in &ordered {
            let dlabel = device_label(device);
            let pending: Vec<_> = phrase_ids
                .iter()
                .flat_map(|phrase| {
                    scenarios
                        .iter()
                        .map(move |sc| ((*phrase).to_string(), (*sc).to_string()))
                })
                .filter(|(phrase, sc)| {
                    !done.contains(&(
                        model.clone(),
                        dlabel.to_string(),
                        phrase.clone(),
                        sc.clone(),
                    ))
                })
                .collect();
            if pending.is_empty() {
                eprintln!("{model:<12} {dlabel:<6} (resume skip)");
                continue;
            }

            let slice_json = PathBuf::from("_workers").join(format!("{model}_{dlabel}.jsonl"));
            let slice_json_abs = args.out_dir.join(&slice_json);
            let _ = std::fs::remove_file(&slice_json_abs);

            let mut cmd = Command::new(&exe);
            cmd.env(WORKER_ENV, "1")
                .arg("run")
                .arg("--no-isolate")
                .arg("-m")
                .arg(model)
                .arg("-d")
                .arg(dlabel)
                .arg("--phrases")
                .arg(&args.phrases)
                .arg("--iters")
                .arg(args.iters.unwrap_or(1).to_string())
                .arg("--warmup")
                .arg(args.warmup.to_string())
                .arg("--seed")
                .arg(args.seed.to_string())
                .arg("--out-dir")
                .arg(&args.out_dir)
                .arg("--json")
                .arg(&slice_json)
                .arg("--html")
                .arg(format!("_workers/{model}_{dlabel}.html"))
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());
            if args.whisper {
                cmd.arg("--whisper");
            }
            if args.spectral {
                cmd.arg("--spectral");
            }
            if args.noise {
                cmd.arg("--noise");
            }
            if args.clone {
                cmd.arg("--clone");
            }
            if let Some(p) = &args.clone_ref {
                cmd.arg("--clone-ref").arg(p);
            }
            if let Some(t) = &args.clone_ref_text {
                cmd.arg("--clone-ref-text").arg(t);
            }
            if let Some(t) = &args.text_short {
                cmd.arg("--text-short").arg(t);
            }
            if let Some(t) = &args.text_long {
                cmd.arg("--text-long").arg(t);
            }
            if let Some(n) = args.fail_under_fox {
                cmd.arg("--fail-under-fox").arg(n.to_string());
            }

            eprintln!(
                "{model:<12} {dlabel:<6} worker start ({} cell(s))",
                pending.len()
            );
            let start = Instant::now();
            let mut child = cmd
                .spawn()
                .with_context(|| format!("spawn worker {model}/{dlabel}"))?;
            let outcome = wait_with_timeout(&mut child, timeout)?;
            let elapsed = start.elapsed();

            match outcome {
                WorkerOutcome::Ok => {
                    if slice_json_abs.is_file() {
                        let part_rows = read_results_jsonl(&slice_json_abs)?;
                        append_results_jsonl(&json_path, &part_rows)?;
                        for r in part_rows {
                            upsert_done(&mut done, &mut rows, r);
                        }
                    }
                    eprintln!(
                        "{model:<12} {dlabel:<6} worker ok ({:.0}s)",
                        elapsed.as_secs_f64()
                    );
                }
                WorkerOutcome::TimedOut => {
                    let _ = child.kill();
                    let _ = child.wait();
                    // Keep any rows the worker flushed before hang.
                    if slice_json_abs.is_file() {
                        if let Ok(part_rows) = read_results_jsonl(&slice_json_abs) {
                            let mut got = HashSet::new();
                            for r in &part_rows {
                                got.insert((r.phrase.clone(), r.scenario.clone()));
                            }
                            append_results_jsonl(&json_path, &part_rows)?;
                            for r in part_rows {
                                upsert_done(&mut done, &mut rows, r);
                            }
                            let fail_rows: Vec<_> = pending
                                .iter()
                                .filter(|(phrase, sc)| !got.contains(&(phrase.clone(), sc.clone())))
                                .map(|(phrase, sc)| {
                                    failed_row(
                                        model,
                                        dlabel,
                                        phrase,
                                        sc,
                                        format!("worker timed out after {}s", args.timeout_secs),
                                    )
                                })
                                .collect();
                            append_results_jsonl(&json_path, &fail_rows)?;
                            for r in fail_rows {
                                upsert_done(&mut done, &mut rows, r);
                            }
                        }
                    } else {
                        let fail_rows: Vec<_> = pending
                            .iter()
                            .map(|(phrase, sc)| {
                                failed_row(
                                    model,
                                    dlabel,
                                    phrase,
                                    sc,
                                    format!("worker timed out after {}s", args.timeout_secs),
                                )
                            })
                            .collect();
                        append_results_jsonl(&json_path, &fail_rows)?;
                        for r in fail_rows {
                            upsert_done(&mut done, &mut rows, r);
                        }
                    }
                    eprintln!("{model:<12} {dlabel:<6} TIMEOUT {}s", args.timeout_secs);
                }
                WorkerOutcome::Failed(code) => {
                    let mut got = HashSet::new();
                    if slice_json_abs.is_file() {
                        let part_rows = read_results_jsonl(&slice_json_abs)?;
                        for r in &part_rows {
                            got.insert((r.phrase.clone(), r.scenario.clone()));
                        }
                        append_results_jsonl(&json_path, &part_rows)?;
                        for r in part_rows {
                            upsert_done(&mut done, &mut rows, r);
                        }
                    }
                    let fail_rows: Vec<_> = pending
                        .iter()
                        .filter(|(phrase, sc)| !got.contains(&(phrase.clone(), sc.clone())))
                        .map(|(phrase, sc)| {
                            failed_row(
                                model,
                                dlabel,
                                phrase,
                                sc,
                                format!("worker exited with status {code}"),
                            )
                        })
                        .collect();
                    if !fail_rows.is_empty() {
                        append_results_jsonl(&json_path, &fail_rows)?;
                        for r in fail_rows {
                            upsert_done(&mut done, &mut rows, r);
                        }
                    }
                    eprintln!("{model:<12} {dlabel:<6} worker fail status={code}");
                }
            }

            let summary = write_summary_json(&summary_path, &rows)?;
            write_html(&html_path, &rows, &summary)?;
            let _ = write_markdown(
                &md_path,
                &rows,
                &format!(
                    "Host: local `rlx-tts-bench` (isolate). Devices: {}.\n",
                    args.devices
                ),
            );
        }
    }

    let summary = write_summary_json(&summary_path, &rows)?;
    write_html(&html_path, &rows, &summary)?;
    write_markdown(
        &md_path,
        &rows,
        &format!(
            "Host: local `rlx-tts-bench` (isolate). Devices: {}.\n",
            args.devices
        ),
    )?;
    eprintln!(
        "wrote {}  {}  {}  {}",
        json_path.display(),
        summary_path.display(),
        html_path.display(),
        md_path.display()
    );
    eprintln!(
        "summary: ok={} skipped={} failed={}",
        summary.n_ok, summary.n_skipped, summary.n_failed
    );
    Ok(rows)
}

fn upsert_done(
    done: &mut HashSet<(String, String, String, String)>,
    rows: &mut Vec<BenchRow>,
    row: BenchRow,
) {
    let key = (
        row.model.clone(),
        row.device.clone(),
        row.phrase.clone(),
        row.scenario.clone(),
    );
    done.insert(key.clone());
    rows.retain(|r| {
        !(r.model == key.0 && r.device == key.1 && r.phrase == key.2 && r.scenario == key.3)
    });
    rows.push(row);
}

fn resolve_out_path(out_dir: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        out_dir.join(p)
    }
}

enum WorkerOutcome {
    Ok,
    TimedOut,
    Failed(i32),
}

fn wait_with_timeout(child: &mut std::process::Child, timeout: Duration) -> Result<WorkerOutcome> {
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                return Ok(if status.success() {
                    WorkerOutcome::Ok
                } else {
                    WorkerOutcome::Failed(status.code().unwrap_or(-1))
                });
            }
            None => {
                if start.elapsed() >= timeout {
                    return Ok(WorkerOutcome::TimedOut);
                }
                thread::sleep(Duration::from_millis(250));
            }
        }
    }
}
