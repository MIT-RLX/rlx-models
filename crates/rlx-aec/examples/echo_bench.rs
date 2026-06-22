//! RLX AEC echo bench — synthetic MSE improvement, latency, optional fdaf-aec comparison.

use anyhow::{Context, Result, bail};
use rlx_aec::{
    FdafConfig, FdafNlms, apply_echo, correlation, embedded_residual_weights, mse_improvement_db,
};
use serde::Serialize;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize)]
struct BenchRow {
    label: String,
    mse_improve_db: f32,
    correlation: f32,
    us_per_frame: f64,
    residual: bool,
}

#[derive(Serialize)]
struct BenchReport {
    rows: Vec<BenchRow>,
}

fn synth(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut clean = Vec::with_capacity(n);
    let mut far = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / 16_000.0;
        clean.push(0.35 * (2.0 * std::f32::consts::PI * 280.0 * t).sin());
        far.push(0.42 * (2.0 * std::f32::consts::PI * 520.0 * t).sin());
    }
    (clean, far)
}

fn bench_rlx(
    mic: &[f32],
    far: &[f32],
    clean: &[f32],
    residual: bool,
    label: &str,
) -> Result<BenchRow> {
    let delay = 200;
    let mut aligned = vec![0.0f32; far.len() + delay];
    aligned[delay..delay + far.len()].copy_from_slice(far);
    let aligned = aligned[..mic.len()].to_vec();

    let cfg = FdafConfig {
        n_fft: 1024,
        frame_samples: 512,
        step_size: 0.05,
        use_residual: residual,
        adapt: true,
    };
    let residual_w = if residual {
        Some(embedded_residual_weights()?)
    } else {
        None
    };
    let mut fdaf = FdafNlms::new(cfg, residual_w)?;
    let mut out = vec![0.0f32; mic.len()];

    let hop = 512;
    for _ in 0..20 {
        fdaf.process_buffer(mic, &aligned, &mut out)?;
    }

    let frames = mic.len() / hop;
    let t0 = Instant::now();
    fdaf.process_buffer(mic, &aligned, &mut out)?;
    let us_per_frame = (t0.elapsed().as_secs_f64() * 1e6) / frames.max(1) as f64;

    Ok(BenchRow {
        label: label.to_string(),
        mse_improve_db: mse_improvement_db(mic, &out, clean),
        correlation: correlation(clean, &out),
        us_per_frame,
        residual,
    })
}

fn bench_fdaf_aec(mic: &[f32], far: &[f32], clean: &[f32]) -> Result<BenchRow> {
    use fdaf_aec::FdafAec;
    const FFT: usize = 1024;
    const HOP: usize = FFT / 2;
    let delay = 200;
    let mut aligned = vec![0.0f32; far.len() + delay];
    aligned[delay..delay + far.len()].copy_from_slice(far);
    let aligned = aligned[..mic.len()].to_vec();

    let mut aec = FdafAec::new(FFT, 0.05);
    let mut out = Vec::new();
    for pos in (0..mic.len()).step_by(HOP) {
        let end = (pos + HOP).min(mic.len());
        let mut mf = vec![0.0f32; HOP];
        let mut ff = vec![0.0f32; HOP];
        mf[..end - pos].copy_from_slice(&mic[pos..end]);
        ff[..end - pos].copy_from_slice(&aligned[pos..end]);
        let chunk = aec.process(&ff, &mf);
        out.extend_from_slice(&chunk[..end - pos]);
    }
    Ok(BenchRow {
        label: "fdaf-aec".to_string(),
        mse_improve_db: mse_improvement_db(mic, &out, clean),
        correlation: correlation(clean, &out),
        us_per_frame: 0.0,
        residual: false,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut json_out: Option<PathBuf> = None;
    let mut compare_fdaf = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(&args[i]));
                i += 1;
            }
            "--compare-fdaf" => {
                compare_fdaf = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("echo_bench [--json-out PATH] [--compare-fdaf]");
                return Ok(());
            }
            other => bail!("unknown arg: {other}"),
        }
    }

    let n = 32_000;
    let (clean, far) = synth(n);
    let mic = apply_echo(&clean, &far, 200, 0.65);

    let mut rows = vec![
        bench_rlx(&mic, &far, &clean, false, "rlx-aec-linear")?,
        bench_rlx(&mic, &far, &clean, true, "rlx-aec+residual")?,
    ];

    if compare_fdaf {
        rows.push(bench_fdaf_aec(&mic, &far, &clean)?);
    }

    println!("label              MSE+ dB  corr    us/frame");
    for r in &rows {
        println!(
            "{:<18} {:7.2}   {:5.3}   {:7.1}",
            r.label, r.mse_improve_db, r.correlation, r.us_per_frame
        );
    }

    let report = BenchReport { rows };
    if let Some(path) = json_out {
        let json = serde_json::to_string_pretty(&report).context("json")?;
        std::fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
        println!("json → {}", path.display());
    }

    Ok(())
}
