// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Localize where a GPU backend (CUDA/Metal/MLX) diverges from CPU inside the
//! Kitten graph. Compiles the same multi-probe graph on CPU and on the target
//! device, runs identical inputs (fixed duration carry so alignment is not a
//! variable), and reports the per-node max-abs diff, worst first.
//!
//! Device via `KITTEN_PROBE_DEVICE` (default `cuda`). Optional `PROBE_FILTER`.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    build_cached_graphs_from_import, compile_multi_probe_graph, import_from_bundle_cached,
    probe_output_f32_at, run_parity_inputs_with_duration,
};
use kitten_tts_mini_rlx::probe_watch::{self, WATCH};
use rlx_runtime::Device;

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = (a[i] - ma) as f64;
        let xb = (b[i] - mb) as f64;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    (num / (da.sqrt() * db.sqrt())) as f32
}

fn f32_out(outs: &[(Vec<u8>, rlx_runtime::DType)], idx: usize) -> Vec<f32> {
    outs.get(idx)
        .map(|(b, _)| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_device(name: &str) -> Device {
    match name.trim().to_ascii_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" => Device::Gpu,
        "rocm" => Device::Rocm,
        other => {
            eprintln!("unknown KITTEN_PROBE_DEVICE={other:?}, using cuda");
            Device::Cuda
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let n = a.len().min(b.len());
    let mut max = 0.0f32;
    let mut idx = 0usize;
    for j in 0..n {
        let d = (a[j] - b[j]).abs();
        if d > max {
            max = d;
            idx = j;
        }
    }
    (max, idx)
}

fn main() -> anyhow::Result<()> {
    let device =
        parse_device(&std::env::var("KITTEN_PROBE_DEVICE").unwrap_or_else(|_| "cuda".into()));
    let filter = std::env::var("PROBE_FILTER").ok();
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");

    // Framed həˈloʊ: `[pad] + phonemes + [ellipsis=10] + [pad]`.
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 10, 0];
    let token_len = ids.len();
    let seq = token_len;
    let max_wave = token_len
        .saturating_mul(600)
        .saturating_mul(8)
        .saturating_add(12_000)
        .max(24_000);
    // Deterministic style row (no python/npz dependency): tiny ramp.
    let style: Vec<f32> = (0..256).map(|i| (i as f32 % 17.0) * 0.01 - 0.08).collect();
    // ORT duration for framed həˈloʊ (sum=34 alignment frames).
    let ort_dur: Vec<i64> = vec![3, 2, 2, 3, 4, 4, 13, 2, 1];

    let graph_opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: max_wave,
    };

    let import = import_from_bundle_cached(&bundle_dir, &graph_opts)?;

    // Waveform-only mode: compile plain full graphs on Cpu and `device`, run the
    // same inputs, and report the true output-waveform correlation (probe taps can
    // be unreliable on f32-uniform arenas, so this is the decisive signal).
    if std::env::var("KITTEN_PROBE_WAVEFORM").is_ok() {
        kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
        let cpu_g = build_cached_graphs_from_import(
            Device::Cpu,
            "probe_cpu",
            &import,
            None,
            seq,
            max_wave,
        )?;
        let dev_g =
            build_cached_graphs_from_import(device, "probe_dev", &import, None, seq, max_wave)?;
        let cpu_wave = {
            let mut g = cpu_g.full.lock().unwrap();
            let o = run_parity_inputs_with_duration(
                &mut g,
                seq,
                ids.len(),
                &ids,
                &style,
                Some(&ort_dur),
            );
            f32_out(&o, 0)
        };
        let dev_wave = {
            let mut g = dev_g.full.lock().unwrap();
            let o = run_parity_inputs_with_duration(
                &mut g,
                seq,
                ids.len(),
                &ids,
                &style,
                Some(&ort_dur),
            );
            f32_out(&o, 0)
        };
        let peak = |w: &[f32]| w.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        eprintln!(
            "waveform: cpu.len={} {device:?}.len={} corr={:.4} cpu_peak={:.4e} dev_peak={:.4e}",
            cpu_wave.len(),
            dev_wave.len(),
            corr(&cpu_wave, &dev_wave),
            peak(&cpu_wave),
            peak(&dev_wave),
        );
        eprintln!("cpu[0..6]={:?}", &cpu_wave[..cpu_wave.len().min(6)]);
        eprintln!("dev[0..6]={:?}", &dev_wave[..dev_wave.len().min(6)]);
        return Ok(());
    }

    // Tap the largest HIR node for each aliased ONNX name (the real op output).
    let all_probes: Vec<_> = WATCH
        .iter()
        .filter(|(hir, _)| probe_watch::matches_filter(hir, filter.as_deref()))
        .filter_map(|(hir, _)| {
            import
                .hir
                .nodes()
                .iter()
                .filter(|n| n.name.as_deref() == Some(*hir))
                .max_by_key(|n| {
                    n.shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product::<usize>()
                })
                .map(|n| (n.id, *hir))
        })
        .collect();

    eprintln!("compiling {} probes on Cpu …", all_probes.len());
    let mut cpu_graph =
        compile_multi_probe_graph(Device::Cpu, &bundle_dir, &graph_opts, &import, &all_probes)?;
    eprintln!("compiling {} probes on {device:?} …", all_probes.len());
    let mut dev_graph =
        compile_multi_probe_graph(device, &bundle_dir, &graph_opts, &import, &all_probes)?;

    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
    let cpu_outs = run_parity_inputs_with_duration(
        &mut cpu_graph,
        seq,
        ids.len(),
        &ids,
        &style,
        Some(&ort_dur),
    );
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
    let dev_outs = run_parity_inputs_with_duration(
        &mut dev_graph,
        seq,
        ids.len(),
        &ids,
        &style,
        Some(&ort_dur),
    );

    struct Row {
        label: String,
        max_abs: f32,
        idx: usize,
        cpu0: Vec<f32>,
        dev0: Vec<f32>,
    }
    let mut rows = Vec::new();
    for (i, (_, hir_name)) in all_probes.iter().enumerate() {
        let (Some(cpu), Some(dev)) = (
            probe_output_f32_at(&cpu_outs, i),
            probe_output_f32_at(&dev_outs, i),
        ) else {
            continue;
        };
        let (max_abs, idx) = max_abs_diff(&cpu, &dev);
        let short = hir_name
            .rsplit('/')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/");
        rows.push(Row {
            label: short,
            max_abs,
            idx,
            cpu0: cpu.iter().take(4).copied().collect(),
            dev0: dev.iter().take(4).copied().collect(),
        });
    }

    // Graph output waveform peak on each side.
    let peak = |outs: &[(Vec<u8>, rlx_runtime::DType)]| {
        outs.first()
            .map(|(b, _)| {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()).abs())
                    .fold(0.0f32, f32::max)
            })
            .unwrap_or(0.0)
    };
    eprintln!(
        "\nwaveform peak: cpu={:.4e} {device:?}={:.4e}",
        peak(&cpu_outs),
        peak(&dev_outs)
    );

    rows.sort_by(|a, b| b.max_abs.partial_cmp(&a.max_abs).unwrap());
    eprintln!("\n=== Cpu vs {device:?} — worst first ===");
    for r in rows.iter().take(24) {
        eprintln!(
            "{:>48} max={:.5} idx={} cpu0={:?} dev0={:?}",
            r.label, r.max_abs, r.idx, r.cpu0, r.dev0
        );
    }
    Ok(())
}
