// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Matrix report: MiniCPM5-1B GGUF quants × RLX backends (packed prefill).
//
// ```sh
// just fetch-minicpm5-gguf-all
// cargo test -p rlx-models --test minicpm5_quant_matrix --features all-backends --release \
//   minicpm5_quant_matrix -- --nocapture
// ```

use rlx_minicpm5::{MINICPM5_GGUF_FILES, MiniCpm5Runner};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn gguf_dir() -> PathBuf {
    std::env::var("RLX_MINICPM5_GGUF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/MiniCPM5-1B-GGUF"))
}

fn gguf_path(quant: &str) -> Option<PathBuf> {
    let env_key = format!("RLX_MINICPM5_GGUF_{quant}");
    if let Ok(p) = std::env::var(&env_key) {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    let filename = MINICPM5_GGUF_FILES
        .iter()
        .find(|(label, _)| *label == quant)
        .map(|(_, f)| *f)?;
    let path = gguf_dir().join(filename);
    path.is_file().then_some(path)
}

fn backends() -> Vec<(&'static str, Device)> {
    [
        ("cpu", Device::Cpu),
        #[cfg(feature = "metal")]
        ("metal", Device::Metal),
        #[cfg(feature = "mlx")]
        ("mlx", Device::Mlx),
        #[cfg(feature = "cuda")]
        ("cuda", Device::Cuda),
        #[cfg(feature = "rocm")]
        ("rocm", Device::Rocm),
        #[cfg(feature = "gpu")]
        ("wgpu", Device::Gpu),
        #[cfg(feature = "vulkan")]
        ("vulkan", Device::Vulkan),
    ]
    .into_iter()
    .filter(|(_, device)| *device == Device::Cpu || rlx_runtime::is_available(*device))
    .collect()
}

fn predict_run(path: &Path, device: Device) -> Result<(f64, Vec<f32>), String> {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let path = path.to_path_buf();
    let outcome = catch_unwind(AssertUnwindSafe(move || {
        let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let t0 = Instant::now();
        let mut runner = MiniCpm5Runner::builder()
            .weights(&path)
            .device(device)
            .max_seq(64)
            .packed_weights(true)
            .build()
            .map_err(|e| format!("build: {e:#}"))?;
        let logits = runner
            .predict_logits(&prompt)
            .map_err(|e| format!("predict: {e:#}"))?;
        if !logits.iter().all(|v| v.is_finite()) {
            return Err("non-finite logits".into());
        }
        Ok((t0.elapsed().as_secs_f64() * 1000.0, logits))
    }));
    match outcome {
        Ok(inner) => inner,
        Err(_) => Err("panic".into()),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

#[test]
fn minicpm5_quant_matrix() {
    let bks = backends();
    if bks.is_empty() {
        eprintln!("no backends");
        return;
    }

    eprintln!("\n=== MiniCPM5-1B GGUF quant × backend (prefill ms) ===\n");

    let mut header = String::from("quant      ");
    for (name, _) in &bks {
        header.push_str(&format!("{name:>10} "));
    }
    header.push_str(" parity");
    eprintln!("{header}");

    let mut any_gguf = false;
    for (quant, _) in MINICPM5_GGUF_FILES {
        let Some(path) = gguf_path(quant) else {
            eprintln!("{quant:<10} (missing — just fetch-minicpm5-gguf {quant})");
            continue;
        };
        any_gguf = true;

        let mut row = format!("{quant:<10} ");
        let mut min_cos = 1.0f32;
        let mut cpu_logits: Option<Vec<f32>> = None;

        for (name, device) in &bks {
            match predict_run(&path, *device) {
                Ok((ms, logits)) => {
                    if *device == Device::Cpu {
                        cpu_logits = Some(logits);
                    } else if let Some(cpu) = cpu_logits.as_ref() {
                        min_cos = min_cos.min(cosine(cpu, &logits));
                    }
                    row.push_str(&format!("{ms:>9.1} "));
                }
                Err(e) => {
                    row.push_str(&format!("{:>10} ", "ERR"));
                    eprintln!("  [{quant}/{name}] {e}");
                }
            }
        }
        row.push_str(&format!(
            " {}",
            if min_cos >= 0.999 { "ok" } else { "MISMATCH" }
        ));
        eprintln!("{row}");
        eprintln!("           ({})", path.display());
    }

    if !any_gguf {
        eprintln!("\nno GGUF files — run: just fetch-minicpm5-gguf-all");
    }
}
