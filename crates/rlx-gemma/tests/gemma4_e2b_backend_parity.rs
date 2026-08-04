// RLX — GPLv3. E2B QAT backend parity: resolved Metal→MLX matches CPU hidden states.

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_core::weight_loader::WeightLoader; // tensor_bytes_borrowed for packed-Borrow upload
use rlx_gemma::builder::PackedSrc;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::gemma_e2b::{compile_e2b_prefill, resolve_e2b_device};
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;

fn dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = std::path::Path::new(&home).join(
        ".cache/huggingface/hub/\
         models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("config.json").is_file().then_some(snap)
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn hidden_last_token(device: Device, dir: &std::path::Path, cfg: &GemmaConfig) -> Option<Vec<f32>> {
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let loader = GemmaQatLoader::open(dir).ok()?;
    let ple = loader.compute_per_layer_inputs(cfg, &ids).ok()?;
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let mut bld = GemmaQatLoader::open(dir).ok()?;
    let mut packed = HashMap::new();
    let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        cfg,
        &mut bld,
        1,
        ids.len(),
        false,
        false,
        // with_kv_outputs: RLX_E2B_KV_OUT=1 forces K/V to be graph outputs,
        // extending their liveness to graph-end — tests whether the wgpu arena
        // is aliasing the long-lived shared K/V buffer (layers 20–34).
        std::env::var("RLX_E2B_KV_OUT").is_ok(),
        &mut packed,
        None,
        None,
    )
    .ok()?;
    let _exec = resolve_e2b_device(device);
    // RLX_E2B_PLAIN_COMPILE=1 bypasses compile_e2b_prefill (gemma_prefill
    // profile + fusion.skip + Metal guard) and uses the plain Session::compile
    // — the same path the (passing) single-DequantMatMul wgpu parity test uses.
    let mut c = if std::env::var("RLX_E2B_PLAIN_COMPILE").is_ok() {
        let mut c = rlx_runtime::Session::new(device).compile(g);
        for (name, data) in &p {
            c.set_param(name, data.as_slice());
        }
        c
    } else {
        compile_e2b_prefill(device, g, p).ok()?
    };
    // Provide the packed (Q4_K-repacked QAT) projection weights — the builder
    // records each as a `PackedSrc`: `Owned` (embed table) uploads its resident
    // bytes; `Borrow` recipes concat one or more loader tensors from the mmap
    // (mirrors production `upload_packed_borrowed`); the `F32` sentinel is a
    // f32-fallback weight — skip it, uploading empty U8 would clobber the param.
    let mut scratch: Vec<u8> = Vec::new();
    for (k, (src, _scheme, _shape)) in &packed {
        match src {
            PackedSrc::Owned(b) => c.set_param_typed(k, b, rlx_ir::DType::U8),
            PackedSrc::Borrow { keys, nbytes } => {
                scratch.clear();
                scratch.reserve(*nbytes);
                for key in keys {
                    scratch.extend_from_slice(
                        bld.tensor_bytes_borrowed(key)
                            .expect("packed borrow: missing loader tensor"),
                    );
                }
                c.set_param_typed(k, &scratch, rlx_ir::DType::U8);
            }
            PackedSrc::F32 => {}
        }
    }
    let outs = c.run(&[
        ("input_ids", ids_f32.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ]);
    // RLX_TAP_L0=1 + RLX_TAP_LAYER=N: the builder appends layer-N taps as extra
    // graph outputs. Cache the CPU tap VECTORS, then for each device compute the
    // cross-backend per-tap L2 (rms hides the direction error) — the tap whose
    // relL2 JUMPS above the prior tap is the amplifying op.
    if std::env::var("RLX_TAP_L0").is_ok() || std::env::var("RLX_TAP_ALL").is_ok() {
        use std::sync::OnceLock;
        static CPU_TAPS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
        if device == Device::Cpu {
            let _ = CPU_TAPS.set(outs.clone());
        } else if let Some(cpu_taps) = CPU_TAPS.get() {
            for (i, (co, go)) in cpu_taps.iter().zip(&outs).enumerate() {
                let cn = co.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let d = co
                    .iter()
                    .zip(go)
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                eprintln!(
                    "[e2b tap {i:2} {device:?}] len={} relL2={:.3e} L2={d:.3e} cpu_norm={cn:.3e}",
                    co.len(),
                    d / cn.max(1e-12)
                );
            }
        }
    }
    let out = outs.into_iter().next()?;
    let h = cfg.hidden_size;
    let last = ids.len() - 1;
    Some(out[last * h..(last + 1) * h].to_vec())
}

// CPU baseline is expensive (~140s); cache it across the per-device tests
// (they run sequentially in one process under `--test-threads=1`).
fn cpu_hidden(d: &std::path::Path, cfg: &GemmaConfig) -> Vec<f32> {
    use std::sync::OnceLock;
    static CPU: OnceLock<Vec<f32>> = OnceLock::new();
    CPU.get_or_init(|| hidden_last_token(Device::Cpu, d, cfg).expect("cpu forward"))
        .clone()
}

fn check_e2b(dev: Device, tag: &str) {
    let Some(d) = dir() else {
        eprintln!("[{tag}] no checkpoint — skip");
        return;
    };
    if !rlx_runtime::is_available(dev) {
        eprintln!("[{tag}] {dev:?} unavailable — skip");
        return;
    }
    if dev == Device::Gpu {
        // Gemma 4 E2B's text arena (~8.4 GiB) exceeds wgpu's 4 GiB max_buffer_size
        // (rlx-wgpu has no arena partitioning on the text path yet), so buffer
        // planning panics. Skip *before* building — attempting the plan just to
        // catch the panic leaves wgpu in a state that hangs on teardown.
        eprintln!("[{tag}] E2B text arena exceeds wgpu 4 GiB buffer cap — skip");
        return;
    }
    let cfg = GemmaConfig::from_file(&d.join("config.json")).expect("config");
    let cpu = cpu_hidden(&d, &cfg);
    let resolved = hidden_last_token(dev, &d, &cfg).expect("resolved forward");
    let exec = resolve_e2b_device(dev);
    let drift = l2(&cpu, &resolved);
    // Also report cosine + relative L2: for GPU dequant (which can only be
    // ULP-close to the host dequant the CPU ref reuses), raw L2 over a 35-layer
    // stack overstates the functional error. cos≈1 ⇒ same direction (right token).
    let cpu_norm = cpu.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    let res_norm = resolved.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    let dot: f32 = cpu.iter().zip(&resolved).map(|(a, b)| a * b).sum();
    let cos = dot / (cpu_norm * res_norm);
    eprintln!(
        "[{tag}] exec={exec:?} L2={drift:.4e} relL2={:.4e} cos={cos:.6} cpu_norm={cpu_norm:.4e}",
        drift / cpu_norm
    );
    assert!(
        drift < 0.02,
        "resolved E2B device {exec:?} hidden drift {drift} too high vs CPU"
    );
}

#[test]
fn e2b_resolved_metal_matches_cpu_hidden() {
    check_e2b(Device::Metal, "e2b metal");
}

#[test]
fn e2b_resolved_mlx_matches_cpu_hidden() {
    check_e2b(Device::Mlx, "e2b mlx");
}

#[test]
fn e2b_resolved_wgpu_matches_cpu_hidden() {
    check_e2b(Device::Gpu, "e2b wgpu");
}

#[test]
fn e2b_resolved_coreml_matches_cpu_hidden() {
    check_e2b(Device::Ane, "e2b coreml");
}

#[test]
fn e2b_resolved_cuda_matches_cpu_hidden() {
    check_e2b(Device::Cuda, "e2b cuda");
}
