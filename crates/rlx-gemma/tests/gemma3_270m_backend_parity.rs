//! Env-gated functional parity: packed Gemma 3 270M greedy decode on every
//! accelerator backend must match the CPU reference token stream.
//!
//! ```sh
//! RLX_GEMMA3_GGUF=/tmp/rlx-weights/gemma-3-270m.gguf \
//! cargo test -p rlx-gemma --test gemma3_270m_backend_parity \
//!   --features apple-silicon --release -- --test-threads=1 --nocapture
//! ```

use rlx_gemma::GemmaRunner;
use rlx_qwen3::SampleOpts;
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

const HF_CHAT_IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 1156, 2915, 1156, 236881, 25685, 528, 886, 2822, 13315, 236761,
    106, 107, 105, 4368, 107,
];

const GREEDY_STEPS: usize = 16;

fn weights() -> Option<PathBuf> {
    std::env::var("RLX_GEMMA3_GGUF").ok().map(PathBuf::from)
}

fn greedy_on(device: Device) -> Vec<u32> {
    let weights = weights().expect("RLX_GEMMA3_GGUF");
    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(device)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()
        .expect("build runner");
    runner
        .generate(HF_CHAT_IDS, GREEDY_STEPS, |_| {})
        .expect("greedy generate")
}

fn cpu_reference_greedy() -> Vec<u32> {
    greedy_on(Device::Cpu)
}

// Exercised only by backend-feature-gated tests below; dead under default features.
#[allow(dead_code)]
fn prefill_top1(device: Device) -> u32 {
    let weights = weights().expect("RLX_GEMMA3_GGUF");
    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(device)
        .max_seq(512)
        .build()
        .expect("build runner");
    let logits = runner.predict_logits(HF_CHAT_IDS).expect("prefill logits");
    let vocab = runner.config().vocab_size;
    logits
        .iter()
        .take(vocab)
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// Only invoked under backend-feature cfgs; unused under default features.
#[allow(unused_macros)]
macro_rules! backend_greedy_matches_cpu {
    ($name:ident, $device:expr, $label:literal) => {
        #[test]
        fn $name() {
            let Some(_) = weights() else {
                eprintln!("skip: set RLX_GEMMA3_GGUF");
                return;
            };
            let device = $device;
            if !is_available(device) {
                eprintln!("skip {}: {:?} unavailable", $label, device);
                return;
            }
            let cpu = cpu_reference_greedy();
            let other = greedy_on(device);
            eprintln!("{} greedy = {:?}", $label, other);
            eprintln!("cpu greedy   = {:?}", cpu);
            assert_eq!(
                other, cpu,
                "packed Gemma 3 270M greedy on {} must match CPU",
                $label
            );
        }
    };
}

#[test]
fn cpu_reference_greedy_is_stable() {
    let Some(_) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    let a = cpu_reference_greedy();
    let b = cpu_reference_greedy();
    assert_eq!(a, b, "CPU greedy must be deterministic");
    assert_eq!(a[0], 11634, "HF chat greedy step 0");
}

#[cfg(feature = "metal")]
backend_greedy_matches_cpu!(
    gemma3_270m_greedy_matches_cpu_on_metal,
    Device::Metal,
    "Metal"
);

#[cfg(feature = "mlx")]
backend_greedy_matches_cpu!(gemma3_270m_greedy_matches_cpu_on_mlx, Device::Mlx, "MLX");

#[cfg(feature = "gpu")]
backend_greedy_matches_cpu!(gemma3_270m_greedy_matches_cpu_on_wgpu, Device::Gpu, "wgpu");

#[cfg(feature = "cuda")]
backend_greedy_matches_cpu!(gemma3_270m_greedy_matches_cpu_on_cuda, Device::Cuda, "CUDA");

#[cfg(feature = "coreml")]
backend_greedy_matches_cpu!(
    gemma3_270m_greedy_matches_cpu_on_coreml,
    Device::Ane,
    "CoreML"
);

#[cfg(feature = "vulkan")]
backend_greedy_matches_cpu!(
    gemma3_270m_greedy_matches_cpu_on_vulkan,
    Device::Vulkan,
    "Vulkan"
);

#[cfg(feature = "gpu")]
#[test]
fn gemma3_270m_prefill_top1_matches_cpu_on_wgpu() {
    let Some(_) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    if !is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let cpu = prefill_top1(Device::Cpu);
    let gpu = prefill_top1(Device::Gpu);
    eprintln!("wgpu prefill top1={gpu} cpu top1={cpu}");
    assert_eq!(gpu, cpu, "wgpu packed prefill top-1 must match CPU");
}

#[cfg(feature = "cuda")]
#[test]
fn gemma3_270m_prefill_top1_matches_cpu_on_cuda() {
    let Some(_) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    if !is_available(Device::Cuda) {
        eprintln!("skip: CUDA unavailable");
        return;
    }
    let cpu = prefill_top1(Device::Cpu);
    let gpu = prefill_top1(Device::Cuda);
    eprintln!("CUDA prefill top1={gpu} cpu top1={cpu}");
    assert_eq!(gpu, cpu, "CUDA packed prefill top-1 must match CPU");
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma3_270m_prefill_top1_matches_cpu_on_vulkan() {
    let Some(_) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    if !is_available(Device::Vulkan) {
        eprintln!("skip: Vulkan unavailable");
        return;
    }
    let cpu = prefill_top1(Device::Cpu);
    let gpu = prefill_top1(Device::Vulkan);
    eprintln!("Vulkan prefill top1={gpu} cpu top1={cpu}");
    assert_eq!(gpu, cpu, "Vulkan packed prefill top-1 must match CPU");
}

#[test]
fn gemma3_270m_prefill_top1_finite_on_metal() {
    if !cfg!(feature = "metal") {
        eprintln!("skip: metal feature disabled");
        return;
    }
    let Some(weights) = weights() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Metal)
        .max_seq(512)
        .build()
        .expect("build");

    let logits = runner.predict_logits(HF_CHAT_IDS).expect("prefill logits");
    let vocab = runner.config().vocab_size;
    let slice = &logits[..vocab];
    assert!(
        slice.iter().all(|v| v.is_finite()),
        "Metal session prefill logits must be finite"
    );
    let top1 = slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    assert_eq!(top1, 11634, "Metal prefill top-1 vs CPU reference");
}
