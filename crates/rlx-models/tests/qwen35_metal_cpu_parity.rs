// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Cross-backend qwen35 correctness: does prefill/decode on each GPU backend
// match CPU on small synthetic hybrid models? Reproduces (and guards) the
// Metal MPSGraph leading-batch-dim axis bug in Narrow / FusedSwiGLU that
// degenerated Bonsai-27B's gated-DeltaNet prefill.
//
// NOTE: the synth `ramp` weights grow with tensor size and OVERFLOW for
// hidden dims ~5120 (softplus exp → inf → NaN → degenerate logits on ALL
// backends). So these configs stay at small hidden (<=64) where the cos
// metric is meaningful; large-hidden repros need realistic (bounded)
// weights, not `ramp`.

use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen35::synth;
use rlx_runtime::Device;

fn last_logits(device: Device) -> Vec<f32> {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg, weights)
        .device(device)
        .max_seq(12)
        .last_logits_only(true)
        .build()
        .expect("runner");
    runner
        .prefill_get_last_logits(&[1, 2, 3, 4])
        .expect("prefill")
}

/// RLX_QWEN35_HOST_EMBED must be a pure memory optimization: gathering the
/// embedding rows host-side and feeding `inputs_embeds` has to produce the
/// EXACT same logits as the in-graph gather from the resident F32 table.
/// (Guards the Bonsai-27B 4.7 GiB token_embd off-load.)
#[test]
fn qwen35_host_embed_matches_gather_cpu() {
    let baseline = last_logits(Device::Cpu);
    // SAFETY: single-threaded test process (see run invocation).
    unsafe { std::env::set_var("RLX_QWEN35_HOST_EMBED", "1") };
    let host = last_logits(Device::Cpu);
    unsafe { std::env::remove_var("RLX_QWEN35_HOST_EMBED") };
    assert_eq!(
        baseline.len(),
        host.len(),
        "host-embed changed output length"
    );
    let max_abs = baseline
        .iter()
        .zip(&host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("qwen35 host-embed vs gather CPU: max_abs={max_abs:.3e}");
    assert!(
        max_abs < 1e-5,
        "host-embed diverges from in-graph gather: max_abs={max_abs}"
    );

    // Decode too: generated token IDs must be identical with host-embed on/off.
    let base_gen = generated_tokens(Device::Cpu);
    unsafe { std::env::set_var("RLX_QWEN35_HOST_EMBED", "1") };
    let host_gen = generated_tokens(Device::Cpu);
    unsafe { std::env::remove_var("RLX_QWEN35_HOST_EMBED") };
    eprintln!("qwen35 host-embed gen: base={base_gen:?} host={host_gen:?}");
    assert_eq!(
        base_gen, host_gen,
        "host-embed decode diverges from in-graph gather"
    );
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "macos", feature = "mlx"),
    feature = "gpu",
    feature = "cuda"
))]
fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "macos", feature = "mlx"),
    feature = "gpu",
    feature = "cuda"
))]
fn check(device: Device, name: &str) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip qwen35 {name}: backend unavailable");
        return;
    }
    let cpu = last_logits(Device::Cpu);
    let gpu = last_logits(device);
    let c = cos(&cpu, &gpu);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("qwen35 tiny prefill {name}: cos={c:.6} max_abs={max_abs:.4e}");
    assert!(
        c > 0.999,
        "{name} diverges from cpu: cos={c} max_abs={max_abs}"
    );
}

/// Prefill parity on a caller-supplied config (small hidden only).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn cfg_cos(cfg: rlx_models::Qwen35Config, name: &str) -> f32 {
    let ll = |device: Device| -> Vec<f32> {
        let weights = synth::synth_weights(&cfg);
        let mut runner = Qwen35RunnerBuilder::default()
            .inline_weights(cfg.clone(), weights)
            .device(device)
            .max_seq(16)
            .last_logits_only(true)
            .build()
            .expect("runner");
        runner
            .prefill_get_last_logits(&[1, 2, 3, 4])
            .expect("prefill")
    };
    let cpu = ll(Device::Cpu);
    let gpu = ll(Device::Metal);
    let c = cos(&cpu, &gpu);
    eprintln!("qwen35 {name} metal prefill: cos={c:.6}");
    c
}

/// Greedy decode (bucketed) — exercises the decode path prefill-only misses.
fn generated_tokens(device: Device) -> Vec<u32> {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(cfg, weights)
        .device(device)
        .max_seq(16)
        .bucketed_decode(true)
        .build()
        .expect("runner");
    runner
        .generate_with_opts(
            &[1, 2, 3, 4],
            6,
            rlx_models::qwen35::SampleOpts::greedy(),
            |_| true,
        )
        .expect("generate")
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    feature = "gpu",
    feature = "cuda"
))]
fn check_gen(device: Device, name: &str) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip qwen35 gen {name}: unavailable");
        return;
    }
    let cpu = generated_tokens(Device::Cpu);
    let gpu = generated_tokens(device);
    eprintln!("qwen35 tiny gen {name}: cpu={cpu:?} {name}={gpu:?}");
    assert_eq!(cpu, gpu, "{name} bucketed decode diverges from cpu");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_tiny_metal_matches_cpu_prefill() {
    check(Device::Metal, "metal");
}

/// PADDED prefill: short prompt into a large max_seq bucket (like the 27B's
/// max_seq 96), triggering the metal active-extent / padded-prefill path
/// (a known NaN/aliasing source >~30 tokens). Short-seq tests never hit it.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_padded_prefill_metal_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let ll = |device: Device| -> Vec<f32> {
        let cfg = synth::medium_cfg();
        let weights = synth::synth_weights(&cfg);
        let mut runner = Qwen35RunnerBuilder::default()
            .inline_weights(cfg, weights)
            .device(device)
            .max_seq(96)
            .last_logits_only(true)
            .build()
            .expect("runner");
        runner
            .prefill_get_last_logits(&[1, 2, 3, 4])
            .expect("prefill")
    };
    let cpu = ll(Device::Cpu);
    let metal = ll(Device::Metal);
    let c = cos(&cpu, &metal);
    eprintln!("qwen35 padded-prefill(max_seq=96) metal: cos={c:.6}");
    assert!(c > 0.999, "padded prefill metal diverges: cos={c}");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_tiny_metal_matches_cpu_decode() {
    check_gen(Device::Metal, "metal");
}

/// Metal F32 vs AutoMixed f16 residual (`RLX_QWEN35_AMP=1`) on packed Q1_0.
/// Synth dense AMP still has residual f16 gaps; this guards the Bonsai path.
/// Env: `RLX_BONSAI27B_GGUF` (skip when unset).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_bonsai_metal_amp_matches_f32() {
    let Some(path) = std::env::var("RLX_BONSAI27B_GGUF").ok() else {
        eprintln!("skip: RLX_BONSAI27B_GGUF not set");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: metal unavailable");
        return;
    }
    let prompt = [1u32, 2, 3, 4];
    let generate = |amp: bool| -> Vec<u32> {
        if amp {
            unsafe { std::env::set_var("RLX_QWEN35_AMP", "1") };
        } else {
            unsafe { std::env::remove_var("RLX_QWEN35_AMP") };
        }
        let mut runner = Qwen35RunnerBuilder::default()
            .weights(&path)
            .packed_weights(true)
            .device(Device::Metal)
            .max_seq(32)
            .bucketed_decode(true)
            .build()
            .expect("runner");
        let out = runner
            .generate_with_opts(&prompt, 4, rlx_models::qwen35::SampleOpts::greedy(), |_| {
                true
            })
            .expect("generate");
        unsafe { std::env::remove_var("RLX_QWEN35_AMP") };
        out
    };
    let f32_gen = generate(false);
    let amp_gen = generate(true);
    eprintln!("qwen35 bonsai metal AMP vs F32: f32={f32_gen:?} amp={amp_gen:?}");
    assert_eq!(
        f32_gen, amp_gen,
        "Bonsai AMP decode tokens diverge from F32"
    );
}

/// Deep synth model (24 layers) — exercises depth like the 64-layer 27B.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_deep_metal_matches_cpu_prefill() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let mut cfg = synth::medium_cfg();
    cfg.num_hidden_layers = 24;
    cfg.nextn_predict_layers = 0;
    let c = cfg_cos(cfg, "deep24");
    assert!(c > 0.999, "deep metal diverges: cos={c}");
}

/// MRoPE sections — the multimodal RoPE path.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_mrope_metal_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let mut cfg = synth::medium_cfg();
    cfg.num_hidden_layers = 4;
    cfg.nextn_predict_layers = 0;
    cfg.rope_dim_sections = vec![3, 3, 2, 0];
    let c = cfg_cos(cfg, "mrope");
    assert!(c > 0.999, "MRoPE metal diverges: cos={c}");
}

/// Bonsai's large GDN/ssm dims (at small hidden, so no overflow).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_bigssm_metal_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let mut cfg = synth::medium_cfg();
    cfg.num_hidden_layers = 4;
    cfg.nextn_predict_layers = 0;
    cfg.ssm_state_size = 128;
    cfg.ssm_group_count = 16;
    cfg.ssm_time_step_rank = 48;
    cfg.ssm_inner_size = 6144;
    let c = cfg_cos(cfg, "bigssm");
    assert!(c > 0.999, "big-ssm metal diverges: cos={c}");
}

/// Bonsai's head_dim 256 (at small hidden).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_bighead_metal_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let mut cfg = synth::medium_cfg();
    cfg.num_hidden_layers = 4;
    cfg.nextn_predict_layers = 0;
    cfg.key_length = 256;
    cfg.value_length = 256;
    let c = cfg_cos(cfg, "bighead256");
    assert!(c > 0.999, "big head_dim metal diverges: cos={c}");
}

/// Real Bonsai-27B prefill parity, Metal vs CPU. Env-gated on
/// RLX_BONSAI27B_GGUF. With RLX_CPU_DUMP_NODES + RLX_METAL_DUMP_NODES +
/// RLX_ARENA_NO_REUSE + RLX_MPSGRAPH_MIN_FLOPS=<huge> it prints per-node
/// dumps to diff for the first divergent op. Runs cpu then metal serially.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_bonsai27b_metal_matches_cpu() {
    let Some(path) = std::env::var("RLX_BONSAI27B_GGUF").ok() else {
        eprintln!("skip: RLX_BONSAI27B_GGUF not set");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: metal unavailable");
        return;
    }
    let max_seq: usize = std::env::var("RLX_TEST_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let n_prompt: usize = std::env::var("RLX_TEST_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let prompt: Vec<u32> = (0..n_prompt).map(|i| (i as u32) + 1).collect();
    let ll = |device: Device| -> Vec<f32> {
        let mut runner = Qwen35RunnerBuilder::default()
            .weights(&path)
            .packed_weights(true)
            .device(device)
            .max_seq(max_seq)
            .last_logits_only(true)
            .build()
            .expect("runner");
        runner.prefill_get_last_logits(&prompt).expect("prefill")
    };
    let cpu = ll(Device::Cpu);
    let metal = ll(Device::Metal);
    let c = cos(&cpu, &metal);
    let am = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    eprintln!(
        "BONSAI-27B prefill metal: cos={c:.6} cpu_argmax={} metal_argmax={}",
        am(&cpu),
        am(&metal)
    );
    assert!(c > 0.99, "Bonsai-27B metal prefill diverges: cos={c}");
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn qwen35_tiny_mlx_matches_cpu_prefill() {
    check(Device::Mlx, "mlx");
}

#[cfg(feature = "gpu")]
#[test]
fn qwen35_tiny_wgpu_matches_cpu_prefill() {
    check(Device::Gpu, "wgpu");
}

#[cfg(feature = "gpu")]
#[test]
fn qwen35_tiny_wgpu_matches_cpu_decode() {
    check_gen(Device::Gpu, "wgpu");
}

#[cfg(feature = "cuda")]
#[test]
fn qwen35_tiny_cuda_matches_cpu_prefill() {
    check(Device::Cuda, "cuda");
}

#[cfg(feature = "cuda")]
#[test]
fn qwen35_tiny_cuda_matches_cpu_decode() {
    check_gen(Device::Cuda, "cuda");
}
