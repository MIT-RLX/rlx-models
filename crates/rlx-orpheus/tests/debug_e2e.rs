// Env-gated debug for Orpheus LM prompt + greedy token generation.
mod support;

use rlx_core::weight_loader::load_from_path;
use rlx_gguf::GgufFile;
use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
use rlx_orpheus::backbone::{BackboneLoadOptions, BackboneModel, DEFAULT_N_CTX};
use rlx_orpheus::tokens::{CUSTOM_TOKEN_BASE, build_prompt};
use rlx_qwen35::decode_ids_from_gguf;
use rlx_runtime::Device;
use support::orpheus_gguf_path;

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn top_tokens(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored
}

fn load_generator(
    device: Device,
    gguf: &std::path::Path,
    prefill: MetalGgufPrefillMode,
) -> Llama32Generator {
    let path = gguf.to_str().expect("utf8 path");
    let mut loader = load_from_path(path).expect("open gguf");
    let raw = GgufFile::from_path(gguf).expect("parse gguf");
    let cfg = llama32_cfg_from_gguf(&raw).expect("cfg");
    let g = Llama32Generator::from_loader_at_mode(cfg, loader.as_mut(), device, gguf, prefill)
        .expect("generator")
        .with_compile_seq_cap(64)
        .with_decode_cache(64);
    if std::env::var("ORPHEUS_DYNAMIC_PREFILL").ok().as_deref() == Some("1") {
        g.with_dynamic_prefill_cache(8)
    } else {
        g.with_prefill_cache(8)
    }
}

#[test]
#[ignore]
fn debug_orpheus_reference_prefill_metal_vs_cpu() {
    let Some(gguf) = orpheus_gguf_path() else {
        eprintln!("skip: no gguf");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: metal unavailable");
        return;
    }

    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    eprintln!("prompt len={}", prompt.len());

    let (cpu_tok, cpu_logits) = {
        let mut cpu = load_generator(Device::Cpu, &gguf, MetalGgufPrefillMode::CpuF32);
        let cpu_logits = cpu.prefill_get_last_logits(&prompt).expect("cpu prefill");
        let cpu_tok = greedy_argmax(&cpu_logits);
        eprintln!(
            "cpu argmax={cpu_tok} custom={}",
            cpu_tok >= CUSTOM_TOKEN_BASE
        );
        (cpu_tok, cpu_logits)
    };

    let metal_logits = {
        let mut metal = load_generator(Device::Metal, &gguf, MetalGgufPrefillMode::CpuF32);
        metal
            .prefill_get_last_logits(&prompt)
            .expect("metal+cpu-f32 prefill")
    };
    let metal_tok = greedy_argmax(&metal_logits);
    eprintln!(
        "metal argmax={metal_tok} custom={}",
        metal_tok >= CUSTOM_TOKEN_BASE
    );

    let max_abs = cpu_logits
        .iter()
        .zip(metal_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("reference prefill max_abs={max_abs:.6}");
    assert!(
        max_abs < 0.05,
        "reference prefill diverged (max_abs={max_abs})"
    );
    assert_eq!(cpu_tok, metal_tok, "reference prefill argmax mismatch");
    assert!(
        metal_tok >= CUSTOM_TOKEN_BASE,
        "reference prefill argmax {metal_tok} is not a speech token"
    );
}

#[test]
#[ignore]
fn debug_orpheus_prefill_logits_metal_vs_cpu() {
    let Some(gguf) = orpheus_gguf_path() else {
        eprintln!("skip: no gguf");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: metal unavailable");
        return;
    }

    let prefill_mode = if std::env::var("ORPHEUS_DEBUG_METAL_F32").ok().as_deref() == Some("1") {
        MetalGgufPrefillMode::MetalF32
    } else {
        MetalGgufPrefillMode::CpuF32
    };

    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    eprintln!("prompt len={} prefill_mode={prefill_mode:?}", prompt.len());

    let run_cpu = std::env::var("ORPHEUS_DEBUG_CPU").ok().as_deref() != Some("0");
    let mut cpu_logits = None;
    if run_cpu {
        let mut cpu = load_generator(Device::Cpu, &gguf, MetalGgufPrefillMode::CpuF32);
        let logits = cpu.prefill_get_last_logits(&prompt).expect("cpu prefill");
        let cpu_tok = greedy_argmax(&logits);
        eprintln!(
            "cpu argmax={cpu_tok} custom={}",
            cpu_tok >= CUSTOM_TOKEN_BASE
        );
        for (id, v) in top_tokens(&logits, 8) {
            let piece =
                decode_ids_from_gguf(&gguf, std::slice::from_ref(&id), false).unwrap_or_default();
            eprintln!("  cpu top {id} ({piece:?}) = {v:.4}");
        }
        cpu_logits = Some(logits);
    }

    let mut metal = load_generator(Device::Metal, &gguf, prefill_mode);
    let metal_logits = metal
        .prefill_get_last_logits(&prompt)
        .expect("metal prefill");
    let metal_tok = greedy_argmax(&metal_logits);
    eprintln!(
        "metal argmax={metal_tok} custom={}",
        metal_tok >= CUSTOM_TOKEN_BASE
    );
    for (id, v) in top_tokens(&metal_logits, 8) {
        let piece =
            decode_ids_from_gguf(&gguf, std::slice::from_ref(&id), false).unwrap_or_default();
        eprintln!("  metal top {id} ({piece:?}) = {v:.4}");
    }

    let n = cpu_logits
        .as_ref()
        .map(|c| c.len().min(metal_logits.len()))
        .unwrap_or(metal_logits.len());

    if let Some(cpu_logits) = cpu_logits.as_ref() {
        let max_abs = cpu_logits[..n]
            .iter()
            .zip(metal_logits[..n].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("prefill+kv logits max_abs={max_abs:.6} len={n}");
    }

    // Prefill without KV outputs (same graph as `step()`).
    let mut metal_no_kv = load_generator(Device::Metal, &gguf, prefill_mode);
    metal_no_kv.prefill(&prompt);
    let metal_step = metal_no_kv
        .step(rlx_qwen3::SampleOpts::greedy())
        .expect("metal step");
    eprintln!(
        "metal step() no-kv first token={metal_step} custom={}",
        metal_step >= CUSTOM_TOKEN_BASE
    );

    if run_cpu && std::env::var("ORPHEUS_DEBUG_CPU_STEP").ok().as_deref() == Some("1") {
        let mut cpu = load_generator(Device::Cpu, &gguf, MetalGgufPrefillMode::CpuF32);
        cpu.prefill(&prompt);
        let t = cpu.step(rlx_qwen3::SampleOpts::greedy()).expect("cpu step");
        eprintln!(
            "cpu step() no-kv first token={t} custom={}",
            t >= CUSTOM_TOKEN_BASE
        );
        assert_eq!(t, metal_step, "Metal step() != CPU step()");
    }

    if let Some(cpu_logits) = cpu_logits.as_ref() {
        let max_abs = cpu_logits[..n]
            .iter()
            .zip(metal_logits[..n].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 0.05,
            "Metal prefill+kv logits diverged from CPU (max_abs={max_abs})"
        );
        assert_eq!(
            greedy_argmax(&cpu_logits[..n]),
            metal_tok,
            "Metal prefill+kv argmax != CPU"
        );
        assert_eq!(
            metal_step, metal_tok,
            "Metal prefill+kv argmax != step() no-kv"
        );
    } else {
        assert!(
            metal_tok >= CUSTOM_TOKEN_BASE,
            "Metal prefill+kv argmax {metal_tok} is not a custom speech token"
        );
    }
}

#[test]
#[ignore]
fn debug_orpheus_packed_prefill_metal_vs_cpu() {
    let Some(gguf) = orpheus_gguf_path() else {
        eprintln!("skip: no gguf");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: metal unavailable");
        return;
    }
    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");

    let mut cpu = rlx_llama32::Llama32RunnerBuilder::default()
        .weights(&gguf)
        .device(Device::Cpu)
        .packed_weights(true)
        .max_seq(64)
        .stream(false)
        .build()
        .expect("cpu packed runner");
    let cpu_logits = cpu.predict_logits(&prompt).expect("cpu packed");
    let cpu_tok = greedy_argmax(&cpu_logits);
    eprintln!(
        "cpu packed argmax={cpu_tok} custom={}",
        cpu_tok >= CUSTOM_TOKEN_BASE
    );

    let mut metal = rlx_llama32::Llama32RunnerBuilder::default()
        .weights(&gguf)
        .device(Device::Metal)
        .packed_weights(true)
        .max_seq(64)
        .stream(false)
        .build()
        .expect("metal packed runner");
    let metal_logits = metal.predict_logits(&prompt).expect("metal packed");
    let metal_tok = greedy_argmax(&metal_logits);
    eprintln!(
        "metal packed argmax={metal_tok} custom={}",
        metal_tok >= CUSTOM_TOKEN_BASE
    );

    let max_abs = cpu_logits
        .iter()
        .zip(metal_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("packed prefill max_abs={max_abs:.6}");
    assert!(
        max_abs < 0.05,
        "packed Metal prefill diverged (max_abs={max_abs})"
    );
    assert_eq!(cpu_tok, metal_tok);
}

#[test]
#[ignore]
fn debug_prompt_and_greedy_tokens() {
    let Some(gguf) = orpheus_gguf_path() else {
        eprintln!("skip: no gguf");
        return;
    };

    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    eprintln!("prompt ids ({}): {:?}", prompt.len(), prompt);
    for &id in &prompt {
        let piece =
            decode_ids_from_gguf(&gguf, std::slice::from_ref(&id), false).unwrap_or_default();
        eprintln!("  {id} -> {piece:?}");
    }

    for device in [Device::Metal, Device::Cpu] {
        if device == Device::Metal && !rlx_runtime::is_available(Device::Metal) {
            continue;
        }
        eprintln!("\n=== device {device:?} ===");
        unsafe { std::env::set_var("ORPHEUS_GREEDY", "1") };
        unsafe { std::env::set_var("ORPHEUS_DEBUG_TOKENS", "1") };
        let backbone = BackboneModel::load_on_with(
            &gguf,
            DEFAULT_N_CTX,
            device,
            BackboneLoadOptions::reference_parity(),
        )
        .expect("load");
        let cfg = rlx_orpheus::GenerationConfig {
            max_new_tokens: 32,
            ..Default::default()
        };
        let codes = backbone
            .generate_codes_from_prompt(&prompt, &cfg)
            .expect("generate");
        eprintln!(
            "codes ({}): {:?}",
            codes.len(),
            &codes[..codes.len().min(20)]
        );
    }
}
