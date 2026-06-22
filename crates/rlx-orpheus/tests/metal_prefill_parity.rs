// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Fast Orpheus GGUF prefill parity (no full 56-token LM run).

mod support;

use rlx_orpheus::tokens::{CUSTOM_TOKEN_BASE, build_prompt, mask_logits_for_snac_slot};
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;

fn orpheus_sample_opts() -> SampleOpts {
    if std::env::var("ORPHEUS_GREEDY").ok().as_deref() == Some("1") {
        SampleOpts::greedy()
    } else {
        SampleOpts::temperature(0.6, 42)
            .with_top_p(0.8)
            .with_repetition_penalty(1.3)
    }
}

fn generate_masked_codes(
    generator: &mut rlx_llama32::Llama32Generator,
    sample: SampleOpts,
    steps: u64,
) -> Vec<i32> {
    use rlx_orpheus::tokens::mask_logits_for_snac_slot;
    use rlx_orpheus::tokens::{custom_token_id_to_code, is_snac_slot_token, use_snac_logit_mask};
    use rlx_qwen3::apply_repetition_penalty;

    let mut stream_index = 0usize;
    let mut codes = Vec::new();
    let mut token_counts = std::collections::HashMap::<u32, u32>::new();
    let apply_penalty = sample.repetition_penalty > 1.0;
    for step in 0..steps {
        let slot_ix = stream_index;
        let tok = generator
            .step_cached_adjust(sample, step, |logits| {
                if use_snac_logit_mask() {
                    mask_logits_for_snac_slot(logits, slot_ix);
                }
                if apply_penalty {
                    apply_repetition_penalty(logits, &token_counts, sample.repetition_penalty);
                }
            })
            .expect("decode step");
        if let Some(code) = custom_token_id_to_code(tok, stream_index) {
            if is_snac_slot_token(tok, stream_index) && code > 0 {
                codes.push(code);
            }
        }
        if is_snac_slot_token(tok, stream_index) {
            stream_index += 1;
        }
        *token_counts.entry(tok).or_insert(0) += 1;
    }
    codes
}

#[test]
#[cfg(all(feature = "llama", feature = "metal"))]
fn metal_reference_prefill_first_token_is_speech() {
    let Some(gguf) = support::orpheus_gguf_path() else {
        eprintln!("skip: run `just fetch-orpheus`");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    use rlx_core::weight_loader::load_from_path;
    use rlx_gguf::GgufFile;
    use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
    use rlx_orpheus::DEFAULT_COMPILE_SEQ_CAP;

    let sample = orpheus_sample_opts();
    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    let path = gguf.to_str().expect("utf8");
    let mut loader = load_from_path(path).expect("open gguf");
    let raw = GgufFile::from_path(&gguf).expect("parse gguf");
    let cfg = llama32_cfg_from_gguf(&raw).expect("cfg");
    let mut generator = Llama32Generator::from_loader_at_mode(
        cfg,
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::CpuF32,
    )
    .expect("generator")
    .with_compile_seq_cap(DEFAULT_COMPILE_SEQ_CAP as usize)
    .with_prefill_cache(4)
    .with_decode_cache(DEFAULT_COMPILE_SEQ_CAP as usize + 8);

    generator.prefill(&prompt);
    let first = generator
        .step_cached_adjust(sample, 0, |logits| mask_logits_for_snac_slot(logits, 0))
        .expect("first cached decode step");
    eprintln!("first token={first} custom={}", first >= CUSTOM_TOKEN_BASE);
    assert!(
        first >= CUSTOM_TOKEN_BASE,
        "expected speech custom token, got {first}"
    );
}

#[test]
#[cfg(all(feature = "llama", feature = "metal"))]
fn metal_reference_generates_packable_codes_for_hi() {
    let Some(gguf) = support::orpheus_gguf_path() else {
        eprintln!("skip: run `just fetch-orpheus`");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    use rlx_core::weight_loader::load_from_path;
    use rlx_gguf::GgufFile;
    use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
    use rlx_orpheus::DEFAULT_COMPILE_SEQ_CAP;
    use rlx_orpheus::tokens::pack_orpheus_codes;

    let sample = orpheus_sample_opts();
    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    let path = gguf.to_str().expect("utf8");
    let mut loader = load_from_path(path).expect("open gguf");
    let raw = GgufFile::from_path(&gguf).expect("parse gguf");
    let cfg = llama32_cfg_from_gguf(&raw).expect("cfg");
    let cap = DEFAULT_COMPILE_SEQ_CAP as usize;
    let mut generator = Llama32Generator::from_loader_at_mode(
        cfg,
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::CpuF32,
    )
    .expect("generator")
    .with_compile_seq_cap(cap)
    .with_prefill_cache(4)
    .with_decode_cache(cap + 8);

    generator.prefill(&prompt);
    let codes = generate_masked_codes(&mut generator, sample, 28);
    eprintln!(
        "codes len={} first={:?}",
        codes.len(),
        &codes[..codes.len().min(8)]
    );
    if let Ok(path) = std::env::var("ORPHEUS_DUMP_CODES") {
        let body = format!(
            "{}\n{}",
            codes.len(),
            codes
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = std::fs::write(&path, body);
    }
    let generated = generator.tokens()[prompt.len()..].to_vec();
    eprintln!(
        "generated {} tokens: {:?}",
        generated.len(),
        &generated[..generated.len().min(16)]
    );
    assert!(
        codes.len() >= 7,
        "expected at least one SNAC frame, got {}",
        codes.len()
    );
    assert!(
        pack_orpheus_codes(&codes).is_some(),
        "codes must pack for SNAC decode"
    );
}

#[test]
#[cfg(feature = "llama")]
fn cpu_eager_oneshot_generates_packable_codes_for_hi() {
    let Some(gguf) = support::orpheus_gguf_path() else {
        eprintln!("skip: run `just fetch-orpheus`");
        return;
    };

    use rlx_core::weight_loader::load_from_path;
    use rlx_gguf::GgufFile;
    use rlx_llama32::{Llama32Generator, llama32_cfg_from_gguf};
    use rlx_orpheus::DEFAULT_COMPILE_SEQ_CAP;
    use rlx_orpheus::tokens::pack_orpheus_codes;
    use rlx_runtime::Device;

    let sample = orpheus_sample_opts();
    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    let path = gguf.to_str().expect("utf8");
    let mut loader = load_from_path(path).expect("open gguf");
    let raw = GgufFile::from_path(&gguf).expect("parse gguf");
    let cfg = llama32_cfg_from_gguf(&raw).expect("cfg");
    let cap = DEFAULT_COMPILE_SEQ_CAP as usize;
    let mut generator = Llama32Generator::from_loader(cfg, loader.as_mut(), Device::Cpu)
        .expect("generator")
        .with_decode_cache(cap + 8);

    generator.prefill(&prompt);
    let codes = generate_masked_codes(&mut generator, sample, 14);
    eprintln!(
        "cpu oneshot codes len={} first={:?}",
        codes.len(),
        &codes[..codes.len().min(8)]
    );
    assert!(codes.len() >= 7);
    assert!(
        pack_orpheus_codes(&codes).is_some(),
        "cpu oneshot must pack"
    );
}

#[test]
#[cfg(all(feature = "llama", feature = "metal"))]
fn metal_packed_prefill_first_token_is_speech() {
    let Some(gguf) = support::orpheus_gguf_path() else {
        eprintln!("skip: run `just fetch-orpheus`");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    use rlx_core::weight_loader::load_from_path;
    use rlx_gguf::GgufFile;
    use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
    use rlx_orpheus::DEFAULT_COMPILE_SEQ_CAP;

    let sample = orpheus_sample_opts();
    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    let path = gguf.to_str().expect("utf8");
    let mut loader = load_from_path(path).expect("open gguf");
    let raw = GgufFile::from_path(&gguf).expect("parse gguf");
    let cfg = llama32_cfg_from_gguf(&raw).expect("cfg");
    let mut generator = Llama32Generator::from_loader_at_mode(
        cfg,
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::PackedGguf,
    )
    .expect("generator")
    .with_compile_seq_cap(DEFAULT_COMPILE_SEQ_CAP as usize)
    .with_prefill_cache(4)
    .with_decode_cache(DEFAULT_COMPILE_SEQ_CAP as usize + 8);

    generator.prefill(&prompt);
    let first = generator
        .step_cached_adjust(sample, 0, |logits| mask_logits_for_snac_slot(logits, 0))
        .expect("first cached decode step");
    eprintln!(
        "packed first token={first} custom={}",
        first >= CUSTOM_TOKEN_BASE
    );
    assert!(
        first >= CUSTOM_TOKEN_BASE,
        "expected speech custom token, got {first}"
    );
}

fn generate_masked_tokens(
    generator: &mut rlx_llama32::Llama32Generator,
    sample: SampleOpts,
    steps: u64,
) -> Vec<u32> {
    use rlx_orpheus::tokens::{is_custom_token_id, mask_logits_for_snac_slot};
    use rlx_qwen3::apply_repetition_penalty;

    let mut stream_index = 0usize;
    let mut tokens = Vec::new();
    let mut token_counts = std::collections::HashMap::<u32, u32>::new();
    let apply_penalty = sample.repetition_penalty > 1.0;
    for step in 0..steps {
        let slot_ix = stream_index;
        let tok = generator
            .step_cached_adjust(sample, step, |logits| {
                mask_logits_for_snac_slot(logits, slot_ix);
                if apply_penalty {
                    apply_repetition_penalty(logits, &token_counts, sample.repetition_penalty);
                }
            })
            .expect("decode step");
        tokens.push(tok);
        if is_custom_token_id(tok) {
            stream_index += 1;
        }
        *token_counts.entry(tok).or_insert(0) += 1;
    }
    tokens
}

#[test]
#[cfg(all(feature = "llama", feature = "metal"))]
fn metal_kv_masked_decode_matches_oneshot() {
    let Some(gguf) = support::orpheus_gguf_path() else {
        eprintln!("skip: run `just fetch-orpheus`");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    use rlx_core::weight_loader::load_from_path;
    use rlx_gguf::GgufFile;
    use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
    use rlx_orpheus::DEFAULT_COMPILE_SEQ_CAP;

    let sample = orpheus_sample_opts();
    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    let path = gguf.to_str().expect("utf8");
    let cap = DEFAULT_COMPILE_SEQ_CAP as usize;
    let steps = 14u64;

    let mut loader = load_from_path(path).expect("open gguf");
    let raw = GgufFile::from_path(&gguf).expect("parse gguf");
    let cfg = llama32_cfg_from_gguf(&raw).expect("cfg");

    let mut bucketed = Llama32Generator::from_loader_at_mode(
        cfg.clone(),
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::CpuF32,
    )
    .expect("generator")
    .with_compile_seq_cap(cap)
    .with_decode_cache(cap + 8);
    bucketed.prefill(&prompt);
    let bucketed_tokens = generate_masked_tokens(&mut bucketed, sample, steps);

    let mut loader = load_from_path(path).expect("open gguf");
    let mut oneshot = Llama32Generator::from_loader_at_mode(
        cfg,
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::CpuF32,
    )
    .expect("generator")
    .with_compile_seq_cap(cap);
    oneshot.prefill(&prompt);
    let oneshot_tokens = generate_masked_tokens(&mut oneshot, sample, steps);

    eprintln!(
        "bucketed: {:?}",
        &bucketed_tokens[..bucketed_tokens.len().min(8)]
    );
    eprintln!(
        "oneshot:  {:?}",
        &oneshot_tokens[..oneshot_tokens.len().min(8)]
    );
    assert_eq!(
        bucketed_tokens, oneshot_tokens,
        "bucketed KV masked decode diverged from oneshot on Orpheus GGUF"
    );
}

#[test]
#[cfg(all(feature = "llama", feature = "metal"))]
fn metal_synthesis_codes_decode_to_audible_pcm() {
    let Some(gguf) = support::orpheus_gguf_path() else {
        eprintln!("skip: run `just fetch-orpheus`");
        return;
    };
    let Some(snac_path) = support::snac_decoder_path() else {
        eprintln!("skip: set ORPHEUS_SNAC_PATH");
        return;
    };
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    use rlx_core::weight_loader::load_from_path;
    use rlx_gguf::GgufFile;
    use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
    use rlx_orpheus::DEFAULT_COMPILE_SEQ_CAP;
    use rlx_orpheus::{
        SnacBackend, SnacLoadOptions, decode_orpheus_codes, tokens::pack_orpheus_codes,
    };

    let sample = orpheus_sample_opts();
    let prompt = build_prompt(&gguf, "Hi.", Some("tara")).expect("prompt");
    let path = gguf.to_str().expect("utf8");
    let mut loader = load_from_path(path).expect("open gguf");
    let raw = GgufFile::from_path(&gguf).expect("parse gguf");
    let cfg = llama32_cfg_from_gguf(&raw).expect("cfg");
    let cap = DEFAULT_COMPILE_SEQ_CAP as usize;
    let mut generator = Llama32Generator::from_loader_at_mode(
        cfg,
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::CpuF32,
    )
    .expect("generator")
    .with_compile_seq_cap(cap)
    .with_decode_cache(cap + 8);
    generator.prefill(&prompt);
    let codes = generate_masked_codes(&mut generator, sample, 56);
    assert!(
        pack_orpheus_codes(&codes).is_some(),
        "generated codes must pack for SNAC"
    );

    let snac = SnacBackend::open(&snac_path, SnacLoadOptions::default()).expect("snac");
    let pcm = decode_orpheus_codes(&snac, &codes).expect("decode");
    let peak = pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    eprintln!("codes={} pcm_len={} peak={peak:.4}", codes.len(), pcm.len());
    assert!(
        peak > 0.05,
        "expected audible PCM (peak > 0.05), got peak={peak:.6}"
    );
}
