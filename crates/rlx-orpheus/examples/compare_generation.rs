//! Compare KV-cached vs full-recompute token generation for Orpheus.
use rlx_core::weight_loader::load_from_path;
use rlx_gguf::GgufFile;
use rlx_llama32::{
    Llama32Generator, Llama32RunnerBuilder, MetalGgufPrefillMode, llama32_cfg_from_gguf,
};
use rlx_orpheus::tokens::{
    build_prompt, is_snac_slot_token, mask_logits_for_snac_slot, use_snac_logit_mask,
};
use rlx_orpheus::{DEFAULT_COMPILE_SEQ_CAP, SnacBackend, SnacLoadOptions, decode_orpheus_codes};
use rlx_qwen3::{SampleOpts, apply_repetition_penalty, sample_token_at};
use rlx_runtime::Device;

fn sample_opts() -> SampleOpts {
    SampleOpts::temperature(0.6, 42)
        .with_top_p(0.8)
        .with_repetition_penalty(1.3)
}

fn tokens_to_codes(tokens: &[u32]) -> Vec<i32> {
    use rlx_orpheus::tokens::custom_token_id_to_code;
    let mut stream_index = 0usize;
    let mut codes = Vec::new();
    for &tok in tokens {
        if let Some(code) = custom_token_id_to_code(tok, stream_index) {
            if is_snac_slot_token(tok, stream_index) && code > 0 {
                codes.push(code);
            }
        }
        if is_snac_slot_token(tok, stream_index) {
            stream_index += 1;
        }
    }
    codes
}

fn peak_pcm(snac: &SnacBackend, codes: &[i32]) -> f32 {
    decode_orpheus_codes(snac, codes)
        .map(|pcm| pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max))
        .unwrap_or(0.0)
}

fn main() -> anyhow::Result<()> {
    let gguf = std::env::var("ORPHEUS_GGUF_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf")
        });
    let snac_path = std::env::var("ORPHEUS_SNAC_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors")
        });
    let text = std::env::var("ORPHEUS_TEXT").unwrap_or_else(|_| "Hello from RLX.".into());
    let voice = std::env::var("ORPHEUS_VOICE").unwrap_or_else(|_| "tara".into());
    let steps: u64 = std::env::var("ORPHEUS_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(56);
    let prompt = build_prompt(&gguf, &text, Some(&voice))?;
    let path = gguf.to_str().unwrap();
    let raw = GgufFile::from_path(&gguf)?;
    let cfg = llama32_cfg_from_gguf(&raw)?;
    let cap = DEFAULT_COMPILE_SEQ_CAP as usize;
    let sample = sample_opts();
    let snac = SnacBackend::open(&snac_path, SnacLoadOptions::default())?;

    eprintln!("prompt len={} steps={steps}", prompt.len());

    // KV-cached path (current default).
    let mut loader = load_from_path(path)?;
    let mut kv = Llama32Generator::from_loader_at_mode(
        cfg.clone(),
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::CpuF32,
    )?
    .with_compile_seq_cap(cap)
    .with_decode_cache(cap + 8);
    kv.prefill(&prompt);
    let start = kv.tokens().len();
    let mut token_counts = std::collections::HashMap::<u32, u32>::new();
    let mut stream_index = 0usize;
    for step in 0..steps {
        let slot_ix = stream_index;
        let tok = kv.step_cached_adjust(sample, step, |logits| {
            if use_snac_logit_mask() {
                mask_logits_for_snac_slot(logits, slot_ix);
            }
            apply_repetition_penalty(logits, &token_counts, sample.repetition_penalty);
        })?;
        if is_snac_slot_token(tok, stream_index) {
            stream_index += 1;
        }
        *token_counts.entry(tok).or_insert(0) += 1;
    }
    let kv_tokens = kv.tokens()[start..].to_vec();
    let kv_codes = tokens_to_codes(&kv_tokens);
    eprintln!(
        "kv_cached: tokens={} codes={} peak={:.4} head={:?}",
        kv_tokens.len(),
        kv_codes.len(),
        peak_pcm(&snac, &kv_codes),
        &kv_codes[..kv_codes.len().min(8)]
    );

    // Full prefill recompute via packed runner (O(n^2), reference).
    let upper = (prompt.len() + steps as usize).next_power_of_two().min(cap);
    let mut runner = Llama32RunnerBuilder::default()
        .weights(&gguf)
        .max_seq(upper.max(prompt.len() + 1))
        .device(Device::Metal)
        .packed_weights(true)
        .build()?;
    let mut history = prompt.clone();
    let mut token_counts = std::collections::HashMap::<u32, u32>::new();
    let mut stream_index = 0usize;
    let mut packed_tokens = Vec::new();
    for step in 0..steps {
        let slot_ix = stream_index;
        let mut logits = runner.predict_logits(&history)?;
        if use_snac_logit_mask() {
            mask_logits_for_snac_slot(&mut logits, slot_ix);
        }
        apply_repetition_penalty(&mut logits, &token_counts, sample.repetition_penalty);
        let tok = sample_token_at(&logits, sample, step) as u32;
        packed_tokens.push(tok);
        history.push(tok);
        if is_snac_slot_token(tok, stream_index) {
            stream_index += 1;
        }
        *token_counts.entry(tok).or_insert(0) += 1;
    }
    let packed_codes = tokens_to_codes(&packed_tokens);
    eprintln!(
        "packed_recompute: tokens={} codes={} peak={:.4} head={:?}",
        packed_tokens.len(),
        packed_codes.len(),
        peak_pcm(&snac, &packed_codes),
        &packed_codes[..packed_codes.len().min(8)]
    );
    eprintln!(
        "packed tokens: {:?}",
        &packed_tokens[..packed_tokens.len().min(12)]
    );
    eprintln!("tokens_match={}", kv_tokens == packed_tokens);

    // Oneshot decode reference (no bucket cache) — slow but should match logits.
    let mut loader = load_from_path(path)?;
    let mut oneshot = Llama32Generator::from_loader_at_mode(
        cfg,
        loader.as_mut(),
        Device::Metal,
        &gguf,
        MetalGgufPrefillMode::CpuF32,
    )?
    .with_compile_seq_cap(cap);
    oneshot.prefill(&prompt);
    let start = oneshot.tokens().len();
    let mut token_counts = std::collections::HashMap::<u32, u32>::new();
    let mut stream_index = 0usize;
    for step in 0..steps {
        let slot_ix = stream_index;
        let tok = oneshot.step_cached_adjust(sample, step, |logits| {
            if use_snac_logit_mask() {
                mask_logits_for_snac_slot(logits, slot_ix);
            }
            apply_repetition_penalty(logits, &token_counts, sample.repetition_penalty);
        })?;
        if is_snac_slot_token(tok, stream_index) {
            stream_index += 1;
        }
        *token_counts.entry(tok).or_insert(0) += 1;
    }
    let oneshot_tokens = oneshot.tokens()[start..].to_vec();
    let oneshot_codes = tokens_to_codes(&oneshot_tokens);
    eprintln!(
        "oneshot_kv: tokens={} codes={} peak={:.4} head={:?}",
        oneshot_tokens.len(),
        oneshot_codes.len(),
        peak_pcm(&snac, &oneshot_codes),
        &oneshot_codes[..oneshot_codes.len().min(8)]
    );
    eprintln!("kv_vs_oneshot={}", kv_tokens == oneshot_tokens);
    Ok(())
}
