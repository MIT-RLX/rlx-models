// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Real-weights validation of the **packed** LM path + the multimodal
//! embed-splice redesign, on-device (Metal/MLX). Uses `Llama32Runner` directly
//! (the bartowski Mistral-Small-3.1 GGUF is tagged `general.architecture=llama`,
//! which `MistralRunner`'s arch gate rejects; the packed embed-override lives in
//! `Llama32Runner`, so this exercises the real mechanism on the real 24B model).
//!
//! Proves: (1) a 24B Q4_K_M loads + runs **packed** (K-quant, ~14 GB — no 96 GB
//! F32 dequant), (2) `set_multimodal_embed_override` splices host embeddings
//! into the packed `input_embeddings` prefill and is consumed on-device.
//!
//! ```text
//! MISTRAL_LM=/path/to/Q4_K_M.gguf \
//!   cargo test -p rlx-mistral-vl --features metal --test real_mistral_lm -- --ignored --nocapture
//! ```

use rlx_llama32::Llama32Runner;
use rlx_runtime::Device;

fn pick_device() -> Device {
    use rlx_runtime::device_ext::is_available;
    for d in [Device::Metal, Device::Mlx] {
        if is_available(d) {
            return d;
        }
    }
    Device::Cpu
}

fn load(path: &str, device: Device) -> Llama32Runner {
    Llama32Runner::builder()
        .weights(path)
        .device(device)
        .max_seq(256)
        .build()
        .expect("build packed Llama32Runner")
}

#[test]
#[ignore = "needs a real 24B quant GGUF via MISTRAL_LM"]
fn packed_lm_text_smoke() {
    let Ok(path) = std::env::var("MISTRAL_LM") else {
        eprintln!("MISTRAL_LM not set — skipping");
        return;
    };
    let device = pick_device();
    eprintln!("device = {device:?}");
    let mut lm = load(&path, device);
    let vocab = lm.config().vocab_size;
    eprintln!(
        "loaded packed: hidden={} layers={} vocab={}",
        lm.config().hidden_size,
        lm.config().num_hidden_layers,
        vocab
    );

    // Single prefill + short decode. NB: a 24B model needs ~14 GB packed plus a
    // Metal compile peak — keep to ONE prefill so a memory-constrained box (this
    // one idles at only ~10 GB free) doesn't get SIGKILLed. Run one test per
    // process (`--test <name>`); two 24B models resident at once will OOM.
    let prompt = [1u32, 733, 16289, 28747, 22478, 349, 264, 2475];
    let mut got = Vec::new();
    lm.generate(&prompt, 8, |t| got.push(t)).expect("generate");
    eprintln!("decoded 8 ids: {got:?}");
    assert_eq!(got.len(), 8);
    assert!(got.iter().all(|&t| (t as usize) < vocab));
}

/// The redesign: splice host embeddings into the packed `input_embeddings`
/// prefill (the vision path, exercised here with a synthetic soft-token block).
#[test]
#[ignore = "needs a real 24B quant GGUF via MISTRAL_LM"]
fn packed_lm_embed_override() {
    let Ok(path) = std::env::var("MISTRAL_LM") else {
        eprintln!("MISTRAL_LM not set — skipping");
        return;
    };
    let device = pick_device();
    let mut lm = load(&path, device);
    let hidden = lm.config().hidden_size;

    // Prompt: [text] [3 placeholder rows for "vision"] [text].
    let before = [1u32, 733, 16289];
    let n_vision = 3usize;
    let after = [28747, 22478, 349];
    let vision_start = before.len();
    let mut ids: Vec<u32> = Vec::new();
    ids.extend_from_slice(&before);
    ids.extend(std::iter::repeat_n(0u32, n_vision));
    ids.extend_from_slice(&after);

    // Synthetic "vision" soft tokens — small distinctive values in embed space.
    let vision: Vec<f32> = (0..n_vision * hidden)
        .map(|i| 0.01 * ((i % 7) as f32 - 3.0))
        .collect();

    lm.set_multimodal_embed_override(vision_start, vision);
    assert!(
        lm.multimodal_override_pending(),
        "override should be pending pre-run"
    );

    let mut got = Vec::new();
    lm.generate(&ids, 6, |t| got.push(t))
        .expect("generate with splice");

    // Consumed on-device → packed prefill actually ran with the splice.
    assert!(
        !lm.multimodal_override_pending(),
        "embed override was NOT consumed — packed prefill path not taken (vision would be dropped)"
    );
    eprintln!("spliced-prefill decoded 6 ids: {got:?}");
    assert_eq!(got.len(), 6);
    assert!(got.iter().all(|&t| (t as usize) < lm.config().vocab_size));
}
