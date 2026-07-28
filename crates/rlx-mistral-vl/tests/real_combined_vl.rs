// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Full combined VL path with REAL weights: Pixtral vision encode → splice →
//! packed LM generate. Uses `PixtralVisionEncoder` + `Llama32Runner` directly
//! (the bartowski LM GGUF is `llama`-arch, which `MistralRunner`'s gate rejects;
//! the splice mechanism is arch-independent and lives in `Llama32Runner`).
//!
//! Memory-safe on a constrained box: encode the image and DROP the vision
//! encoder *before* loading the 14 GB packed LM, so peak ≈ max(1.7, 14) GB, not
//! the sum. Run one process, single-threaded.
//!
//! ```text
//! MISTRAL_MMPROJ=/path/to/mmproj-f16.gguf MISTRAL_LM=/path/to/Q4_K_M.gguf \
//!   cargo test -p rlx-mistral-vl --features metal --test real_combined_vl -- --ignored --nocapture --test-threads=1
//! ```

use rlx_llama32::Llama32Runner;
use rlx_mistral_vl::PixtralVisionEncoder;
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

#[test]
#[ignore = "needs MISTRAL_MMPROJ + MISTRAL_LM (real Pixtral mmproj + 24B quant GGUF)"]
fn combined_vl_packed_generate() {
    let (Ok(mmproj), Ok(lm_path)) = (std::env::var("MISTRAL_MMPROJ"), std::env::var("MISTRAL_LM"))
    else {
        eprintln!("MISTRAL_MMPROJ / MISTRAL_LM not set — skipping");
        return;
    };

    // ── 1. Vision encode on CPU, then drop the encoder to free ~1.7 GB before
    //       the LM loads (keeps peak memory to one model at a time). ──
    let (vision_embd, n_vision, proj_dim) = {
        let mut enc =
            PixtralVisionEncoder::from_mmproj_on_device(&mmproj, Device::Cpu).expect("load mmproj");
        let cfg = enc.config().clone();
        // Thin wide strip → long edge to image_size, short edge tiny → small grid.
        let w = cfg.image_size;
        let h = cfg.align_size() * 2;
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = (x % 256) as u8;
                rgb[i + 1] = (y % 256) as u8;
                rgb[i + 2] = ((x + y) % 256) as u8;
            }
        }
        let out = enc.encode_rgb(&rgb, w, h).expect("encode_rgb");
        let proj = cfg.projector_output_dim;
        (out.clone(), out.len() / proj, proj)
    }; // ← encoder dropped here, weights freed
    eprintln!("vision: n_vision={n_vision} proj_dim={proj_dim}");
    assert!(n_vision > 0 && vision_embd.iter().all(|v| v.is_finite()));

    // ── 2. Load the packed LM (14 GB, Metal). Vision encoder is gone now. ──
    let device = pick_device();
    let mut lm = Llama32Runner::builder()
        .weights(&lm_path)
        .device(device)
        .max_seq(512)
        .build()
        .expect("build packed LM");
    let hidden = lm.config().hidden_size;
    let vocab = lm.config().vocab_size;
    eprintln!("lm: device={device:?} hidden={hidden} vocab={vocab}");
    assert_eq!(
        proj_dim, hidden,
        "projector dim {proj_dim} != LM hidden {hidden} — mmproj/LM mismatch"
    );

    // ── 3. Sequence: [text] [n_vision placeholders] [text]. Placeholder ids get
    //       overwritten by the vision splice, so any in-vocab id works. ──
    let before = [1u32, 3, 4, 5];
    let after = [6u32, 7, 8];
    let vision_start = before.len();
    let mut ids: Vec<u32> = Vec::with_capacity(before.len() + n_vision + after.len());
    ids.extend_from_slice(&before);
    ids.extend(std::iter::repeat_n(0u32, n_vision));
    ids.extend_from_slice(&after);
    eprintln!(
        "seq_len={} (vision {}..{})",
        ids.len(),
        vision_start,
        vision_start + n_vision
    );

    // ── 4. Splice vision soft tokens into the packed prefill + generate. ──
    lm.set_multimodal_embed_override(vision_start, vision_embd);
    let mut got = Vec::new();
    lm.generate(&ids, 12, |t| got.push(t))
        .expect("combined generate");

    assert!(
        !lm.multimodal_override_pending(),
        "vision splice NOT consumed — packed prefill path not taken"
    );
    eprintln!("combined VL decoded 12 ids: {got:?}");
    assert_eq!(got.len(), 12);
    assert!(
        got.iter().all(|&t| (t as usize) < vocab),
        "decoded id out of vocab"
    );
}
