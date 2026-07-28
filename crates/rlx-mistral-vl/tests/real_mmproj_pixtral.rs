// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Real-weights smoke test for the Pixtral mmproj vision encoder.
//!
//! Skipped unless `MISTRAL_MMPROJ` points at a real Pixtral mmproj GGUF, so a
//! plain `cargo test` stays weight-free. Run with:
//!
//! ```text
//! MISTRAL_MMPROJ=/path/to/mmproj-f16.gguf \
//!   cargo test -p rlx-mistral-vl --test real_mmproj_pixtral -- --ignored --nocapture
//! ```
//!
//! OOM note: the ViT uses full O(n_pos^2) attention and `encode_rgb` always
//! upscales the long edge to `image_size` (~1540 → ~110 patches). A square
//! image would blow up to ~12k patches. We feed a thin wide strip so only the
//! short edge stays tiny — a few hundred patches, a few MB of attention.

use rlx_mistral_vl::PixtralVisionEncoder;
use rlx_runtime::Device;

#[test]
#[ignore = "needs a real Pixtral mmproj GGUF via MISTRAL_MMPROJ"]
fn pixtral_encoder_real_weights_smoke() {
    let Ok(path) = std::env::var("MISTRAL_MMPROJ") else {
        eprintln!("MISTRAL_MMPROJ not set — skipping real-weights smoke test");
        return;
    };

    // 1) Real config parse from the GGUF `clip.*` keys.
    let mut enc =
        PixtralVisionEncoder::from_mmproj_on_device(&path, Device::Cpu).expect("load mmproj");
    let cfg = enc.config().clone();
    eprintln!(
        "pixtral cfg: hidden={} layers={} heads={} head_dim={} inter={} \
         image_size={} patch={} merge={} proj={} silu={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.head_dim(),
        cfg.intermediate_size,
        cfg.image_size,
        cfg.patch_size,
        cfg.spatial_merge_size,
        cfg.projector_output_dim,
        cfg.use_silu,
    );

    // 2) Thin wide strip: long edge → image_size, short edge → 2 merge-rows so we
    //    exercise exactly one img_break. Keeps grid ~ (image_size/patch) x 4.
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

    // 3) Full path: preprocess → patch-embed → compile ViT+merger+projector → run.
    let out = enc.encode_rgb(&rgb, w, h).expect("encode_rgb");

    let proj = cfg.projector_output_dim;
    assert_eq!(out.len() % proj, 0, "output not a whole number of tokens");
    let n_tokens = out.len() / proj;

    let finite = out.iter().all(|v| v.is_finite());
    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for &v in &out {
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
    }
    eprintln!(
        "encoded: n_tokens={n_tokens} dim={proj} finite={finite} \
         min={mn:.4} max={mx:.4} mean={:.5}",
        sum / out.len() as f64
    );

    assert!(
        finite,
        "non-finite soft tokens — numeric blowup in the vision tower"
    );
    assert!(n_tokens > 0, "no tokens produced");
    // A dead / disconnected graph produces a flat (all-equal) output.
    assert!(
        mx - mn > 1e-3,
        "projector output is ~constant ({mn}..{mx}) — likely a dead path"
    );
}
