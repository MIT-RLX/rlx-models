// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! End-to-end through the REAL `MistralVlRunner` wrapper (the shipping path used
//! by `auto_runner_with_mmproj` / skill-llm `try_mistral_vl_runner`): build with
//! weights + mmproj, then `generate_multimodal_rgb`. Exercises the packed
//! embed-splice inside the actual runner, incl. the `.accept_llama_arch(true)`
//! path (the bartowski GGUF is `llama`-tagged; the mmproj confirms Mistral-3 VL).
//!
//! NB memory: the wrapper keeps BOTH the vision encoder (~1.7 GB) and the packed
//! LM (~14 GB) resident, so peak is higher than the drop-then-load combined
//! test. One process, single-threaded, and ideally on an otherwise-idle box.
//!
//! ```text
//! MISTRAL_MMPROJ=/path/to/mmproj-f16.gguf MISTRAL_LM=/path/to/Q4_K_M.gguf \
//!   cargo test -p rlx-mistral-vl --features metal --test real_vl_wrapper -- --ignored --nocapture --test-threads=1
//! ```

use rlx_mistral_vl::MistralVlRunner;
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
#[ignore = "needs MISTRAL_MMPROJ + MISTRAL_LM; loads both models (~16 GB peak)"]
fn mistral_vl_runner_packed_generate() {
    let (Ok(mmproj), Ok(lm)) = (std::env::var("MISTRAL_MMPROJ"), std::env::var("MISTRAL_LM"))
    else {
        eprintln!("MISTRAL_MMPROJ / MISTRAL_LM not set — skipping");
        return;
    };
    let device = pick_device();

    // Real shipping constructor — `.accept_llama_arch` is set internally because
    // an mmproj is present, so the `llama`-tagged GGUF is accepted.
    let mut runner = MistralVlRunner::builder()
        .weights(&lm)
        .mmproj(&mmproj)
        .device(device)
        .max_seq(512)
        .build()
        .expect("build MistralVlRunner (llama-arch LM + Pixtral mmproj)");
    assert!(runner.has_vision(), "vision encoder not attached");
    eprintln!("MistralVlRunner built on {device:?} (packed LM + Pixtral mmproj)");

    // Thin wide strip so the long edge → image_size, short edge stays tiny
    // (small ViT grid). w matches the model's image_size (1540).
    let (w, h) = (1540usize, 56usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb[i] = (x % 256) as u8;
            rgb[i + 1] = (y % 256) as u8;
            rgb[i + 2] = ((x + y) % 256) as u8;
        }
    }

    let mut got = Vec::new();
    let ids = runner
        .generate_multimodal_rgb("describe <image>", &rgb, w, h, None, 12, |t| {
            got.push(t);
            true
        })
        .expect("generate_multimodal_rgb");

    eprintln!("MistralVlRunner decoded {} ids: {ids:?}", ids.len());
    assert_eq!(ids.len(), 12, "wrong token count");
}
