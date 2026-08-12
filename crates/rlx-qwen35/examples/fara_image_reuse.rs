//! Validate the STATIC hidden-prefill reuse (the image-TTFT optimization):
//! load Fara1.5-4B GGUF + mmproj, run several DIFFERENT images, and check
//!   (a) each description is correct (shape + colour), and
//!   (b) images after the first REUSE the compiled hidden-prefill graph instead
//!       of paying the ~5.6s MPSGraph recompile every turn.
//!
//! This lives in rlx-qwen35 (not the skill) so it builds without the rest of the
//! model zoo. Run:
//!
//! ```bash
//! FARA_IMG_DIR=/path/to/pngs \
//! cargo run -p rlx-qwen35 --example fara_image_reuse --release \
//!   --features "apple-silicon,qwen35-vlm,tokenizer" -- --device metal
//! ```

use anyhow::{Context, Result};
use rlx_qwen35::{MEDIA_MARKER, Qwen35Runner, decode_ids_from_gguf};
use rlx_runtime::{Device, parse_device};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn load_rgb(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w as usize, h as usize))
}

fn main() -> Result<()> {
    let mut device = Device::Metal;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "--device" {
            device = parse_device(&it.next().context("--device needs a value")?)?;
        }
    }

    let gguf = std::env::var("FARA_GGUF")
        .unwrap_or_else(|_| "/Users/Shared/weights/fara1.5-4b-gguf/Fara1.5-4B-Q4_K_M.gguf".into());
    let mmproj = std::env::var("FARA_MMPROJ").unwrap_or_else(|_| {
        "/Users/Shared/weights/fara1.5-4b-gguf/mmproj-Fara1.5-4B-f16.gguf".into()
    });
    let dir = std::env::var("FARA_IMG_DIR").context("set FARA_IMG_DIR to the png directory")?;
    let gguf_path = PathBuf::from(&gguf);

    eprintln!("[probe] loading Fara GGUF + mmproj on {device:?} …");
    let mut runner = Qwen35Runner::builder()
        .weights(&gguf_path)
        .device(device)
        .packed_weights(true)
        .max_seq(1408)
        .skip_warm(true)
        .mmproj(&mmproj)
        .build()?;

    let q = "What is the main shape and its colour? Answer in one short sentence.";
    let run = |runner: &mut Qwen35Runner, file: &str| -> Result<(String, f64)> {
        let (rgb, w, h) = load_rgb(&Path::new(&dir).join(file))?;
        let prompt =
            format!("<|im_start|>user\n{MEDIA_MARKER}{q}<|im_end|>\n<|im_start|>assistant\n");
        let mut ids = Vec::new();
        let t = Instant::now();
        // Mimic the skill's stop guard: end the turn at the first `</think>` on
        // real content (the reasoning model otherwise loops answer + </think>).
        runner.generate_multimodal(&prompt, &rgb, w, h, None, 40, |tok| {
            ids.push(tok);
            let so_far = decode_ids_from_gguf(&gguf_path, &ids, true).unwrap_or_default();
            !so_far.contains("</think>")
        })?;
        let secs = t.elapsed().as_secs_f64();
        let text = decode_ids_from_gguf(&gguf_path, &ids, true)?;
        // Trim at the stop marker for a clean single answer, like the skill does.
        let text = text
            .split("</think>")
            .next()
            .unwrap_or(&text)
            .trim()
            .to_string();
        Ok((text, secs))
    };

    // First image pays the one-time hidden-prefill compile; each subsequent image
    // is a DIFFERENT one (different seq/content) that must reuse the kept static
    // graph — validating both correctness and the no-recompile fast path.
    let cases = [
        ("blue_circle_128.png", "blue", "circle"),
        ("red_square.png", "red", "square"),
        ("green_circle.png", "green", "circle"),
        ("yellow_tri.png", "yellow", ""),
    ];
    let mut times = Vec::new();
    for (i, (file, colour, shape)) in cases.iter().enumerate() {
        let (text, secs) = run(&mut runner, file)?;
        println!("[img{} {file}] {secs:.2}s :: {text:?}", i + 1);
        let lc = text.to_lowercase();
        assert!(
            !lc.contains('\u{0120}') && !lc.contains('\u{010a}'),
            "byte-level markers leaked: {text:?}"
        );
        assert!(
            lc.contains(colour),
            "img{} missing colour `{colour}`: {text:?}",
            i + 1
        );
        if !shape.is_empty() {
            assert!(
                lc.contains(shape),
                "img{} missing shape `{shape}`: {text:?}",
                i + 1
            );
        }
        times.push(secs);
    }

    let first = times[0];
    let rest_max = times[1..].iter().cloned().fold(0.0f64, f64::max);
    println!(
        "\nreuse: img1={:.2}s, img2..={:?} (max {:.2}s). Recompile-saved ≈ {:.2}s",
        first,
        &times[1..],
        rest_max,
        first - rest_max
    );
    // The recompile is ~5.6s; reused turns should be materially faster than the
    // first. Require at least a 2s edge so a flaky measurement doesn't pass by luck.
    assert!(
        rest_max + 2.0 < first,
        "expected reused images to skip the hidden-prefill recompile, but they were not faster: img1={first:.2}s rest_max={rest_max:.2}s"
    );
    println!("OK — static hidden prefill reused across images, output correct.");
    Ok(())
}
