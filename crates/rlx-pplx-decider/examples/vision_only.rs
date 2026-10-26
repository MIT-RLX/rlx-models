//! Run only the vision tower on one image — for measuring its time and
//! memory apart from the backbone.
//!
//! `cargo run -p rlx-pplx-decider --release --example vision_only -- MMPROJ IMAGE [DEVICE]`

use rlx_pplx_decider::vision::{RgbImage, preprocess};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mmproj, image, rest @ ..] = &args[..] else {
        anyhow::bail!("usage: vision_only MMPROJ IMAGE [DEVICE]");
    };
    let device = rlx_cli::parse_device(rest.first().map_or("cpu", String::as_str))?;
    let img = preprocess(&RgbImage::open(image, None)?, 65_536, 1_638_400)?;
    let mut enc = rlx_qwen35::Qwen35VisionEncoder::load(mmproj, device)?;
    for i in 0..2 {
        let t = std::time::Instant::now();
        let out = enc.encode_nchw(&img.nchw, img.resized.width, img.resized.height)?;
        eprintln!(
            "run {i}: {} tokens in {:.2?} on {:?}",
            out.n_tokens,
            t.elapsed(),
            enc.device()
        );
    }
    Ok(())
}
