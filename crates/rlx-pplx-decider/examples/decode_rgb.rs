//! Decode an image the way the decider does and write raw RGB8 — for
//! comparing decoders against PIL.
//!
//! `cargo run -p rlx-pplx-decider --release --example decode_rgb -- IMAGE OUT.rgb`

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [image, out] = &args[..] else {
        anyhow::bail!("usage: decode_rgb IMAGE OUT.rgb");
    };
    let img = rlx_pplx_decider::RgbImage::open(image, None)?;
    std::fs::write(out, &img.pixels)?;
    eprintln!("{}x{}", img.width, img.height);
    Ok(())
}
