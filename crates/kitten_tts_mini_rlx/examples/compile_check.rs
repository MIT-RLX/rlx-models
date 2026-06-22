//! Fast native compile check (no inference). Usage:
//!   KITTEN_RLX_WEIGHTS=weights cargo run --example compile_check --release --features native

fn main() -> anyhow::Result<()> {
    let weights = std::env::var("KITTEN_RLX_WEIGHTS").unwrap_or_else(|_| "weights".into());
    let opts = kitten_tts_mini_rlx::GraphOptions {
        sequence_length: 128,
        max_waveform_samples: 24_000,
    };
    kitten_tts_mini_rlx::compile(
        rlx_runtime::Device::Cpu,
        std::path::Path::new(&weights),
        &opts,
    )?;
    eprintln!("compile ok");
    Ok(())
}
