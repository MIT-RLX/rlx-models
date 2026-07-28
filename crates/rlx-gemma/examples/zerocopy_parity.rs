//! Minimal greedy-parity check for the packed Gemma decode path — used to
//! validate the zero-copy packed-weight migration (CPU vs Metal must produce
//! identical token IDs).
//!
//! Usage:
//!   cargo run -p rlx-gemma --features <backends> --example zerocopy_parity -- <gguf> <cpu|metal> [n_new]
use anyhow::Result;
use rlx_gemma::GemmaRunner;
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;

// A fixed Gemma-3 chat-templated prompt (token IDs), borrowed from gemma_bench.
const IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 1156, 2915, 1156, 236881, 25685, 528, 886, 2822, 13315, 236761,
    106, 107, 105, 4368, 107,
];

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: zerocopy_parity <gguf> <cpu|metal> [n_new]");
    let dev = match std::env::args().nth(2).as_deref() {
        Some("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let n_new: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let mut r = GemmaRunner::builder()
        .weights(&path)
        .packed_weights(true)
        .device(dev)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()?;
    let toks = r.generate(IDS, n_new, |_| {})?;
    println!("device={dev:?} n_new={n_new} toks={toks:?}");
    Ok(())
}
