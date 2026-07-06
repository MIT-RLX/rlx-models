//! Compare one-shot prefill vs incremental decode step-1 logits.
//!
//!   RLX_GEMMA_GRAPH_LM_HEAD=1 cargo run --release -p rlx-gemma --example decode_one_step -- \
//!     /tmp/rlx-weights/gemma-3-270m.gguf

use anyhow::{Context, Result};
use rlx_gemma::GemmaRunner;
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::PathBuf;

const HF_CHAT_IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 1156, 2915, 1156, 236881, 25685, 528, 886, 2822, 13315, 236761,
    106, 107, 105, 4368, 107,
];

fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(5);
    v
}

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .context("usage: decode_one_step <gguf>")?
        .into();

    let mut runner = GemmaRunner::builder()
        .weights(&path)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()?;

    let prefill = runner.predict_logits(HF_CHAT_IDS)?;
    let step0 = prefill
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    println!("prefill top1 = {step0}");

    let mut extended = HF_CHAT_IDS.to_vec();
    extended.push(step0);
    let one_shot = runner.predict_logits(&extended)?;
    println!("one-shot step1 top5: {:?}", top5(&one_shot));

    let mut runner2 = GemmaRunner::builder()
        .weights(&path)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(512)
        .sample(SampleOpts::greedy())
        .build()?;
    let toks = runner2.generate(HF_CHAT_IDS, 2, |_| {})?;
    println!("incremental tokens = {toks:?}");

    Ok(())
}
