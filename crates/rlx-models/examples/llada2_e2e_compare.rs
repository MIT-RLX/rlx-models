// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! End-to-end forward logits parity: RLX vs PyTorch (`llada2_full_parity_reference.py`).

use rlx_cli::parse_llada2_device;
use rlx_models::llada2::{
    LLaDA2Runner, load_llada2_from_dir, mask::block_diffusion_attention_mask,
};
use rlx_runtime::Device;
use serde::Deserialize;
use std::env;
use std::process::Command;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 1.0;
    }
    dot / (na * nb)
}

#[derive(Debug, Deserialize)]
struct ReferenceOut {
    #[allow(dead_code)]
    test: String,
    #[allow(dead_code)]
    seq_len: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    logits: Vec<Vec<f32>>,
    #[allow(dead_code)]
    error: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let model_dir = env::var("LLADA2_MODEL_DIR")
        .map_err(|_| anyhow::anyhow!("set LLADA2_MODEL_DIR to checkpoint directory"))?;
    let seq_len: usize = env::var("LLADA2_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let block_length: usize = env::var("LLADA2_BLOCK_LENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let device = env::var("RLX_DEVICE")
        .ok()
        .map(|s| parse_llada2_device(&s))
        .transpose()?
        .unwrap_or(Device::Cpu);

    let (cfg, weights) = load_llada2_from_dir(model_dir.as_ref())?;
    let mut runner = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(device)
        .batch_seq(1, seq_len)
        .build()?;

    let prompt = [1u32, 2, 3];
    let mut ids = vec![cfg.mask_token_id as f32; seq_len];
    let mut pos = vec![0f32; seq_len];
    for i in 0..seq_len {
        pos[i] = i as f32;
        if i < prompt.len() {
            ids[i] = prompt[i] as f32;
        }
    }
    let mask = block_diffusion_attention_mask(1, seq_len, block_length);
    let mut full_mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for r in 0..seq_len {
        for c in 0..seq_len {
            full_mask[r * seq_len + c] = mask[r * seq_len + c];
        }
    }

    let rlx_logits = runner.forward_logits(&ids, &pos, &full_mask)?;
    let vocab = cfg.vocab_size;

    let out = Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/llada2_full_parity_reference.py"
        ))
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--prompt-ids")
        .arg("1,2,3")
        .arg("--seq-len")
        .arg(seq_len.to_string())
        .arg("--block-length")
        .arg(block_length.to_string())
        .output()?;
    if !out.status.success() {
        anyhow::bail!("reference failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let reference: ReferenceOut = serde_json::from_slice(&out.stdout)?;

    let mut min_cos = 1.0f32;
    let mut max_abs = 0.0f32;
    for pos in 0..seq_len {
        let base = pos * vocab;
        let rlx_row = &rlx_logits[base..base + vocab];
        let ref_row = &reference.logits[pos];
        let cos = cosine_similarity(rlx_row, ref_row);
        min_cos = min_cos.min(cos);
        for (a, b) in rlx_row.iter().zip(ref_row.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        eprintln!("pos={pos} cosine={cos:.6} max_abs={max_abs:.6}");
    }
    eprintln!("e2e summary device={device:?} min_cosine={min_cos:.6} max_abs={max_abs:.6}");
    if min_cos < 0.99 {
        anyhow::bail!("min cosine {min_cos} below 0.99");
    }
    Ok(())
}
