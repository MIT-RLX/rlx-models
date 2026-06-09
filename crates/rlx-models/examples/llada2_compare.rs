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

//! Compare RLX LLaDA2 forward logits vs PyTorch reference (`llada2_parity_reference.py`).

use rlx_models::llada2::mask::block_diffusion_attention_mask;
use rlx_models::llada2::{LLaDA2Runner, synth};
use rlx_runtime::Device;
use std::env;
use std::process::Command;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let top_k: usize = args
        .iter()
        .position(|a| a == "--top-k")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let seq_len: usize = args
        .iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let mut runner = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch_seq(1, seq_len)
        .build()?;

    let prompt: &[u32] = &[1u32, 2, 3];
    let mut ids = vec![0f32; seq_len];
    let mut pos = vec![0f32; seq_len];
    for i in 0..seq_len {
        pos[i] = i as f32;
        ids[i] = if i < prompt.len() {
            prompt[i] as f32
        } else {
            cfg.mask_token_id as f32
        };
    }
    let mask = block_diffusion_attention_mask(1, seq_len, seq_len.min(4));
    let mut full_mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for r in 0..seq_len {
        for c in 0..seq_len {
            full_mask[r * seq_len + c] = mask[r * seq_len + c];
        }
    }

    let logits = runner.forward_logits(&ids, &pos, &full_mask)?;
    let vocab = cfg.vocab_size;
    for p in 0..seq_len {
        let base = p * vocab;
        let row = &logits[base..base + vocab];
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, &ti) in order.iter().take(top_k).enumerate() {
            println!(
                "RLX_LOGIT pos={p} rank={rank} token={ti} value={:.6}",
                row[ti]
            );
        }
    }

    if env::var("LLADA2_SKIP_reference").is_ok() {
        return Ok(());
    }

    let denoiser_ref = Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/llada2_parity_reference.py"
        ))
        .arg("--prompt-ids")
        .arg("1,2,3")
        .arg("--seq-len")
        .arg(seq_len.to_string())
        .arg("--top-k")
        .arg(top_k.to_string())
        .output();

    match denoiser_ref {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            print!("{s}");
        }
        Ok(out) => {
            eprintln!("reference failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        Err(e) => eprintln!("reference spawn skipped: {e}"),
    }

    Ok(())
}
