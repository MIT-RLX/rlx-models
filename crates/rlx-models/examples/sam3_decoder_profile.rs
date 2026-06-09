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

// Per-stage timing of the IR decoder — measure where the 200ms+ on CPU
// actually goes so we know what's worth optimizing.

use anyhow::Result;
use rlx_models::sam3::detector_decoder_ir::Sam3CompiledDecoder;
use rlx_models::sam3::{Sam3, Sam3Config};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<()> {
    let weights = rlx_ir::env::var("RLX_SAM3_WEIGHTS")
        .ok_or_else(|| anyhow::anyhow!("set RLX_SAM3_WEIGHTS"))?;
    let ref_dir: PathBuf = rlx_ir::env::var("RLX_SAM3_REF_DIR")
        .unwrap_or_else(|| "/var/folders/9_/pjm86g5j44l4cdv5mld3wd9c0000gn/T/tmp.0NBLOovOZD".into())
        .into();

    let model = Sam3::from_safetensors(&weights, Sam3Config::base())?;
    let batch = 1;
    let c = 256;
    let h = 72;
    let w = 72;
    let seq = 32;

    let mem_seq_first = read_f32(&ref_dir.join("encoder_memory.f32"));
    let pos_nchw = read_f32(&ref_dir.join("encoder_pos.f32"));
    let prompt = read_f32(&ref_dir.join("encoder_prompt.f32"));
    let prompt_mask = std::fs::read(ref_dir.join("encoder_prompt_mask.u8"))?;

    let mut memory_bf = vec![0f32; batch * h * w * c];
    for l in 0..h * w {
        for b in 0..batch {
            let s = (l * batch + b) * c;
            let d = (b * h * w + l) * c;
            memory_bf[d..d + c].copy_from_slice(&mem_seq_first[s..s + c]);
        }
    }
    let mut memory_pos = vec![0f32; batch * h * w * c];
    for b in 0..batch {
        for y in 0..h {
            for xc in 0..w {
                for ch in 0..c {
                    memory_pos[(b * h * w + y * w + xc) * c + ch] =
                        pos_nchw[((b * c + ch) * h + y) * w + xc];
                }
            }
        }
    }

    for (dev_name, dev) in [
        ("CPU", Device::Cpu),
        #[cfg(feature = "metal")]
        ("Metal", Device::Metal),
        #[cfg(feature = "mlx")]
        ("MLX", Device::Mlx),
    ] {
        eprintln!("--- {dev_name} ---");
        let t_compile = Instant::now();
        let mut dec = Sam3CompiledDecoder::new_with_profile(
            model.decoder_weights(),
            batch,
            h * w,
            seq,
            dev,
            model.compile_profile(),
        )?;
        let compile_ms = t_compile.elapsed().as_secs_f32() * 1000.0;

        // Warmup
        let _ = dec.run(&memory_bf, &memory_pos, &prompt, &prompt_mask, h, w)?;

        // 3 measured runs
        let mut totals = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let _ = dec.run(&memory_bf, &memory_pos, &prompt, &prompt_mask, h, w)?;
            totals.push(t.elapsed().as_secs_f32() * 1000.0);
        }
        let avg = totals.iter().sum::<f32>() / totals.len() as f32;
        let mn = totals.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = totals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "{dev_name:>6}: compile={compile_ms:.1}ms  decoder_run avg={avg:.1}ms  min={mn:.1}ms  max={mx:.1}ms"
        );
    }
    Ok(())
}
