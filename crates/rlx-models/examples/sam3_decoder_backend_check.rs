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

// Verifies the IR decoder produces parity-class output on every Apple
// backend the build is compiled with: CPU always, Metal under `--features
// metal`, MLX under `--features mlx`. Uses the same reference dump as
// the parity test.

use anyhow::Result;
use rlx_models::sam3::detector_decoder_ir::Sam3CompiledDecoder;
use rlx_models::sam3::{Sam3, Sam3Config};
use rlx_runtime::Device;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn diff(a: &[f32], b: &[f32]) -> (f32, f64) {
    let n = a.len().min(b.len());
    let mut mad = 0.0f32;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > mad {
            mad = d;
        }
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let cos = if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (1.0 - dot / (na * nb).sqrt()).max(0.0)
    };
    (mad, cos)
}

fn run_on(model: &Sam3, dev: Device, dev_name: &str, ref_dir: &Path) -> Result<()> {
    let batch = 1;
    let c = 256;
    let h = 72;
    let w = 72;
    let seq = 32;
    let nq = 200;
    let num_layers = 6;
    let mem_seq_first = read_f32(&ref_dir.join("encoder_memory.f32"));
    let pos_nchw = read_f32(&ref_dir.join("encoder_pos.f32"));
    let prompt = read_f32(&ref_dir.join("encoder_prompt.f32"));
    let prompt_mask = std::fs::read(ref_dir.join("encoder_prompt_mask.u8"))?;
    let int_ref = read_f32(&ref_dir.join("decoder_intermediate.f32"));
    let ref_boxes_ref = read_f32(&ref_dir.join("decoder_ref_boxes.f32"));

    // memory seq-first -> batch-first
    let mut memory_bf = vec![0f32; batch * h * w * c];
    for l in 0..h * w {
        for b in 0..batch {
            let s = (l * batch + b) * c;
            let d = (b * h * w + l) * c;
            memory_bf[d..d + c].copy_from_slice(&mem_seq_first[s..s + c]);
        }
    }
    // pos NCHW -> [B, hw, C]
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

    eprintln!("--- {dev_name} ---");
    let t0 = Instant::now();
    let mut dec = Sam3CompiledDecoder::new_with_profile(
        model.decoder_weights(),
        batch,
        h * w,
        seq,
        dev,
        model.compile_profile(),
    )?;
    let compile_s = t0.elapsed().as_secs_f32();
    let t = Instant::now();
    let (int_out, ref_boxes, _, _) =
        dec.run(&memory_bf, &memory_pos, &prompt, &prompt_mask, h, w)?;
    let run_ms = t.elapsed().as_secs_f32() * 1000.0;

    let (mad_i, cos_i) = diff(&int_out, &int_ref);
    let (mad_r, cos_r) = diff(&ref_boxes, &ref_boxes_ref);

    println!(
        "{dev_name:>8}  compile={compile_s:.1}s  run={run_ms:7.1}ms  intermediate cos={cos_i:.3e} mad={mad_i:.4}  ref_boxes cos={cos_r:.3e} mad={mad_r:.4}"
    );
    let _ = (num_layers, nq);
    if cos_i > 1e-3 || cos_r > 1e-3 {
        eprintln!("FAIL: {dev_name} cosine distance too large");
    } else {
        eprintln!("OK:   {dev_name} parity within 1e-3");
    }
    Ok(())
}

fn main() -> Result<()> {
    let weights = rlx_ir::env::var("RLX_SAM3_WEIGHTS")
        .ok_or_else(|| anyhow::anyhow!("set RLX_SAM3_WEIGHTS"))?;
    let ref_dir: PathBuf = rlx_ir::env::var("RLX_SAM3_REF_DIR")
        .unwrap_or_else(|| "/var/folders/9_/pjm86g5j44l4cdv5mld3wd9c0000gn/T/tmp.0NBLOovOZD".into())
        .into();
    let model = Sam3::from_safetensors(&weights, Sam3Config::base())?;
    run_on(&model, Device::Cpu, "CPU", &ref_dir)?;
    #[cfg(feature = "metal")]
    run_on(&model, Device::Metal, "Metal", &ref_dir)?;
    #[cfg(feature = "mlx")]
    run_on(&model, Device::Mlx, "MLX", &ref_dir)?;
    Ok(())
}
