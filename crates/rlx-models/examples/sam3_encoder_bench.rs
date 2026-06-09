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

//
//! Compile-once / run-many detector-encoder benchmark on CPU, Metal,
//! and the legacy host path. Precomputes vision + text outside the
//! measured region so we time only the encoder.

use anyhow::Result;
use rlx_models::sam3::detector_encoder::forward_encoder;
use rlx_models::sam3::detector_encoder_ir::Sam3CompiledEncoder;
use rlx_models::sam3::{SAM3_IMG_SIZE, Sam3, Sam3Config};
use rlx_runtime::Device;
use std::env;
use std::time::Instant;

fn synthesize_image() -> Vec<u8> {
    let n = SAM3_IMG_SIZE * SAM3_IMG_SIZE * 3;
    let mut v = vec![0u8; n];
    for y in 0..SAM3_IMG_SIZE {
        for x in 0..SAM3_IMG_SIZE {
            for c in 0..3 {
                let fx = x as f32 / SAM3_IMG_SIZE as f32;
                let fy = y as f32 / SAM3_IMG_SIZE as f32;
                let phase = (c as f32) * 0.7;
                let s = (std::f32::consts::TAU * fx + phase).sin()
                    * (std::f32::consts::PI * fy + phase).cos();
                let val = ((s + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                v[(y * SAM3_IMG_SIZE + x) * 3 + c] = val;
            }
        }
    }
    v
}

fn fmt(vs: &[f32]) -> String {
    let avg = vs.iter().sum::<f32>() / vs.len() as f32;
    let min = vs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    format!("avg={avg:8.1}ms  min={min:8.1}ms  max={max:8.1}ms")
}

fn main() -> Result<()> {
    let weights = rlx_ir::env::var("RLX_SAM3_WEIGHTS")
        .ok_or_else(|| anyhow::anyhow!("set RLX_SAM3_WEIGHTS"))?;
    let iters: usize = env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let warmup: usize = env::var("BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    let mut model = Sam3::from_safetensors(&weights, Sam3Config::base())?;
    let image = synthesize_image();
    let tokens: Vec<u32> = if let Some(p) = rlx_ir::env::var("RLX_SAM3_TOKENS_BIN") {
        std::fs::read(&p)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect()
    } else {
        vec![0u32; 32]
    };
    let seq = tokens.len();

    eprintln!("precomputing vision + neck + text once…");
    let t = Instant::now();
    let neck = model.predict_neck(&image, SAM3_IMG_SIZE, SAM3_IMG_SIZE)?;
    eprintln!("vision+neck precompute: {:.1}s", t.elapsed().as_secs_f32());
    let text = model.encode_text_tokens(&tokens, 1, seq)?;
    let lvl = &neck[2];
    let hw = lvl.h * lvl.w;

    // ── Host backend ──────────────────────────────────────────────────
    eprintln!("benching host (per-head sgemm loop) …");
    let mut host_times = Vec::new();
    for i in 0..(warmup + iters) {
        let t = Instant::now();
        let _ = forward_encoder(
            model.encoder_weights(),
            &lvl.features,
            &lvl.pos,
            &text.text_memory_resized,
            &text.attention_mask,
            1,
            lvl.h,
            lvl.w,
            seq,
            None,
        )?;
        if i >= warmup {
            host_times.push(t.elapsed().as_secs_f32() * 1000.0);
        }
    }

    // ── IR on CPU ─────────────────────────────────────────────────────
    eprintln!("compiling IR encoder for CPU …");
    let t = Instant::now();
    let mut cpu_enc = Sam3CompiledEncoder::new_with_profile(
        model.encoder_weights(),
        1,
        hw,
        seq,
        Device::Cpu,
        model.compile_profile(),
    )?;
    eprintln!("CPU compile: {:.1}ms", t.elapsed().as_secs_f32() * 1000.0);
    let mut cpu_times = Vec::new();
    for i in 0..(warmup + iters) {
        let t = Instant::now();
        let _ = cpu_enc.run(
            &lvl.features,
            &lvl.pos,
            &text.text_memory_resized,
            &text.attention_mask,
            lvl.h,
            lvl.w,
        )?;
        if i >= warmup {
            cpu_times.push(t.elapsed().as_secs_f32() * 1000.0);
        }
    }

    // ── IR on Metal (if available) ────────────────────────────────────
    let metal_times = if cfg!(feature = "metal") {
        eprintln!("compiling IR encoder for Metal …");
        let t = Instant::now();
        let mut metal_enc = Sam3CompiledEncoder::new_with_profile(
            model.encoder_weights(),
            1,
            hw,
            seq,
            Device::Metal,
            model.compile_profile(),
        )?;
        eprintln!("Metal compile: {:.1}ms", t.elapsed().as_secs_f32() * 1000.0);
        let mut ts = Vec::new();
        for i in 0..(warmup + iters) {
            let t = Instant::now();
            let _ = metal_enc.run(
                &lvl.features,
                &lvl.pos,
                &text.text_memory_resized,
                &text.attention_mask,
                lvl.h,
                lvl.w,
            )?;
            if i >= warmup {
                ts.push(t.elapsed().as_secs_f32() * 1000.0);
            }
        }
        ts
    } else {
        Vec::new()
    };

    println!(
        "# SAM3 detector encoder bench (release, {} iters, {} warmup)",
        iters, warmup
    );
    println!("  host (per-head sgemm) : {}", fmt(&host_times));
    println!("  IR  / CPU             : {}", fmt(&cpu_times));
    if !metal_times.is_empty() {
        println!("  IR  / Metal           : {}", fmt(&metal_times));
    }
    let host_avg = host_times.iter().sum::<f32>() / host_times.len() as f32;
    let cpu_avg = cpu_times.iter().sum::<f32>() / cpu_times.len() as f32;
    println!("# speedup IR/CPU   vs host: {:.2}×", host_avg / cpu_avg);
    if !metal_times.is_empty() {
        let m_avg = metal_times.iter().sum::<f32>() / metal_times.len() as f32;
        println!("# speedup IR/Metal vs host: {:.2}×", host_avg / m_avg);
        println!("# speedup IR/Metal vs IR/CPU: {:.2}×", cpu_avg / m_avg);
    }
    Ok(())
}
