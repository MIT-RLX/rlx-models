// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Decode the prototype's KNOWN-GOOD codes (`proto_codes_*.npy`, from
//! onnxruntime) with rlx-dac — isolates whether low synthesis amplitude is the
//! DAC (weights/config) or the AR loop's codes.
//!
//! ```text
//! RLX_DAC_DIR=weights/tts/parler-dac cargo run -p rlx-parlertts --example dac_check --features native -- weights/tts/parlertts/proto_codes_description.npy
//! ```

use anyhow::{Result, anyhow};
use rlx_dac::codec::DacCodec;
use rlx_dac::codes::DacCodes;
use rlx_runtime::Device;

/// Minimal .npy reader for a 2-D int64 array (shape [K, T], C-order).
fn read_npy_i64(path: &str) -> Result<(Vec<i64>, Vec<usize>)> {
    let buf = std::fs::read(path)?;
    if &buf[..6] != b"\x93NUMPY" {
        return Err(anyhow!("not an npy file"));
    }
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = std::str::from_utf8(&buf[10..10 + hlen])?;
    // parse shape from "'shape': (K, T)"
    let shp = header
        .split("'shape':")
        .nth(1)
        .ok_or_else(|| anyhow!("no shape"))?;
    let inner = shp.trim_start().trim_start_matches('(');
    let dims: Vec<usize> = inner
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data_off = 10 + hlen;
    let raw = &buf[data_off..];
    let vals: Vec<i64> = raw
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((vals, dims))
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/tts/parlertts/proto_codes_description.npy".into());
    let dac_dir = std::env::var("RLX_DAC_DIR").unwrap_or_else(|_| "weights/tts/parler-dac".into());

    let (vals, dims) = read_npy_i64(&path)?;
    let (k, t) = (dims[0], dims[1]);
    eprintln!(
        "codes {path}: shape [{k},{t}], range [{},{}]",
        vals.iter().min().unwrap(),
        vals.iter().max().unwrap()
    );
    // [K,T] C-order → rows[k][t]
    let mut rows: Vec<Vec<u32>> = vec![Vec::with_capacity(t); k];
    for ki in 0..k {
        for ti in 0..t {
            rows[ki].push(vals[ki * t + ti].clamp(0, 1023) as u32);
        }
    }
    let dac = DacCodec::open_on(&dac_dir, Device::Cpu)?;
    let codes = DacCodes::from_quantizer_layout(rows);
    let pcm = dac.decode_codes(&codes)?;
    let peak = pcm.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let rms = (pcm.iter().map(|&v| v * v).sum::<f32>() / pcm.len().max(1) as f32).sqrt();
    eprintln!(
        "DAC → {} samples = {:.2}s @ {}Hz, peak {:.3}, rms {:.4}",
        pcm.len(),
        pcm.len() as f32 / dac.sample_rate() as f32,
        dac.sample_rate(),
        peak,
        rms
    );
    Ok(())
}
