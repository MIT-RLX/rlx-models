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

//! A tiny VGSL network written as a Kraken ≥ 6 `.safetensors` (header
//! metadata + UUID-prefixed state dict) runs end to end without weights
//! from anywhere.

use rlx_kraken::KrakenRunner;
use rlx_runtime::Device;

const PREFIX: &str = "00000000-0000-0000-0000-000000000000";

/// Minimal safetensors writer: `u64` header length, JSON header, f32 data.
fn write_safetensors(
    path: &std::path::Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    meta: &str,
) {
    let mut hdr = String::from("{\"__metadata__\":{\"kraken_meta\":");
    hdr.push_str(&serde_json::to_string(meta).unwrap());
    hdr.push('}');
    let mut data = Vec::new();
    for (name, shape, v) in tensors {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        hdr.push_str(&format!(
            ",\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    hdr.push('}');
    let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
    out.extend(hdr.as_bytes());
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

#[test]
fn tiny_safetensors_model_runs() {
    // 8 px high → conv (4 ch) → 2×2 pool → 4 rows folded into 16 features
    // → BiLSTM(5) → 4 classes.
    let vgsl = "[1,8,0,1 Cr{C_0}3,3,4 Mp{Mp_1}2,2 S{S_2}1(1x0)1,3 Lbx{L_3}5 O{O_4}1c4]";
    let meta = serde_json::json!({ PREFIX: {
        "model_type": ["recognition"],
        "vgsl": vgsl,
        "codec": {" ": [1], "a": [2], "b": [3]},
        "hyper_params": {"pad": 2},
    }})
    .to_string();
    let mut seed = 1u32;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    };
    let p = |s: &str| format!("{PREFIX}.nn.{s}");
    let mut t = vec![
        (p("C_0.co.weight"), vec![4, 1, 3, 3], rnd(36)),
        (p("C_0.co.bias"), vec![4], rnd(4)),
        (p("O_4.lin.weight"), vec![4, 10], rnd(40)),
        (p("O_4.lin.bias"), vec![4], rnd(4)),
    ];
    for sfx in ["", "_reverse"] {
        t.push((
            p(&format!("L_3.layer.weight_ih_l0{sfx}")),
            vec![20, 16],
            rnd(320),
        ));
        t.push((
            p(&format!("L_3.layer.weight_hh_l0{sfx}")),
            vec![20, 5],
            rnd(100),
        ));
        t.push((p(&format!("L_3.layer.bias_ih_l0{sfx}")), vec![20], rnd(20)));
        t.push((p(&format!("L_3.layer.bias_hh_l0{sfx}")), vec![20], rnd(20)));
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.safetensors");
    write_safetensors(&path, &t, &meta);

    let mut r = KrakenRunner::open(&path, Device::Cpu).unwrap();
    assert_eq!(r.padding(), 2);
    let img = image::DynamicImage::ImageLuma8(image::GrayImage::from_fn(30, 8, |x, y| {
        image::Luma([((x * 7 + y * 13) % 256) as u8])
    }));
    let rec = r.recognize(&img).unwrap();
    // 30 px + 2·2 padding, halved by the pool.
    assert_eq!(rec.frames, 17);
    assert!(rec.text.chars().all(|c| " ab".contains(c)));
}
