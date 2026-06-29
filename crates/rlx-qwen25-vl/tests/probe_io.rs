// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use rlx_qwen25_vl::{AifProbe, load_probe_sample, sanitize_sample_id};
use std::io::Write;

#[test]
fn load_probe_sample_roundtrip() {
    let dir = std::env::temp_dir().join(format!("rlx_probe_io_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let sid = sanitize_sample_id("q/1");
    let n_vision = 3usize;
    let n_layers = 2usize;
    let dynamics: Vec<f32> = (0..(n_vision * n_layers))
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();

    write_npy_f32(
        &dir.join(format!("{sid}_vision_dynamics.npy")),
        &dynamics,
        &[n_vision as i64, n_layers as i64],
    );
    let meta = serde_json::json!({
        "n_vision_tokens": n_vision,
        "aif_n_layers": n_layers,
    });
    std::fs::write(
        dir.join(format!("{sid}_meta.json")),
        serde_json::to_vec(&meta).unwrap(),
    )
    .unwrap();

    let probe = load_probe_sample(&dir, "q/1").expect("load");
    assert_eq!(probe.mu.len(), n_vision);
    assert_eq!(probe.dynamics.len(), n_vision);
    assert_eq!(probe.dynamics[0].len(), n_layers);

    let rebuilt = AifProbe::build(probe.dynamics.clone());
    assert_eq!(rebuilt.mu, probe.mu);
}

fn write_npy_f32(path: &std::path::Path, data: &[f32], shape: &[i64]) {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Pad header to align to 64 bytes (numpy v1.0).
    let prefix_len = 10; // magic + version + header_len u16
    let pad = (16 - ((prefix_len + header.len() + 1) % 16)) % 16;
    header.push_str(&" ".repeat(pad));
    header.push('\n');

    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    let hlen = header.len() as u16;
    out.extend_from_slice(&hlen.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&out).unwrap();
}
