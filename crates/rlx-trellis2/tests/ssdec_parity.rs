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

//! Real-weight parity: host dense sparse-structure decoder vs PyTorch.
//! Gated on `RLX_TRELLIS2_SSDEC_CKPT` + `RLX_TRELLIS2_SSDEC_REF`
//! (`scripts/ss_dec_ref.py`).

use rlx_trellis2::config::SparseStructureVaeArgs;
use rlx_trellis2::conv3d::Vol;
use rlx_trellis2::ss_decoder::decode_occupancy;
use safetensors::SafeTensors;
use std::path::Path;

fn read_f32(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st.tensor(name).unwrap();
    let v = t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v, t.shape().to_vec())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[test]
fn ssdec_matches_pytorch() {
    let (Ok(ckpt), Ok(refp)) = (
        std::env::var("RLX_TRELLIS2_SSDEC_CKPT"),
        std::env::var("RLX_TRELLIS2_SSDEC_REF"),
    ) else {
        eprintln!("skipping ssdec parity: set RLX_TRELLIS2_SSDEC_CKPT + RLX_TRELLIS2_SSDEC_REF");
        return;
    };
    let cfg: SparseStructureVaeArgs = serde_json::from_str(
        r#"{"out_channels":1,"latent_channels":8,"num_res_blocks":2,
            "num_res_blocks_middle":2,"channels":[512,128,32],"use_fp16":true}"#,
    )
    .unwrap();
    let wm = rlx_core::load_weight_map(Path::new(&ckpt), &[]).expect("load ckpt");
    let bytes = std::fs::read(&refp).expect("read golden");
    let st = SafeTensors::deserialize(&bytes).expect("parse golden");

    let (latent, lshape) = read_f32(&st, "latent"); // [1,8,16,16,16]
    let (occ_ref, _) = read_f32(&st, "occ"); // [1,1,64,64,64]
    let res = lshape[2];
    let lv = Vol {
        c: lshape[1],
        d: res,
        h: res,
        w: res,
        data: latent,
    };

    let occ = decode_occupancy(&cfg, &wm, &lv).expect("decode");
    let c = cosine(&occ.data, &occ_ref);
    let maxabs = occ
        .data
        .iter()
        .zip(&occ_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = maxabs / occ_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "occ: cos {c:.6} maxabs {maxabs:.3e} rel {rel:.3e} (occ range shape {:?})",
        occ.data.len()
    );
    assert!(c > 0.9999, "ssdec cos too low: {c}");
    assert!(rel < 5e-3, "ssdec relative error too high: {rel}");
}
