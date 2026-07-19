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

//! Real-weight parity: host shape VAE decoder vs a pure-Python transcription
//! (`scripts/shape_dec_ref.py`, submanifold conv == dense conv3d).
//! Gated on `RLX_TRELLIS2_SHAPEDEC_CKPT` + `RLX_TRELLIS2_SHAPEDEC_REF`.

use rlx_trellis2::config::SparseVaeConfig;
use rlx_trellis2::shape_decoder::decode;
use rlx_trellis2::sparse::SparseTensor;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

fn f32s(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st.tensor(name).unwrap();
    let v = t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v, t.shape().to_vec())
}
fn i32s(st: &SafeTensors, name: &str) -> Vec<i32> {
    st.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn shape_dec_cfg() -> SparseVaeConfig {
    SparseVaeConfig::from_json_str(
        r#"{"name":"FlexiDualGridVaeDecoder","args":{
            "resolution":256,"latent_channels":32,
            "model_channels":[1024,512,256,128,64],"num_blocks":[4,16,8,4,0],
            "block_type":["SparseConvNeXtBlock3d","SparseConvNeXtBlock3d","SparseConvNeXtBlock3d","SparseConvNeXtBlock3d","SparseConvNeXtBlock3d"],
            "up_block_type":["SparseResBlockC2S3d","SparseResBlockC2S3d","SparseResBlockC2S3d","SparseResBlockC2S3d"],
            "block_args":[{},{},{},{},{}],"use_fp16":true}}"#,
    )
    .unwrap()
}

#[test]
fn shape_decoder_matches_transcription() {
    let (Ok(ckpt), Ok(refp)) = (
        std::env::var("RLX_TRELLIS2_SHAPEDEC_CKPT"),
        std::env::var("RLX_TRELLIS2_SHAPEDEC_REF"),
    ) else {
        eprintln!(
            "skipping shape dec parity: set RLX_TRELLIS2_SHAPEDEC_CKPT + RLX_TRELLIS2_SHAPEDEC_REF"
        );
        return;
    };
    let cfg = shape_dec_cfg();
    let wm = rlx_core::load_weight_map(Path::new(&ckpt), &[]).expect("load ckpt");
    let bytes = std::fs::read(&refp).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();

    let in_coords = i32s(&st, "in_coords");
    let (in_latent, lshape) = f32s(&st, "in_latent");
    let (out_feats, oshape) = f32s(&st, "out_feats");
    let out_coords = i32s(&st, "out_coords");
    let lat_c = lshape[1];
    let out_c = oshape[1];

    let coords: Vec<[i32; 3]> = in_coords
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let latent = SparseTensor::new(in_latent, coords, lat_c);

    let dec = decode(&cfg, &wm, &latent, None).expect("decode");
    println!(
        "rust: {} voxels {} ch | ref: {} voxels {} ch",
        dec.voxels.n(),
        dec.voxels.c,
        oshape[0],
        out_c
    );
    assert_eq!(dec.voxels.n(), oshape[0], "voxel count mismatch");
    assert_eq!(dec.voxels.c, out_c);

    // match by coordinate (both should produce identical coord sets)
    let ref_coords: Vec<[i32; 3]> = out_coords
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let mut ref_map: HashMap<[i32; 3], usize> = HashMap::new();
    for (i, c) in ref_coords.iter().enumerate() {
        ref_map.insert(*c, i);
    }
    let mut maxabs = 0.0f32;
    let mut dot = 0.0f64;
    let (mut na, mut nb) = (0.0f64, 0.0f64);
    for i in 0..dec.voxels.n() {
        let c = dec.voxels.coords[i];
        let &j = ref_map
            .get(&c)
            .unwrap_or_else(|| panic!("rust coord {c:?} not in reference"));
        for k in 0..out_c {
            let a = dec.voxels.feats[i * out_c + k];
            let b = out_feats[j * out_c + k];
            maxabs = maxabs.max((a - b).abs());
            dot += a as f64 * b as f64;
            na += (a * a) as f64;
            nb += (b * b) as f64;
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    println!("shape dec: maxabs {maxabs:.3e} cos {cos:.6}");
    assert!(cos > 0.9999, "shape decoder cos too low: {cos}");
}
