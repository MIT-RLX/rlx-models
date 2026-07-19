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

//! Structure-stage e2e parity vs Python TRELLIS.2 dumps.
//!
//! Gated on env:
//!   `RLX_TRELLIS2_SSFLOW_CKPT`     — `ss_flow_img_dit_*.safetensors` stem or file
//!   `RLX_TRELLIS2_SSDEC_CKPT`      — `ss_dec_conv3d_16l8_fp16` stem
//!   `RLX_TRELLIS2_SS_INJECT`       — `ss_inject_steps12.safetensors` (`cond`,`noise`,`neg_cond`)
//!   `RLX_TRELLIS2_SS_OCC_REF`      — `ss_occ_steps12.safetensors` (`occ_xyz` i32 [N,3])
//!
//! Optional:
//!   `RLX_TRELLIS2_SS_SAMPLE`       — sample latent (+ optional `occ_xyz`) for cosine / decode checks
//!   `RLX_TRELLIS2_SS_ONESTEP`      — one-step DiT dump (`x`,`t`,`cond`,`out`)
//!   `RLX_TRELLIS2_DIT_DEVICE=metal` — compiled DiT on Metal
//!   `RLX_TRELLIS2_PRE_RGB`         — `preprocessed_rgb.npy` for DINO vs inject cond check
//!   `RLX_TRELLIS2_DINOV3`          — dinov3 weights dir / safetensors
//!   `RLX_TRELLIS2_OCC_DUMP`        — write rust occupancy i32 xyz bytes

use rlx_runtime::Device;
use rlx_trellis2::conv3d::Vol;
use rlx_trellis2::dit_host::dit_forward;
use rlx_trellis2::rope::grid_coords;
use rlx_trellis2::sampler::{SamplerConfig, flow_euler_sample};
use rlx_trellis2::ss_decoder::{decode_occupancy, occupancy_to_coords};
use safetensors::SafeTensors;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn read_f32(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
    let bytes = t.data();
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v, t.shape().to_vec())
}

fn read_i32(st: &SafeTensors, name: &str) -> (Vec<i32>, Vec<usize>) {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
    let bytes = t.data();
    let v: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v, t.shape().to_vec())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn cdhw_to_tokens(x: &[f32], n_pos: usize, ch: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n_pos * ch];
    for c in 0..ch {
        for p in 0..n_pos {
            t[p * ch + c] = x[c * n_pos + p];
        }
    }
    t
}

fn tokens_to_cdhw(t: &[f32], n_pos: usize, ch: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; ch * n_pos];
    for c in 0..ch {
        for p in 0..n_pos {
            x[c * n_pos + p] = t[p * ch + c];
        }
    }
    x
}

fn stem_path(p: &str) -> PathBuf {
    let pb = PathBuf::from(p);
    if pb.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        pb.with_extension("")
    } else {
        pb
    }
}

fn iou_coords(a: &[[i32; 3]], b: &[[i32; 3]]) -> (f32, usize, usize, usize) {
    let sa: BTreeSet<_> = a.iter().copied().collect();
    let sb: BTreeSet<_> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count();
    let uni = sa.union(&sb).count();
    let iou = if uni == 0 {
        1.0
    } else {
        inter as f32 / uni as f32
    };
    (iou, inter, sa.len(), sb.len())
}

#[test]
fn structure_inject_matches_python_occ() {
    let (Ok(flow_p), Ok(dec_p), Ok(inj_p), Ok(occ_p)) = (
        std::env::var("RLX_TRELLIS2_SSFLOW_CKPT"),
        std::env::var("RLX_TRELLIS2_SSDEC_CKPT"),
        std::env::var("RLX_TRELLIS2_SS_INJECT"),
        std::env::var("RLX_TRELLIS2_SS_OCC_REF"),
    ) else {
        eprintln!(
            "skipping: set RLX_TRELLIS2_SSFLOW_CKPT, RLX_TRELLIS2_SSDEC_CKPT, \
             RLX_TRELLIS2_SS_INJECT, RLX_TRELLIS2_SS_OCC_REF"
        );
        return;
    };

    let inj_bytes = std::fs::read(&inj_p).expect("read inject");
    let inj = SafeTensors::deserialize(&inj_bytes).expect("parse inject");
    let (cond, cond_shape) = read_f32(&inj, "cond");
    let (noise, noise_shape) = read_f32(&inj, "noise");
    let (neg, _) = read_f32(&inj, "neg_cond");
    assert_eq!(cond_shape.len(), 3, "cond [1,L,C]");
    let n_cond = cond_shape[1];
    let cond_ch = cond_shape[2];
    assert_eq!(noise_shape, vec![1, 8, 16, 16, 16]);

    let occ_bytes = std::fs::read(&occ_p).expect("read occ");
    let occ_st = SafeTensors::deserialize(&occ_bytes).expect("parse occ");
    let (occ_flat, occ_shape) = read_i32(&occ_st, "occ_xyz");
    assert_eq!(occ_shape, vec![occ_flat.len() / 3, 3]);
    let py_xyz: Vec<[i32; 3]> = occ_flat
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    let mut flow = rlx_trellis2::weights::load_dit(&stem_path(&flow_p)).expect("load dit");
    let ss_dec = rlx_trellis2::weights::load_ss_decoder(&stem_path(&dec_p)).expect("load ssdec");

    let res = flow.cfg.args.resolution;
    let in_ch = flow.cfg.args.in_channels;
    let out_ch = flow.cfg.args.out_channels;
    let n_pos = res * res * res;
    let coords_rope = grid_coords(res);
    assert_eq!(noise.len(), in_ch * n_pos);
    assert_eq!(cond_ch, flow.cfg.args.cond_channels);

    let scfg = SamplerConfig {
        sigma_min: 1e-5,
        steps: 12,
        guidance_strength: 7.5,
        guidance_rescale: 0.7,
        guidance_interval: [0.6, 1.0],
        rescale_t: 5.0,
    };

    let device = if std::env::var("RLX_TRELLIS2_DIT_DEVICE").as_deref() == Ok("metal") {
        Device::Metal
    } else {
        Device::Cpu
    };
    let use_compiled = device != Device::Cpu;
    eprintln!(
        "structure inject parity: n_cond={n_cond} steps={} device={device:?} compiled={use_compiled}",
        scfg.steps
    );

    let mut step_i = 0usize;
    let t0 = std::time::Instant::now();
    let model_v = |x_t: &[f32], t_scaled: f32, cnd: &[f32]| -> Vec<f32> {
        step_i += 1;
        let tokens = cdhw_to_tokens(x_t, n_pos, in_ch);
        let s0 = std::time::Instant::now();
        let out = if use_compiled {
            flow.forward_compiled(device, &tokens, &coords_rope, n_pos, cnd, n_cond, t_scaled)
                .expect("compiled")
        } else {
            dit_forward(
                &flow.cfg,
                &flow.weights,
                &tokens,
                &coords_rope,
                n_pos,
                cnd,
                n_cond,
                t_scaled,
                None,
            )
            .expect("host")
        };
        eprintln!(
            "  ss dit #{step_i} t={t_scaled:.1} {:.1}s",
            s0.elapsed().as_secs_f64()
        );
        tokens_to_cdhw(&out, n_pos, out_ch)
    };

    let sample = flow_euler_sample(model_v, &noise, &cond, &neg, &scfg);
    eprintln!("ss sample done in {:.1}s", t0.elapsed().as_secs_f64());

    if let Ok(sample_p) = std::env::var("RLX_TRELLIS2_SS_SAMPLE") {
        let bytes = std::fs::read(&sample_p).expect("read sample ref");
        let st = SafeTensors::deserialize(&bytes).expect("parse sample");
        let (py_sample, _) = read_f32(&st, "sample");
        let cos = cosine(&sample, &py_sample);
        let maxabs = sample
            .iter()
            .zip(&py_sample)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("sample vs Python: cos={cos:.6} maxabs={maxabs:.4e}");
        // Prefer a float32 Python reference (`convert_to(float32)`); bf16-torso
        // MPS dumps sit ~0.991 vs host f32 even when the sampler is correct.
        assert!(cos > 0.999, "sample cosine {cos} too low (maxabs={maxabs})");
    }

    let latent = Vol {
        c: out_ch,
        d: res,
        h: res,
        w: res,
        data: sample,
    };
    let occ = decode_occupancy(&ss_dec.cfg, &ss_dec.weights, &latent).expect("decode");
    let with_batch = occupancy_to_coords(&occ, 32);
    let rust_xyz: Vec<[i32; 3]> = with_batch.into_iter().map(|c| [c[1], c[2], c[3]]).collect();

    let (iou, inter, n_r, n_p) = iou_coords(&rust_xyz, &py_xyz);
    eprintln!("occupancy IoU={iou:.4} inter={inter} rust={n_r} python={n_p}");

    // Persist for offline comparison / mesh preview.
    if let Ok(dump) = std::env::var("RLX_TRELLIS2_OCC_DUMP") {
        let mut flat = Vec::with_capacity(rust_xyz.len() * 3);
        for c in &rust_xyz {
            flat.extend_from_slice(c);
        }
        let mut bytes = Vec::new();
        for v in &flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&dump, &bytes).ok();
        eprintln!(
            "wrote rust occ i32 xyz to {dump} ({} voxels)",
            rust_xyz.len()
        );
    }

    assert!(
        iou > 0.95,
        "structure occupancy IoU {iou} too low (rust={n_r} py={n_p} inter={inter})"
    );
}

#[test]
fn ssdec_on_python_sample_matches_occ() {
    let (Ok(dec_p), Ok(sample_p)) = (
        std::env::var("RLX_TRELLIS2_SSDEC_CKPT"),
        std::env::var("RLX_TRELLIS2_SS_SAMPLE"),
    ) else {
        eprintln!("skipping: set RLX_TRELLIS2_SSDEC_CKPT + RLX_TRELLIS2_SS_SAMPLE");
        return;
    };
    let bytes = std::fs::read(&sample_p).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let (sample, shape) = read_f32(&st, "sample");
    assert_eq!(shape, vec![1, 8, 16, 16, 16]);

    let ss_dec = rlx_trellis2::weights::load_ss_decoder(&stem_path(&dec_p)).unwrap();
    let latent = Vol {
        c: 8,
        d: 16,
        h: 16,
        w: 16,
        data: sample.clone(),
    };
    let occ = decode_occupancy(&ss_dec.cfg, &ss_dec.weights, &latent).unwrap();
    let with_batch = occupancy_to_coords(&occ, 32);
    let rust_xyz: Vec<[i32; 3]> = with_batch.into_iter().map(|c| [c[1], c[2], c[3]]).collect();

    if st.tensor("occ_xyz").is_ok() {
        let (occ_flat, _) = read_i32(&st, "occ_xyz");
        let py_xyz: Vec<[i32; 3]> = occ_flat
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let (iou, inter, n_r, n_p) = iou_coords(&rust_xyz, &py_xyz);
        eprintln!("ssdec-only IoU={iou:.6} inter={inter} rust={n_r} py={n_p}");
        assert!(iou > 0.999, "decoder IoU {iou}");
    } else {
        eprintln!(
            "ssdec-only: no occ_xyz in sample file; decoded {} voxels",
            rust_xyz.len()
        );
    }

    if let Ok(out_p) = std::env::var("RLX_TRELLIS2_SS_SAMPLE_OUT") {
        // Write sample + rust-decoded occ for inject parity refs.
        let mut flat = Vec::with_capacity(rust_xyz.len() * 3);
        for c in &rust_xyz {
            flat.extend_from_slice(c);
        }
        write_sample_occ_safetensors(&out_p, &sample, &flat).expect("write sample+occ");
        eprintln!("wrote {out_p} ({} voxels)", rust_xyz.len());
    }
}

fn write_sample_occ_safetensors(
    path: &str,
    sample: &[f32],
    occ_xyz: &[i32],
) -> std::io::Result<()> {
    use safetensors::{Dtype, tensor::TensorView};
    use std::collections::HashMap;
    let sample_shape = vec![1usize, 8, 16, 16, 16];
    let n = occ_xyz.len() / 3;
    let occ_shape = vec![n, 3usize];
    let sample_bytes =
        unsafe { std::slice::from_raw_parts(sample.as_ptr() as *const u8, sample.len() * 4) };
    let occ_bytes =
        unsafe { std::slice::from_raw_parts(occ_xyz.as_ptr() as *const u8, occ_xyz.len() * 4) };
    let mut tensors = HashMap::new();
    tensors.insert(
        "sample".to_string(),
        TensorView::new(Dtype::F32, sample_shape, sample_bytes).unwrap(),
    );
    tensors.insert(
        "occ_xyz".to_string(),
        TensorView::new(Dtype::I32, occ_shape, occ_bytes).unwrap(),
    );
    let serialized = safetensors::serialize(tensors, None).unwrap();
    std::fs::write(path, serialized)
}

#[test]
fn dinov3_matches_python_cond_optional() {
    let (Ok(inj_p), Ok(rgb_p), Ok(dino_w)) = (
        std::env::var("RLX_TRELLIS2_SS_INJECT"),
        std::env::var("RLX_TRELLIS2_PRE_RGB"),
        std::env::var("RLX_TRELLIS2_DINOV3"),
    ) else {
        eprintln!("skipping DINO check: set RLX_TRELLIS2_SS_INJECT, PRE_RGB, DINOV3");
        return;
    };

    let (h, w, pixels) = read_npy_u8_hwc(&rgb_p);

    let inj_bytes = std::fs::read(&inj_p).unwrap();
    let inj = SafeTensors::deserialize(&inj_bytes).unwrap();
    let (py_cond, py_shape) = read_f32(&inj, "cond");
    let n_cond = py_shape[1];
    let dim = py_shape[2];

    let cfg_path = Path::new(&dino_w).join("config.json");
    let weights = if Path::new(&dino_w).is_dir() {
        Path::new(&dino_w).join("model.safetensors")
    } else {
        PathBuf::from(&dino_w)
    };
    let mut cfg = if cfg_path.is_file() {
        rlx_dinov3::DinoV3Config::from_file(&cfg_path).expect("dino cfg")
    } else {
        rlx_dinov3::DinoV3Config::vit_l16(512)
    };
    cfg.image_size = 512;
    cfg.final_layer_norm_affine = false;

    let mut runner = rlx_dinov3::DinoV3Runner::builder()
        .weights(&weights)
        .config(cfg)
        .device(Device::Cpu)
        .img_size(512)
        .build()
        .expect("dino build");
    let out = runner.predict_image(&pixels, h, w).expect("dino forward");
    let rust = &out.tokens[0];
    assert_eq!(rust.len(), n_cond * dim);
    let cos = cosine(rust, &py_cond);
    eprintln!("DINOv3 (Trellis non-affine LN) vs Python cond cosine={cos:.6}");
    assert!(cos > 0.99, "DINO cond cosine {cos} too low");

    // Also compare against the same NCHW the Python dump used (isolates resize).
    if let Ok(refs) = std::env::var("RLX_TRELLIS2_DINO_REFS") {
        let bytes = std::fs::read(&refs).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let (nchw, nchw_shape) = read_f32(&st, "nchw");
        assert_eq!(nchw_shape, vec![1, 3, 512, 512]);
        let out2 = runner.forward_nchw(&nchw).expect("nchw");
        let (py_local, _) = read_f32(&st, "cond_local");
        let cos2 = cosine(&out2.tokens[0], &py_local);
        eprintln!("DINOv3 vs Python on shared NCHW cosine={cos2:.6}");
        // Bilinear vs exact HF stack can sit a bit under 1; require strong agreement.
        assert!(cos2 > 0.85, "shared-NCHW DINO cosine {cos2} too low");
    }
}

/// Minimal `.npy` reader for C-order uint8 3-D arrays (H,W,3).
fn read_npy_u8_hwc(path: &str) -> (usize, usize, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read npy");
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let ver = bytes[6];
    let header_len = if ver == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };
    let header_start = if ver == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).unwrap();
    let shape_idx = header.find('(').expect("shape");
    let shape_end = header[shape_idx..].find(')').unwrap() + shape_idx;
    let nums: Vec<usize> = header[shape_idx + 1..shape_end]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert_eq!(nums.len(), 3);
    let (h, w, c) = (nums[0], nums[1], nums[2]);
    assert_eq!(c, 3);
    let data_start = header_start + header_len;
    let data = bytes[data_start..].to_vec();
    assert_eq!(data.len(), h * w * 3);
    (h, w, data)
}

#[test]
fn dit_onestep_inject_matches_python() {
    let (Ok(flow_p), Ok(ref_p)) = (
        std::env::var("RLX_TRELLIS2_SSFLOW_CKPT"),
        std::env::var("RLX_TRELLIS2_SS_ONESTEP"),
    ) else {
        eprintln!("skipping: set SSFLOW_CKPT + SS_ONESTEP");
        return;
    };
    let bytes = std::fs::read(&ref_p).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let (x, x_shape) = read_f32(&st, "x");
    let (t, _) = read_f32(&st, "t");
    let (cond, cshape) = read_f32(&st, "cond");
    let (g_out, _) = read_f32(&st, "out");
    let n_cond = cshape[1];
    let res = x_shape[2];
    let n_pos = res * res * res;
    let in_ch = x_shape[1];
    let tokens = cdhw_to_tokens(&x, n_pos, in_ch);
    let coords = grid_coords(res);
    let flow = rlx_trellis2::weights::load_dit(&stem_path(&flow_p)).unwrap();
    let out = dit_forward(
        &flow.cfg,
        &flow.weights,
        &tokens,
        &coords,
        n_pos,
        &cond,
        n_cond,
        t[0],
        None,
    )
    .unwrap();
    let out_cdhw = tokens_to_cdhw(&out, n_pos, flow.cfg.args.out_channels);
    let cos = cosine(&out_cdhw, &g_out);
    let maxabs = out_cdhw
        .iter()
        .zip(&g_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("onestep inject host vs py: cos={cos:.6} maxabs={maxabs:.4e}");
    assert!(cos > 0.9999, "onestep cos {cos}");
}
