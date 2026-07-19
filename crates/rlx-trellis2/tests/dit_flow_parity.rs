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

//! Compiled DiT (`dit_flow`) vs host (`dit_host`) parity on a tiny synthetic net.
//!
//! Optional real-weight / Metal timing when `RLX_TRELLIS2_SSFLOW_CKPT` is set.

use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;
use rlx_trellis2::config::{DitConfig, DitKind};
use rlx_trellis2::dit_flow::{compile_dit, dit_forward_compiled};
use rlx_trellis2::dit_host::dit_forward;
use rlx_trellis2::rope::grid_coords;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

fn tiny_cfg() -> DitConfig {
    DitConfig {
        kind: DitKind::SparseStructureFlow,
        args: serde_json::from_str(
            r#"{
                "resolution": 2,
                "in_channels": 4,
                "out_channels": 4,
                "model_channels": 32,
                "cond_channels": 16,
                "num_blocks": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "pe_mode": "rope",
                "share_mod": true,
                "initialization": "scaled",
                "qk_rms_norm": true,
                "qk_rms_norm_cross": true,
                "dtype": "float32"
            }"#,
        )
        .unwrap(),
    }
}

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.017 + seed).sin() * 0.05)
        .collect()
}

fn synth_weights(cfg: &DitConfig) -> WeightMap {
    let c = cfg.args.model_channels;
    let in_ch = cfg.args.in_channels;
    let out_ch = cfg.args.out_channels;
    let cond = cfg.args.cond_channels;
    let nh = cfg.num_heads();
    let hd = cfg.head_dim();
    let mlp_h = cfg.mlp_hidden();
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut s = 1.0f32;
    let mut put = |k: &str, shape: Vec<usize>| {
        let n = shape.iter().product();
        t.insert(k.into(), (fill(n, s), shape));
        s += 0.37;
    };

    put("input_layer.weight", vec![c, in_ch]);
    put("input_layer.bias", vec![c]);
    put("t_embedder.mlp.0.weight", vec![c, 256]);
    put("t_embedder.mlp.0.bias", vec![c]);
    put("t_embedder.mlp.2.weight", vec![c, c]);
    put("t_embedder.mlp.2.bias", vec![c]);
    put("adaLN_modulation.1.weight", vec![6 * c, c]);
    put("adaLN_modulation.1.bias", vec![6 * c]);

    for blk in 0..cfg.args.num_blocks {
        let p = format!("blocks.{blk}");
        put(&format!("{p}.modulation"), vec![6 * c]);
        put(&format!("{p}.norm2.weight"), vec![c]);
        put(&format!("{p}.norm2.bias"), vec![c]);
        put(&format!("{p}.self_attn.to_qkv.weight"), vec![3 * c, c]);
        put(&format!("{p}.self_attn.to_qkv.bias"), vec![3 * c]);
        put(&format!("{p}.self_attn.q_rms_norm.gamma"), vec![nh * hd]);
        put(&format!("{p}.self_attn.k_rms_norm.gamma"), vec![nh * hd]);
        put(&format!("{p}.self_attn.to_out.weight"), vec![c, c]);
        put(&format!("{p}.self_attn.to_out.bias"), vec![c]);
        put(&format!("{p}.cross_attn.to_q.weight"), vec![c, c]);
        put(&format!("{p}.cross_attn.to_q.bias"), vec![c]);
        put(&format!("{p}.cross_attn.to_kv.weight"), vec![2 * c, cond]);
        put(&format!("{p}.cross_attn.to_kv.bias"), vec![2 * c]);
        put(&format!("{p}.cross_attn.q_rms_norm.gamma"), vec![nh * hd]);
        put(&format!("{p}.cross_attn.k_rms_norm.gamma"), vec![nh * hd]);
        put(&format!("{p}.cross_attn.to_out.weight"), vec![c, c]);
        put(&format!("{p}.cross_attn.to_out.bias"), vec![c]);
        put(&format!("{p}.mlp.mlp.0.weight"), vec![mlp_h, c]);
        put(&format!("{p}.mlp.mlp.0.bias"), vec![mlp_h]);
        put(&format!("{p}.mlp.mlp.2.weight"), vec![c, mlp_h]);
        put(&format!("{p}.mlp.mlp.2.bias"), vec![c]);
    }
    put("out_layer.weight", vec![out_ch, c]);
    put("out_layer.bias", vec![out_ch]);
    WeightMap::from_tensors(t)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[test]
fn dit_flow_matches_host_synthetic_cpu() {
    let cfg = tiny_cfg();
    let wm = synth_weights(&cfg);
    let res = cfg.args.resolution;
    let n_pos = res * res * res;
    let n_cond = 3usize;
    let tokens = fill(n_pos * cfg.args.in_channels, 0.2);
    let cond = fill(n_cond * cfg.args.cond_channels, 0.5);
    let coords = grid_coords(res);
    let t = 500.0f32;

    let host =
        dit_forward(&cfg, &wm, &tokens, &coords, n_pos, &cond, n_cond, t, None).expect("host");
    let mut compiled = compile_dit(&cfg, &wm, Device::Cpu, n_pos, n_cond).expect("compile");
    let graph = dit_forward_compiled(
        &mut compiled,
        &cfg,
        &wm,
        &tokens,
        &coords,
        n_pos,
        &cond,
        n_cond,
        t,
    )
    .expect("compiled");

    let cos = cosine(&host, &graph);
    eprintln!("dit_flow synthetic CPU cosine={cos:.6}");
    assert!(
        cos > 0.999,
        "compiled vs host cosine {cos} (len {})",
        host.len()
    );
}

#[test]
fn dit_flow_ss_real_weights_optional() {
    let Ok(ckpt) = std::env::var("RLX_TRELLIS2_SSFLOW_CKPT") else {
        eprintln!("skipping: set RLX_TRELLIS2_SSFLOW_CKPT for real-weight dit_flow parity");
        return;
    };
    let cfg = DitConfig {
        kind: DitKind::SparseStructureFlow,
        args: serde_json::from_str(
            r#"{"resolution":16,"in_channels":8,"out_channels":8,"model_channels":1536,
                "cond_channels":1024,"num_blocks":30,"num_heads":12,"mlp_ratio":5.3334,
                "pe_mode":"rope","share_mod":true,"initialization":"scaled",
                "qk_rms_norm":true,"qk_rms_norm_cross":true,"dtype":"bfloat16"}"#,
        )
        .unwrap(),
    };
    let wm = rlx_core::load_weight_map(Path::new(&ckpt), &[]).expect("load ckpt");
    let res = 16usize;
    let n_pos = res * res * res;
    let n_cond = 8usize;
    let tokens = fill(n_pos * 8, 0.11);
    let cond = fill(n_cond * 1024, 0.3);
    let coords = grid_coords(res);
    let t = 500.0f32;

    let device = if std::env::var("RLX_TRELLIS2_DIT_DEVICE").as_deref() == Ok("metal") {
        Device::Metal
    } else {
        Device::Cpu
    };

    let t_host = Instant::now();
    let host =
        dit_forward(&cfg, &wm, &tokens, &coords, n_pos, &cond, n_cond, t, None).expect("host");
    let host_s = t_host.elapsed().as_secs_f64();

    let t_c = Instant::now();
    let mut compiled = compile_dit(&cfg, &wm, device, n_pos, n_cond).expect("compile");
    let compile_s = t_c.elapsed().as_secs_f64();
    // Warmup (Metal/MLX shader / buffer setup).
    let _ = dit_forward_compiled(
        &mut compiled,
        &cfg,
        &wm,
        &tokens,
        &coords,
        n_pos,
        &cond,
        n_cond,
        t,
    )
    .expect("compiled warmup");
    let t_r = Instant::now();
    let graph = dit_forward_compiled(
        &mut compiled,
        &cfg,
        &wm,
        &tokens,
        &coords,
        n_pos,
        &cond,
        n_cond,
        t,
    )
    .expect("compiled");
    let run_s = t_r.elapsed().as_secs_f64();

    let cos = cosine(&host, &graph);
    eprintln!(
        "dit_flow real SS device={device:?} cosine={cos:.6} host={host_s:.2}s compile={compile_s:.2}s run(after warmup)={run_s:.2}s"
    );
    assert!(cos > 0.99, "compiled vs host cosine {cos}");
}
