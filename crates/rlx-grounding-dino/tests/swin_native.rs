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

//! End-to-end smoke test for the native Swin backbone on a tiny synthetic
//! checkpoint: validates patch-embed → stages → patch-merge → shifted-window
//! attention → multi-scale output maps, with correct shapes and finite values.

use rlx_core::weight_map::WeightMap;
use rlx_grounding_dino::config::SwinConfig;
use rlx_grounding_dino::swin::SwinBackbone;
use std::collections::HashMap;

fn det(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 11) as f32 - 5.0) * 0.05)
        .collect()
}

fn tiny_cfg() -> SwinConfig {
    SwinConfig {
        embed_dim: 2,
        depths: vec![2, 2],
        num_heads: vec![1, 2],
        window_size: 2,
        image_size: 16,
        patch_size: 2,
        mlp_ratio: 4.0,
        num_channels: 3,
        out_indices: vec![1, 2],
        layer_norm_eps: 1e-5,
        qkv_bias: true,
    }
}

fn build_synth(cfg: &SwinConfig) -> WeightMap {
    let p = "model.backbone.conv_encoder.model.";
    let ps = cfg.patch_size;
    let ws = cfg.window_size;
    let ws2 = ws * ws;
    let rel_rows = (2 * ws - 1) * (2 * ws - 1);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut put = |k: String, data: Vec<f32>, shape: Vec<usize>| {
        t.insert(k, (data, shape));
    };

    let ed = cfg.embed_dim;
    put(
        format!("{p}embeddings.patch_embeddings.projection.weight"),
        det(ed * 3 * ps * ps, 1),
        vec![ed, 3, ps, ps],
    );
    put(
        format!("{p}embeddings.patch_embeddings.projection.bias"),
        det(ed, 2),
        vec![ed],
    );
    put(
        format!("{p}embeddings.norm.weight"),
        vec![1.0; ed],
        vec![ed],
    );
    put(format!("{p}embeddings.norm.bias"), vec![0.0; ed], vec![ed]);

    for s in 0..cfg.num_stages() {
        let dim = cfg.stage_dim(s);
        let heads = cfg.num_heads[s];
        let inter = dim * 4;
        for b in 0..cfg.depths[s] {
            let bp = format!("{p}encoder.layers.{s}.blocks.{b}.");
            put(
                format!("{bp}layernorm_before.weight"),
                vec![1.0; dim],
                vec![dim],
            );
            put(
                format!("{bp}layernorm_before.bias"),
                vec![0.0; dim],
                vec![dim],
            );
            for (name, seed) in [("query", 10), ("key", 20), ("value", 30)] {
                put(
                    format!("{bp}attention.self.{name}.weight"),
                    det(dim * dim, seed + s + b),
                    vec![dim, dim],
                );
                put(
                    format!("{bp}attention.self.{name}.bias"),
                    vec![0.0; dim],
                    vec![dim],
                );
            }
            put(
                format!("{bp}attention.self.relative_position_bias_table"),
                det(rel_rows * heads, 40),
                vec![rel_rows, heads],
            );
            // index values must be < rel_rows
            put(
                format!("{bp}attention.self.relative_position_index"),
                (0..ws2 * ws2).map(|i| (i % rel_rows) as f32).collect(),
                vec![ws2, ws2],
            );
            put(
                format!("{bp}attention.output.dense.weight"),
                det(dim * dim, 50 + s),
                vec![dim, dim],
            );
            put(
                format!("{bp}attention.output.dense.bias"),
                vec![0.0; dim],
                vec![dim],
            );
            put(
                format!("{bp}layernorm_after.weight"),
                vec![1.0; dim],
                vec![dim],
            );
            put(
                format!("{bp}layernorm_after.bias"),
                vec![0.0; dim],
                vec![dim],
            );
            put(
                format!("{bp}intermediate.dense.weight"),
                det(inter * dim, 60),
                vec![inter, dim],
            );
            put(
                format!("{bp}intermediate.dense.bias"),
                vec![0.0; inter],
                vec![inter],
            );
            put(
                format!("{bp}output.dense.weight"),
                det(dim * inter, 70),
                vec![dim, inter],
            );
            put(format!("{bp}output.dense.bias"), vec![0.0; dim], vec![dim]);
        }
        if s < cfg.num_stages() - 1 {
            let dp = format!("{p}encoder.layers.{s}.downsample.");
            put(
                format!("{dp}norm.weight"),
                vec![1.0; 4 * dim],
                vec![4 * dim],
            );
            put(format!("{dp}norm.bias"), vec![0.0; 4 * dim], vec![4 * dim]);
            put(
                format!("{dp}reduction.weight"),
                det(2 * dim * 4 * dim, 80),
                vec![2 * dim, 4 * dim],
            );
        }
    }
    for &idx in &cfg.out_indices {
        let dim = cfg.stage_dim(idx - 1);
        put(
            format!("{p}hidden_states_norms.stage{idx}.weight"),
            vec![1.0; dim],
            vec![dim],
        );
        put(
            format!("{p}hidden_states_norms.stage{idx}.bias"),
            vec![0.0; dim],
            vec![dim],
        );
    }
    WeightMap::from_tensors(t)
}

#[test]
fn swin_native_forward_shapes() {
    let cfg = tiny_cfg();
    let wm = build_synth(&cfg);
    let backbone = SwinBackbone::from_weights(&wm, cfg.clone()).unwrap();

    // 16x16 image → patch(2) → 8x8 grid.
    let (h, w) = (16, 16);
    let img: Vec<f32> = (0..3 * h * w)
        .map(|i| (i % 13) as f32 * 0.02 - 0.1)
        .collect();
    let maps = backbone.forward(&img, h, w);

    // out_indices [1,2] → two feature maps.
    assert_eq!(maps.len(), 2);
    // stage1 (dim 2) before downsample: 8x8.
    assert_eq!((maps[0].c, maps[0].h, maps[0].w), (2, 8, 8));
    // stage2 (dim 4) after one downsample: 4x4.
    assert_eq!((maps[1].c, maps[1].h, maps[1].w), (4, 4, 4));
    for m in &maps {
        assert_eq!(m.data.len(), m.c * m.h * m.w);
        assert!(m.data.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn swin_native_handles_non_window_multiple() {
    // Grid not a multiple of the window size exercises the pad/crop paths.
    let cfg = tiny_cfg();
    let wm = build_synth(&cfg);
    let backbone = SwinBackbone::from_weights(&wm, cfg).unwrap();
    // 20x12 image → patch(2) → 10x6 grid (10 not a multiple of window 2? it is;
    // use 18x14 → 9x7 grid, both odd → exercises window pad + merge pad).
    let (h, w) = (18, 14);
    let img: Vec<f32> = (0..3 * h * w).map(|i| (i % 7) as f32 * 0.03).collect();
    let maps = backbone.forward(&img, h, w);
    assert_eq!(maps.len(), 2);
    // 9x7 grid → stage1 map 9x7; downsample (pad to 10x8) → 5x4.
    assert_eq!((maps[0].h, maps[0].w), (9, 7));
    assert_eq!((maps[1].h, maps[1].w), (5, 4));
    for m in &maps {
        assert!(m.data.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn swin_graph_matches_native() {
    let cfg = tiny_cfg();
    let wm = build_synth(&cfg);
    let backbone = SwinBackbone::from_weights(&wm, cfg).unwrap();
    // Odd grid (9x7) exercises window pad/crop, cyclic shift, and merge pad.
    let (h, w) = (18, 14);
    let img: Vec<f32> = (0..3 * h * w).map(|i| (i % 7) as f32 * 0.03).collect();

    let graph = backbone.forward(&img, h, w); // graph (default)
    let native = backbone.forward_native(&img, h, w); // eager oracle

    assert_eq!(graph.len(), native.len());
    for (g, n) in graph.iter().zip(&native) {
        assert_eq!((g.c, g.h, g.w), (n.c, n.h, n.w));
        let e = g
            .data
            .iter()
            .zip(&n.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e < 1e-3, "swin graph vs native max_err={e}");
    }
}
