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

//! Quantized expert stores: the host expert path over a store must compute
//! exactly the MLP its dequantized weights define. Quantization error is
//! measured elsewhere (`examples/expert_formats.rs`); this checks the reads,
//! the block kernels and the routing sum.

use rlx_flint::experts::{ExpertWeights, add_routed};
use rlx_flint::moe::expert_key;
use rlx_flint::quant::{ExpertScheme, QuantStore, write_store};
use rlx_flint::synth::{SynthDtype, tiny_config_json, write_checkpoint};
use rlx_flint::{Checkpoint, FlintConfig};

/// The tiny model, widened so expert rows are whole 32-weight blocks.
fn config() -> serde_json::Value {
    let mut c = tiny_config_json();
    c["hidden_size"] = 64.into();
    c["head_dim"] = 16.into();
    c["intermediate_size"] = 96.into();
    c["moe_intermediate_size"] = 32.into();
    c["shared_expert_intermediate_size"] = 32.into();
    c
}

fn dequantized(store: &QuantStore, name: &str) -> (Vec<f64>, usize, usize) {
    let e = store.entry(name).unwrap();
    let rb = store.scheme.row_bytes(e.cols);
    let mut raw = vec![0u8; e.rows * rb];
    store.read_rows(e, 0, &mut raw).unwrap();
    let mut w = vec![0f32; e.rows * e.cols];
    for (row, out) in raw.chunks_exact(rb).zip(w.chunks_exact_mut(e.cols)) {
        store.scheme.dequantize_row(row, out);
    }
    (w.into_iter().map(f64::from).collect(), e.rows, e.cols)
}

/// `W x` for `W` `[rows, cols]`.
fn matvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
        .collect()
}

#[test]
fn a_store_computes_the_mlp_of_its_dequantized_weights() {
    for dtype in [SynthDtype::F32, SynthDtype::BF16] {
        let ck = tempfile::tempdir().unwrap();
        write_checkpoint(ck.path(), &config(), dtype, 2, 7).unwrap();
        let cfg = FlintConfig::from_value(&config()).unwrap();
        let ckpt = Checkpoint::open(ck.path()).unwrap();
        let layer = cfg.moe_layers().next().unwrap();
        let h = cfg.hidden_size;
        let x: Vec<f32> = (0..2 * h)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.07)
            .collect();
        let ids = [1.0f32, 3.0, 5.0, 3.0, 0.0, 7.0];
        let w = [0.5f32, 0.3, 0.2, 0.6, 0.25, 0.15];
        for scheme in [ExpertScheme::Q8_0, ExpertScheme::Q4_0] {
            let dir = tempfile::tempdir().unwrap();
            write_store(&ckpt, &cfg, scheme, dir.path()).unwrap();
            let store = QuantStore::open(dir.path()).unwrap();
            let mut got = vec![0f32; 2 * h];
            add_routed(
                ExpertWeights::Quantized(&store),
                &cfg,
                layer,
                &x,
                &ids,
                &w,
                2,
                &mut got,
            )
            .unwrap();
            for r in 0..2 {
                let xr: Vec<f64> = x[r * h..(r + 1) * h].iter().map(|&v| v as f64).collect();
                let mut want = vec![0f64; h];
                for k in 0..3 {
                    let e = ids[r * 3 + k] as usize;
                    let (g, gi, gc) = dequantized(&store, &expert_key(layer, e, "gate_proj"));
                    let (u, ui, uc) = dequantized(&store, &expert_key(layer, e, "up_proj"));
                    let (d, di, dc) = dequantized(&store, &expert_key(layer, e, "down_proj"));
                    let (gx, ux) = (matvec(&g, gi, gc, &xr), matvec(&u, ui, uc, &xr));
                    let act: Vec<f64> = gx
                        .iter()
                        .zip(&ux)
                        .map(|(&a, &b)| a / (1.0 + (-a).exp()) * b)
                        .collect();
                    for (o, y) in want.iter_mut().zip(matvec(&d, di, dc, &act)) {
                        *o += w[r * 3 + k] as f64 * y;
                    }
                }
                let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                for (j, (&g, &wv)) in got[r * h..(r + 1) * h].iter().zip(&want).enumerate() {
                    assert!(
                        (g as f64 - wv).abs() <= 1e-5 * scale.max(1.0),
                        "{dtype:?} {scheme:?} row {r} col {j}: {g} vs {wv}"
                    );
                }
            }
        }
    }
}
