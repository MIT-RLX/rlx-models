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

//! Cross-implementation parity: load the *same* weights into both
//! `rlx_mamba::Mamba1Block` and `burn_mamba::mamba1::Mamba1`, run the
//! parallel `forward`, and compare outputs element-wise.
//!
//! Both implementations follow the same Mamba1 algorithm (sequential
//! reference scan, depthwise causal conv1d k=4, x_proj split, softplus
//! dt, gated SiLU), so they should agree to within fp32 rounding —
//! roughly a few ULP × O(seq × d_inner × d_state) accumulated error.
//!
//! Weight conventions:
//! - Burn's `Linear` weight is `[in, out]` (`LinearConfig::new(in, out)`)
//!   and computes `y = x @ W`. rlx-mamba uses the same convention.
//! - Burn's `Conv1d` weight is `[out, in/groups, kernel]`. With
//!   `groups = d_inner`, `in/groups = 1`, so the weight collapses to
//!   `[d_inner, 1, k]` — same storage as rlx-mamba's `[d_inner, k]`.

use burn::backend::NdArray;
use burn::module::Param;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{Linear, PaddingConfig1d};
use burn::tensor::{Tensor, TensorData};
use burn_mamba::mamba1::prelude::{Mamba1, Mamba1Config as BurnMamba1Config};
use rlx_mamba::{Mamba1Block, Mamba1Config};

type B = NdArray<f32>;

fn det_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn mamba1_block_forward_matches_burn_mamba() {
    let d_model: usize = 16;
    let d_state: usize = 8;
    let d_conv: usize = 4;
    let expand: usize = 2;
    let d_inner: usize = expand * d_model;
    let dt_rank: usize = d_model.div_ceil(d_state);
    let batch: usize = 1;
    let seq: usize = 12;

    // Generate the shared weight pool. Use small scales so the SSM scan
    // doesn't saturate exp() / softplus().
    let in_proj_w = det_vec(d_model * 2 * d_inner, 0x01, 0.05);
    let conv1d_w = det_vec(d_inner * d_conv, 0x02, 0.05);
    let conv1d_b = det_vec(d_inner, 0x03, 0.02);
    let x_proj_w = det_vec(d_inner * (dt_rank + 2 * d_state), 0x04, 0.05);
    let dt_proj_w = det_vec(dt_rank * d_inner, 0x05, 0.05);
    let dt_proj_b = det_vec(d_inner, 0x06, 0.02);
    // a_log = log(arange(1..=d_state)) per channel (burn-mamba's init).
    let a_log: Vec<f32> = (0..d_inner)
        .flat_map(|_| (1..=d_state).map(|i| (i as f32).ln()))
        .collect();
    let d_skip = vec![1.0f32; d_inner];
    let out_proj_w = det_vec(d_inner * d_model, 0x07, 0.05);

    let input = det_vec(batch * seq * d_model, 0xFE, 0.1);

    // ----- rlx-mamba side -----
    let mut cfg = Mamba1Config::new(d_model);
    cfg.d_state = d_state;
    cfg.d_conv = d_conv;
    cfg.expand = expand;
    assert_eq!(cfg.d_inner(), d_inner);
    assert_eq!(cfg.dt_rank(), dt_rank);
    let block = Mamba1Block::from_weights(
        cfg,
        in_proj_w.clone(),
        vec![0.0; 2 * d_inner], // in_proj bias = 0 (burn default: bias=false)
        conv1d_w.clone(),
        conv1d_b.clone(),
        x_proj_w.clone(),
        dt_proj_w.clone(),
        dt_proj_b.clone(),
        a_log.clone(),
        d_skip.clone(),
        out_proj_w.clone(),
        vec![0.0; d_model],
    )
    .unwrap();
    let rlx_out = block.forward(&input, batch, seq).unwrap();

    // ----- burn-mamba side -----
    let device = Default::default();
    let burn_cfg = BurnMamba1Config::new(d_model)
        .with_d_state(d_state)
        .with_d_conv(d_conv)
        .with_expand(expand);
    let mut model: Mamba1<B> = burn_cfg.init(&device);

    // Overwrite each Param with our shared weights. Burn's Linear has
    // weight shape [in, out]; we store our buffer the same way, so the
    // raw TensorData layout transfers 1:1.
    model.in_proj = Linear {
        weight: Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(in_proj_w, [d_model, 2 * d_inner]),
            &device,
        )),
        bias: None,
    };
    // Conv1d weight shape [d_inner, 1, d_conv] = same flat layout as our [d_inner, d_conv].
    model.conv1d = Conv1dConfig::new(d_inner, d_inner, d_conv)
        .with_padding(PaddingConfig1d::Explicit(d_conv - 1, d_conv - 1))
        .with_groups(d_inner)
        .with_bias(true)
        .init(&device);
    model.conv1d.weight = Param::from_tensor(Tensor::<B, 3>::from_data(
        TensorData::new(conv1d_w, [d_inner, 1, d_conv]),
        &device,
    ));
    model.conv1d.bias = Some(Param::from_tensor(Tensor::<B, 1>::from_data(
        TensorData::new(conv1d_b, [d_inner]),
        &device,
    )));

    model.x_proj = Linear {
        weight: Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(x_proj_w, [d_inner, dt_rank + 2 * d_state]),
            &device,
        )),
        bias: None,
    };

    model.dt_proj = Linear {
        weight: Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(dt_proj_w, [dt_rank, d_inner]),
            &device,
        )),
        bias: Some(Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(dt_proj_b, [d_inner]),
            &device,
        ))),
    };

    model.a_log = Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(a_log, [d_inner, d_state]),
        &device,
    ));

    model.d = Param::from_tensor(Tensor::<B, 1>::from_data(
        TensorData::new(d_skip, [d_inner]),
        &device,
    ));

    model.out_proj = Linear {
        weight: Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(out_proj_w, [d_inner, d_model]),
            &device,
        )),
        bias: None,
    };

    let x = Tensor::<B, 3>::from_data(
        TensorData::new(input.clone(), [batch, seq, d_model]),
        &device,
    );
    let burn_out_tensor = model.forward(x);
    let burn_out: Vec<f32> = burn_out_tensor.into_data().convert::<f32>().to_vec().unwrap();

    // Compare.
    assert_eq!(rlx_out.len(), burn_out.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut idx = 0;
    for (i, (a, b)) in rlx_out.iter().zip(burn_out.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = abs / (a.abs().max(b.abs()).max(1e-8));
        if abs > max_abs {
            max_abs = abs;
            idx = i;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    println!(
        "rlx_mamba vs burn_mamba: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, \
         worst at idx {idx} (rlx={}, burn={})",
        rlx_out[idx], burn_out[idx]
    );
    assert!(
        max_abs < 1e-4,
        "parity failed: max_abs = {max_abs:.3e}, max_rel = {max_rel:.3e}"
    );
}
