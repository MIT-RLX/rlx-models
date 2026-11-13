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

//! Synthetic-weight checks (no checkpoint needed) plus a check of the
//! real `Teklia/pylaia-rimes` `model` file when it is present.

use rlx_pylaia::weights::TensorTable;
use rlx_pylaia::{Activation, ConvBlockConfig, PylaiaConfig, PylaiaRunner, PylaiaWeights};
use rlx_pylaia::{RnnKind, Sequencer, SymbolTable};
use rlx_runtime::Device;
use std::path::Path;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

fn tiny(rnn: RnnKind, sequencer: Sequencer) -> PylaiaConfig {
    let block = |c, pool| ConvBlockConfig {
        out_channels: c,
        kernel: [3, 3],
        stride: [1, 1],
        dilation: [1, 1],
        pool,
        batchnorm: true,
        activation: Activation::LeakyRelu(0.01),
    };
    PylaiaConfig {
        in_channels: 1,
        blocks: vec![block(4, Some([2, 2])), block(6, Some([2, 2]))],
        sequencer,
        rnn,
        rnn_units: 5,
        rnn_layers: 2,
        num_labels: 7,
    }
}

/// Every tensor a `LaiaCRNN` of `cfg` holds, filled deterministically.
fn table(cfg: &PylaiaConfig) -> TensorTable {
    let mut seed = 7u64;
    let mut t = TensorTable::new();
    let mut put = |name: String, shape: Vec<usize>, positive: bool| {
        let n = shape.iter().product();
        let data = (0..n)
            .map(|_| {
                let v = lcg(&mut seed);
                if positive { 0.5 + v.abs() } else { v }
            })
            .collect();
        t.insert(name, (data, shape));
    };
    let mut cin = cfg.in_channels;
    for (i, b) in cfg.blocks.iter().enumerate() {
        let c = b.out_channels;
        put(format!("conv.{i}.conv.weight"), vec![c, cin, 3, 3], false);
        for (k, pos) in [("weight", true), ("bias", false), ("running_mean", false)] {
            put(format!("conv.{i}.batchnorm.{k}"), vec![c], pos);
        }
        put(format!("conv.{i}.batchnorm.running_var"), vec![c], true);
        cin = c;
    }
    let u = cfg.rnn_units;
    let rows = cfg.rnn.gates() * u;
    for l in 0..cfg.rnn_layers {
        let input = if l == 0 { cfg.frame_features() } else { 2 * u };
        for s in ["", "_reverse"] {
            put(format!("rnn.weight_ih_l{l}{s}"), vec![rows, input], false);
            put(format!("rnn.weight_hh_l{l}{s}"), vec![rows, u], false);
            put(format!("rnn.bias_ih_l{l}{s}"), vec![rows], false);
            put(format!("rnn.bias_hh_l{l}{s}"), vec![rows], false);
        }
    }
    put("linear.weight".into(), vec![cfg.num_labels, 2 * u], false);
    put("linear.bias".into(), vec![cfg.num_labels], false);
    t
}

fn symbols(n: usize) -> SymbolTable {
    let mut s = String::from("<ctc> 0\n");
    for i in 1..n {
        s.push_str(&format!("{} {i}\n", char::from(b'a' + i as u8)));
    }
    SymbolTable::parse(&s).unwrap()
}

fn run(cfg: PylaiaConfig, height: usize, width: usize) -> Vec<f32> {
    let w = PylaiaWeights::from_table(&cfg, table(&cfg)).unwrap();
    let labels = cfg.num_labels;
    let mut r = PylaiaRunner::from_parts(cfg, w, symbols(labels), Device::Cpu);
    let x: Vec<f32> = (0..height * width)
        .map(|i| (i % 13) as f32 / 13.0)
        .collect();
    r.logits(&x, height, width).unwrap()
}

#[test]
fn lstm_fixed_sequencer_shapes() {
    // 16 rows → two 2× pools → 4 = the `none-4` sequencer height.
    let logits = run(tiny(RnnKind::Lstm, Sequencer::Fixed(4)), 16, 24);
    assert_eq!(logits.len(), 6 * 7);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn gru_avgpool_sequencer_shapes() {
    // 20 rows → 5 after pooling, adaptively averaged to 3.
    let logits = run(tiny(RnnKind::Gru, Sequencer::AvgPool(3)), 20, 16);
    assert_eq!(logits.len(), 4 * 7);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn fixed_sequencer_rejects_wrong_height() {
    let cfg = tiny(RnnKind::Lstm, Sequencer::Fixed(4));
    let w = PylaiaWeights::from_table(&cfg, table(&cfg)).unwrap();
    assert!(rlx_pylaia::graph::build(&cfg, &w, 20, 24).is_err());
}

#[test]
fn unread_checkpoint_tensors_are_an_error() {
    let cfg = tiny(RnnKind::Lstm, Sequencer::Fixed(4));
    let mut t = table(&cfg);
    t.insert("extra.weight".into(), (vec![0.0], vec![1]));
    assert!(PylaiaWeights::from_table(&cfg, t).is_err());
}

/// The RIMES release's `model` file decodes to the documented architecture.
#[test]
fn rimes_model_file() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/ocr/pylaia-rimes");
    if !dir.join("model").exists() {
        eprintln!(
            "skipping: {} not present (`just fetch-pylaia rimes`)",
            dir.display()
        );
        return;
    }
    let cfg = PylaiaConfig::from_model_file(&dir.join("model")).unwrap();
    assert_eq!(cfg, PylaiaConfig::teklia_default(100));
}
