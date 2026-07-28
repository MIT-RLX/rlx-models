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

//! Synthetic striding-Conformer encoder: build → compile → run on CPU.

use anyhow::Result;
use rlx_conformer_ctc::config::AsrConfig;
use rlx_conformer_ctc::encoder::build_encoder_hir;
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_flow::WeightSource;
use rlx_nemo::NemoConfig;
use rlx_runtime::Device;

struct SynthWeights {
    d: usize,
    ff: usize,
    nh: usize,
    hd: usize,
    k: usize,
    c: usize,
    freq_out: usize,
    num_classes: usize,
}

impl SynthWeights {
    fn new(cfg: &AsrConfig) -> Self {
        Self {
            d: cfg.d_model,
            ff: cfg.ff_dim(),
            nh: cfg.n_heads,
            hd: cfg.head_dim(),
            k: cfg.conv_kernel,
            c: cfg.subsampling_conv_channels,
            freq_out: cfg.freq_after_subsample(),
            num_classes: cfg.num_classes,
        }
    }

    fn shape_for(&self, key: &str) -> Vec<usize> {
        let (d, ff, nh, hd, k, c) = (self.d, self.ff, self.nh, self.hd, self.k, self.c);
        if key == "encoder.pre_encode.out.weight" {
            return vec![d, c * self.freq_out];
        }
        if key == "encoder.pre_encode.out.bias" {
            return vec![d];
        }
        if let Some(rest) = key.strip_prefix("encoder.pre_encode.conv.") {
            let mut it = rest.splitn(2, '.');
            let idx: usize = it.next().unwrap().parse().unwrap();
            let suffix = it.next().unwrap();
            return match (idx, suffix) {
                (0, "weight") => vec![c, 1, 3, 3],
                (_, "weight") => vec![c, c, 3, 3],
                (_, "bias") => vec![c],
                _ => vec![c],
            };
        }
        if key == "decoder.decoder_layers.0.weight" {
            return vec![self.num_classes, d, 1];
        }
        if key == "decoder.decoder_layers.0.bias" {
            return vec![self.num_classes];
        }
        let suffix = key
            .strip_prefix("encoder.layers.")
            .and_then(|s| s.split_once('.').map(|(_, r)| r))
            .unwrap_or(key);
        match suffix {
            "norm_feed_forward1.weight" | "norm_feed_forward1.bias" => vec![d],
            "feed_forward1.linear1.weight" => vec![ff, d],
            "feed_forward1.linear1.bias" => vec![ff],
            "feed_forward1.linear2.weight" => vec![d, ff],
            "feed_forward1.linear2.bias" => vec![d],
            "norm_self_att.weight" | "norm_self_att.bias" => vec![d],
            "self_attn.linear_q.weight"
            | "self_attn.linear_k.weight"
            | "self_attn.linear_v.weight"
            | "self_attn.linear_out.weight"
            | "self_attn.linear_pos.weight" => vec![d, d],
            "self_attn.linear_q.bias"
            | "self_attn.linear_k.bias"
            | "self_attn.linear_v.bias"
            | "self_attn.linear_out.bias" => vec![d],
            "self_attn.pos_bias_u" | "self_attn.pos_bias_v" => vec![nh, hd],
            "norm_conv.weight" | "norm_conv.bias" => vec![d],
            "conv.pointwise_conv1.weight" => vec![2 * d, d, 1],
            "conv.pointwise_conv1.bias" => vec![2 * d],
            "conv.depthwise_conv.weight" => vec![d, 1, k],
            "conv.depthwise_conv.bias" => vec![d],
            "conv.batch_norm.weight"
            | "conv.batch_norm.bias"
            | "conv.batch_norm.running_mean"
            | "conv.batch_norm.running_var" => vec![d],
            "conv.pointwise_conv2.weight" => vec![d, d, 1],
            "conv.pointwise_conv2.bias" => vec![d],
            "norm_feed_forward2.weight" | "norm_feed_forward2.bias" => vec![d],
            "feed_forward2.linear1.weight" => vec![ff, d],
            "feed_forward2.linear1.bias" => vec![ff],
            "feed_forward2.linear2.weight" => vec![d, ff],
            "feed_forward2.linear2.bias" => vec![d],
            "norm_out.weight" | "norm_out.bias" => vec![d],
            other => panic!("synthetic weights: unhandled key suffix {other:?} (full {key:?})"),
        }
    }
}

impl WeightSource for SynthWeights {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let shape = self.shape_for(key);
        let n: usize = shape.iter().product();
        let data: Vec<f32> = if key.ends_with("running_var") {
            vec![1.0; n]
        } else if key.ends_with("running_mean") {
            vec![0.0; n]
        } else if key.contains("norm") && key.ends_with("weight") {
            vec![1.0; n]
        } else {
            (0..n).map(|i| 0.01 * ((i % 7) as f32 - 3.0)).collect()
        };
        if transpose && shape.len() == 2 {
            let (r, cc) = (shape[0], shape[1]);
            let mut out = vec![0.0; n];
            for i in 0..r {
                for j in 0..cc {
                    out[j * r + i] = data[i * cc + j];
                }
            }
            return Ok((out, vec![cc, r]));
        }
        Ok((data, shape))
    }

    fn has(&self, _key: &str) -> bool {
        true
    }
}

fn tiny_cfg() -> AsrConfig {
    let yaml = br#"
preprocessor:
  features: 16
  n_fft: 256
  window_size: 0.025
  window_stride: 0.01
encoder:
  d_model: 32
  n_layers: 2
  n_heads: 2
  ff_expansion_factor: 2
  conv_kernel_size: 3
  subsampling: striding
  subsampling_factor: 4
  subsampling_conv_channels: 8
decoder:
  num_classes: 16
"#;
    AsrConfig::from_nemo(&NemoConfig::from_yaml_bytes(yaml).unwrap()).unwrap()
}

#[test]
fn encoder_builds_compiles_and_runs() {
    let cfg = tiny_cfg();
    let mel_frames = 64usize; // -> ~16 encoder frames with striding×4
    let mut w = SynthWeights::new(&cfg);

    let (hir, params, t) = build_encoder_hir(&cfg, &mut w, mel_frames).expect("build encoder HIR");
    assert_eq!(t, cfg.enc_frames(mel_frames));

    let built = built_from_hir(hir, params).expect("built model");
    let saved = built.params().clone();
    let mut cg = compile_built(built, Device::Cpu).expect("compile encoder");
    for (n, d) in &saved {
        cg.set_param(n, d);
    }

    let mel: Vec<f32> = (0..cfg.n_mels * mel_frames)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();
    let out = cg
        .run(&[("mel", mel.as_slice())])
        .into_iter()
        .next()
        .expect("encoder output");

    assert_eq!(out.len(), t * cfg.d_model, "output is [t, d_model]");
    assert!(out.iter().all(|x| x.is_finite()), "all outputs finite");
}
