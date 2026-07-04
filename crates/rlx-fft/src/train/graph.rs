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

//! RLX training graphs — forward + MSE loss + autodiff backward.

use crate::model::butterfly::{
    ParamSlot, append_forward_butterfly, build_butterfly_forward_graph,
    build_butterfly_inverse_graph,
};
use crate::model::config::{FftLearnConfig, TransformDir};
use crate::model::reference::roundtrip_scale;
use crate::model::twiddle::TwiddleSet;
use anyhow::Result;
use rlx_autodiff::grad_with_loss;
use rlx_compile::legalize_broadcast;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

#[derive(Debug)]
pub struct SupervisedTrainGraph {
    pub forward: Graph,
    pub backward: Graph,
    pub params: Vec<ParamSlot>,
    pub loss: NodeId,
    pub data_input: &'static str,
    pub target_input: &'static str,
}

#[derive(Debug)]
pub struct EncDecTrainGraph {
    pub forward: Graph,
    pub backward: Graph,
    pub encoder_params: Vec<ParamSlot>,
    pub decoder_params: Vec<ParamSlot>,
    pub loss: NodeId,
    pub spectrum_weight: f32,
}

fn mse_loss(g: &mut Graph, pred: NodeId, target: NodeId, flat_len: i64) -> NodeId {
    let diff = g.sub(pred, target);
    let sq = g.mul(diff, diff);
    let flat = g.reshape_(sq, vec![flat_len]);
    g.mean(flat, vec![0], false)
}

fn attach_supervised_loss(
    mut g: Graph,
    pred: NodeId,
    target_shape: Shape,
    params: Vec<ParamSlot>,
    flat_len: i64,
) -> Result<(Graph, Graph, Vec<ParamSlot>, NodeId)> {
    let target = g.input("target", target_shape);
    let loss = mse_loss(&mut g, pred, target, flat_len);
    g.set_outputs(vec![loss]);
    let loss_node = loss;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    let mut params: Vec<ParamSlot> = params
        .into_iter()
        .map(|mut p| {
            p.param = remap[&p.param];
            p
        })
        .collect();
    let wrt: Vec<NodeId> = params.iter().map(|p| p.param).collect();
    let bwd = grad_with_loss(&g, &wrt);
    let grad_ids: Vec<NodeId> = bwd.outputs[1..=params.len()].to_vec();
    for (slot, grad) in params.iter_mut().zip(grad_ids) {
        slot.grad = Some(grad);
    }
    Ok((g, bwd, params, remap[&loss_node]))
}

pub fn build_supervised_train_graph(
    cfg: &FftLearnConfig,
    dir: TransformDir,
) -> Result<SupervisedTrainGraph> {
    let built = if dir.is_forward() {
        build_butterfly_forward_graph(cfg)?
    } else {
        build_butterfly_inverse_graph(cfg)?
    };
    let (data_input, target_shape) = if dir.is_forward() {
        ("signal", Shape::new(&[cfg.batch, cfg.n_fft, 2], DType::F32))
    } else {
        (
            "spectrum",
            Shape::new(&[cfg.batch, cfg.n_fft, 2], DType::F32),
        )
    };
    let (forward, backward, params, loss) = {
        let flat_len = (cfg.batch * cfg.n_fft * 2) as i64;
        attach_supervised_loss(
            built.graph,
            built.spectrum_out,
            target_shape,
            built.params,
            flat_len,
        )?
    };
    Ok(SupervisedTrainGraph {
        forward,
        backward,
        params,
        loss,
        data_input,
        target_input: "target",
    })
}

/// Fused encoder–decoder training graph: butterfly enc + conj + butterfly dec + dual MSE in one IR.
pub fn build_encdec_train_graph(
    cfg: &FftLearnConfig,
    spectrum_weight: f32,
) -> Result<EncDecTrainGraph> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let scale = roundtrip_scale(n);
    let mut g = Graph::new("fft_encdec_train");
    let signal_in = g.input("signal", Shape::new(&[batch, n], f));

    let zeros = g.sub(signal_in, signal_in);
    let re = g.reshape_(signal_in, vec![batch as i64, n as i64, 1]);
    let im = g.reshape_(zeros, vec![batch as i64, n as i64, 1]);
    let enc_state = g.concat_(vec![re, im], 2);
    let (mut encoder_params, spectrum) =
        append_forward_butterfly(&mut g, cfg, enc_state, TwiddleSet::Encoder)?;

    let spec_re = g.narrow_(spectrum, 2, 0, 1);
    let spec_im = g.narrow_(spectrum, 2, 1, 1);
    let conj_im = g.neg(spec_im);
    let dec_in = g.concat_(vec![spec_re, conj_im], 2);
    let (mut decoder_params, dec_state) =
        append_forward_butterfly(&mut g, cfg, dec_in, TwiddleSet::Decoder)?;
    let out_re = g.narrow_(dec_state, 2, 0, 1);
    let out_im = g.narrow_(dec_state, 2, 1, 1);
    let out_conj_im = g.neg(out_im);
    let recovered = g.concat_(vec![out_re, out_conj_im], 2);

    let scale_node = g.add_node(
        Op::Constant {
            data: scale.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], f),
    );
    let sig_re = g.reshape_(signal_in, vec![batch as i64, n as i64, 1]);
    let target_re = g.mul(sig_re, scale_node);
    let target_im = g.sub(sig_re, sig_re);
    let target = g.concat_(vec![target_re, target_im], 2);
    let flat_len = (batch * n * 2) as i64;
    let mut loss = mse_loss(&mut g, recovered, target, flat_len);

    if spectrum_weight > 0.0 {
        let ref_spec = g.input("target_spectrum", Shape::new(&[batch, n, 2], f));
        let spec_w = g.add_node(
            Op::Constant {
                data: spectrum_weight.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], f),
        );
        let spec_loss = mse_loss(&mut g, spectrum, ref_spec, flat_len);
        let weighted = g.mul(spec_w, spec_loss);
        loss = g.add(loss, weighted);
    }

    g.set_outputs(vec![loss]);
    let loss_node = loss;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    for p in &mut encoder_params {
        p.param = remap[&p.param];
    }
    for p in &mut decoder_params {
        p.param = remap[&p.param];
    }
    let mut params: Vec<ParamSlot> = encoder_params
        .iter()
        .chain(decoder_params.iter())
        .cloned()
        .collect();
    let wrt: Vec<NodeId> = params.iter().map(|p| p.param).collect();
    let bwd = grad_with_loss(&g, &wrt);
    let grad_ids: Vec<NodeId> = bwd.outputs[1..=params.len()].to_vec();
    for (slot, grad) in params.iter_mut().zip(grad_ids) {
        slot.grad = Some(grad);
    }

    let enc_len = encoder_params.len();
    Ok(EncDecTrainGraph {
        forward: g,
        backward: bwd,
        encoder_params: params[..enc_len].to_vec(),
        decoder_params: params[enc_len..].to_vec(),
        loss: remap[&loss_node],
        spectrum_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::FftLearnConfig;

    #[test]
    fn supervised_train_graph_builds() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        build_supervised_train_graph(&cfg, TransformDir::Forward).expect("fft train graph");
        build_supervised_train_graph(&cfg, TransformDir::Inverse).expect("ifft train graph");
    }

    #[test]
    fn encdec_train_graph_builds() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        build_encdec_train_graph(&cfg, 1.0).expect("encdec train graph");
    }
}
