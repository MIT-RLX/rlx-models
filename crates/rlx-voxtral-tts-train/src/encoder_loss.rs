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

//! Combined encoder training loss graph.

use rlx_autodiff::grad_with_loss;
use rlx_compile::legalize_broadcast;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_voxtral_tts::config::CodecArgs;

use crate::codec_graph::{CodecForwardGraph, CodecGraphLayout, ParamSlot, build_codec_recon_graph};

#[derive(Debug)]
pub struct EncoderTrainGraph {
    pub forward: Graph,
    pub backward: Graph,
    pub loss: NodeId,
    pub params: Vec<ParamSlot>,
    pub d_output: NodeId,
    pub fwd: CodecForwardGraph,
}

pub fn build_encoder_train_graph(
    cfg: &CodecArgs,
    layout: &CodecGraphLayout,
    mel_weight: f32,
    stft_weight: f32,
    commitment_delta: f32,
    diversity_weight: f32,
    gan_weight: f32,
    asr_weight: f32,
) -> EncoderTrainGraph {
    let mut fwd = build_codec_recon_graph(cfg, layout).expect("codec forward");
    let param_slots: Vec<ParamSlot> = fwd.params.clone();
    let mut g = std::mem::replace(&mut fwd.graph, Graph::new("empty"));
    let f = DType::F32;

    let recon = fwd.recon_wav;
    let target = g.input(
        "target_wav",
        Shape::new(&[layout.patch_size, layout.wav_t], f),
    );
    let diff = g.sub(recon, target);
    let abs = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Abs),
        vec![diff],
        g.shape(diff).clone(),
    );
    let l1 = mean_all(&mut g, abs);

    let mel_in = g.input("mel_basis", Shape::new(&[64, layout.wav_t.max(1)], f));
    let recon_t = g.transpose_(recon, vec![1, 0]);
    let target_t = g.transpose_(target, vec![1, 0]);
    let pred_mel = g.mm(mel_in, recon_t);
    let tgt_mel = g.mm(mel_in, target_t);
    let mel_diff = g.sub(pred_mel, tgt_mel);
    let mel_abs = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Abs),
        vec![mel_diff],
        g.shape(mel_diff).clone(),
    );
    let mel_mean = mean_all(&mut g, mel_abs);
    let mel_w = scalar(&mut g, mel_weight);
    let mel_loss = g.mul(mel_w, mel_mean);

    let stft_in = g.input("stft_basis", Shape::new(&[128, layout.wav_t.max(1)], f));
    let pred_stft = g.mm(stft_in, recon_t);
    let tgt_stft = g.mm(stft_in, target_t);
    let stft_diff = g.sub(pred_stft, tgt_stft);
    let stft_abs = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Abs),
        vec![stft_diff],
        g.shape(stft_diff).clone(),
    );
    let stft_mean = mean_all(&mut g, stft_abs);
    let stft_w = scalar(&mut g, stft_weight);
    let stft_loss = g.mul(stft_w, stft_mean);

    let commit_diff = g.sub(fwd.latent, fwd.quantized);
    let commit_sq = g.mul(commit_diff, commit_diff);
    let commit_mean = mean_all(&mut g, commit_sq);
    let commit_w = scalar(&mut g, commitment_delta);
    let commit = g.mul(commit_w, commit_mean);

    let codebook = param_slots
        .iter()
        .find(|p| p.name == "quantizer.semantic_codebook.embedding")
        .map(|p| p.param)
        .expect("semantic codebook param");
    let sem = g.narrow_(fwd.latent, 0, 0, cfg.semantic_dim);
    let sem_t = g.transpose_(sem, vec![1, 0]);
    let cb_t = g.transpose_(codebook, vec![1, 0]);
    let dist = g.mm(sem_t, cb_t);
    let probs = g.sm(dist, -1);
    let log_p = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Log),
        vec![probs],
        g.shape(probs).clone(),
    );
    let pl = g.mul(probs, log_p);
    let ent_axis = g.mean(pl, vec![1], true);
    let ent_scalar = g.mean(ent_axis, vec![0], false);
    let entropy = g.neg(ent_scalar);
    let div_w = scalar(&mut g, diversity_weight);
    let div_loss = g.mul(div_w, entropy);

    let gan_in = g.input("d_fake", Shape::new(&[1], f));
    let gan_w = scalar(&mut g, gan_weight);
    let gan_loss = g.mul(gan_w, gan_in);

    let asr_in = g.input("asr_mse", Shape::new(&[1], f));
    let asr_w = scalar(&mut g, asr_weight);
    let asr_loss = g.mul(asr_w, asr_in);

    let mut loss = g.add(l1, mel_loss);
    loss = g.add(loss, stft_loss);
    loss = g.add(loss, commit);
    loss = g.add(loss, div_loss);
    loss = g.add(loss, gan_loss);
    loss = g.add(loss, asr_loss);
    g.set_outputs(vec![loss]);
    let loss_node = loss;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    let mut params: Vec<ParamSlot> = param_slots
        .into_iter()
        .filter(|p| p.trainable)
        .map(|mut p| {
            p.param = remap[&p.param];
            p
        })
        .collect();

    let wrt: Vec<NodeId> = params.iter().map(|p| p.param).collect();
    let bwd = grad_with_loss(&g, &wrt);
    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output");
    let grad_ids: Vec<NodeId> = bwd.outputs[1..=params.len()].to_vec();
    for (slot, grad) in params.iter_mut().zip(grad_ids) {
        slot.grad = Some(grad);
    }

    fwd.graph = g.clone();
    EncoderTrainGraph {
        forward: g,
        backward: bwd,
        loss: remap[&loss_node],
        params,
        d_output,
        fwd,
    }
}

fn mean_all(g: &mut Graph, x: NodeId) -> NodeId {
    let m0 = g.mean(x, vec![0], true);
    g.mean(m0, vec![1], false)
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

#[cfg(test)]
mod grad_tests {
    use super::*;
    use rlx_autodiff::prepare_graph_for_ad;

    fn sample_codec() -> CodecArgs {
        CodecArgs {
            channels: 1,
            sampling_rate: 24000,
            pretransform_patch_size: 240,
            patch_proj_kernel_size: 7,
            semantic_codebook_size: 128,
            semantic_dim: 256,
            acoustic_codebook_size: 21,
            acoustic_dim: 36,
            dim: 1024,
            hidden_dim: 4096,
            head_dim: 128,
            n_heads: 8,
            n_kv_heads: 8,
            attn_sliding_window_size: 16,
            encoder_transformer_lengths_str: "1,1".into(),
            encoder_convs_kernels_str: "4,3".into(),
            encoder_convs_strides_str: "2,1".into(),
            decoder_transformer_lengths_str: "1,1".into(),
            decoder_convs_kernels_str: "3,4".into(),
            decoder_convs_strides_str: "1,2".into(),
        }
    }

    fn check_topo(g: &Graph, label: &str) {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for node in g.nodes() {
            for inp in &node.inputs {
                assert!(
                    seen.contains(inp),
                    "{label}: node {} op {:?} missing input {}",
                    node.id.0,
                    std::mem::discriminant(&node.op),
                    inp.0
                );
            }
            seen.insert(node.id);
        }
    }

    #[test]
    fn encoder_train_graph_topo() {
        let cfg = sample_codec();
        let layout = CodecGraphLayout::new(&cfg, 8);
        let train = build_encoder_train_graph(&cfg, &layout, 1.0, 1.0, 0.1, 0.1, 0.0, 0.0);
        check_topo(&train.forward, "forward");
        let prep = prepare_graph_for_ad(train.forward.clone());
        check_topo(&prep, "prepared");
    }
}
