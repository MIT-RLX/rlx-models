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

//! MTP speculative decoding for Qwen3.5.
//!
//! **Two-phase decode:** `propose` prefills on `context`, runs the MTP
//! draft loop on a checkpointed cache, then restores the checkpoint so
//! rejected tokens never touch GDN conv/SSM state. After accept/reject,
//! `SpecDecoder::step` calls [`Speculator::commit`] on draft and target
//! so GDN state advances only for accepted tokens.
//!
//! **Target cache reuse:** [`Qwen35TrunkTarget`] keeps decode state across
//! spec rounds when `context` extends the prior prefix — verify skips
//! reprefill when the synced prefix matches.

use crate::runner::Qwen35Runner;
use anyhow::Result;
use rlx_qwen3::sampling::softmax_logits;
use rlx_runtime::spec_decode::{DraftProposal, SpecDecoder, Speculator, VerifyResult};

fn truncate_logits(logits: &[f32], vocab: usize) -> &[f32] {
    &logits[..logits.len().min(vocab)]
}

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Draft proposer: samples from MTP head logits when available.
pub struct Qwen35MtpDraft {
    inner: Qwen35Runner,
}

impl Qwen35MtpDraft {
    pub fn new(runner: Qwen35Runner) -> Self {
        Self { inner: runner }
    }

    pub fn runner(&self) -> &Qwen35Runner {
        &self.inner
    }
    pub fn runner_mut(&mut self) -> &mut Qwen35Runner {
        &mut self.inner
    }
}

impl Speculator for Qwen35MtpDraft {
    fn propose(&mut self, context: &[u32], n: usize) -> DraftProposal {
        self.propose_inner(context, n)
            .expect("Qwen35MtpDraft::propose failed")
    }

    fn verify(&mut self, _context: &[u32], _proposed: &[u32]) -> VerifyResult {
        VerifyResult { probs: vec![] }
    }

    fn commit(&mut self, context: &[u32], accepted: &[u32]) {
        if accepted.is_empty() {
            return;
        }
        self.commit_inner(context, accepted)
            .expect("Qwen35MtpDraft::commit failed");
    }
}

impl Qwen35MtpDraft {
    fn propose_inner(&mut self, context: &[u32], n: usize) -> Result<DraftProposal> {
        if n == 0 {
            return Ok(DraftProposal {
                tokens: vec![],
                probs: vec![],
            });
        }

        self.inner.reset_decode_cache();
        let seed = self.inner.prefill_seed_for_decode(context)?;
        let checkpoint = self.inner.decode_cache_checkpoint();

        let vocab = self.inner.lm_vocab_size();
        let mut logits = seed.mtp_logits.unwrap_or(seed.trunk_logits);
        let mut tokens = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);

        for i in 0..n {
            let row = truncate_logits(&logits, vocab);
            let p = softmax_logits(row);
            let tok = argmax(row);
            tokens.push(tok);
            probs.push(p);
            if i + 1 < n {
                logits = self.inner.decode_get_mtp_logits(tok)?;
            }
        }

        self.inner.restore_decode_cache(checkpoint);
        Ok(DraftProposal { tokens, probs })
    }

    fn commit_inner(&mut self, context: &[u32], accepted: &[u32]) -> Result<()> {
        self.inner.reset_decode_cache();
        self.inner.prefill_seed_for_decode(context)?;
        self.inner.commit_decode_tokens(accepted)?;
        Ok(())
    }
}

/// Verification target: trunk LM logits with persistent decode cache.
pub struct Qwen35TrunkTarget {
    inner: Qwen35Runner,
    /// Prefix the runner decode cache + `pending_logits` correspond to.
    synced_context: Vec<u32>,
    /// Logits predicting the token after `synced_context` (when warm).
    pending_logits: Option<Vec<f32>>,
}

impl Qwen35TrunkTarget {
    pub fn new(runner: Qwen35Runner) -> Self {
        Self {
            inner: runner,
            synced_context: Vec::new(),
            pending_logits: None,
        }
    }

    pub fn runner(&self) -> &Qwen35Runner {
        &self.inner
    }
    pub fn runner_mut(&mut self) -> &mut Qwen35Runner {
        &mut self.inner
    }
}

impl Speculator for Qwen35TrunkTarget {
    fn propose(&mut self, _context: &[u32], _n: usize) -> DraftProposal {
        DraftProposal {
            tokens: vec![],
            probs: vec![],
        }
    }

    fn verify(&mut self, context: &[u32], proposed: &[u32]) -> VerifyResult {
        self.verify_inner(context, proposed)
            .expect("Qwen35TrunkTarget::verify failed")
    }

    fn commit(&mut self, context: &[u32], accepted: &[u32]) {
        if accepted.is_empty() {
            return;
        }
        self.commit_inner(context, accepted)
            .expect("Qwen35TrunkTarget::commit failed");
    }
}

impl Qwen35TrunkTarget {
    fn seed_from_context(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        self.inner.reset_decode_cache();
        let seed = self.inner.prefill_seed_for_decode(context)?;
        self.synced_context = context.to_vec();
        self.pending_logits = Some(seed.trunk_logits.clone());
        Ok(seed.trunk_logits)
    }

    fn verify_inner(&mut self, context: &[u32], proposed: &[u32]) -> Result<VerifyResult> {
        let n = proposed.len();
        if n == 0 {
            return Ok(VerifyResult { probs: vec![] });
        }

        let warm = self.synced_context == context
            && self.pending_logits.is_some()
            && self.inner.decode_cache_checkpoint().is_some();

        let start_logits = if warm {
            self.pending_logits.clone().unwrap()
        } else {
            self.seed_from_context(context)?
        };

        let checkpoint = self.inner.decode_cache_checkpoint();
        let mut probs = Vec::with_capacity(n);
        let start_for_restore = start_logits.clone();
        let mut logits = start_logits;
        for i in 0..n {
            probs.push(softmax_logits(&logits));
            if i + 1 < n {
                logits = self.inner.decode_get_logits(proposed[i])?;
            }
        }
        self.inner.restore_decode_cache(checkpoint);
        self.pending_logits = Some(start_for_restore);
        Ok(VerifyResult { probs })
    }

    fn commit_inner(&mut self, context: &[u32], accepted: &[u32]) -> Result<()> {
        if self.synced_context != context {
            self.seed_from_context(context)?;
        }
        let mut logits = self
            .pending_logits
            .take()
            .ok_or_else(|| anyhow::anyhow!("qwen35 target commit without pending logits"))?;
        for &tok in accepted {
            logits = self.inner.decode_get_logits(tok)?;
        }
        self.synced_context = context.to_vec();
        self.synced_context.extend_from_slice(accepted);
        self.pending_logits = Some(logits);
        Ok(())
    }
}

/// Convenience: one speculative-decoding round (MTP draft + trunk verify).
pub fn speculative_decode_round(
    draft: Qwen35MtpDraft,
    target: Qwen35TrunkTarget,
    context: &[u32],
    n: usize,
    seed: u64,
) -> Vec<u32> {
    let mut dec = SpecDecoder::new(draft, target, n, seed);
    dec.step(context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::Qwen35RunnerBuilder;
    use crate::{
        MatWeight, Qwen35Config, Qwen35FullAttnLayer, Qwen35LayerFfn, Qwen35LinearLayer,
        Qwen35MtpLayer, Qwen35TrunkLayer, Qwen35Weights,
    };
    use rlx_runtime::Device;

    fn mat(data: Vec<f32>) -> MatWeight {
        MatWeight::F32(data)
    }

    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
    }

    fn tiny_cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 4,
            nextn_predict_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            key_length: 4,
            value_length: 4,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_dim_count: 4,
            rope_dim_sections: vec![],
            full_attention_interval: 3,
            ssm_conv_kernel: 4,
            ssm_group_count: 2,
            ssm_inner_size: 8,
            ssm_state_size: 4,
            ssm_time_step_rank: 2,
            tie_word_embeddings: true,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    fn dense_ffn(cfg: &Qwen35Config) -> Qwen35LayerFfn {
        let n_embd = cfg.hidden_size;
        let n_ff = cfg.intermediate_size;
        Qwen35LayerFfn::Dense {
            gate: mat(ramp(n_embd * n_ff, 0.01)),
            down: mat(ramp(n_ff * n_embd, 0.01)),
            up: mat(ramp(n_embd * n_ff, 0.01)),
        }
    }

    fn linear_layer(cfg: &Qwen35Config) -> Qwen35LinearLayer {
        let n_embd = cfg.hidden_size;
        let n_state = cfg.ssm_state_size;
        let n_k_heads = cfg.ssm_group_count;
        let n_v_heads = cfg.ssm_time_step_rank;
        let key_dim = n_state * n_k_heads;
        let value_dim = n_state * n_v_heads;
        let conv_channels = key_dim * 2 + value_dim;
        let _n_ff = cfg.intermediate_size;
        Qwen35LinearLayer {
            attn_norm: vec![1.0; n_embd],
            attn_post_norm: vec![1.0; n_embd],
            attn_qkv: mat(ramp(n_embd * conv_channels, 0.01)),
            attn_gate: mat(ramp(n_embd * value_dim, 0.01)),
            ssm_conv1d: ramp(cfg.ssm_conv_kernel * conv_channels, 0.02),
            ssm_dt_bias: ramp(n_v_heads, 0.05),
            ssm_a: vec![-1.0; n_v_heads],
            ssm_beta: mat(ramp(n_embd * n_v_heads, 0.01)),
            ssm_alpha: mat(ramp(n_embd * n_v_heads, 0.01)),
            ssm_norm: vec![1.0; n_state],
            ssm_out: mat(ramp(value_dim * n_embd, 0.01)),
            ffn: dense_ffn(cfg),
        }
    }

    fn full_attn_layer(cfg: &Qwen35Config) -> Qwen35FullAttnLayer {
        let n_embd = cfg.hidden_size;
        let n_head = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.key_length;
        let _n_ff = cfg.intermediate_size;
        Qwen35FullAttnLayer {
            attn_norm: vec![1.0; n_embd],
            attn_post_norm: vec![1.0; n_embd],
            attn_q_gate: mat(ramp(n_embd * n_head * hd * 2, 0.01)),
            attn_k: mat(ramp(n_embd * n_kv * hd, 0.01)),
            attn_v: mat(ramp(n_embd * n_kv * hd, 0.01)),
            attn_output: mat(ramp(n_head * hd * n_embd, 0.01)),
            attn_q_norm: vec![1.0; hd],
            attn_k_norm: vec![1.0; hd],
            ffn: dense_ffn(cfg),
        }
    }

    fn synth_weights(cfg: &Qwen35Config) -> Qwen35Weights {
        let n_embd = cfg.hidden_size;
        let n_vocab = cfg.vocab_size;
        let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let interval = cfg.full_attention_interval.max(1);
        let mut trunk = Vec::new();
        for il in 0..n_main {
            let is_full = ((il + 1) % interval) == 0;
            trunk.push(if is_full {
                Qwen35TrunkLayer::FullAttn(full_attn_layer(cfg))
            } else {
                Qwen35TrunkLayer::Linear(linear_layer(cfg))
            });
        }
        Qwen35Weights {
            token_embd: std::sync::Arc::from(ramp(n_vocab * n_embd, 0.001)),
            output_norm: vec![1.0; n_embd],
            output: None,
            token_embd_lm: None,
            trunk_layers: trunk,
            mtp_layers: vec![Qwen35MtpLayer {
                base: full_attn_layer(cfg),
                eh_proj: mat(ramp(2 * n_embd * n_embd, 0.01)),
                enorm: vec![1.0; n_embd],
                hnorm: vec![1.0; n_embd],
                embed_tokens: None,
                shared_head_head: None,
                shared_head_norm: None,
            }],
        }
    }

    fn make_target(cfg: &Qwen35Config) -> Qwen35TrunkTarget {
        Qwen35TrunkTarget::new(
            Qwen35RunnerBuilder::default()
                .inline_weights(cfg.clone(), synth_weights(cfg))
                .device(Device::Cpu)
                .max_seq(32)
                .last_logits_only(true)
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn trunk_target_verify_matches_cold_repredict() {
        let cfg = tiny_cfg();
        let mut target = make_target(&cfg);
        let context = vec![1u32, 2, 3];
        let proposed = vec![4u32, 5];

        let warm = target.verify(&context, &proposed);
        target.inner.reset_decode_cache();
        target.synced_context.clear();
        target.pending_logits = None;
        let cold = target.verify(&context, &proposed);
        assert_eq!(warm.probs.len(), cold.probs.len());
        for (a, b) in warm.probs.iter().zip(&cold.probs) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-5, "verify prob mismatch");
            }
        }
    }

    #[test]
    fn trunk_target_reuses_cache_after_commit() {
        let cfg = tiny_cfg();
        let mut target = make_target(&cfg);
        let context = vec![1u32, 2, 3];
        let _ = target.verify(&context, &[4, 5]);
        target.commit(&context, &[4]);
        let extended: Vec<u32> = context.iter().copied().chain(std::iter::once(4)).collect();
        assert_eq!(target.synced_context, extended);
        let _ = target.verify(&extended, &[6]);
        assert_eq!(target.synced_context, extended);
    }
}
