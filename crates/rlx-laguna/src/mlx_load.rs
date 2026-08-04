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

//! Load a **Laguna mlx-community checkpoint dir** (HF `config.json` + affine
//! safetensors) into [`LagunaPackedWeights`], the mirror of the GGUF
//! [`LagunaPackedWeights::from_loader`] path. Quantized linears stay packed as
//! [`MatWeight::PackedMlx`] (dequantized transiently one matrix at a time by the
//! host forward — never a full-model F32 expand); norms / router bias load as
//! native F32.
//!
//! Tensor names follow the mlx-community Laguna layout, under the
//! `language_model.` prefix (Laguna is a VLM; this is its text tower):
//!
//! ```text
//! language_model.model.embed_tokens.weight, …model.norm.weight, language_model.lm_head.weight
//! per layer …model.layers.{i}.:
//!   input_layernorm / post_attention_layernorm
//!   self_attn.{q,k,v,o}_proj + q_norm / k_norm + g_proj (attn gate)
//!   dense: mlp.{gate,up,down}_proj
//!   moe:   mlp.gate.proj (router) + mlp.gate.e_score_correction_bias
//!          + mlp.shared_expert.{gate,up,down}_proj
//!          + mlp.switch_mlp.{gate,up,down}_proj (stacked [E,out,in])
//! ```

use crate::config::LagunaConfig;
use crate::packed::{
    LagunaPackedFfn, LagunaPackedLayer, LagunaPackedWeights, MatWeight, PackedParams,
};
use anyhow::{Context, Result, anyhow};
use rlx_core::weight_loader::{MlxLoader, WeightLoader};
use std::collections::HashSet;

const LM: &str = "language_model.model";
const HEAD: &str = "language_model.lm_head.weight";

/// Required quantized linear → [`MatWeight::PackedMlx`] (stays packed).
fn packed(loader: &mut MlxLoader, key: &str) -> Result<MatWeight> {
    let p = loader
        .take_packed_mlx(key)?
        .ok_or_else(|| anyhow!("laguna mlx: `{key}` expected packed-affine (got dense/absent)"))?;
    Ok(MatWeight::PackedMlx(Box::new(p)))
}

/// Linear that may be packed (large) or native float (small routers): prefer
/// packed, else load F32.
fn packed_or_f32(loader: &mut MlxLoader, key: &str) -> Result<MatWeight> {
    if let Some(p) = loader.take_packed_mlx(key)? {
        return Ok(MatWeight::PackedMlx(Box::new(p)));
    }
    let (data, _shape) = loader
        .take(key)
        .with_context(|| format!("laguna mlx: `{key}` not packed and not native-float"))?;
    Ok(MatWeight::F32(data))
}

/// Native-float side tensor (RMSNorm gain, router bias) → F32.
fn f32_vec(loader: &mut MlxLoader, key: &str) -> Result<Vec<f32>> {
    let (data, _shape) = loader
        .take(key)
        .with_context(|| format!("laguna mlx norm `{key}`"))?;
    Ok(data)
}

/// Build [`LagunaPackedWeights`] from a Laguna mlx-community directory. `cfg`
/// comes from [`LagunaConfig::from_json_path`] on the same dir's `config.json`.
pub fn load_mlx_weights(dir: &str, cfg: &LagunaConfig) -> Result<LagunaPackedWeights> {
    let mut loader = MlxLoader::open(dir)?;
    let avail: HashSet<String> = loader.remaining_keys().into_iter().collect();
    let has = |k: &str| avail.contains(k);

    let token_embd = packed(&mut loader, &format!("{LM}.embed_tokens.weight"))?;
    let output_norm = f32_vec(&mut loader, &format!("{LM}.norm.weight"))?;
    let output = if cfg.tie_word_embeddings || !has(HEAD) {
        None
    } else {
        Some(packed(&mut loader, HEAD)?)
    };

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for il in 0..cfg.num_hidden_layers {
        let p = format!("{LM}.layers.{il}");
        let sa = format!("{p}.self_attn");
        let attn_norm = f32_vec(&mut loader, &format!("{p}.input_layernorm.weight"))?;
        let ffn_norm = f32_vec(&mut loader, &format!("{p}.post_attention_layernorm.weight"))?;
        let qn_key = format!("{sa}.q_norm.weight");
        let kn_key = format!("{sa}.k_norm.weight");
        let q_norm = if has(&qn_key) {
            Some(f32_vec(&mut loader, &qn_key)?)
        } else {
            None
        };
        let k_norm = if has(&kn_key) {
            Some(f32_vec(&mut loader, &kn_key)?)
        } else {
            None
        };

        let wq = packed(&mut loader, &format!("{sa}.q_proj.weight"))?;
        let wk = packed(&mut loader, &format!("{sa}.k_proj.weight"))?;
        let wv = packed(&mut loader, &format!("{sa}.v_proj.weight"))?;
        let wo = packed(&mut loader, &format!("{sa}.o_proj.weight"))?;
        let g_key = format!("{sa}.g_proj.weight");
        let wg = if has(&g_key) {
            Some(packed(&mut loader, &g_key)?)
        } else {
            None
        };

        let mp = format!("{p}.mlp");
        let ffn = if cfg.is_dense_mlp(il) {
            LagunaPackedFfn::Dense {
                gate: packed(&mut loader, &format!("{mp}.gate_proj.weight"))?,
                up: packed(&mut loader, &format!("{mp}.up_proj.weight"))?,
                down: packed(&mut loader, &format!("{mp}.down_proj.weight"))?,
            }
        } else {
            let bias_key = format!("{mp}.gate.e_score_correction_bias");
            let gate_bias = if has(&bias_key) {
                Some(f32_vec(&mut loader, &bias_key)?)
            } else {
                None
            };
            LagunaPackedFfn::Moe {
                router: packed_or_f32(&mut loader, &format!("{mp}.gate.proj.weight"))?,
                gate_bias,
                gate_exps: packed(&mut loader, &format!("{mp}.switch_mlp.gate_proj.weight"))?,
                up_exps: packed(&mut loader, &format!("{mp}.switch_mlp.up_proj.weight"))?,
                down_exps: packed(&mut loader, &format!("{mp}.switch_mlp.down_proj.weight"))?,
                shared_gate: packed(&mut loader, &format!("{mp}.shared_expert.gate_proj.weight"))?,
                shared_up: packed(&mut loader, &format!("{mp}.shared_expert.up_proj.weight"))?,
                shared_down: packed(&mut loader, &format!("{mp}.shared_expert.down_proj.weight"))?,
            }
        };

        layers.push(LagunaPackedLayer {
            attn_norm,
            ffn_norm,
            q_norm,
            k_norm,
            wq,
            wk,
            wv,
            wo,
            wg,
            ffn,
        });
    }

    Ok(LagunaPackedWeights {
        token_embd,
        output_norm,
        output,
        layers,
        // Host mlx-affine forward reads MatWeight bytes inline — no GGUF
        // mmap side-channel, so the packed-metadata map stays empty.
        packed_params: PackedParams::new(),
        packed_tensor_count: 0,
        native_f32_bytes: 0,
    })
}
