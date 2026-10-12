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

//! Reading a peft LoRA adapter and folding it into the base weights.
//!
//! Merging rather than keeping the adapter live is what kev's own server
//! does by default (`KEV_MERGE=1`), and it is not only a speed choice: in
//! fp32 the merge is exact, and when serving bf16 the merged weights are
//! *closer* to the fp32 reference than the unmerged adapter (max |Δp| 0.017
//! vs 0.029 over kev's 24 dev records, 0 argmax flips vs 1).
//!
//! # Layout
//!
//! peft stores `lora_A` as `[r, in]` and `lora_B` as `[out, r]`, and RLX
//! keeps matmul weights as row-major `[out, in]` — the same orientation as
//! `nn.Linear.weight`. So the fold is a plain
//!
//! ```text
//! W += (alpha / r) · B · A
//! ```
//!
//! with no transposes anywhere. That is worth stating explicitly because a
//! silently transposed merge produces a model that still runs, still emits
//! plausible probabilities, and is entirely wrong.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

/// One target module's `A` / `B` pair.
#[derive(Debug, Clone)]
pub struct LoraPair {
    /// `[r, in]`, row-major.
    pub a: Vec<f32>,
    /// `[out, r]`, row-major.
    pub b: Vec<f32>,
    pub r: usize,
    pub in_features: usize,
    pub out_features: usize,
}

/// A peft LoRA adapter directory (`adapter_config.json` +
/// `adapter_model.safetensors`).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub r: usize,
    pub alpha: f32,
    pub use_rslora: bool,
    pub target_modules: Vec<String>,
    /// Module path (`layers.0.self_attn.q_proj`) → weights.
    pairs: HashMap<String, LoraPair>,
}

impl LoraAdapter {
    /// `alpha / r`, or `alpha / sqrt(r)` under rsLoRA.
    pub fn scale(&self) -> f32 {
        if self.use_rslora {
            self.alpha / (self.r as f32).sqrt()
        } else {
            self.alpha / self.r as f32
        }
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn modules(&self) -> impl Iterator<Item = &str> {
        self.pairs.keys().map(String::as_str)
    }

    pub fn get(&self, module_path: &str) -> Option<&LoraPair> {
        self.pairs.get(module_path)
    }

    /// An adapter that targets nothing — the cold start for training, where
    /// the base weights are used as they are.
    pub fn empty() -> Self {
        Self {
            r: 0,
            alpha: 0.0,
            use_rslora: false,
            target_modules: Vec::new(),
            pairs: HashMap::new(),
        }
    }

    /// Read `dir/adapter_config.json` and `dir/adapter_model.safetensors`.
    pub fn open(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("adapter_config.json");
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .with_context(|| format!("reading {}", cfg_path.display()))?,
        )
        .with_context(|| format!("parsing {}", cfg_path.display()))?;

        let r = cfg["r"]
            .as_u64()
            .ok_or_else(|| anyhow!("{}: no `r`", cfg_path.display()))? as usize;
        let alpha = cfg["lora_alpha"]
            .as_f64()
            .ok_or_else(|| anyhow!("{}: no `lora_alpha`", cfg_path.display()))?
            as f32;
        let use_rslora = cfg["use_rslora"].as_bool().unwrap_or(false);
        let target_modules = cfg["target_modules"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        // A token-trained adapter carries embedding rows that a weight merge
        // cannot represent; kev refuses to merge those, and so do we.
        if !cfg["trainable_token_indices"].is_null() {
            bail!(
                "{}: adapter has trainable_token_indices; it cannot be merged \
                 into the base weights (an unmerged path is not implemented)",
                cfg_path.display()
            );
        }
        if cfg["use_dora"].as_bool().unwrap_or(false) {
            bail!("{}: DoRA adapters are not supported", cfg_path.display());
        }

        let st_path = dir.join("adapter_model.safetensors");
        let bytes =
            std::fs::read(&st_path).with_context(|| format!("reading {}", st_path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow!("{}: {e}", st_path.display()))?;

        let mut a_by_module: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut b_by_module: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for (name, view) in st.tensors() {
            let Some((module, which)) = split_lora_key(&name) else {
                continue;
            };
            let shape = view.shape().to_vec();
            let data = rlx_core::safetensors_checkpoint::tensor_view_to_f32(&name, view)?;
            match which {
                Side::A => a_by_module.insert(module, (data, shape)),
                Side::B => b_by_module.insert(module, (data, shape)),
            };
        }

        let mut pairs = HashMap::with_capacity(a_by_module.len());
        for (module, (a, a_shape)) in a_by_module {
            let (b, b_shape) = b_by_module.remove(&module).ok_or_else(|| {
                anyhow!("{}: {module} has lora_A but no lora_B", st_path.display())
            })?;
            if a_shape.len() != 2 || b_shape.len() != 2 {
                bail!("{module}: lora_A {a_shape:?} / lora_B {b_shape:?} must be rank 2");
            }
            let (ra, in_features) = (a_shape[0], a_shape[1]);
            let (out_features, rb) = (b_shape[0], b_shape[1]);
            if ra != rb {
                bail!("{module}: lora_A rank {ra} != lora_B rank {rb}");
            }
            if ra != r {
                bail!("{module}: rank {ra} disagrees with adapter_config r={r}");
            }
            pairs.insert(
                module,
                LoraPair {
                    a,
                    b,
                    r: ra,
                    in_features,
                    out_features,
                },
            );
        }
        if !b_by_module.is_empty() {
            let orphan = b_by_module.keys().next().cloned().unwrap_or_default();
            bail!("{}: {orphan} has lora_B but no lora_A", st_path.display());
        }
        if pairs.is_empty() {
            bail!(
                "{}: no lora_A/lora_B pairs found (is this a peft adapter?)",
                st_path.display()
            );
        }

        Ok(Self {
            r,
            alpha,
            use_rslora,
            target_modules,
            pairs,
        })
    }

    /// Fold this module's delta into a row-major `[out, in]` weight.
    ///
    /// Returns `false` when the adapter does not target `module_path`, so a
    /// caller can distinguish "no adapter here" from "merged".
    pub fn merge_into(
        &self,
        module_path: &str,
        w: &mut [f32],
        out: usize,
        inn: usize,
    ) -> Result<bool> {
        let Some(p) = self.pairs.get(module_path) else {
            return Ok(false);
        };
        if p.out_features != out || p.in_features != inn {
            bail!(
                "{module_path}: adapter is [{}, {}] but the base weight is [{out}, {inn}]",
                p.out_features,
                p.in_features
            );
        }
        if w.len() != out * inn {
            bail!(
                "{module_path}: base weight has {} values, expected {}",
                w.len(),
                out * inn
            );
        }
        let scale = self.scale();
        // W[o, i] += scale * sum_k B[o, k] * A[k, i]
        for o in 0..out {
            let brow = &p.b[o * p.r..(o + 1) * p.r];
            let wrow = &mut w[o * inn..(o + 1) * inn];
            for (k, bk) in brow.iter().enumerate() {
                if *bk == 0.0 {
                    continue;
                }
                let s = scale * bk;
                let arow = &p.a[k * inn..(k + 1) * inn];
                for (wi, ai) in wrow.iter_mut().zip(arow) {
                    *wi += s * ai;
                }
            }
        }
        Ok(true)
    }
}

enum Side {
    A,
    B,
}

/// `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`
/// → `("layers.0.self_attn.q_proj", A)`.
///
/// peft optionally inserts the adapter name (`lora_A.default.weight`), and
/// the `base_model.model.` prefix depends on what was wrapped, so both are
/// handled rather than assumed.
fn split_lora_key(key: &str) -> Option<(String, Side)> {
    let (head, side) = match strip_lora(key, "lora_A") {
        Some(h) => (h, Side::A),
        None => (strip_lora(key, "lora_B")?, Side::B),
    };
    let head = head.strip_prefix("base_model.model.").unwrap_or(head);
    let head = head.strip_prefix("base_model.").unwrap_or(head);
    Some((head.to_string(), side))
}

fn strip_lora<'a>(key: &'a str, marker: &str) -> Option<&'a str> {
    let idx = key.find(&format!(".{marker}."))?;
    let tail = &key[idx + marker.len() + 2..];
    // Everything after the marker must be `weight` or `<adapter>.weight`.
    let ok = tail == "weight" || tail.ends_with(".weight");
    ok.then(|| &key[..idx])
}

/// The peft module path for a GGUF-style tensor name.
///
/// Goes through the shared GGUF↔HF map so the two name spaces cannot drift:
/// `blk.0.attn_q.weight` → `layers.0.self_attn.q_proj`.
pub fn module_path_for_gguf(gguf: &str) -> Option<String> {
    let hf = rlx_core::weight_loader::gguf_to_hf_qwen35_name(gguf)?;
    let hf = hf.strip_suffix(".weight").unwrap_or(&hf);
    for prefix in ["model.language_model.", "language_model.model.", "model."] {
        if let Some(rest) = hf.strip_prefix(prefix) {
            return Some(rest.to_string());
        }
    }
    Some(hf.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_peft_keys_with_and_without_adapter_name() {
        let (m, s) = split_lora_key("base_model.model.layers.3.self_attn.q_proj.lora_A.weight")
            .expect("parsed");
        assert_eq!(m, "layers.3.self_attn.q_proj");
        assert!(matches!(s, Side::A));

        let (m, s) =
            split_lora_key("base_model.model.layers.3.linear_attn.out_proj.lora_B.default.weight")
                .expect("parsed");
        assert_eq!(m, "layers.3.linear_attn.out_proj");
        assert!(matches!(s, Side::B));

        assert!(split_lora_key("layers.0.self_attn.q_proj.weight").is_none());
    }

    /// The DeltaNet projections kev targets must round-trip through the
    /// shared GGUF↔HF map, or a merge would silently skip them and leave
    /// the linear-attention layers un-adapted.
    #[test]
    fn maps_every_kev_target_from_gguf_names() {
        let cases = [
            ("blk.0.attn_q.weight", "layers.0.self_attn.q_proj"),
            ("blk.1.attn_k.weight", "layers.1.self_attn.k_proj"),
            ("blk.2.attn_v.weight", "layers.2.self_attn.v_proj"),
            ("blk.3.attn_output.weight", "layers.3.self_attn.o_proj"),
            ("blk.4.ffn_gate.weight", "layers.4.mlp.gate_proj"),
            ("blk.5.ffn_up.weight", "layers.5.mlp.up_proj"),
            ("blk.6.ffn_down.weight", "layers.6.mlp.down_proj"),
            ("blk.7.attn_qkv.weight", "layers.7.linear_attn.in_proj_qkv"),
            ("blk.8.attn_gate.weight", "layers.8.linear_attn.in_proj_z"),
            ("blk.9.ssm_alpha.weight", "layers.9.linear_attn.in_proj_a"),
            ("blk.10.ssm_beta.weight", "layers.10.linear_attn.in_proj_b"),
            ("blk.11.ssm_out.weight", "layers.11.linear_attn.out_proj"),
        ];
        for (gguf, want) in cases {
            assert_eq!(
                module_path_for_gguf(gguf).as_deref(),
                Some(want),
                "{gguf} should map to {want}"
            );
        }
    }

    #[test]
    fn merge_is_w_plus_scaled_b_times_a() {
        // out=2, in=3, r=1.  B = [[1],[2]], A = [[1, 0, -1]]
        let pair = LoraPair {
            a: vec![1.0, 0.0, -1.0],
            b: vec![1.0, 2.0],
            r: 1,
            in_features: 3,
            out_features: 2,
        };
        let mut pairs = HashMap::new();
        pairs.insert("m".to_string(), pair);
        let ad = LoraAdapter {
            r: 1,
            alpha: 2.0,
            use_rslora: false,
            target_modules: vec![],
            pairs,
        };
        assert_eq!(ad.scale(), 2.0);
        let mut w = vec![0.0f32; 6];
        assert!(ad.merge_into("m", &mut w, 2, 3).expect("merge"));
        // scale * B A = 2 * [[1,0,-1],[2,0,-2]]
        assert_eq!(w, vec![2.0, 0.0, -2.0, 4.0, 0.0, -4.0]);
        assert!(!ad.merge_into("absent", &mut w, 2, 3).expect("no-op"));
    }
}
