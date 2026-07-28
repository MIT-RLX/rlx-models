// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates **structure-only build for MLX-quantized (packed) weights** — the
//! last gap for running a real 4-bit V4 across nodes without the coordinator
//! OOMing. A synthetic MLX-affine loader serves the projection weights as packed
//! (codes + scales + biases). We build the DeepSeek-V4 graph twice:
//!   * normally — the big `w_q` codes land in the `packed` map;
//!   * structure-only (`StructureLoader`) — the codes are DEFERRED (empty in
//!     `packed`, param sized from metadata), scales/biases kept, manifest marks
//!     each proj weight `PackedMlx`.
//!
//! Then `ManifestParamSource` re-fetches each deferred shard and we assert the
//! reloaded codes are byte-identical to the normal build. (We inspect the maps
//! rather than run the dequant forward — synthetic 4-bit packing isn't a valid
//! numeric checkpoint; the f32 forward parity is covered by the sibling probe.)
//!
//!   cargo run --release -p rlx-models-core --example dsv4_packed_structure_probe

use anyhow::Result;
use rlx_distributed::Param;
use rlx_distributed::ParamSource;
use rlx_ir::quant::QuantScheme;
use rlx_models_core::distributed_bridge::{LoadKind, ManifestParamSource, StructureLoader};
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_prefill};
use rlx_models_core::weight_loader::{MlxPackedLinear, WeightLoader};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    ((x - x.floor()) as f32 - 0.5) * 0.3
}

/// A proj weight (goes through `load_proj` → packed). Excludes the compressor /
/// indexer `wkv` (those use `load_p`, stay f32).
fn is_proj(key: &str) -> bool {
    if key.contains(".compressor.") || key.contains(".indexer.") {
        return false;
    }
    key == "lm_head.weight"
        || [
            ".attn.wq_a.weight",
            ".attn.wq_b.weight",
            ".attn.wkv.weight",
            ".ffn.gate_proj.weight",
            ".ffn.up_proj.weight",
            ".ffn.down_proj.weight",
        ]
        .iter()
        .any(|s| key.ends_with(s))
}

/// f32 for everything + synthetic MLX-affine packed for proj weights.
struct SynthPackedLoader {
    t: HashMap<String, (Vec<f32>, Vec<usize>)>,
}
impl WeightLoader for SynthPackedLoader {
    fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.t
            .get(k)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing {k}"))
    }
    fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(k)?;
        let (r, c) = (s[0], s[1]);
        let mut o = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = d[i * c + j];
            }
        }
        Ok((o, vec![c, r]))
    }
    fn take_packed_mlx(&mut self, k: &str) -> Result<Option<MlxPackedLinear>> {
        if !is_proj(k) {
            return Ok(None);
        }
        let (_d, shape) = self.take(k)?; // [out, in]
        let (out, k_in) = (shape[0], shape[1]);
        // 4-bit codes: out*in/2 bytes; deterministic by key hash.
        let seed = k.bytes().map(|b| b as f64).sum::<f64>() + 1.0;
        let w_q: Vec<u8> = (0..out * k_in / 2)
            .map(|i| ((rnd(seed, i) * 255.0) as i32 & 0xff) as u8)
            .collect();
        // n_groups = in/group_size = 1 (group_size = in). scales/biases: out f32 LE.
        let scales: Vec<u8> = (0..out)
            .flat_map(|i| (0.05 + 0.1 * rnd(seed + 1.0, i)).to_le_bytes())
            .collect();
        let biases: Vec<u8> = (0..out)
            .flat_map(|i| (0.01 * rnd(seed + 2.0, i)).to_le_bytes())
            .collect();
        Ok(Some(MlxPackedLinear {
            w_q,
            scales,
            biases,
            scheme: QuantScheme::MlxAffine {
                bits: 4,
                group_size: k_in as u32,
            },
            out_shape: vec![out, k_in],
        }))
    }
    fn len(&self) -> usize {
        self.t.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.t.keys().cloned().collect()
    }
}

fn spec() -> DeepseekV4Spec {
    // in-features of every proj weight = 8 (== group_size ⇒ n_groups 1).
    DeepseekV4Spec {
        vocab_size: 16,
        dim: 8,
        n_layers: 2,
        hc_mult: 2,
        n_heads: 2,
        head_dim: 4,
        rope_head_dim: 2,
        q_lora_rank: 8,
        n_groups: 2,
        o_lora_rank: 3,
        compress_ratios: vec![4; 2],
        index_head_dim: 4,
        index_n_heads: 2,
        index_topk: 2,
        window_size: 64,
        first_k_dense_replace: 2,
        n_hash_layers: 0,
        moe_intermediate_size: 8,
        n_routed_experts: 4,
        n_activated_experts: 2,
        n_shared_experts: 1,
        intermediate_size: 8,
        route_scale: 1.0,
        rope_theta: 10000.0,
        compress_rope_theta: 160000.0,
        swiglu_limit: 10.0,
        rms_norm_eps: 1e-5,
        hc_sinkhorn_iters: 5,
        hc_eps: 1e-6,
    }
}

fn tensor_map(s: &DeepseekV4Spec) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let (vocab, dim, nl, hc, nh, hd, ql) = (
        s.vocab_size,
        s.dim,
        s.n_layers,
        s.hc_mult,
        s.n_heads,
        s.head_dim,
        s.q_lora_rank,
    );
    let (ngrp, olora, inter) = (s.n_groups, s.o_lora_rank, s.intermediate_size);
    let (ihd, in_heads, ratio) = (s.index_head_dim, s.index_n_heads, 4usize);
    let (mix_hc, hcd, dpg) = ((2 + hc) * hc, hc * dim, nh * hd / ngrp);
    let mut t = HashMap::new();
    let mut sd = 1.0;
    let mut put = |k: String, shape: Vec<usize>| {
        sd += 1.0;
        let n: usize = shape.iter().product();
        t.insert(k, ((0..n).map(|i| rnd(sd, i) + 0.05).collect(), shape));
    };
    put("model.embed_tokens.weight".into(), vec![vocab, dim]);
    for il in 0..nl {
        let p = format!("model.layers.{il}");
        for (suf, sh) in [
            ("attn_hc.fn", vec![mix_hc, hcd]),
            ("attn_hc.scale", vec![3]),
            ("attn_hc.base", vec![mix_hc]),
            ("attn_norm.weight", vec![dim]),
            ("attn.wq_a.weight", vec![ql, dim]),
            ("attn.q_norm.weight", vec![ql]),
            ("attn.wq_b.weight", vec![nh * hd, ql]),
            ("attn.wkv.weight", vec![hd, dim]),
            ("attn.kv_norm.weight", vec![hd]),
            ("attn.attn_sink", vec![nh]),
            ("attn.wo_a.weight", vec![ngrp * olora, dpg]),
            ("attn.wo_b.weight", vec![dim, ngrp * olora]),
            ("attn.compressor.wkv.weight", vec![2 * hd, dim]),
            ("attn.compressor.wgate.weight", vec![2 * hd, dim]),
            ("attn.compressor.ape", vec![ratio, 2 * hd]),
            ("attn.compressor.norm.weight", vec![hd]),
            ("attn.indexer.compressor.wkv.weight", vec![2 * ihd, dim]),
            ("attn.indexer.compressor.wgate.weight", vec![2 * ihd, dim]),
            ("attn.indexer.compressor.ape", vec![ratio, 2 * ihd]),
            ("attn.indexer.compressor.norm.weight", vec![ihd]),
            ("attn.indexer.wq_b.weight", vec![in_heads * ihd, ql]),
            ("attn.indexer.weights_proj.weight", vec![in_heads, dim]),
            ("ffn_hc.fn", vec![mix_hc, hcd]),
            ("ffn_hc.scale", vec![3]),
            ("ffn_hc.base", vec![mix_hc]),
            ("ffn_norm.weight", vec![dim]),
            ("ffn.gate_proj.weight", vec![inter, dim]),
            ("ffn.up_proj.weight", vec![inter, dim]),
            ("ffn.down_proj.weight", vec![dim, inter]),
        ] {
            put(format!("{p}.{suf}"), sh);
        }
    }
    put("model.hc_head.fn".into(), vec![hc, hcd]);
    put("model.hc_head.scale".into(), vec![1]);
    put("model.hc_head.base".into(), vec![hc]);
    put("model.norm.weight".into(), vec![dim]);
    put("lm_head.weight".into(), vec![vocab, dim]);
    t
}

type Packed = HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>;

fn main() -> Result<()> {
    let spec = spec();
    let seq = 16;

    // ── Normal build: proj codes land in `packed` ──
    let mut ln = SynthPackedLoader {
        t: tensor_map(&spec),
    };
    let mut packed_n: Packed = HashMap::new();
    let (_g1, _p1) = build_deepseek_v4_prefill(&spec, &mut ln, seq, &mut packed_n)?;

    // ── Structure-only build: proj codes DEFERRED ──
    let mut ls = SynthPackedLoader {
        t: tensor_map(&spec),
    };
    let (params_s, packed_s, manifest) = {
        let mut sl = StructureLoader::new(&mut ls);
        let mut ps: Packed = HashMap::new();
        let (_g2, p2) = build_deepseek_v4_prefill(&spec, &mut sl, seq, &mut ps)?;
        (p2, ps, std::mem::take(&mut sl.manifest))
    };

    // Every proj weight: real codes normally, EMPTY (deferred) structure-only,
    // param sized identically, manifest marks it PackedMlx.
    let proj: Vec<String> = packed_n.keys().filter(|k| is_proj(k)).cloned().collect();
    assert!(!proj.is_empty(), "no proj weights went packed");
    let mut deferred_bytes = 0usize;
    for k in &proj {
        let real = &packed_n[k].0;
        assert!(!real.is_empty(), "{k}: normal build should hold codes");
        assert!(
            packed_s[k].0.is_empty(),
            "{k}: structure build must DEFER codes"
        );
        assert_eq!(packed_n[k].2, packed_s[k].2, "{k}: out_shape differs");
        assert_eq!(
            manifest.get(k),
            Some(&LoadKind::PackedMlx),
            "{k}: manifest not PackedMlx"
        );
        deferred_bytes += real.len();
    }

    // Held (small) vs deferred (big).
    let held_f32: usize = params_s.values().map(|v| v.len() * 4).sum();
    let held_packed: usize = packed_s.values().map(|(b, _, _)| b.len()).sum();

    // Reload: ManifestParamSource re-fetches each deferred shard byte-identically.
    let mut lr = SynthPackedLoader {
        t: tensor_map(&spec),
    };
    let synth: HashMap<String, Vec<f32>> = params_s
        .into_iter()
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let synth_packed: HashMap<String, Vec<u8>> = packed_s
        .iter()
        .filter(|(_, (b, _, _))| !b.is_empty())
        .map(|(k, (b, _, _))| (k.clone(), b.clone()))
        .collect();
    let mut src = ManifestParamSource {
        loader: &mut lr,
        manifest,
        synth,
        synth_packed,
    };
    let mut reload_ok = 0usize;
    for k in &proj {
        match src.get(k) {
            Some(Param::Typed(bytes, rlx_ir::DType::U8)) if bytes == packed_n[k].0 => {
                reload_ok += 1
            }
            other => panic!(
                "{k}: reload mismatch: {:?}",
                other.map(|p| matches!(p, Param::Typed(..)))
            ),
        }
    }

    println!("── DeepSeek-V4 structure-only build for MLX-PACKED weights ──");
    println!(
        "proj weights: {} deferred ({} B of codes not held at build); structure kept only \
         {} B f32 + {} B packed (scales/biases + synth consts)",
        proj.len(),
        deferred_bytes,
        held_f32,
        held_packed
    );
    println!(
        "reload: {reload_ok}/{} deferred shards re-fetched byte-identical",
        proj.len()
    );
    if reload_ok == proj.len() && deferred_bytes > held_packed {
        println!("✅ packed codes deferred at build + re-loaded exactly per shard");
        Ok(())
    } else {
        Err(anyhow::anyhow!("packed structure-only failed"))
    }
}
