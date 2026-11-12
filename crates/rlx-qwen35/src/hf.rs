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

//! Reading HF-native (safetensors) Qwen3.5 checkpoints with a builder written
//! against llama.cpp GGUFs: the one tensor whose values differ, and the trunk
//! depth the config misreports.

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;

/// Number of real decoder layers in an HF checkpoint, from the tensor names.
///
/// `Qwen35Config::from_hf_config_value` computes the trunk depth as
/// `num_hidden_layers - mtp_num_hidden_layers`. That is right for a
/// GGUF-derived layout, where llama.cpp appends the MTP block as the last
/// `blk.N` — but **wrong for HF safetensors**, where `Qwen3.5-0.8B-Base`
/// carries `model.language_model.layers.0..23` *and* a separate `mtp.*`
/// module. Trusting the config there drops the last decoder layer and the
/// model still runs, one block short.
///
/// So count the layers that actually exist. `mtp.layers.N` is excluded: it is
/// a different module, and we never build the MTP head. So are the layer
/// stacks of a vision or audio tower sharing the file: MiniCPM-V 4.6/4.7 ship
/// `model.vision_tower.encoder.layers.0..26` next to a 24-layer trunk, and
/// counting them would ask for three decoder layers that do not exist.
pub fn hf_decoder_layer_count<'a>(keys: impl Iterator<Item = &'a str>) -> Option<usize> {
    const TOWERS: [&str; 4] = ["vision", "visual", "audio", "merger"];
    let mut max_idx: Option<usize> = None;
    for k in keys {
        if k.starts_with("mtp.") || k.contains(".mtp.") {
            continue;
        }
        let mut parts = k.splitn(2, ".layers.");
        let (Some(owner), Some(rest)) = (parts.next(), parts.next()) else {
            continue;
        };
        if TOWERS.iter().any(|t| owner.contains(t)) {
            continue;
        }
        let Some(idx) = rest.split('.').next().and_then(|d| d.parse::<usize>().ok()) else {
            continue;
        };
        max_idx = Some(max_idx.map_or(idx, |m: usize| m.max(idx)));
    }
    max_idx.map(|m| m + 1)
}

/// Converts HF-native tensors into the form the `rlx-qwen35` builder expects.
///
/// The builder was written against llama.cpp-converted GGUFs, and the GGUF
/// converter does not just rename tensors — it *transforms* one. The Gated
/// DeltaNet decay coefficient is stored in HF as `linear_attn.A_log`, and
/// llama.cpp writes `A = -exp(A_log)` into `blk.N.ssm_a`. The builder then
/// uses `ssm_a` directly (`decay = exp(A · softplus(dt))`), with a comment
/// recording that applying `-exp` a second time was a real bug once.
///
/// Feeding raw `A_log` through that path is silently catastrophic in a way
/// that is easy to miss: at position 0 the recurrent state is zero, so the
/// decay is unobservable and the first token is *exact*. Every later position
/// compounds the error. On Qwen3.5-0.8B the raw `A_log` even contains a
/// positive entry, so that head's state **grows** instead of decaying.
///
/// The second transform is an ordering one. When a Gated DeltaNet layer has
/// more value heads than key heads (Qwen3.5-4B: 32 and 16), HF expands q/k with
/// `repeat_interleave`, so value head `j` reads key head `j / factor`; the
/// builder tiles instead (`repeat_heads`: head `j` reads `j % n_k`), as
/// llama.cpp does, and the GGUF converter reorders the value heads to match.
/// [`HfQwen35Fixups::gdn_heads`] applies that reorder. With as many value heads
/// as key heads (0.8B, 2B) both orders are the same and nothing changes.
pub struct HfQwen35Fixups<L: WeightLoader> {
    inner: L,
    converted: usize,
    heads: Option<GdnHeads>,
    reordered: usize,
}

/// Gated DeltaNet head geometry for the value-head reorder.
#[derive(Clone, Copy, Debug)]
struct GdnHeads {
    n_k: usize,
    n_v: usize,
    /// Key (and q) head width.
    d_k: usize,
    /// Value head width.
    d_v: usize,
}

impl GdnHeads {
    /// Old (HF, grouped) head for new (tiled) head `j`: new head `j` pairs with
    /// key head `j % n_k` and is that group's `j / n_k`-th value head.
    fn source(&self, j: usize) -> usize {
        let factor = self.n_v / self.n_k;
        (j % self.n_k) * factor + j / self.n_k
    }
}

impl<L: WeightLoader> HfQwen35Fixups<L> {
    pub fn new(inner: L) -> Self {
        Self {
            inner,
            converted: 0,
            heads: None,
            reordered: 0,
        }
    }

    /// Also reorder the Gated DeltaNet value heads from HF's grouped order to
    /// the tiled order the builder uses (see the type docs). Needed for any
    /// checkpoint with more value heads than key heads; a no-op otherwise.
    pub fn gdn_heads(mut self, cfg: &crate::Qwen35Config) -> Self {
        let (n_k, n_v) = (cfg.ssm_group_count, cfg.ssm_time_step_rank);
        if n_k > 0 && n_v > n_k && n_v % n_k == 0 {
            let d_k = cfg.ssm_state_size;
            let d_v = if cfg.ssm_inner_size > 0 { cfg.ssm_inner_size / n_v } else { d_k };
            self.heads = Some(GdnHeads { n_k, n_v, d_k, d_v });
        }
        self
    }

    /// How many tensors had their value heads reordered.
    pub fn reordered(&self) -> usize {
        self.reordered
    }

    /// How many `ssm_a` tensors were converted from `A_log`.
    pub fn converted(&self) -> usize {
        self.converted
    }

    /// `ssm_a` is the only tensor whose *values* differ between the HF and
    /// GGUF layouts.
    fn is_a_log(key: &str) -> bool {
        key.ends_with(".ssm_a") || key.ends_with("linear_attn.A_log")
    }

    fn fix(&mut self, key: &str, data: &mut [f32], shape: &[usize], transposed: bool) -> Result<()> {
        self.reorder(key, data, shape, transposed)?;
        if !Self::is_a_log(key) {
            return Ok(());
        }
        for v in data.iter_mut() {
            *v = -v.exp();
        }
        self.converted += 1;
        Ok(())
    }

    /// Value-head reorder of one tensor, if it has a value-head axis. In HF
    /// layout every such tensor is `[rows, cols]` row-major with the heads on
    /// one axis, in a run of `n_v` blocks of `block` starting at `offset`.
    fn reorder(&mut self, key: &str, data: &mut [f32], shape: &[usize], transposed: bool) -> Result<()> {
        let Some(h) = self.heads else { return Ok(()) };
        let key_dim = h.n_k * h.d_k;
        let suffix = key.rsplit_once('.').map_or(key, |(head, last)| {
            if last == "weight" || last == "bias" { head.rsplit_once('.').map_or(head, |(_, s)| s) } else { last }
        });
        // (axis in HF layout, offset of the value-head run on it, block per head)
        let (axis, offset, block) = match suffix {
            // in_proj_qkv rows: q | k | v.
            "attn_qkv" => (0, 2 * key_dim, h.d_v),
            // conv1d channels: the same q | k | v split, `[channels, 1, kernel]`.
            "ssm_conv1d" => (0, 2 * key_dim, h.d_v),
            "attn_gate" => (0, 0, h.d_v),
            "ssm_beta" | "ssm_alpha" | "ssm_a" | "ssm_dt" => (0, 0, 1),
            "A_log" | "dt_bias" | "in_proj_b" | "in_proj_a" | "in_proj_z" => {
                anyhow::bail!("{key}: untranslated HF name; wrap an HfTranslatingLoader first")
            }
            // out_proj takes the value heads in on its columns.
            "ssm_out" => (1, 0, h.d_v),
            _ => return Ok(()),
        };
        if !key.starts_with("blk.") {
            return Ok(());
        }
        // Rank-1 tensors are a single row; rank >= 2 flatten trailing axes.
        let (rows, cols) = match shape {
            [n] => (*n, 1),
            [r, rest @ ..] => (*r, rest.iter().product()),
            [] => return Ok(()),
        };
        let (rows, cols, axis) = if transposed && shape.len() == 2 { (rows, cols, 1 - axis) } else { (rows, cols, axis) };
        let along = if axis == 0 { rows } else { cols };
        anyhow::ensure!(
            offset + h.n_v * block <= along,
            "{key}: shape {shape:?} has no {} value heads of {block} at {offset}",
            h.n_v
        );
        let old = data.to_vec();
        for j in 0..h.n_v {
            let src = offset + h.source(j) * block;
            let dst = offset + j * block;
            if axis == 0 {
                data[dst * cols..(dst + block) * cols].copy_from_slice(&old[src * cols..(src + block) * cols]);
            } else {
                for r in 0..rows {
                    data[r * cols + dst..r * cols + dst + block]
                        .copy_from_slice(&old[r * cols + src..r * cols + src + block]);
                }
            }
        }
        self.reordered += 1;
        Ok(())
    }
}

impl<L: WeightLoader> WeightLoader for HfQwen35Fixups<L> {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.inner.take(key)?;
        self.fix(key, &mut data, &shape, false)?;
        Ok((data, shape))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.inner.take_transposed(key)?;
        self.fix(key, &mut data, &shape, true)?;
        Ok((data, shape))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
    fn arch_hint(&self) -> Option<&str> {
        self.inner.arch_hint()
    }
}

/// The loader every HF safetensors read goes through: HF tensor names served
/// under the GGUF names the builder asks for, with `A_log` converted.
///
/// The runner reads an HF checkpoint twice — once to build, and again after a
/// Metal/MLX upload released the host copy of the projections. The second read
/// once skipped the conversion, so every graph compiled after the reload saw a
/// raw `A_log` (Qwen3.5-0.8B `predict_logits` off by 10.9 on Metal and MLX).
///
/// Without the config this cannot reorder Gated DeltaNet value heads, so it is
/// only right for models with as many value as key heads; prefer
/// [`hf_weight_loader_for`].
pub fn hf_weight_loader(
    inner: Box<dyn WeightLoader>,
) -> HfQwen35Fixups<rlx_core::HfTranslatingLoader<Box<dyn WeightLoader>>> {
    HfQwen35Fixups::new(rlx_core::HfTranslatingLoader::new(inner))
}

/// [`hf_weight_loader`] for `cfg`: also reorders the Gated DeltaNet value heads
/// (see [`HfQwen35Fixups`]), which every model with more value than key heads needs.
pub fn hf_weight_loader_for(
    inner: Box<dyn WeightLoader>,
    cfg: &crate::Qwen35Config,
) -> HfQwen35Fixups<rlx_core::HfTranslatingLoader<Box<dyn WeightLoader>>> {
    hf_weight_loader(inner).gdn_heads(cfg)
}

/// The text model of an HF Qwen3.5 directory (a VLM's language model
/// included), as the runner reads it: the config with the trunk depth taken
/// from the tensors, and the weights through [`hf_weight_loader`].
pub fn hf_text_model(
    dir: &std::path::Path,
) -> anyhow::Result<(crate::Qwen35Config, crate::Qwen35Weights)> {
    let mut cfg = crate::Qwen35Config::from_hf_config_json(&dir.join("config.json"))?;
    let inner: Box<dyn WeightLoader> = Box::new(rlx_core::SafetensorsMmapLoader::open(dir)?);
    let keys = inner.remaining_keys();
    if let Some(found) = hf_decoder_layer_count(keys.iter().map(String::as_str)) {
        cfg.num_hidden_layers = found;
        cfg.nextn_predict_layers = 0;
    }
    let mut loader = hf_weight_loader_for(inner, &cfg);
    let weights = crate::Qwen35Weights::from_loader(&mut loader, &cfg)?;
    Ok((cfg, weights))
}

#[cfg(test)]
mod tests {
    use super::{HfQwen35Fixups, hf_decoder_layer_count};
    use anyhow::Result;
    use rlx_core::weight_loader::WeightLoader;
    use std::collections::HashMap;

    struct Map(HashMap<String, (Vec<f32>, Vec<usize>)>);

    impl WeightLoader for Map {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            self.0.remove(key).ok_or_else(|| anyhow::anyhow!("no {key}"))
        }
        fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
            let (d, s) = self.take(key)?;
            let (r, c) = (s[0], s[1]);
            Ok(((0..r * c).map(|i| d[(i % r) * c + i / r]).collect(), vec![c, r]))
        }
        fn remaining_keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    /// 4 key heads, 8 value heads of width 16 (`synth::medium_cfg`). HF's grouped
    /// value head `k * 2 + r` must land at tiled position `r * 4 + k`.
    #[test]
    fn value_heads_are_reordered_from_grouped_to_tiled() {
        let cfg = crate::synth::medium_cfg();
        let (n_k, n_v, d) = (cfg.ssm_group_count, cfg.ssm_time_step_rank, cfg.ssm_state_size);
        assert_eq!((n_k, n_v, d, cfg.ssm_inner_size), (4, 8, 16, 128));
        let src = |j: usize| (j % n_k) * (n_v / n_k) + j / n_k;
        let (key_dim, value_dim, n_in) = (n_k * d, n_v * d, 3);
        let qkv_rows = 2 * key_dim + value_dim;
        let mut m = HashMap::new();
        m.insert("blk.0.ssm_dt.bias".to_string(), (ramp(n_v), vec![n_v]));
        m.insert("blk.0.attn_qkv.weight".to_string(), (ramp(qkv_rows * n_in), vec![qkv_rows, n_in]));
        m.insert("blk.0.ssm_out.weight".to_string(), (ramp(n_in * value_dim), vec![n_in, value_dim]));
        m.insert("blk.0.ssm_conv1d.weight".to_string(), (ramp(qkv_rows * 4), vec![qkv_rows, 1, 4]));
        m.insert("blk.0.ssm_norm.weight".to_string(), (ramp(d), vec![d]));
        m.insert("blk.0.ssm_beta.weight".to_string(), (ramp(n_v * n_in), vec![n_v, n_in]));
        let mut l = HfQwen35Fixups::new(Map(m)).gdn_heads(&cfg);

        let (dt, _) = l.take("blk.0.ssm_dt.bias").unwrap();
        assert_eq!(dt, (0..n_v).map(|j| src(j) as f32).collect::<Vec<_>>());
        assert_eq!(dt, vec![0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]);

        let (qkv, _) = l.take("blk.0.attn_qkv.weight").unwrap();
        let row = |r: usize| (r * n_in) as f32;
        for r in 0..2 * key_dim {
            assert_eq!(qkv[r * n_in], row(r), "q/k row {r} moved");
        }
        for j in 0..n_v {
            for i in 0..d {
                let (new, old) = (2 * key_dim + j * d + i, 2 * key_dim + src(j) * d + i);
                assert_eq!(qkv[new * n_in..(new + 1) * n_in], ramp(qkv_rows * n_in)[old * n_in..(old + 1) * n_in]);
            }
        }

        let (conv, _) = l.take("blk.0.ssm_conv1d.weight").unwrap();
        let (new, old) = (2 * key_dim + d, 2 * key_dim + src(1) * d);
        assert_eq!(conv[new * 4], (old * 4) as f32);

        // `take_transposed` serves `[cols, rows]`: the heads are on the other axis.
        let (out_t, shape) = l.take_transposed("blk.0.ssm_out.weight").unwrap();
        assert_eq!(shape, vec![value_dim, n_in]);
        for j in 0..n_v {
            let (new, old) = (j * d, src(j) * d);
            assert_eq!(out_t[new * n_in], old as f32, "out_proj column block {j}");
        }

        let (beta, _) = l.take("blk.0.ssm_beta.weight").unwrap();
        assert_eq!(beta[n_in], (src(1) * n_in) as f32);
        // Shared across heads: untouched.
        assert_eq!(l.take("blk.0.ssm_norm.weight").unwrap().0, ramp(d));
        assert_eq!(l.reordered(), 5);
    }

    /// Without `gdn_heads`, or with as many value as key heads, nothing moves.
    #[test]
    fn value_heads_stay_put_unless_asked_and_needed() {
        let one = || {
            let mut m = HashMap::new();
            m.insert("blk.0.ssm_dt.bias".to_string(), (ramp(8), vec![8]));
            Map(m)
        };
        let mut plain = HfQwen35Fixups::new(one());
        assert_eq!(plain.take("blk.0.ssm_dt.bias").unwrap().0, ramp(8));
        let mut equal = crate::synth::medium_cfg();
        equal.ssm_group_count = equal.ssm_time_step_rank;
        let mut l = HfQwen35Fixups::new(one()).gdn_heads(&equal);
        assert_eq!(l.take("blk.0.ssm_dt.bias").unwrap().0, ramp(8));
        assert_eq!(l.reordered(), 0);
    }

    #[test]
    fn layer_count_ignores_vision_tower_layers() {
        // MiniCPM-V 4.7-1B: a 27-layer SigLIP tower beside a 24-layer trunk.
        let keys = [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.23.mlp.up_proj.weight",
            "model.vision_tower.encoder.layers.26.mlp.fc1.weight",
            "model.merger.mlp.0.linear_1.weight",
            "mtp.layers.0.input_layernorm.weight",
        ];
        assert_eq!(hf_decoder_layer_count(keys.into_iter()), Some(24));
    }
}
