// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The checkpoint contract, checked against the real published tensor list —
//! **all 48 blocks**, from ~50 KB of header data and no weights at all.
//!
//! A GGUF's tensor *index* is header data, so the name, rank and dims of every
//! tensor in `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S can be read with two
//! HTTP range requests. That is the only way to get whole-model coverage of a
//! checkpoint whose smallest quantization is tens of gigabytes.
//!
//! Three things get triangulated here, and it is the combination that matters:
//!
//! 1. [`Qwen4ExpConfig::block_tensor_names`] states the contract in config
//!    terms — "a linear layer has these tensors, an attention layer has those".
//! 2. Against the fixture, that prediction must reproduce the real file
//!    **exactly**: no missing tensor and no invented one, for every block.
//! 3. Against the emitters, a flow built over a weight map keyed by exactly
//!    those names must consume them and ask for nothing else.
//!
//! (1)+(2) catches a wrong layer schedule; (2)+(3) catches the contract and the
//! emitters drifting apart. Either alone would let a whole class of error
//! through — the manifest could be a fiction that happens to match the
//! emitters, or the emitters could agree with a manifest that does not match
//! reality.

use rlx_qwen4exp::config::LayerKind;
use rlx_qwen4exp::{Qwen4ExpConfig, synth};
use std::collections::{BTreeMap, BTreeSet};

const MANIFEST: &str = include_str!("fixtures/qwen38_flash_next_tensors.tsv");
const HF_CONFIG: &str = include_str!("fixtures/qwen38_flash_next_config.json");

/// `name -> (dims in GGML order, ggml type)`.
fn published() -> BTreeMap<String, (Vec<usize>, String)> {
    MANIFEST
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            let name = f.next().expect("name").to_string();
            let dims = f
                .next()
                .expect("dims")
                .split(',')
                .map(|d| d.parse().expect("dim"))
                .collect();
            let ty = f.next().expect("type").to_string();
            (name, (dims, ty))
        })
        .collect()
}

fn cfg() -> Qwen4ExpConfig {
    Qwen4ExpConfig::from_hf_json(HF_CONFIG).expect("parse the published config")
}

#[test]
fn fixture_is_the_whole_checkpoint() {
    assert_eq!(
        published().len(),
        1224,
        "the published UD-IQ1_S set is 1224 tensors"
    );
}

/// The published config and the hand-written [`Qwen4ExpConfig::qwen38_flash_next`]
/// must be the same model — the latter is what the unit tests check the PLE
/// hash against, so a drift between them would quietly invalidate those.
#[test]
fn the_hand_written_config_matches_the_published_one() {
    let mut hf = cfg();
    // The two differ only where the format differs: a GGUF has its norm gains
    // pre-offset and carries the hash constants, HF does not.
    hf.rms_norm_offset = false;
    hf.ple_constants = None;
    assert_eq!(hf, Qwen4ExpConfig::qwen38_flash_next());
}

/// The contract must reproduce the real tensor list exactly, block by block.
///
/// Reported per block rather than as one set difference, so a failure names the
/// layer and the direction (missing vs invented) instead of dumping 1224 names.
#[test]
fn predicted_tensors_match_the_published_checkpoint() {
    let cfg = cfg();
    let p = published();

    let mut real: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let mut globals: BTreeSet<String> = BTreeSet::new();
    for name in p.keys() {
        match name.strip_prefix("blk.") {
            Some(rest) => {
                let (idx, suffix) = rest.split_once('.').expect("blk.N.suffix");
                real.entry(idx.parse().expect("block index"))
                    .or_default()
                    .insert(suffix.to_string());
            }
            None => {
                globals.insert(name.clone());
            }
        }
    }

    // Note what is NOT here: `output_norm.weight`. The final hyper-connection
    // mixer is the output norm.
    assert_eq!(
        globals,
        [
            "output.weight",
            "output_hc_down.weight",
            "output_hc_norm.weight",
            "output_hc_up.weight",
            "per_layer_token_embd.weight",
            "token_embd.weight",
        ]
        .map(String::from)
        .into_iter()
        .collect::<BTreeSet<_>>(),
        "unexpected non-block tensors"
    );
    assert_eq!(
        real.len(),
        cfg.block_count(),
        "config predicts {} blocks, the checkpoint has {}",
        cfg.block_count(),
        real.len()
    );

    for (block, actual) in &real {
        let want: BTreeSet<String> = cfg.block_tensor_names(*block).into_iter().collect();
        let missing: Vec<_> = want.difference(actual).collect();
        let extra: Vec<_> = actual.difference(&want).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "blk.{block}: predicted-but-absent {missing:?}, present-but-unpredicted {extra:?}"
        );
    }
}

/// The layer schedule the config derives must be the one the checkpoint's
/// tensors imply — `ssm_a` marks a linear layer, `attn_q_norm` an attention
/// one, and `ple_key` the single PLE module.
///
/// This is the same fact `config_parsing` reads off the metadata
/// (`attention.compress_ratios`), but read off the *weights* instead, so a
/// converter that disagreed with itself would show up.
#[test]
fn layer_schedule_agrees_with_the_tensors_present() {
    let cfg = cfg();
    let p = published();
    let has = |b: usize, s: &str| p.contains_key(&format!("blk.{b}.{s}"));

    let (mut linear, mut indexed, mut ple) = (0, 0, 0);
    for i in 0..cfg.num_hidden_layers {
        let looks_linear = has(i, "ssm_a");
        let looks_indexed = has(i, "indexer.q_proj.weight");
        assert!(
            looks_linear ^ looks_indexed,
            "blk.{i} is neither cleanly linear nor cleanly indexed"
        );
        match cfg.layer_kind(i) {
            LayerKind::Linear => {
                assert!(looks_linear, "blk.{i}: config says linear, tensors say QSA");
                linear += 1;
            }
            LayerKind::Indexed => {
                assert!(
                    looks_indexed,
                    "blk.{i}: config says QSA, tensors say linear"
                );
                indexed += 1;
            }
        }
        assert_eq!(
            cfg.ple_layer_index(i).is_some(),
            has(i, "ple_key.weight"),
            "blk.{i}: PLE disagreement between config and tensors"
        );
        if has(i, "ple_key.weight") {
            ple += 1;
        }
        // Every block is MoE — there is no leading dense stretch.
        assert!(has(i, "ffn_gate_exps.weight"), "blk.{i} must be MoE");
        assert!(has(i, "ffn_gate_inp_shexp.weight"), "blk.{i} shared gate");
    }
    assert_eq!(
        (linear, indexed),
        (36, 12),
        "Qwen3.8-Flash-Next is 36 linear + 12 QSA layers"
    );
    assert_eq!(ple, 1, "exactly one PLE module, on layer 1");
    assert_eq!(cfg.ple_layers, vec![1]);
}

/// Shapes, for every block — and in both directions.
///
/// The fixture keeps GGML's innermost-first order, which the loader reverses.
/// [`synth::tensor_shapes`] states the *reversed* shape the emitters assume, so
/// reversing the fixture and comparing checks the orientation of every tensor
/// in the model at once. That is the check that would have caught a transposed
/// `ple_key` or an expert bank read as `[out, in, experts]`.
#[test]
fn published_shapes_match_the_emitters_orientation() {
    let cfg = cfg();
    let p = published();
    let predicted: BTreeMap<String, Vec<usize>> = synth::tensor_shapes(&cfg)
        .into_iter()
        .filter(|(n, _)| n != "per_layer_token_embd.weight")
        .collect();

    for (name, want) in &predicted {
        let (ggml_dims, _) = p
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not in the published checkpoint"));
        let mut reversed = ggml_dims.clone();
        reversed.reverse();
        assert_eq!(
            reversed, *want,
            "{name}: the file stores {ggml_dims:?}, so the loader reports \
             {reversed:?}, but the emitters expect {want:?}"
        );
    }

    // The n-gram table is sized by the hash, not by a config field, so check it
    // on its own: 16 heads × a distinct prime near 20 M, padded to 128.
    let hash = rlx_qwen4exp::PleHash::derive(&cfg, 0);
    let (dims, ty) = &p["per_layer_token_embd.weight"];
    assert_eq!(dims, &vec![cfg.ple_head_dim, hash.padded_rows as usize]);
    assert_eq!(ty, "IQ4_NL", "the table is quantized, not f32");
}

/// What the emitters actually read, against what the contract promises.
///
/// Builds the whole tiny model over a weight map keyed by exactly
/// [`Qwen4ExpConfig::tensor_manifest`] and asserts nothing is left over except
/// the indexer tensors a provably-dense QSA layer legitimately skips.
#[test]
fn the_emitters_read_exactly_the_manifest() {
    use rlx_core::weight_map::WeightMap;
    use rlx_qwen4exp::build_qwen4exp_text_flow;

    let cfg = synth::tiny_config(2048);
    let manifest: BTreeSet<String> = cfg.tensor_manifest().into_iter().collect();
    let mut tensors = synth::synth_tensors(&cfg, 3);
    // The table is a host-side gather, never a graph parameter.
    tensors.remove("per_layer_token_embd.weight");
    let synthesized: BTreeSet<String> = tensors.keys().cloned().collect();
    assert!(
        synthesized.is_subset(&manifest),
        "the fixture invented tensors the manifest does not list: {:?}",
        synthesized.difference(&manifest).collect::<Vec<_>>()
    );

    let mut weights = WeightMap::from_tensors(tensors);
    build_qwen4exp_text_flow(&cfg, &mut weights, 4, true).expect("build the tiny model");

    let leftover: Vec<String> = weights.keys().map(str::to_string).collect();
    assert!(
        leftover.iter().all(|n| n.contains(".indexer.")),
        "the flow left model weights unread: {leftover:?}"
    );
    // ...and the indexer leftovers are exactly the ones the dense short-circuit
    // skips, not a missing projection.
    assert_eq!(
        leftover.len(),
        4,
        "one dense QSA layer should skip exactly its 4 indexer tensors, got {leftover:?}"
    );
}
