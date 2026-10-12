// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Which of a checkpoint's tensors one block of a split model needs, and
//! how to read only those off disk.
//!
//! A model split across machines is usually split *because* no one machine
//! can hold it. Loading the whole checkpoint and then keeping a slice of it
//! defeats the point: the peak is what fails, not the steady state. So the
//! decision about what a block needs has to come before the read, not
//! after it.
//!
//! [`block_needs`] is that decision, and it is the only place it is made.
//! [`load`] uses it to read a block's tensors and nothing else;
//! [`filter`] uses it to narrow tensors already in memory, which is what a
//! single machine does when it holds the whole checkpoint anyway. Those two
//! must select the same tensors — a block that loads a weight its graph
//! does not want wastes memory, and one that skips a weight its graph does
//! want fails at graph-building on that rank alone, with the model loaded
//! and every shape agreeing.
//!
//! One tensor is deliberately loaded twice. Tied embeddings make the
//! embedding table the output head as well, and those are opposite ends of
//! the pipeline — so the first and last blocks each hold a copy of what is
//! frequently the largest tensor in the model. That is the price of tying
//! rather than a fault here, but it is worth knowing when working out how
//! many layers a given machine can take: the end blocks carry a
//! vocab-sized tensor that the middle ones do not.

use crate::pipeline_block::Tensors;
use anyhow::{Context, Result, anyhow, bail};
use rlx_distributed::{BlockRole, block_role};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Does a block holding `layers` need the tensor called `key`?
///
/// `key` is the canonical (HF-style) name — run GGUF names through
/// [`crate::weight_loader::gguf_to_hf_name`] first, which is what makes one
/// set of names enough for both formats.
///
/// `embed` and `logits` are the ends of the model: the block that turns
/// token ids into hidden states, and the block that turns hidden states
/// back into logits. On one machine both are the same block; split across
/// four, two blocks are neither.
pub fn block_needs(
    key: &str,
    layers: &Range<usize>,
    embed: bool,
    logits: bool,
    tied_embeddings: bool,
) -> bool {
    if let Some(rest) = key.strip_prefix("model.layers.") {
        // Parsing the index rather than matching a prefix: "model.layers.1"
        // is a prefix of "model.layers.10", and a block that took the
        // latter for the former would hold nine layers it was never asked
        // for and fail on a shape much later.
        let Some(idx) = rest.split('.').next().and_then(|n| n.parse::<usize>().ok()) else {
            return false;
        };
        return layers.contains(&idx);
    }
    match key {
        // Tied embeddings mean the embedding table *is* the output head, so
        // the block holding the head needs the table even though it embeds
        // nothing.
        "model.embed_tokens.weight" => embed || (logits && tied_embeddings),
        // The final norm belongs with the head. A block that applies it
        // without owning the head would hand the next block a hidden state
        // that has already been normalised.
        "model.norm.weight" => logits,
        "lm_head.weight" => logits && !tied_embeddings,
        _ => false,
    }
}

/// Which ends of the model a rank holds.
///
/// Ranks count backwards: rank 0 holds the *last* layers and the head, so
/// it is the only rank with logits and the only one that can answer a
/// request.
pub fn block_ends(rank: u32, world: u32) -> (bool, bool) {
    match block_role(rank, world) {
        BlockRole::Single => (true, true),
        BlockRole::First => (true, false),
        BlockRole::Middle => (false, false),
        BlockRole::Last => (false, true),
    }
}

/// Narrow tensors already in memory to the ones this block needs.
pub fn filter(
    all: Tensors,
    layers: &Range<usize>,
    embed: bool,
    logits: bool,
    tied: bool,
) -> Tensors {
    all.into_iter()
        .filter(|(k, _)| block_needs(k, layers, embed, logits, tied))
        .collect()
}

/// What a block load touched, for the caller to report.
///
/// Returned rather than logged because this crate does not log: which rank
/// this is and whether anyone wants to hear about it are both the caller's
/// business.
#[derive(Debug, Clone)]
pub struct BlockLoad {
    pub tensors: Tensors,
    /// How many of the checkpoint's tensors were passed over. With
    /// `tensors.len()` this gives the share of the model this block holds,
    /// which is the number worth printing: it is the claim that sharding is
    /// doing anything at all.
    pub skipped: usize,
    pub files: usize,
    pub elapsed: Duration,
}

impl BlockLoad {
    pub fn loaded(&self) -> usize {
        self.tensors.len()
    }
}

/// Every weight file a model is stored in, in a stable order.
///
/// A checkpoint large enough to be worth sharding is itself usually split
/// over several files, and a block's layers do not line up with that split
/// — so each block reads what it needs from each file rather than assuming
/// one.
pub fn weight_files(weights: &Path) -> Result<Vec<PathBuf>> {
    if weights.is_file() {
        return Ok(vec![weights.to_path_buf()]);
    }
    if !weights.is_dir() {
        bail!("no weights at {weights:?}");
    }
    let mut safetensors = Vec::new();
    let mut gguf = Vec::new();
    for entry in std::fs::read_dir(weights).with_context(|| format!("reading {weights:?}"))? {
        let path = entry?.path();
        match path.extension().and_then(|e| e.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("safetensors") => safetensors.push(path),
            Some(e) if e.eq_ignore_ascii_case("gguf") => gguf.push(path),
            _ => {}
        }
    }
    // Safetensors wins where both are present, matching what the
    // whole-model loader picks for the same directory.
    let mut files = if safetensors.is_empty() {
        gguf
    } else {
        safetensors
    };
    if files.is_empty() {
        bail!("no .safetensors or .gguf file in {weights:?}");
    }
    files.sort();
    Ok(files)
}

/// Read only the tensors this rank's block needs.
///
/// `layers` must be the same division every rank was given. Two ranks
/// disagreeing is the failure this module cannot catch: each one loads a
/// self-consistent set of weights, builds a graph that compiles, and the
/// pipeline runs some layer twice and another never.
pub fn load(
    weights: &Path,
    layers: Range<usize>,
    rank: u32,
    world: u32,
    tied_embeddings: bool,
) -> Result<BlockLoad> {
    let (embed, logits) = block_ends(rank, world);
    let files = weight_files(weights)?;
    let started = Instant::now();

    // One thread per weight file. A checkpoint worth sharding comes in many
    // files, and reading them one after another leaves the disk idle while
    // each tensor is dequantised to f32 — which is the expensive half. The
    // files are independent, so there is nothing to coordinate.
    let per_file: Vec<Result<(Tensors, usize)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = files
            .iter()
            .map(|file| {
                let layers = layers.clone();
                scope.spawn(move || read_needed(file, &layers, embed, logits, tied_embeddings))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow!("a weight reader panicked")))
            })
            .collect()
    });

    let mut tensors = Tensors::new();
    let mut skipped = 0usize;
    for result in per_file {
        let (part, missed) = result?;
        skipped += missed;
        tensors.extend(part);
    }

    if tensors.is_empty() {
        bail!(
            "rank {rank} of {world} found none of the {} tensors it needs in \
             {weights:?} — is every rank pointed at the same model?",
            layers.len()
        );
    }
    Ok(BlockLoad {
        tensors,
        skipped,
        files: files.len(),
        elapsed: started.elapsed(),
    })
}

/// Take this block's tensors out of one weight file.
///
/// Returns what was read and how many tensors were passed over.
fn read_needed(
    file: &Path,
    layers: &Range<usize>,
    embed: bool,
    logits: bool,
    tied: bool,
) -> Result<(Tensors, usize)> {
    let path = file
        .to_str()
        .ok_or_else(|| anyhow!("weights path {file:?} is not valid UTF-8"))?;
    let mut loader =
        crate::weight_loader::load_from_path(path).with_context(|| format!("opening {file:?}"))?;

    let mut out = Tensors::new();
    let mut skipped = 0usize;
    for raw in loader.remaining_keys() {
        let canonical = crate::weight_loader::gguf_to_hf_name(&raw).unwrap_or_else(|| raw.clone());
        if !block_needs(&canonical, layers, embed, logits, tied) {
            skipped += 1;
            continue;
        }
        let t = loader
            .take(&raw)
            .with_context(|| format!("reading {raw}"))?;
        out.insert(canonical, t);
    }

    // The tensors are ours now, held as f32. The loader's mapping of the
    // file is a second, full-size copy of everything it touched, and on
    // macOS the page cache will not give it back on its own. Letting it go
    // matters most in exactly the case sharding is for: a block of a
    // checkpoint far larger than the machine's memory.
    loader.release_mapped_pages();
    Ok((out, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_distributed::pipeline_layer_range;

    fn needs(key: &str, layers: Range<usize>, embed: bool, logits: bool, tied: bool) -> bool {
        block_needs(key, &layers, embed, logits, tied)
    }

    #[test]
    fn a_block_takes_its_own_layers_and_no_others() {
        let mid = 9..18;
        assert!(needs(
            "model.layers.9.self_attn.q_proj.weight",
            mid.clone(),
            false,
            false,
            false
        ));
        assert!(needs(
            "model.layers.17.mlp.down_proj.weight",
            mid.clone(),
            false,
            false,
            false
        ));
        assert!(!needs(
            "model.layers.8.mlp.down_proj.weight",
            mid.clone(),
            false,
            false,
            false
        ));
        assert!(!needs(
            "model.layers.18.mlp.down_proj.weight",
            mid,
            false,
            false,
            false
        ));
    }

    #[test]
    fn the_layer_index_is_not_matched_by_prefix() {
        // "model.layers.1" is a prefix of "model.layers.10". Matching on it
        // would give a block nine layers it never asked for, and a shape
        // error much later with nothing pointing here.
        let one = 1..2;
        assert!(needs(
            "model.layers.1.mlp.up_proj.weight",
            one.clone(),
            false,
            false,
            false
        ));
        assert!(!needs(
            "model.layers.10.mlp.up_proj.weight",
            one.clone(),
            false,
            false,
            false
        ));
        assert!(!needs(
            "model.layers.12.mlp.up_proj.weight",
            one,
            false,
            false,
            false
        ));
    }

    #[test]
    fn only_the_embedding_block_reads_the_embedding_table() {
        let l = 0..4;
        let k = "model.embed_tokens.weight";
        assert!(needs(k, l.clone(), true, false, false));
        assert!(!needs(k, l.clone(), false, false, false));
        // Untied, the block with the head has its own lm_head.weight and
        // does not want the table.
        assert!(!needs(k, l, false, true, false));
    }

    #[test]
    fn a_tied_head_pulls_the_table_onto_the_last_block() {
        // The case that broke a real pipeline: tied weights mean
        // `lm_head.weight` does not exist, so the block holding the head
        // needs the embedding table even though it embeds nothing.
        let l = 24..28;
        assert!(needs(
            "model.embed_tokens.weight",
            l.clone(),
            false,
            true,
            true
        ));
        assert!(!needs("lm_head.weight", l.clone(), false, true, true));
        assert!(needs("lm_head.weight", l, false, true, false));
    }

    #[test]
    fn the_final_norm_goes_with_the_head() {
        let k = "model.norm.weight";
        assert!(needs(k, 24..28, false, true, false));
        assert!(!needs(k, 0..4, true, false, false));
        assert!(!needs(k, 4..8, false, false, false));
    }

    #[test]
    fn an_unknown_tensor_is_nobodys() {
        // Better to leave a tensor unloaded than to give every block a copy
        // of something none of them can place.
        for (embed, logits) in [(true, true), (true, false), (false, false), (false, true)] {
            assert!(!needs("something.new.weight", 0..4, embed, logits, false));
            assert!(!needs(
                "model.layers.notanumber.weight",
                0..4,
                embed,
                logits,
                false
            ));
        }
    }

    #[test]
    fn ranks_count_backwards_so_rank_zero_holds_the_head() {
        assert_eq!(block_ends(0, 1), (true, true));
        assert_eq!(block_ends(0, 2), (false, true));
        assert_eq!(block_ends(1, 2), (true, false));
        assert_eq!(block_ends(0, 4), (false, true));
        assert_eq!(block_ends(1, 4), (false, false));
        assert_eq!(block_ends(2, 4), (false, false));
        assert_eq!(block_ends(3, 4), (true, false));
    }

    /// The names a llama-shaped checkpoint actually contains.
    fn checkpoint(layers: usize, tied: bool) -> Vec<String> {
        let mut keys = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
        ];
        if !tied {
            keys.push("lm_head.weight".into());
        }
        for i in 0..layers {
            for part in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ] {
                keys.push(format!("model.layers.{i}.{part}"));
            }
        }
        keys
    }

    #[test]
    fn every_tensor_is_claimed_by_exactly_one_block_bar_a_tied_table() {
        // The property that makes selective loading safe: a tensor no block
        // claims is a weight that is simply not in the model any more, and
        // the first rank to need it fails on its own while the others sit
        // waiting.
        //
        // One tensor is claimed twice, and has to be. Tied embeddings make
        // the table both the input embedding and the output head, and those
        // are the two ends of the pipeline — different machines. So a
        // sharded tied model holds two copies of what is often its single
        // largest tensor, and that is the cost of tying, not a mistake
        // here. It is worth remembering when deciding how many layers a
        // machine can hold: the end blocks carry a vocab-sized tensor the
        // middle ones do not.
        for tied in [false, true] {
            for layers in [4usize, 28, 30, 32] {
                let keys = checkpoint(layers, tied);
                for world in 1..=8u32 {
                    if world as usize > layers {
                        continue;
                    }
                    let mut claims = std::collections::HashMap::new();
                    for rank in 0..world {
                        let range = pipeline_layer_range(layers, rank, world);
                        let (embed, logits) = block_ends(rank, world);
                        for k in &keys {
                            if block_needs(k, &range, embed, logits, tied) {
                                *claims.entry(k.clone()).or_insert(0u32) += 1;
                            }
                        }
                    }
                    for k in &keys {
                        let twice = tied && world > 1 && k == "model.embed_tokens.weight";
                        assert_eq!(
                            claims.get(k).copied().unwrap_or(0),
                            if twice { 2 } else { 1 },
                            "tied={tied} layers={layers} world={world}: {k}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn deciding_before_the_read_selects_what_narrowing_after_it_would() {
        // `load` decides what to read from the key alone; `filter` narrows
        // tensors already in memory. They are the same decision made at
        // two different times, and the whole point of selective loading is
        // that the earlier one is not a different answer. Written as two
        // functions these drifted apart once already — in the sibling case
        // of dividing layers, where two self-consistent definitions of an
        // even split disagreed about the remainder and each passed its own
        // tests.
        for tied in [false, true] {
            for layers in [4usize, 30] {
                let keys = checkpoint(layers, tied);
                let all: Tensors = keys
                    .iter()
                    .map(|k| (k.clone(), (vec![0.0f32], vec![1usize])))
                    .collect();
                for world in 1..=4u32 {
                    if world as usize > layers {
                        continue;
                    }
                    for rank in 0..world {
                        let range = pipeline_layer_range(layers, rank, world);
                        let (embed, logits) = block_ends(rank, world);
                        let narrowed = filter(all.clone(), &range, embed, logits, tied);
                        let chosen: std::collections::BTreeSet<&String> = keys
                            .iter()
                            .filter(|k| block_needs(k, &range, embed, logits, tied))
                            .collect();
                        let got: std::collections::BTreeSet<&String> = narrowed.keys().collect();
                        assert_eq!(
                            chosen, got,
                            "tied={tied} layers={layers} world={world} rank={rank}"
                        );
                    }
                }
            }
        }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("rlx-block-weights-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"").unwrap();
        p
    }

    #[test]
    fn a_single_file_is_taken_as_given() {
        let d = tmpdir("one");
        let f = touch(&d, "model.safetensors");
        assert_eq!(weight_files(&f).unwrap(), vec![f]);
    }

    #[test]
    fn a_sharded_checkpoint_comes_back_in_order() {
        let d = tmpdir("shards");
        touch(&d, "model-00002-of-00002.safetensors");
        touch(&d, "model-00001-of-00002.safetensors");
        touch(&d, "config.json");
        let files = weight_files(&d).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0].to_string_lossy().contains("00001"));
        assert!(files[1].to_string_lossy().contains("00002"));
    }

    #[test]
    fn safetensors_wins_over_gguf_in_the_same_directory() {
        // Whichever the whole-model loader would pick for this directory,
        // so a block does not end up reading a different quantisation of
        // the same model than its peers.
        let d = tmpdir("both");
        touch(&d, "model.gguf");
        touch(&d, "model.safetensors");
        let files = weight_files(&d).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().ends_with(".safetensors"));
    }

    #[test]
    fn a_directory_with_no_weights_says_so() {
        let d = tmpdir("empty");
        touch(&d, "config.json");
        let err = weight_files(&d).unwrap_err().to_string();
        assert!(err.contains("no .safetensors or .gguf"), "{err}");
        let err = weight_files(&d.join("nope")).unwrap_err().to_string();
        assert!(err.contains("no weights at"), "{err}");
    }
}
