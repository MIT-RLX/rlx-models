// RLX models — GGUF quant selection.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Selection of the best GGUF file from a repository's file list, given an
//! optional quantization selector.

use crate::model_ref::gguf_matches_quant_selector;

/// Ordered default preference of quant selectors, used when the caller does
/// not specify one. Earlier entries are preferred.
///
/// The ordering trades size/quality: `Q4_K_M` is the common "good enough,
/// small" default, followed by higher-quality K-quants, then `Q8_0`, then
/// full-precision `F16`.
pub const DEFAULT_QUANT_PREFERENCE: &[&str] = &["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"];

/// Whether `file` is a GGUF file (case-insensitive extension check).
pub fn is_gguf_file(file: &str) -> bool {
    file.to_ascii_lowercase().ends_with(".gguf")
}

/// The first shard of a split GGUF, or a non-split GGUF. Returns `false` for
/// shards `-00002-of-...` and later so the picker only ever surfaces the entry
/// point of a sharded distribution.
fn is_primary_gguf_shard(file: &str) -> bool {
    match crate::model_ref::split_gguf_shard_info(file) {
        Some(shard) => shard.part == "00001",
        None => true,
    }
}

/// Pick the best-matching GGUF file from `files`.
///
/// - If `selector` is `Some`, returns the first file matching that selector.
/// - If `selector` is `None`, walks [`DEFAULT_QUANT_PREFERENCE`] and returns
///   the first file matching a preferred quant; if none match, falls back to
///   the first GGUF file present.
///
/// Only primary shards (or non-sharded files) are considered.
pub fn select_gguf<'a, I, S>(files: I, selector: Option<&str>) -> Option<&'a str>
where
    I: IntoIterator<Item = &'a S>,
    S: AsRef<str> + 'a,
{
    let ggufs: Vec<&'a str> = files
        .into_iter()
        .map(AsRef::as_ref)
        .filter(|file| is_gguf_file(file) && is_primary_gguf_shard(file))
        .collect();

    if let Some(selector) = selector {
        return ggufs
            .iter()
            .copied()
            .find(|file| gguf_matches_quant_selector(file, selector));
    }

    for preferred in DEFAULT_QUANT_PREFERENCE {
        if let Some(found) = ggufs
            .iter()
            .copied()
            .find(|file| gguf_matches_quant_selector(file, preferred))
        {
            return Some(found);
        }
    }

    ggufs.first().copied()
}

/// All GGUF shard files (in the order given) belonging to the same
/// distribution as `primary`. For a non-split GGUF this is just `[primary]`.
///
/// A caller that has selected a primary shard uses this to gather every part
/// that must be downloaded together.
pub fn gguf_shard_group<'a, I, S>(files: I, primary: &str) -> Vec<&'a str>
where
    I: IntoIterator<Item = &'a S>,
    S: AsRef<str> + 'a,
{
    let Some(prefix) = crate::model_ref::split_gguf_shard_info(primary).map(|shard| shard.prefix)
    else {
        // Not a split shard: just this one file.
        return files
            .into_iter()
            .map(AsRef::as_ref)
            .filter(|file| *file == primary)
            .collect();
    };

    files
        .into_iter()
        .map(AsRef::as_ref)
        .filter(|file| {
            crate::model_ref::split_gguf_shard_info(file)
                .is_some_and(|shard| shard.prefix == prefix)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_explicit_selector() {
        let files = vec![
            "Model-Q4_K_M.gguf".to_string(),
            "Model-Q8_0.gguf".to_string(),
        ];
        assert_eq!(select_gguf(&files, Some("Q8_0")), Some("Model-Q8_0.gguf"));
        assert_eq!(
            select_gguf(&files, Some("Q4_K_M")),
            Some("Model-Q4_K_M.gguf")
        );
        assert_eq!(select_gguf(&files, Some("Q2_K")), None);
    }

    #[test]
    fn selects_default_preference_order() {
        // Both present -> prefer Q4_K_M over Q8_0 per DEFAULT_QUANT_PREFERENCE.
        let files = vec![
            "Model-Q8_0.gguf".to_string(),
            "Model-Q4_K_M.gguf".to_string(),
        ];
        assert_eq!(select_gguf(&files, None), Some("Model-Q4_K_M.gguf"));
    }

    #[test]
    fn selects_next_preference_when_top_missing() {
        let files = vec!["Model-Q6_K.gguf".to_string(), "Model-Q8_0.gguf".to_string()];
        // Q4_K_M and Q5_K_M absent -> Q6_K.
        assert_eq!(select_gguf(&files, None), Some("Model-Q6_K.gguf"));
    }

    #[test]
    fn falls_back_to_first_gguf_when_no_preference_matches() {
        let files = vec![
            "config.json".to_string(),
            "Model-Q2_K.gguf".to_string(),
            "Model-Q3_K.gguf".to_string(),
        ];
        assert_eq!(select_gguf(&files, None), Some("Model-Q2_K.gguf"));
    }

    #[test]
    fn ignores_non_primary_shards_when_selecting() {
        let files = vec![
            "Model-Q4_K_M-00002-of-00002.gguf".to_string(),
            "Model-Q4_K_M-00001-of-00002.gguf".to_string(),
        ];
        assert_eq!(
            select_gguf(&files, Some("Q4_K_M")),
            Some("Model-Q4_K_M-00001-of-00002.gguf")
        );
    }

    #[test]
    fn gathers_shard_group_for_split_gguf() {
        let files = vec![
            "config.json".to_string(),
            "Model-Q4_K_M-00001-of-00002.gguf".to_string(),
            "Model-Q4_K_M-00002-of-00002.gguf".to_string(),
        ];
        let group = gguf_shard_group(&files, "Model-Q4_K_M-00001-of-00002.gguf");
        assert_eq!(group.len(), 2);
        assert!(group.contains(&"Model-Q4_K_M-00001-of-00002.gguf"));
        assert!(group.contains(&"Model-Q4_K_M-00002-of-00002.gguf"));
    }

    #[test]
    fn shard_group_for_non_split_is_single_file() {
        let files = vec!["Model-Q4_K_M.gguf".to_string(), "other.gguf".to_string()];
        let group = gguf_shard_group(&files, "Model-Q4_K_M.gguf");
        assert_eq!(group, vec!["Model-Q4_K_M.gguf"]);
    }
}
