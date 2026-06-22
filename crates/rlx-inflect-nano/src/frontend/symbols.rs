//! Symbol table + phoneme→id mapping (mirrors `tiny_tts/text/symbols.py` and
//! `phonemes_to_ids`). Loaded from the exported `frontend/symbols.json` so the
//! id assignment is byte-identical to Python.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct SymbolsJson {
    symbols: Vec<String>,
    num_tones: usize,
    num_languages: usize,
    language_id_map: HashMap<String, i64>,
    language_tone_start_map: HashMap<String, i64>,
}

pub struct Symbols {
    sym_to_id: HashMap<String, i64>,
    unk_id: i64,
    pub num_tones: usize,
    pub num_languages: usize,
    language_id_map: HashMap<String, i64>,
    language_tone_start_map: HashMap<String, i64>,
}

impl Symbols {
    pub fn load(path: &Path) -> Result<Self> {
        let j: SymbolsJson = serde_json::from_str(
            &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
        )?;
        let sym_to_id: HashMap<String, i64> = j
            .symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as i64))
            .collect();
        let unk_id = *sym_to_id.get("UNK").context("symbols.json missing UNK")?;
        Ok(Self {
            sym_to_id,
            unk_id,
            num_tones: j.num_tones,
            num_languages: j.num_languages,
            language_id_map: j.language_id_map,
            language_tone_start_map: j.language_tone_start_map,
        })
    }

    /// `map_phoneme`: punctuation remap, then keep if known else "UNK".
    pub fn map_phoneme(&self, ph: &str) -> String {
        let mapped = match ph {
            "：" | "；" | "，" | "·" | "、" => ",",
            "。" => ".",
            "！" => "!",
            "？" => "?",
            "\n" => ".",
            "..." => "…",
            "v" => "V",
            other => other,
        };
        if self.sym_to_id.contains_key(mapped) {
            mapped.to_string()
        } else {
            "UNK".to_string()
        }
    }

    /// `phonemes_to_ids(phones, tones, language)`.
    pub fn phonemes_to_ids(
        &self,
        phones: &[String],
        tones: &[i64],
        language: &str,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let phone_ids: Vec<i64> = phones
            .iter()
            .map(|s| *self.sym_to_id.get(s.as_str()).unwrap_or(&self.unk_id))
            .collect();
        let tone_start = self.language_tone_start_map[language];
        let tone_ids: Vec<i64> = tones.iter().map(|t| t + tone_start).collect();
        let lang_id = self.language_id_map[language];
        let lang_ids = vec![lang_id; phone_ids.len()];
        (phone_ids, tone_ids, lang_ids)
    }
}

/// `commons.insert_blanks(lst, 0)`: `[0, x0, 0, x1, ..., 0]`.
pub fn insert_blanks(lst: &[i64]) -> Vec<i64> {
    let mut out = vec![0i64; lst.len() * 2 + 1];
    for (i, &v) in lst.iter().enumerate() {
        out[2 * i + 1] = v;
    }
    out
}
