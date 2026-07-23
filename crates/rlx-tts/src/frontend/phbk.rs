
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use super::PhoneMap;

#[derive(Debug, Clone, Deserialize)]
pub struct Phbk {
    #[serde(default)]
    pub feats: Vec<String>,
    #[serde(default)]
    pub name: HashMap<String, Vec<i32>>,
    #[serde(default)]
    pub pos: HashMap<String, Vec<i32>>,
    #[serde(default)]
    pub left_wrd_pos: HashMap<String, Vec<i32>>,
    #[serde(default)]
    pub right_wrd_pos: HashMap<String, Vec<i32>>,
    #[serde(default)]
    pub senttype: HashMap<String, Vec<i32>>,
    #[serde(default)]
    pub question: Vec<String>,
}

impl Phbk {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }

    pub fn validate_against_phone_map(&self, map: &PhoneMap) -> Result<usize> {
        let mut overlap = 0usize;
        for key in self.name.keys() {
            if map.id(key).is_some() {
                overlap += 1;
            }
        }
        ensure!(
            overlap > 0,
            "phbk name table has no overlap with phone_map ({} keys)",
            self.name.len()
        );
        Ok(overlap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_phbk_if_present() {
        let path = std::path::Path::new("weights/tts/rlx-tts/frontend/phbk");
        let sym = std::path::Path::new("weights/tts/rlx-tts/symmap.json");
        if !path.is_file() || !sym.is_file() {
            return;
        }
        let phbk = Phbk::load(path).unwrap();
        assert!(!phbk.name.is_empty());
        let map = PhoneMap::load_symmap(sym).unwrap();
        let n = phbk.validate_against_phone_map(&map).unwrap();
        assert!(n > 20, "expected substantial phone overlap, got {n}");
    }
}
