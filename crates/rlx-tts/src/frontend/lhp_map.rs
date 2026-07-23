
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Debug, Clone, Default)]
pub struct LhpAlphabet {
    /// phone_map symbol → compact LHP fragment
    pub to_compact: HashMap<String, String>,
    /// compact fragment → preferred phone_map symbol (longest-match)
    compact_keys: Vec<String>,
    compact_to_phone: HashMap<String, String>,
}

impl LhpAlphabet {
    pub fn load_to_lhp(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let map: HashMap<String, String> = serde_json::from_str(&text)?;
        let mut compact_to_phone = HashMap::new();
        for (phone, compact) in &map {
            if compact.is_empty() {
                continue;
            }
            // Prefer first mapping; stressed variants often share compact forms.
            compact_to_phone
                .entry(compact.clone())
                .or_insert_with(|| phone.clone());
        }
        let mut compact_keys: Vec<String> = compact_to_phone.keys().cloned().collect();
        compact_keys.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        Ok(Self {
            to_compact: map,
            compact_keys,
            compact_to_phone,
        })
    }

    /// Greedy longest-match decode of a compact LHP string into phone_map tokens.
    pub fn compact_to_phones(&self, compact: &str) -> Vec<String> {
        let mut i = 0usize;
        let chars: Vec<char> = compact.chars().collect();
        let mut out = Vec::new();
        while i < chars.len() {
            let rest: String = chars[i..].iter().collect();
            let mut matched = false;
            for key in &self.compact_keys {
                if rest.starts_with(key.as_str()) {
                    if let Some(phone) = self.compact_to_phone.get(key) {
                        out.push(phone.clone());
                    }
                    i += key.chars().count();
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Fallback: single char if it is itself a phone symbol.
                let ch = chars[i].to_string();
                out.push(ch);
                i += 1;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_compact_roundtrip_if_present() {
        let Some(root) = crate::gguf_bundle::default_extract_dir() else {
            return;
        };
        let path = root.join("frontend/phonetic/to_lhp.json");
        if !path.is_file() {
            return;
        }
        let alpha = LhpAlphabet::load_to_lhp(path).unwrap();
        let phones = alpha.compact_to_phones("hEl'o&U");
        assert!(phones.iter().any(|p| p == "h" || p == "E" || p == "O:"));
    }
}
