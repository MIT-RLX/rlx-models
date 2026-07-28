//! Orthographic rewrite + optional LHP lexicon harvest from `rewrite_rule.dat`.
//!
//! applies a compact `rewrite_map.json` extracted locally via

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use super::rule_dat::RuleDat;

#[derive(Debug, Clone, Default)]
pub struct RewriteRules {
    dat: Option<RuleDat>,
    /// Longest-match orthographic replacements (`Dr.` → `Doctor`, …).
    map: Vec<(String, String)>,
    /// Compact-LHP pronunciations harvested from `\toi=lhp\…` markers.
    lhp_forms: Vec<String>,
}

impl RewriteRules {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let dat = RuleDat::parse(&bytes).ok();
        let lhp_forms = harvest_toi_lhp(&bytes);
        let map = load_rewrite_map(path.with_file_name("rewrite_map.json"));
        Ok(Self {
            dat,
            map,
            lhp_forms,
        })
    }

    /// Harvest `\toi=lhp\…` forms without fully parsing the 7 MB rule engine
    /// (avoids large allocations on constrained hosts).
    pub fn load_harvest_only(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let lhp_forms = harvest_toi_lhp(&bytes);
        let map = load_rewrite_map(path.with_file_name("rewrite_map.json"));
        Ok(Self {
            dat: None,
            map,
            lhp_forms,
        })
    }

    /// Load only the compact `rewrite_map.json` (no 7 MB BinaryGraph parse).
    pub fn load_map_only(map_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            dat: None,
            map: load_rewrite_map(map_path),
            lhp_forms: Vec::new(),
        })
    }

    pub fn apply_literals(&self, text: &str) -> String {
        let mut out = apply_rewrite_map(text, &self.map);
        if let Some(dat) = &self.dat {
            out = dat.apply_literals(&out);
        }
        out
    }

    pub fn map_len(&self) -> usize {
        self.map.len()
    }

    pub fn lhp_form_count(&self) -> usize {
        self.lhp_forms.len()
    }

    /// Build a compact-LHP → itself set for post-rule presence checks.
    pub fn lhp_index(&self) -> HashMap<String, ()> {
        self.lhp_forms.iter().cloned().map(|s| (s, ())).collect()
    }
}

fn load_rewrite_map(path: impl AsRef<Path>) -> Vec<(String, String)> {
    let path = path.as_ref();
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    let Some(obj) = value.as_object() else {
        return Vec::new();
    };
    let mut pairs: Vec<(String, String)> = obj
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect();
    pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
    pairs
}

fn apply_rewrite_map(text: &str, map: &[(String, String)]) -> String {
    if map.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let rest: String = chars[i..].iter().collect();
        let mut hit = None;
        for (lhs, rhs) in map {
            if rest.starts_with(lhs.as_str()) {
                // Prefer token boundary: start of string or non-letter before.
                let ok_start = i == 0 || !chars[i - 1].is_ascii_alphanumeric();
                if ok_start {
                    hit = Some((lhs.chars().count(), rhs.as_str()));
                    break;
                }
            }
        }
        if let Some((n, rhs)) = hit {
            out.push_str(rhs);
            i += n;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn harvest_toi_lhp(bytes: &[u8]) -> Vec<String> {
    // Scan raw bytes (ASCII markers) — avoid `from_utf8_lossy` doubling the 7 MB buffer.
    let marker = b"\\toi=lhp\\";
    let end_marker = b"\\toi=orth\\";
    let mut out = Vec::new();
    let mut start = 0usize;
    while let Some(rel) = find_bytes(&bytes[start..], marker) {
        let abs = start + rel + marker.len();
        if let Some(end_rel) = find_bytes(&bytes[abs..], end_marker) {
            let raw = &bytes[abs..abs + end_rel];
            let cleaned: String = raw
                .iter()
                .copied()
                .filter(|&c| c != 0x1b && !(c < 0x20 && c != b'\t'))
                .map(|c| c as char)
                .collect::<String>()
                .trim()
                .to_string();
            if !cleaned.is_empty() {
                out.push(cleaned);
            }
            start = abs + end_rel + end_marker.len();
        } else {
            break;
        }
    }
    out.sort();
    out.dedup();
    out
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harvest_rewrite_if_present() {
        let Some(root) = crate::gguf_bundle::default_extract_dir() else {
            return;
        };
        let path = root.join("frontend/rewrite_rule.dat");
        if !path.is_file() {
            return;
        }
        let rw = RewriteRules::load(path).unwrap();
        assert!(
            rw.lhp_form_count() > 100,
            "expected many harvested LHP forms"
        );
    }

    #[test]
    fn rewrite_map_dr_if_present() {
        let Some(root) = crate::gguf_bundle::default_extract_dir() else {
            return;
        };
        let path = root.join("frontend/rewrite_map.json");
        if !path.is_file() {
            return;
        }
        let map = load_rewrite_map(path);
        assert!(
            map.iter().any(|(k, v)| k == "Dr." && v == "Doctor"),
            "rewrite_map should include Dr.→Doctor"
        );
        assert_eq!(apply_rewrite_map("Dr. Smith", &map), "Doctor Smith");
    }
}
