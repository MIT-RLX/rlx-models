//! Optional prompt-index sidecar (`gprm_index.json`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Debug, Clone, Default)]
pub struct GprmIndex {
    /// Exact prompt text → stem id.
    by_text: HashMap<String, String>,
}

impl GprmIndex {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        let mut by_text = HashMap::new();
        if let Some(obj) = v.as_object() {
            for (k, val) in obj {
                if let Some(stem) = val.as_str() {
                    by_text.insert(k.clone(), stem.to_string());
                } else if let Some(stem) = val.get("stem").and_then(|x| x.as_str()) {
                    by_text.insert(k.clone(), stem.to_string());
                }
            }
        }
        Ok(Self { by_text })
    }

    pub fn lookup_stem(&self, text: &str) -> Option<&str> {
        self.by_text.get(text.trim()).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.by_text.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_index_if_present() {
        let Some(root) = crate::gguf_bundle::default_extract_dir() else {
            return;
        };
        let path = root.join("frontend/gprm_index.json");
        if !path.is_file() {
            return;
        }
        let idx = GprmIndex::load(path).unwrap();
        assert!(idx.len() > 0);
        assert_eq!(idx.lookup_stem("Hi."), Some("hi"));
    }
}
