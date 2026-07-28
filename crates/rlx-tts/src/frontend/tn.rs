//! Text normalization from `tn_prefix_rule.dat` (literal substitutions).
//!
//! Pure RLX only applies safe operator/literal pairs harvested from R-list
//! tables — enable with `RLX_TTS_LOAD_TN_DAT=1`.

use std::path::Path;

use anyhow::Result;

use super::rule_dat::RuleDat;

#[derive(Debug, Clone, Default)]
pub struct TnPrefix {
    pairs: Vec<(String, String)>,
}

fn safe_tn_pair(lhs: &str, rhs: &str) -> bool {
    if lhs.is_empty() || lhs == rhs {
        return false;
    }
    // Reject tiny fragments that rewrite inside ordinary words (e.g. `m`→…).
    if lhs.chars().count() < 2 {
        return false;
    }
    let ok = |s: &str| {
        s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '\'' | '.' | ','))
    };
    ok(lhs) && ok(rhs)
}

impl TnPrefix {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let dat = RuleDat::load(path)?;
        let mut pairs = Vec::new();
        for table in &dat.tables {
            // Prefer operator-delimited LHS→RHS; fall back to adjacent char-runs
            // only when both sides look like printable word fragments (avoids
            // treating slot/class noise as global replace and mangling text).
            for rule in &table.rules {
                let pair = rule
                    .operator_pair()
                    .or_else(|| rule.literal_pair())
                    .filter(|(lhs, rhs)| safe_tn_pair(lhs, rhs));
                if let Some((lhs, rhs)) = pair {
                    pairs.push((lhs, rhs));
                }
            }
        }
        pairs.sort_by_key(|b| std::cmp::Reverse(b.0.len()));
        pairs.dedup();
        Ok(Self { pairs })
    }

    pub fn apply(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (lhs, rhs) in &self.pairs {
            if out.contains(lhs.as_str()) {
                out = out.replace(lhs, rhs);
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}
