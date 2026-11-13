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

//! PyLaia's `syms.txt`: one `symbol index` pair per line, `<ctc>` at 0.

use anyhow::{Context, Result, anyhow, ensure};
use std::path::Path;

/// The symbol PyLaia writes for a word separator.
pub const SPACE: &str = "<space>";
/// The CTC blank.
pub const BLANK: &str = "<ctc>";

/// Index → symbol table.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolTable {
    syms: Vec<String>,
}

impl SymbolTable {
    /// Read `syms.txt`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading symbols {}", path.display()))?;
        Self::parse(&text)
    }

    /// Parse `symbol index` lines. The symbol may itself be a space-free
    /// token such as `<space>`; the index is the last field.
    pub fn parse(text: &str) -> Result<Self> {
        let mut pairs = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let (sym, idx) = line
                .rsplit_once(' ')
                .ok_or_else(|| anyhow!("syms.txt line {}: {line:?}", n + 1))?;
            let idx: usize = idx
                .trim()
                .parse()
                .with_context(|| format!("syms.txt line {}: index {idx:?}", n + 1))?;
            pairs.push((idx, sym.to_string()));
        }
        let len = pairs.iter().map(|(i, _)| i + 1).max().unwrap_or(0);
        let mut syms = vec![String::new(); len];
        for (i, s) in pairs {
            ensure!(syms[i].is_empty(), "syms.txt: index {i} appears twice");
            syms[i] = s;
        }
        ensure!(
            syms.first().map(String::as_str) == Some(BLANK),
            "syms.txt: index 0 must be {BLANK}"
        );
        Ok(Self { syms })
    }

    /// Number of classes, blank included.
    pub fn len(&self) -> usize {
        self.syms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.syms.is_empty()
    }

    /// Symbol for class `i`.
    pub fn symbol(&self, i: usize) -> Option<&str> {
        self.syms.get(i).map(String::as_str)
    }

    /// Join decoded labels into text, turning `<space>` into `' '` (PyLaia's
    /// `--convert_spaces` output).
    pub fn decode(&self, labels: &[usize]) -> String {
        labels
            .iter()
            .filter_map(|&l| self.symbol(l))
            .map(|s| if s == SPACE { " " } else { s })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_decode() {
        let t = SymbolTable::parse("<ctc> 0\na 1\nb 2\n<space> 3\n").unwrap();
        assert_eq!(t.len(), 4);
        assert_eq!(t.decode(&[1, 3, 2]), "a b");
    }

    #[test]
    fn blank_must_be_first() {
        assert!(SymbolTable::parse("a 0\n<ctc> 1\n").is_err());
    }
}
