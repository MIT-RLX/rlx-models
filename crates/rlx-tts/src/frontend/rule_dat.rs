//!
//! ## Container
//!
//! - `u8` table count
//! - 3-byte magic `91 48 03`
//! - per table: `u32` rule count, NUL-terminated name, optional 4 zero pad,
//!   then rules (or a BinaryGraph body — see below)
//!
//! ## R-list rules (`g2p_post` / `g2p_lhp` / parts of `tn_prefix`)
//!
//! Per rule: NUL-terminated `R…` label, 9×`u32` header (`hdr[2]` = token count),
//! then `hdr[2]` tokens of 12 bytes `(type, value, flags)`.
//!
//! After tokens, a **trailer** of 8×`u32` is always present. Non-final rules may
//! append a 9th `u32` plus optional RHS bytes (`…\0\xff`) before the next `R`
//! label. After the **last** rule of a table, only the 8×`u32` trailer is
//! consumed — the next `u32` is the following table’s rule count.
//!
//!
//! | ty | Role (observed) |
//! |----|-----------------|
//! | 9  | Latin-1 code unit (`value` = code point) |
//! | 5  | Operator / rewrite marker (`value` often 2) |
//! | 3  | Boundary / class op (`value` often 2) |
//! | 1  | Slot / epsilon |
//! | 0  | End / padding (`value` may carry flags) |
//! | 2  | Secondary marker |
//! | 11 | Class / silence-ish (`value` 0/1/2) |
//! | 12 | Class |
//! | 16 | Terminator seen in rewrite R-list blobs |
//! | `1<<28` | LHS/RHS separator bit used as a type in some post rules |
//!
//! `rewrite_rule.dat` is primarily an **FRBinaryGraph** (FastRewriter), not a
//!
//! Literal `ty=9` char-run apply remains a best-effort subset for TN/g2p cleanup.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

const MAGIC: [u8; 3] = [0x91, 0x48, 0x03];
const TOK_CHAR: u32 = 9;
const TRAILER_U32S: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Char,
    OpReplace,
    OpBoundary,
    Slot,
    End,
    Class,
    SepBit,
    Other(u32),
}

#[derive(Debug, Clone)]
pub struct RuleToken {
    pub ty: u32,
    pub value: u32,
    pub flags: u32,
}

impl RuleToken {
    pub fn kind(&self) -> TokenKind {
        match self.ty {
            9 => TokenKind::Char,
            5 => TokenKind::OpReplace,
            3 => TokenKind::OpBoundary,
            1 => TokenKind::Slot,
            0 => TokenKind::End,
            11 | 12 => TokenKind::Class,
            0x1000_0000 => TokenKind::SepBit,
            other => TokenKind::Other(other),
        }
    }

    pub fn as_char(&self) -> Option<char> {
        if self.ty == TOK_CHAR {
            char::from_u32(self.value)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub label: String,
    pub header: [u32; 9],
    pub tokens: Vec<RuleToken>,
    /// Optional RHS / payload bytes between the 8-u32 trailer and the next rule.
    pub trailer_payload: Vec<u8>,
}

impl Rule {
    /// Concatenate type-9 code units (LHS/RHS blobs often encode as char runs).
    pub fn char_runs(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        for tok in &self.tokens {
            if let Some(ch) = tok.as_char() {
                cur.push(ch);
            } else if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// Best-effort string rewrite: first char-run → second char-run when present.
    pub fn literal_pair(&self) -> Option<(String, String)> {
        let runs = self.char_runs();
        if runs.len() >= 2 {
            Some((runs[0].clone(), runs[1].clone()))
        } else if runs.len() == 1 {
            // Single LHS run with RHS in trailer payload (NUL-terminated ASCII).
            if let Some(rhs) = trailer_ascii_rhs(&self.trailer_payload) {
                return Some((runs[0].clone(), rhs));
            }
            None
        } else {
            None
        }
    }

    /// Match LHS char-run before the first operator token; RHS from trailer or
    /// following char-run.
    pub fn operator_pair(&self) -> Option<(String, String)> {
        let mut lhs = String::new();
        let mut saw_op = false;
        let mut rhs = String::new();
        for tok in &self.tokens {
            match tok.kind() {
                TokenKind::Char => {
                    if let Some(ch) = tok.as_char() {
                        if saw_op {
                            rhs.push(ch);
                        } else {
                            lhs.push(ch);
                        }
                    }
                }
                TokenKind::OpReplace | TokenKind::OpBoundary | TokenKind::SepBit => {
                    saw_op = true;
                }
                TokenKind::Slot | TokenKind::End | TokenKind::Class | TokenKind::Other(_) => {}
            }
        }
        if lhs.is_empty() {
            return None;
        }
        if rhs.is_empty() {
            rhs = trailer_ascii_rhs(&self.trailer_payload).unwrap_or_default();
        }
        if saw_op {
            Some((lhs, rhs))
        } else {
            None
        }
    }
}

fn trailer_ascii_rhs(payload: &[u8]) -> Option<String> {
    if payload.is_empty() {
        return None;
    }
    let end = payload
        .iter()
        .position(|&b| b == 0 || b == 0xff)
        .unwrap_or(payload.len());
    let s = std::str::from_utf8(&payload[..end]).ok()?.trim();
    if s.is_empty() || !s.is_ascii() {
        None
    } else {
        Some(s.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct RuleTable {
    pub name: String,
    pub rules: Vec<Rule>,
    /// True when the table body did not start with an `R…` label (BinaryGraph).
    pub binary_graph: bool,
}

#[derive(Debug, Clone)]
pub struct RuleDat {
    pub tables: Vec<RuleTable>,
}

impl RuleDat {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 8, "rule.dat too short");
        let n_tables = bytes[0] as usize;
        ensure!(bytes[1..4] == MAGIC, "bad rule.dat magic {:02x?}", &bytes[1..4]);
        let mut off = 4;
        let mut tables = Vec::with_capacity(n_tables);
        for ti in 0..n_tables {
            if off + 4 > bytes.len() {
                break;
            }
            let n_rules = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            let (name, next) = match read_cstring(bytes, off) {
                Ok(v) => v,
                Err(_) if !tables.is_empty() => break,
                Err(e) => return Err(e),
            };
            off = next;
            // Optional padding / reserved u32 observed as zeros before first rule.
            if off + 4 <= bytes.len() && bytes[off..off + 4] == [0, 0, 0, 0] {
                if off + 5 < bytes.len() && bytes[off + 4].is_ascii_alphanumeric() {
                    off += 4;
                }
            }

            // BinaryGraph body (rewrite / TN ordinal/sms/…): does not start with R.
            if n_rules > 0 && off < bytes.len() && bytes[off] != b'R' {
                tables.push(RuleTable {
                    name,
                    rules: Vec::new(),
                    binary_graph: true,
                });
                if ti + 1 < n_tables {
                    off = find_next_table_start(bytes, off, n_tables - ti - 1)?;
                } else {
                    off = bytes.len();
                }
                continue;
            }

            let mut rules = Vec::with_capacity(n_rules.min(4096));
            let mut rules_ok = true;
            for ri in 0..n_rules {
                let last = ri + 1 == n_rules;
                match parse_rule(bytes, &mut off, last) {
                    Ok(rule) => rules.push(rule),
                    Err(_) if !tables.is_empty() || ri > 0 => {
                        rules_ok = false;
                        break;
                    }
                    Err(e) => {
                        return Err(e).with_context(|| format!("table {ti} ({name}) rule {ri}"));
                    }
                }
            }
            tables.push(RuleTable {
                name,
                rules,
                binary_graph: false,
            });
            if !rules_ok {
                break;
            }
            // TN / multi-table files pad between tables; resync to the next header.
            if ti + 1 < n_tables {
                let next = find_next_table_start(bytes, off, n_tables - ti - 1)?;
                if next < bytes.len() {
                    off = next;
                } else {
                    break;
                }
            }
        }
        Ok(Self { tables })
    }

    pub fn rule_count(&self) -> usize {
        self.tables.iter().map(|t| t.rules.len()).sum()
    }

    /// Apply literal / operator char-run substitutions left-to-right.
    pub fn apply_literals(&self, input: &str) -> String {
        let mut pairs = Vec::new();
        for table in &self.tables {
            for rule in &table.rules {
                if let Some(p) = rule.operator_pair().or_else(|| rule.literal_pair()) {
                    pairs.push(p);
                }
            }
        }
        // Longer LHS first to prefer specific matches.
        pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut out = input.to_string();
        for (lhs, rhs) in pairs {
            if lhs.is_empty() {
                continue;
            }
            out = out.replace(&lhs, &rhs);
        }
        out
    }
}

/// Heuristic: next table starts at `u32` + lowercase name + R-list or graph body.
fn find_next_table_start(bytes: &[u8], mut off: usize, _remaining: usize) -> Result<usize> {
    while off + 8 < bytes.len() {
        let n_rules = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if (1..10_000).contains(&n_rules) {
            if let Ok((name, name_end)) = read_cstring(bytes, off + 4) {
                let looks_name = !name.is_empty()
                    && name.len() < 64
                    && name.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                    && name
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_'));
                if looks_name {
                    let mut body = name_end;
                    if body + 4 <= bytes.len() && bytes[body..body + 4] == [0, 0, 0, 0] {
                        body += 4;
                    }
                    if body < bytes.len() && (bytes[body] == b'R' || bytes[body] == 1) {
                        return Ok(off);
                    }
                }
            }
        }
        off += 1;
    }
    Ok(bytes.len())
}

fn read_cstring(bytes: &[u8], off: usize) -> Result<(String, usize)> {
    ensure!(off < bytes.len(), "cstring past end");
    let end = bytes[off..]
        .iter()
        .position(|&b| b == 0)
        .context("unterminated cstring")?;
    let s = String::from_utf8_lossy(&bytes[off..off + end]).into_owned();
    Ok((s, off + end + 1))
}

fn parse_rule(bytes: &[u8], off: &mut usize, last_in_table: bool) -> Result<Rule> {
    let (label, next) = read_cstring(bytes, *off)?;
    *off = next;
    ensure!(
        label.starts_with('R') || label.chars().next().is_some_and(|c| c.is_ascii_alphanumeric()),
        "expected rule label, got {label:?}"
    );
    ensure!(*off + 36 <= bytes.len(), "truncated rule header for {label}");
    let mut header = [0u32; 9];
    for h in &mut header {
        *h = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
        *off += 4;
    }
    let n_tok = header[2] as usize;
    ensure!(
        *off + n_tok * 12 <= bytes.len(),
        "truncated tokens for {label}: need {n_tok}"
    );
    let mut tokens = Vec::with_capacity(n_tok);
    for _ in 0..n_tok {
        let ty = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
        let value = u32::from_le_bytes(bytes[*off + 4..*off + 8].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[*off + 8..*off + 12].try_into().unwrap());
        *off += 12;
        tokens.push(RuleToken { ty, value, flags });
    }

    let trailer_payload = if last_in_table {
        ensure!(
            *off + TRAILER_U32S * 4 <= bytes.len(),
            "truncated final trailer for {label}"
        );
        *off += TRAILER_U32S * 4;
        Vec::new()
    } else {
        let start = *off;
        skip_to_next_rule_label(bytes, off)?;
        bytes[start..*off].to_vec()
    };

    Ok(Rule {
        label,
        header,
        tokens,
        trailer_payload,
    })
}

fn skip_to_next_rule_label(bytes: &[u8], off: &mut usize) -> Result<()> {
    let start = *off;
    while *off < bytes.len() {
        if bytes[*off] == b'R' {
            if let Ok((lab, _)) = read_cstring(bytes, *off) {
                if lab.starts_with('R') && lab[1..].chars().all(|c| c.is_ascii_digit()) {
                    return Ok(());
                }
            }
        }
        *off += 1;
        // Guard runaway scans on corrupt blobs.
        if *off - start > 4096 {
            bail!("rule trailer exceeds 4KiB without next R label");
        }
    }
    bail!("no next rule label before EOF")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_buffer() {
        assert!(RuleDat::parse(&[1, 2, 3]).is_err());
    }

    #[test]
    fn parse_g2p_post_if_present() {
        let Some(root) = crate::gguf_bundle::default_extract_dir() else {
            return;
        };
        let path = root.join("frontend/g2p_post_rule.dat");
        if !path.is_file() {
            return;
        }
        let dat = RuleDat::load(path).expect("parse g2p_post");
        assert!(dat.tables.len() >= 2, "tables={}", dat.tables.len());
        assert!(
            dat.rule_count() >= 8,
            "expected R-list rules, got {}",
            dat.rule_count()
        );
        let names: Vec<_> = dat.tables.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.iter().any(|n| n.contains("hyphen")),
            "names={names:?}"
        );
        // At least one table may be BinaryGraph-shaped.
        let labels: Vec<_> = dat
            .tables
            .iter()
            .flat_map(|t| t.rules.iter().map(|r| r.label.as_str()))
            .collect();
        assert!(labels.iter().any(|l| l.starts_with('R')));
    }

    #[test]
    fn parse_g2p_lhp_if_present() {
        let Some(root) = crate::gguf_bundle::default_extract_dir() else {
            return;
        };
        let path = root.join("frontend/g2p_lhp_rule.dat");
        if !path.is_file() {
            return;
        }
        let dat = RuleDat::load(path).expect("parse g2p_lhp");
        assert!(!dat.tables.is_empty());
        assert!(dat.rule_count() >= 1 || dat.tables.iter().any(|t| t.binary_graph));
    }
}
