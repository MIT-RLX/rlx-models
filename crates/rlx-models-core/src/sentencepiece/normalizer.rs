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

//! `normalizer.cc`: the precompiled charsmap, user-defined symbols and the
//! whitespace rules, with the normalized → original byte alignment.

use std::borrow::Cow;
use std::collections::HashSet;

use anyhow::{Result, bail};

use super::{NormalizerSpec, SPACE_SYMBOL, char_len};

/// Longest-match set of strings (`PrefixMatcher`).
#[derive(Default)]
pub(super) struct PrefixMatcher {
    set: HashSet<String>,
    max_bytes: usize,
}

impl PrefixMatcher {
    pub fn new(items: impl IntoIterator<Item = String>) -> Self {
        let set: HashSet<String> = items.into_iter().collect();
        let max_bytes = set.iter().map(String::len).max().unwrap_or(0);
        Self { set, max_bytes }
    }

    /// Length of the longest member `s` starts with.
    pub fn longest(&self, s: &str) -> Option<usize> {
        if self.set.is_empty() {
            return None;
        }
        (1..=self.max_bytes.min(s.len()))
            .rev()
            .filter(|&n| s.is_char_boundary(n))
            .find(|&n| self.set.contains(&s[..n]))
    }
}

pub(super) struct Normalizer {
    pub spec: NormalizerSpec,
    charsmap: Option<Charsmap>,
    pub treat_whitespace_as_suffix: bool,
}

impl Normalizer {
    pub fn new(spec: NormalizerSpec, treat_whitespace_as_suffix: bool) -> Result<Self> {
        let charsmap = if spec.precompiled_charsmap.is_empty() {
            None
        } else {
            Some(Charsmap::parse(&spec.precompiled_charsmap)?)
        };
        Ok(Self {
            spec,
            charsmap,
            treat_whitespace_as_suffix,
        })
    }

    /// `Normalizer::Normalize`: the normalized text and, for each of its
    /// bytes plus one past the end, the original byte offset it came from.
    pub fn normalize(&self, text: &str, user: &PrefixMatcher) -> (String, Vec<usize>) {
        let spec = &self.spec;
        let mut input = text;
        let mut consumed = 0;
        let mut out = String::with_capacity(text.len() * 3 / 2);
        let mut n2o = Vec::with_capacity(text.len() * 3 / 2 + 1);
        if spec.remove_extra_whitespaces {
            while !input.is_empty() {
                let (s, n) = self.normalize_prefix(input, user);
                if s != " " {
                    break;
                }
                input = &input[n..];
                consumed += n;
            }
        }
        if input.is_empty() {
            // The C++ returns before recording any offset.
            n2o.push(0);
            return (out, n2o);
        }
        let ws = if spec.escape_whitespaces {
            SPACE_SYMBOL
        } else {
            " "
        };
        let add_ws = |out: &mut String, n2o: &mut Vec<usize>, consumed: usize| {
            out.push_str(ws);
            n2o.extend(std::iter::repeat_n(consumed, ws.len()));
        };
        let mut is_prev_space = spec.remove_extra_whitespaces;
        if !self.treat_whitespace_as_suffix && spec.add_dummy_prefix {
            add_ws(&mut out, &mut n2o, consumed);
        }
        while !input.is_empty() {
            let (s, n) = self.normalize_prefix(input, user);
            let mut s: &str = &s;
            if is_prev_space {
                s = s.trim_start_matches(' ');
            }
            if !s.is_empty() {
                for c in s.chars() {
                    if spec.escape_whitespaces && c == ' ' {
                        out.push_str(SPACE_SYMBOL);
                        n2o.extend(std::iter::repeat_n(consumed, SPACE_SYMBOL.len()));
                    } else {
                        out.push(c);
                        n2o.extend(std::iter::repeat_n(consumed, c.len_utf8()));
                    }
                }
                is_prev_space = s.ends_with(' ');
            }
            consumed += n;
            input = &input[n..];
            if !spec.remove_extra_whitespaces {
                is_prev_space = false;
            }
        }
        if spec.remove_extra_whitespaces {
            while out.ends_with(ws) {
                let len = out.len() - ws.len();
                consumed = n2o[len];
                out.truncate(len);
                n2o.truncate(len);
            }
        }
        if self.treat_whitespace_as_suffix && spec.add_dummy_prefix {
            add_ws(&mut out, &mut n2o, consumed);
        }
        n2o.push(consumed);
        (out, n2o)
    }

    /// `Normalizer::NormalizePrefix`: the replacement of the longest
    /// normalizable prefix of `input` and how many bytes it consumed.
    fn normalize_prefix<'a>(
        &'a self,
        input: &'a str,
        user: &PrefixMatcher,
    ) -> (Cow<'a, str>, usize) {
        if let Some(n) = user.longest(input) {
            return (input[..n].into(), n);
        }
        if let Some(map) = &self.charsmap
            && let Some((n, repl)) = map.longest_prefix(input.as_bytes())
            && input.is_char_boundary(n)
            && let Ok(r) = std::str::from_utf8(repl)
        {
            return (r.into(), n);
        }
        let n = char_len(input);
        (input[..n].into(), n)
    }
}

/// `NormalizerSpec.precompiled_charsmap`: a little-endian `u32` trie size,
/// the darts-clone double array, then NUL-terminated replacement strings the
/// trie values index.
struct Charsmap {
    units: Vec<u32>,
    normalized: Vec<u8>,
}

impl Charsmap {
    fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() < 4 {
            bail!("precompiled charsmap shorter than its header");
        }
        let size = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        let rest = &blob[4..];
        if size > rest.len() || !size.is_multiple_of(4) {
            bail!("precompiled charsmap trie size {size} does not fit the blob");
        }
        let units = rest[..size]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect::<Vec<_>>();
        if units.is_empty() {
            bail!("precompiled charsmap with an empty trie");
        }
        Ok(Self {
            units,
            normalized: rest[size..].to_vec(),
        })
    }

    /// Longest key that prefixes `key`: its length and replacement bytes
    /// (darts-clone `commonPrefixSearch`, longest result).
    fn longest_prefix(&self, key: &[u8]) -> Option<(usize, &[u8])> {
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let label = |u: u32| u & ((1 << 31) | 0xFF);
        let has_leaf = |u: u32| (u >> 8) & 1 == 1;
        let value = |u: u32| (u & ((1 << 31) - 1)) as usize;

        let mut found = None;
        let mut pos = offset(self.units[0]);
        for (i, &c) in key.iter().enumerate() {
            pos ^= c as usize;
            let Some(&unit) = self.units.get(pos) else {
                break;
            };
            if label(unit) != u32::from(c) {
                break;
            }
            pos ^= offset(unit);
            if has_leaf(unit) {
                let Some(&leaf) = self.units.get(pos) else {
                    break;
                };
                found = Some((i + 1, value(leaf)));
            }
        }
        let (len, v) = found?;
        let tail = self.normalized.get(v..)?;
        let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
        Some((len, &tail[..end]))
    }
}
