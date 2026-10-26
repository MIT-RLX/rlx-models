// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Tagger output parsing and verbalizer-input permutations — ports of NeMo
//! 1.2.0's `TokenParser` and `Normalizer` token ordering (Apache-2.0), via
//! `kitten_text_processing`.
//!
//! The tagger writes `tokens { money { integer: "20" currency: "$" } }`. The
//! verbalizer accepts the fields of a token only in some orders, so every
//! permutation is tried (unless `preserve_order: true`) until one verbalizes.

use anyhow::{Result, bail};

const PRESERVE_ORDER_KEY: &str = "preserve_order";

/// A parsed token value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Map(Vec<(String, Value)>),
    Bool,
}

/// Insert-or-replace keeping the first position (Python `OrderedDict`).
fn put(d: &mut Vec<(String, Value)>, k: String, v: Value) {
    match d.iter_mut().find(|(key, _)| *key == k) {
        Some(slot) => slot.1 = v,
        None => d.push((k, v)),
    }
}

/// `TokenParser` over a tagged string.
pub struct Parser {
    chars: Vec<char>,
    index: usize,
    /// `None` = end of input (`EOS`).
    ch: Option<char>,
}

impl Parser {
    pub fn new(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let ch = chars.first().copied();
        Self {
            chars,
            index: 0,
            ch,
        }
    }

    fn read(&mut self) -> bool {
        if self.index + 1 < self.chars.len() {
            self.index += 1;
            self.ch = Some(self.chars[self.index]);
            true
        } else {
            self.ch = None;
            false
        }
    }

    fn ws(&mut self) -> bool {
        let mut not_eos = self.ch.is_some();
        while not_eos && self.ch == Some(' ') {
            not_eos = self.read();
        }
        not_eos
    }

    fn expect(&mut self, c: char) -> Result<()> {
        if self.ch != Some(c) {
            bail!("tagger output: expected {c:?}, found {:?}", self.ch);
        }
        self.read();
        Ok(())
    }

    /// `A -> space F space F … space`.
    pub fn parse(&mut self) -> Result<Vec<Vec<(String, Value)>>> {
        let mut out = Vec::new();
        while self.ws() {
            match self.token()? {
                Some(t) => out.push(t),
                None => break,
            }
        }
        Ok(out)
    }

    fn token(&mut self) -> Result<Option<Vec<(String, Value)>>> {
        let Some(key) = self.key()? else {
            return Ok(None);
        };
        self.ws();
        let value = if key == PRESERVE_ORDER_KEY {
            self.expect(':')?;
            self.ws();
            for c in "true".chars() {
                self.expect(c)?;
            }
            Value::Bool
        } else {
            self.value()?
        };
        Ok(Some(vec![(key, value)]))
    }

    fn value(&mut self) -> Result<Value> {
        match self.ch {
            Some(':') => {
                self.expect(':')?;
                self.ws();
                self.expect('"')?;
                let s = self.string()?;
                self.expect('"')?;
                Ok(Value::Str(s))
            }
            Some('{') => {
                self.expect('{')?;
                let mut d = Vec::new();
                for tok in self.parse()? {
                    for (k, v) in tok {
                        put(&mut d, k, v);
                    }
                }
                self.expect('}')?;
                Ok(Value::Map(d))
            }
            _ => bail!("tagger output: bad token value at {}", self.index),
        }
    }

    fn key(&mut self) -> Result<Option<String>> {
        if self.ch.is_none_or(|c| c.is_whitespace()) {
            bail!("tagger output: key expected");
        }
        let mut s = String::new();
        while let Some(c) = self.ch.filter(|c| c.is_ascii_alphabetic() || *c == '_') {
            s.push(c);
            if !self.read() {
                bail!("tagger output: truncated key");
            }
        }
        Ok((!s.is_empty()).then_some(s))
    }

    /// A value runs until a quote followed by a space.
    fn string(&mut self) -> Result<String> {
        if self.ch.is_none() {
            bail!("tagger output: truncated value");
        }
        let mut s = String::new();
        while self.ch != Some('"') || self.chars.get(self.index + 1) != Some(&' ') {
            s.push(self.ch.unwrap());
            if !self.read() {
                bail!("tagger output: unterminated value");
            }
        }
        // An empty value is `None` upstream, which `_permute` then rejects —
        // kept as an empty string so `permute` can do the same.
        Ok(s)
    }
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

fn permutations_estimate(d: &[(String, Value)]) -> usize {
    let mut n = 1usize;
    for (_, v) in d {
        if let Value::Map(inner) = v {
            n = n.saturating_mul(permutations_estimate(inner));
        }
    }
    n.saturating_mul(factorial(d.len()))
}

/// Split token groups so no split's permutation product exceeds `max`.
pub fn split_groups(
    tokens: Vec<Vec<(String, Value)>>,
    max: usize,
) -> Result<Vec<Vec<Vec<(String, Value)>>>> {
    let mut splits = Vec::new();
    let mut current = Vec::new();
    let mut product = 1usize;
    for t in tokens {
        let n = permutations_estimate(&t);
        if n.saturating_mul(product) > max {
            splits.push(std::mem::take(&mut current));
            product = 1;
        }
        if n > max {
            bail!("unsplittable token group with {n} permutations");
        }
        product = product.saturating_mul(n);
        current.push(t);
    }
    splits.push(current);
    Ok(splits)
}

/// Every key order of `d`, in `itertools.permutations` order.
fn orders(n: usize) -> Vec<Vec<usize>> {
    // Python: permutations of positions in lexicographic index order.
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..n).collect();
    out.push(idx.clone());
    // Step to the next lexicographic permutation until there is none.
    while let Some(i) = (1..idx.len()).rev().find(|&i| idx[i - 1] < idx[i]) {
        let j = (0..idx.len()).rev().find(|&j| idx[j] > idx[i - 1]).unwrap();
        idx.swap(i - 1, j);
        idx[i..].reverse();
        out.push(idx.clone());
    }
    out
}

/// `_permute`: serializations of `d` under every field order.
fn permute(d: &[(String, Value)]) -> Result<Vec<String>> {
    let perms = if d.iter().any(|(k, _)| k == PRESERVE_ORDER_KEY) {
        vec![(0..d.len()).collect()]
    } else {
        orders(d.len())
    };
    let mut out = Vec::new();
    for perm in perms {
        let mut sub = vec![String::new()];
        for &i in &perm {
            let (k, v) = &d[i];
            match v {
                Value::Str(s) if s.is_empty() => bail!("empty value for {k}"),
                Value::Str(s) => sub
                    .iter_mut()
                    .for_each(|x| x.push_str(&format!("{k}: \"{s}\" "))),
                Value::Bool => sub
                    .iter_mut()
                    .for_each(|x| x.push_str(&format!("{k}: true "))),
                Value::Map(inner) => {
                    let rec = permute(inner)?;
                    sub = sub
                        .iter()
                        .flat_map(|x| rec.iter().map(move |r| format!("{x} {k} {{ {r} }} ")))
                        .collect();
                }
            }
        }
        out.extend(sub);
    }
    Ok(out)
}

/// Candidate verbalizer inputs for one split, in `itertools.product` order.
pub fn candidates(group: &[Vec<(String, Value)>]) -> Result<Vec<String>> {
    let parts: Vec<Vec<String>> = group.iter().map(|t| permute(t)).collect::<Result<_>>()?;
    let mut out = vec![String::new()];
    for p in parts {
        out = out
            .iter()
            .flat_map(|x| p.iter().map(move |y| format!("{x}{y}")))
            .collect();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_tokens() {
        let t = Parser::new(r#"tokens { money { integer_part: "twenty" currency: "dollars" } } tokens { name: "for" }"#)
            .parse()
            .unwrap();
        assert_eq!(t.len(), 2);
        let Value::Map(m) = &t[0][0].1 else { panic!() };
        let Value::Map(money) = &m[0].1 else { panic!() };
        assert_eq!(money[1], ("currency".into(), Value::Str("dollars".into())));
    }

    #[test]
    fn permutations_follow_itertools_order() {
        assert_eq!(
            orders(3),
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0]
            ]
        );
        let d = vec![
            ("a".into(), Value::Str("x".into())),
            ("b".into(), Value::Str("y".into())),
        ];
        assert_eq!(
            permute(&d).unwrap(),
            [r#"a: "x" b: "y" "#, r#"b: "y" a: "x" "#]
        );
    }
}
