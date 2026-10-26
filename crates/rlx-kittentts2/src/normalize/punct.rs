// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Moses-style detokenization (`kitten_text_processing._detokenize`, after
//! sacremoses, MIT) and NeMo's `post_process_punct` (Apache-2.0), which puts
//! the spaces around punctuation back the way the input had them.

use super::unicode_data::{ALPHA_RANGES, CURRENCY_SYMBOLS};

fn is_alpha(c: char) -> bool {
    let v = c as u32;
    let i = ALPHA_RANGES.partition_point(|&(a, _)| a <= v);
    i > 0 && v <= ALPHA_RANGES[i - 1].1
}

const CJK: [(u32, u32); 11] = [
    (4352, 4607),
    (11904, 42191),
    (43072, 43135),
    (44032, 55215),
    (63744, 64255),
    (65072, 65103),
    (65381, 65500),
    (94208, 101119),
    (110592, 110895),
    (110960, 111359),
    (131072, 196607),
];

fn is_cjk(c: char) -> bool {
    CJK.iter().any(|&(a, b)| (a..=b).contains(&(c as u32)))
}

/// `detokenize(text, "en")`.
pub fn detokenize_en(text: &str) -> String {
    let padded = format!(" {text} ").replace(" @-@ ", "-");
    let tokens: Vec<&str> = padded.split_whitespace().collect();
    let mut quotes: std::collections::HashMap<String, usize> = Default::default();
    let mut space = " ";
    let mut result = String::new();
    let all_in = |t: &str, set: &str| t.chars().all(|c| set.contains(c));
    for (i, token) in tokens.iter().enumerate() {
        let first = token.chars().next().unwrap();
        if is_cjk(first) {
            let prev_cjk = i > 0 && tokens[i - 1].chars().last().is_some_and(is_cjk);
            result.push_str(if prev_cjk { "" } else { space });
            result.push_str(token);
            space = " ";
        } else if token
            .chars()
            .all(|c| CURRENCY_SYMBOLS.contains(c) || "([{¿¡".contains(c))
        {
            result.push_str(space);
            result.push_str(token);
            space = "";
        } else if all_in(token, ",.?!:;\\%}])")
            || (i > 0
                && token.chars().count() > 1
                && first == '\''
                && token.chars().nth(1).is_some_and(is_alpha))
        {
            result.push_str(token);
            space = " ";
        } else if all_in(token, "'\"„“`") {
            let key = if all_in(token, "„“”") {
                "\"".to_string()
            } else {
                token.to_string()
            };
            let count = quotes.get(&key).copied().unwrap_or(0);
            if count % 2 == 1 || (*token == "'" && i > 0 && tokens[i - 1].ends_with('s')) {
                result.push_str(token);
                space = " ";
                if count % 2 == 1 {
                    quotes.insert(key, count + 1);
                }
            } else {
                result.push_str(space);
                result.push_str(token);
                space = "";
                quotes.insert(key, count + 1);
            }
        } else {
            result.push_str(space);
            result.push_str(token);
            space = " ";
        }
    }
    collapse_spaces(&result, 2).trim().to_string()
}

/// Replace runs of at least `min` spaces with one.
fn collapse_spaces(s: &str, min: usize) -> String {
    let mut out = String::with_capacity(s.len());
    let mut run = 0;
    for c in s.chars() {
        if c == ' ' {
            run += 1;
            continue;
        }
        if run > 0 {
            out.push_str(&" ".repeat(if run >= min { 1 } else { run }));
            run = 0;
        }
        out.push(c);
    }
    if run > 0 {
        out.push_str(&" ".repeat(if run >= min { 1 } else { run }));
    }
    out
}

const ASCII_PUNCT: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// `post_process_punct(input, normalized)`: match the spacing around each
/// ASCII punctuation mark to the input's. Elements of the working list grow
/// to `"x "` or shrink to `""` as upstream's do, which changes what later
/// lookups match — kept as is.
pub fn post_process_punct(input: &str, normalized: &str) -> String {
    let input = if input.contains("``") && !normalized.contains("``") {
        input.replace("``", "\"")
    } else {
        input.to_string()
    };
    let inp: Vec<String> = input.chars().map(String::from).collect();
    let mut out: Vec<String> = normalized.chars().map(String::from).collect();
    for p in ASCII_PUNCT.chars().filter(|c| input.contains(*c)) {
        let p = p.to_string();
        let count = |v: &[String]| v.iter().filter(|x| **x == p).count();
        let equal = count(&inp) == count(&out);
        let (mut i_in, mut i_out) = (0usize, 0usize);
        while inp[i_in.min(inp.len())..].contains(&p) {
            let Some(o) = out[i_out.min(out.len())..]
                .iter()
                .position(|x| *x == p)
                .map(|o| o + i_out)
            else {
                break; // `list.index` raised: skip this mark
            };
            i_out = o;
            i_in += inp[i_in..].iter().position(|x| *x == p).unwrap();
            let valid = (i_out > 0 && i_in > 0 && out[i_out - 1] == inp[i_in - 1])
                || (i_out + 1 < out.len()
                    && i_in + 1 < inp.len()
                    && out[i_out + 1] == inp[i_in + 1]);
            if !equal && !valid {
                i_in += 1;
                continue;
            }
            if i_in > 0 && i_out > 0 {
                if out[i_out - 1] == " " && inp[i_in - 1] != " " {
                    out[i_out - 1].clear();
                } else if out[i_out - 1] != " " && inp[i_in - 1] == " " {
                    out[i_out - 1].push(' ');
                }
            }
            if i_in + 1 < inp.len() && i_out + 1 < out.len() {
                if out[i_out + 1] == " " && inp[i_in + 1] != " " {
                    out[i_out + 1].clear();
                } else if out[i_out + 1] != " " && inp[i_in + 1] == " " {
                    out[i_out].push(' ');
                }
            }
            i_out += 1;
            i_in += 1;
        }
    }
    collapse_spaces(&out.concat(), 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detokenizes_punctuation_and_currency() {
        assert_eq!(detokenize_en("I paid $ 12 , ok ?"), "I paid $12, ok?");
        assert_eq!(
            detokenize_en("he said \" hi \" then"),
            "he said \"hi\" then"
        );
        assert_eq!(detokenize_en("it 's fine"), "it's fine");
    }

    #[test]
    fn restores_input_spacing_around_marks() {
        assert_eq!(
            post_process_punct("12 test' example", "twelve test 'example"),
            "twelve test' example"
        );
    }
}
