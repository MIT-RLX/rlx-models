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

//! Source-text normalizer — the OS's `normalization-pattern-file`.
//!
//! The file is a **sed script**: one substitution per line. `TranslationInference`
//! parses each with `/^s(.)(.+?)\1(.*?)\1.*$/`, i.e.
//!
//! ```text
//!   s<delim><pattern><delim><replacement><delim><flags>
//! ```
//!
//! where the delimiter is whatever character follows the `s`, so `s/a/b/` and
//! `s|a|b|` are both valid. Substitutions apply in file order.
//!
//! `fancy-regex` backs the patterns rather than `regex`: the framework's own
//! parser uses backreferences and lazy quantifiers, so the shipped patterns can
//! too, and `regex` supports neither.

use anyhow::{Context, Result, anyhow, bail};
use fancy_regex::Regex;
use std::path::Path;

/// One parsed `s///` command.
#[derive(Debug)]
pub struct SedCommand {
    /// 1-based line number in the source file, for diagnostics.
    pub line: usize,
    pattern: Regex,
    /// Replacement in `fancy-regex` syntax (`$1`), converted from sed's `\1`.
    replacement: String,
    global: bool,
}

impl SedCommand {
    /// Applies this substitution to `text`.
    pub fn apply(&self, text: &str) -> String {
        if self.global {
            self.pattern
                .replace_all(text, self.replacement.as_str())
                .into_owned()
        } else {
            self.pattern
                .replace(text, self.replacement.as_str())
                .into_owned()
        }
    }
}

/// An ordered list of substitutions.
#[derive(Debug, Default)]
pub struct Normalizer {
    pub commands: Vec<SedCommand>,
}

impl Normalizer {
    /// Loads a normalizer file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading normalizer {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing normalizer {}", path.display()))
    }

    /// Parses a normalizer script. Blank lines and `#` comments are skipped;
    /// anything else must be a well-formed `s///` command, matching the
    /// framework's own "invalid sed command" rejection.
    pub fn parse(text: &str) -> Result<Self> {
        let mut commands = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = i + 1;
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            commands.extend(parse_line(trimmed, line)?);
        }
        Ok(Self { commands })
    }

    /// Applies every substitution, in order.
    pub fn normalize(&self, text: &str) -> String {
        let mut out = text.to_string();
        for cmd in &self.commands {
            out = cmd.apply(&out);
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }
}

/// Parses every `s///` command on one line. sed allows `;` to separate
/// commands and to terminate a trailing one, and the OS's shipped
/// `normalizer.pat` relies on both.
fn parse_line(line: &str, lineno: usize) -> Result<Vec<SedCommand>> {
    let mut out = Vec::new();
    let mut rest = line;
    loop {
        let rest_trimmed = rest.trim_start();
        if rest_trimmed.is_empty() {
            break;
        }
        let (cmd, tail) = parse_command(rest_trimmed, lineno)?;
        out.push(cmd);
        rest = tail;
    }
    if out.is_empty() {
        bail!("line {lineno}: no sed command in {line:?}");
    }
    Ok(out)
}

/// Parses one `s<delim>pat<delim>repl<delim>flags` command, returning it and
/// whatever follows the command's `;` terminator.
fn parse_command(line: &str, lineno: usize) -> Result<(SedCommand, &str)> {
    let mut chars = line.char_indices();
    if chars.next().map(|(_, c)| c) != Some('s') {
        bail!("line {lineno}: normalizer command must start with 's': {line:?}");
    }
    let Some((_, delim)) = chars.next() else {
        bail!("line {lineno}: normalizer command has no delimiter: {line:?}");
    };

    // Split on unescaped delimiters, keeping escapes for the regex engine.
    let mut fields: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut escaped = false;
    let mut tail_start = line.len();
    for (i, c) in chars {
        if escaped {
            // A delimiter escaped in the source is a literal delimiter; any
            // other escape belongs to the pattern and is passed through.
            if c == delim {
                cur.push(c);
            } else {
                cur.push('\\');
                cur.push(c);
            }
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == delim {
            fields.push(std::mem::take(&mut cur));
            // `s<D>pat<D>repl<D>flags` has exactly two delimiters after the
            // opening `s<D>`; everything past the second is flags, up to `;`.
            if fields.len() == 2 {
                tail_start = i + c.len_utf8();
                break;
            }
        } else {
            cur.push(c);
        }
    }
    if fields.len() < 2 {
        bail!(
            "line {lineno}: normalizer command is not s{delim}pat{delim}repl{delim}flags: {line:?}"
        );
    }

    let after = &line[tail_start..];
    let (flags, rest) = match after.find(';') {
        Some(i) => (&after[..i], &after[i + 1..]),
        None => (after, ""),
    };

    let (pattern_src, replacement_src) = (&fields[0], &fields[1]);
    if pattern_src.is_empty() {
        bail!("line {lineno}: normalizer command has an empty pattern: {line:?}");
    }

    let mut global = false;
    let mut case_insensitive = false;
    for f in flags.trim().chars() {
        match f {
            'g' => global = true,
            'i' | 'I' => case_insensitive = true,
            // Reject rather than ignore: a silently dropped flag changes the
            // normalized text, which then shows up as a translation bug.
            other => bail!("line {lineno}: unsupported sed flag {other:?} in {line:?}"),
        }
    }

    let pattern = if case_insensitive {
        format!("(?i){pattern_src}")
    } else {
        pattern_src.clone()
    };
    let pattern = Regex::new(&pattern)
        .map_err(|e| anyhow!("line {lineno}: unsupported regex pattern {pattern_src:?}: {e}"))?;

    Ok((
        SedCommand {
            line: lineno,
            pattern,
            replacement: sed_replacement_to_regex(replacement_src),
            global,
        },
        rest,
    ))
}

/// Translates a sed replacement into `fancy-regex` syntax: `\1`…`\9` become
/// `${1}`…`${9}`, a bare `&` becomes `${0}`, and literal `$` is escaped so it
/// is not read as a capture reference.
fn sed_replacement_to_regex(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(d @ '0'..='9') => {
                    out.push_str("${");
                    out.push(d);
                    out.push('}');
                }
                Some('&') => out.push('&'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(other) => out.push(other),
                None => out.push('\\'),
            },
            '&' => out.push_str("${0}"),
            '$' => out.push_str("$$"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_substitution_applies_once_without_g() {
        let n = Normalizer::parse("s/a/b/").expect("parses");
        assert_eq!(n.normalize("aaa"), "baa");
    }

    #[test]
    fn global_flag_replaces_every_occurrence() {
        let n = Normalizer::parse("s/a/b/g").expect("parses");
        assert_eq!(n.normalize("aaa"), "bbb");
    }

    #[test]
    fn any_character_may_be_the_delimiter() {
        let n = Normalizer::parse("s|foo|bar|g").expect("parses");
        assert_eq!(n.normalize("foo foo"), "bar bar");
        let n = Normalizer::parse("s#x#y#g").expect("parses");
        assert_eq!(n.normalize("xx"), "yy");
    }

    #[test]
    fn backreferences_are_translated_from_sed_syntax() {
        let n = Normalizer::parse(r"s/(\w+)\s+(\w+)/\2 \1/").expect("parses");
        assert_eq!(n.normalize("hello world"), "world hello");
    }

    #[test]
    fn ampersand_inserts_the_whole_match() {
        let n = Normalizer::parse("s/cat/[&]/g").expect("parses");
        assert_eq!(n.normalize("cat cat"), "[cat] [cat]");
    }

    #[test]
    fn escaped_delimiter_is_literal() {
        let n = Normalizer::parse(r"s/a\/b/X/g").expect("parses");
        assert_eq!(n.normalize("a/b"), "X");
    }

    #[test]
    fn literal_dollar_survives_the_replacement_conversion() {
        let n = Normalizer::parse("s/price/$/g").expect("parses");
        assert_eq!(n.normalize("price"), "$");
    }

    #[test]
    fn case_insensitive_flag_is_honoured() {
        let n = Normalizer::parse("s/abc/x/gi").expect("parses");
        assert_eq!(n.normalize("ABC abc"), "x x");
    }

    #[test]
    fn lazy_quantifiers_work_which_is_why_fancy_regex_is_used() {
        let n = Normalizer::parse("s/<.+?>//g").expect("parses");
        assert_eq!(n.normalize("<a>keep<b>"), "keep");
    }

    #[test]
    fn commands_apply_in_file_order() {
        let n = Normalizer::parse("s/a/b/g\ns/b/c/g").expect("parses");
        assert_eq!(n.normalize("a"), "c");
        assert_eq!(n.len(), 2);
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let n = Normalizer::parse("# a comment\n\ns/a/b/g\n").expect("parses");
        assert_eq!(n.len(), 1);
    }

    #[test]
    fn malformed_lines_are_rejected_loudly() {
        assert!(Normalizer::parse("not a command").is_err());
        assert!(Normalizer::parse("s/only-two-fields").is_err());
        assert!(Normalizer::parse("s//empty-pattern/").is_err());
        let err = Normalizer::parse("s/a/b/z").expect_err("unknown flag must fail");
        assert!(err.to_string().contains('z'), "{err}");
    }

    #[test]
    fn trailing_semicolon_terminates_a_command() {
        // the shipped MT/normalizer.pat, verbatim.
        let n = Normalizer::parse("s/^[[:space:]]+//g;\ns/[[:space:]]+$//g;\n")
            .expect("real normalizer.pat must parse");
        assert_eq!(n.len(), 2);
        assert_eq!(n.normalize("   hello world   "), "hello world");
    }

    #[test]
    fn several_commands_may_share_a_line() {
        let n = Normalizer::parse("s/a/b/g; s/b/c/g").expect("parses");
        assert_eq!(n.len(), 2);
        assert_eq!(n.normalize("a"), "c");
    }

    #[test]
    fn posix_classes_are_supported() {
        let n = Normalizer::parse("s/[[:digit:]]+/#/g").expect("parses");
        assert_eq!(n.normalize("abc123def45"), "abc#def#");
    }

    #[test]
    fn shipped_tokenizer_pat_spaces_out_punctuation() {
        // the shipped MT/tokenizer.pat, final two rules.
        let n = Normalizer::parse(
            "s/([\\[\\]！？｡。，；：（）»«《》［］、「」﹁﹂‧—\\-\\.\\!\\?:;,\\/\\\"\\(\\)\\^\\*])/ \\1 /g\ns/[[:space:]]+/ /g",
        )
        .expect("real tokenizer.pat rules must parse");
        assert_eq!(
            n.normalize("Hello, world! (test)"),
            "Hello , world ! ( test ) "
        );
    }

    #[test]
    fn line_numbers_are_reported() {
        let err = Normalizer::parse("s/a/b/g\nbroken\n").expect_err("must fail");
        assert!(err.to_string().contains("line 2"), "{err}");
    }
}
