//! Fast content-TYPE classifier: JSON / code / text / file-listing / UI-chrome.
//!
//! Same lesson as `fastclean`: the distinctions are feature-shaped — brace/
//! quote/colon density (JSON), keyword/operator hits (code), perms/path/size
//! patterns (files), prose (text), chrome-glyph density (UI) — so a branch-light
//! heuristic runs at memory-bandwidth speed with no net and no NaNs. Also ships
//! a labeled-line generator (`gen_typed_line`) so an ML version can be trained
//! and compared head-to-head.

use crate::corpus;
use crate::fastclean::{is_chrome_glyph, strip_ansi};
use crate::rng::Rng;

/// The five line types the fast classifier routes: `Ui` chrome (drop), and
/// `Json` / `Code` / `Text` / `File` (keep, tagged by kind).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CType {
    Ui,
    Json,
    Code,
    File,
    Text,
}
impl CType {
    pub fn name(self) -> &'static str {
        match self {
            CType::Ui => "ui",
            CType::Json => "json",
            CType::Code => "code",
            CType::File => "file",
            CType::Text => "text",
        }
    }
    pub fn idx(self) -> usize {
        match self {
            CType::Ui => 0,
            CType::Json => 1,
            CType::Code => 2,
            CType::File => 3,
            CType::Text => 4,
        }
    }
    pub const ALL: [CType; 5] = [
        CType::Ui,
        CType::Json,
        CType::Code,
        CType::File,
        CType::Text,
    ];
}

const KEYWORDS: &[&str] = &[
    "fn ", "def ", "let ", "const ", "var ", "function", "class ", "import ", "return", "if ",
    "else", "for ", "while ", "pub ", "use ", "impl ", "struct ", "enum ", "match ", "println",
    "print(", "echo ", "#include", "func ", "void ", "public ", "static ", "SELECT ", "assert",
];

fn is_perms(s: &str) -> bool {
    let b = s.trim_start().as_bytes();
    b.len() >= 10
        && matches!(b[0], b'-' | b'd' | b'l' | b'b' | b'c' | b'p' | b's')
        && b[1..10]
            .iter()
            .all(|&c| matches!(c, b'r' | b'w' | b'x' | b's' | b'S' | b't' | b'T' | b'-'))
}
fn has_path(s: &str) -> bool {
    s.split(char::is_whitespace)
        .any(|w| w.matches('/').count() >= 2 && !w.contains("://"))
}
fn has_size(s: &str) -> bool {
    let b = s.as_bytes();
    for i in 0..b.len() {
        if b[i].is_ascii_digit() {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'.') {
                j += 1;
            }
            if j < b.len() && matches!(b[j], b'K' | b'M' | b'G' | b'T') {
                return true;
            }
        }
    }
    false
}

/// Shell prompt / status line: an `alnum@alnum` (user@host) token.
fn is_shell_prompt(s: &str) -> bool {
    let b = s.as_bytes();
    b.iter().enumerate().any(|(i, &c)| {
        c == b'@'
            && i > 0
            && b[i - 1].is_ascii_alphanumeric()
            && i + 1 < b.len()
            && b[i + 1].is_ascii_alphanumeric()
    })
}

/// Classify one line's content type. Priority: UI > File > JSON/Code > Text.
pub fn classify_type(line: &str) -> CType {
    let mut v = String::new();
    strip_ansi(line, &mut v);
    let s = v.trim();
    if s.is_empty() {
        return CType::Ui;
    }
    let n = s.chars().count();
    // Definitive UI: block-element / braille glyphs never appear in real content;
    // otherwise fall back to box-drawing density or a shell/status prompt.
    let has_gfx = s.chars().any(|c| {
        let u = c as u32;
        (0x2580..=0x259F).contains(&u) || (0x2800..=0x28FF).contains(&u)
    });
    let chrome = s.chars().filter(|&c| is_chrome_glyph(c)).count();
    // ASCII borders: a run of >=4 of - = + | ~ (mc/dialog/ASCII rules)
    let ascii_border = {
        let b = s.as_bytes();
        let mut run = 1usize;
        let mut hit = false;
        for i in 1..b.len() {
            if b[i] == b[i - 1] && matches!(b[i], b'-' | b'=' | b'+' | b'|' | b'~') {
                run += 1;
                if run >= 4 {
                    hit = true;
                    break;
                }
            } else {
                run = 1;
            }
        }
        hit
    };
    // function-key bars: `^X Exit  ^S Save ...`
    let caret_keys = s
        .as_bytes()
        .windows(2)
        .filter(|w| w[0] == b'^' && w[1].is_ascii_uppercase())
        .count();
    if has_gfx || chrome * 4 >= n || is_shell_prompt(s) || ascii_border || caret_keys >= 2 {
        return CType::Ui;
    }
    if is_perms(s) || (has_path(s) && (has_size(s) || s.split_whitespace().count() <= 2)) {
        return CType::File;
    }
    let braces = s
        .chars()
        .filter(|&c| matches!(c, '{' | '}' | '[' | ']'))
        .count();
    let quotes = s.chars().filter(|&c| c == '"').count();
    let jcolon = s.matches("\":").count() + s.matches("\" :").count();

    // Keywords only count as code when the line also has code punctuation, so
    // prose ("...for the system...") doesn't trip on the word "for".
    let has_code_sym = s.contains(['(', ')', ';', '='])
        || s.contains("::")
        || s.contains("=>")
        || s.contains("->");
    let kw = if has_code_sym {
        KEYWORDS.iter().filter(|k| s.contains(**k)).count()
    } else {
        0
    };
    let ops = s.matches("=>").count()
        + s.matches("->").count()
        + s.matches("::").count()
        + s.matches("&&").count()
        + s.matches("||").count()
        + s.matches(");").count();
    let ends = usize::from(s.ends_with(';') || s.ends_with('{') || s.ends_with('}'));
    let comment = usize::from(s.starts_with("//") || s.starts_with('#') || s.contains("/*"));
    let parens = s.matches('(').count(); // method calls / signatures
    let bracey = usize::from(braces >= 2 && quotes == 0); // brace-heavy, unquoted = code not json
    let code_score = kw * 3
        + ops * 2
        + ends * 2
        + comment * 2
        + usize::from(has_code_sym)
        + usize::from(parens >= 2)
        + bracey * 2;

    // A quoted-key colon (`"k":`) is a definitive JSON signal unless it's clearly code.
    if jcolon >= 1 && kw == 0 {
        return CType::Json;
    }
    if braces >= 2 && quotes >= 2 && braces * 2 + quotes >= code_score {
        return CType::Json;
    }
    if code_score >= 3 {
        return CType::Code;
    }
    if jcolon >= 1 {
        return CType::Json;
    }
    CType::Text
}

// ---------------------------------------------------------------------------
// Labeled-line generator (for evaluation + optional ML training)
// ---------------------------------------------------------------------------

/// Richer code templates across languages (rust/py/js/c/sql/go/sh/html).
pub static CODE_T: &[&str] = &[
    "let mut total: u64 = 0;",
    "for i in 0..n { total += arr[i]; }",
    "fn parse(s: &str) -> Result<i32, Error> {",
    "def foo(x): return x * 2 + [1, 2, 3]",
    "if a and b or c == 42: pass",
    "function add(a, b) { return a + b; }",
    "const xs = data.filter(x => x > 0).map(f);",
    "printf(\"%d: %s\\n\", i, name);",
    "SELECT id, name FROM users WHERE age > 18;",
    "for k, v := range m { fmt.Println(k, v) }",
    "echo \"$HOME/bin\" && ls -la | grep foo",
    "impl Iterator for Counter { type Item = u32; }",
    "self.buf[idx] = (b >> 4) & 0x0F;",
    "return a.max(b).min(255);",
    "x = {k: v for k, v in items if v}",
    "public static void main(String[] args) {",
    "<div class=\"row\"><span>{{ name }}</span></div>",
    "grep -rn 'TODO' src/ | wc -l",
    "data.iter().filter(|&&x| x != 0).sum::<i64>()",
    "assert_eq!(result, expected, \"mismatch\");",
];

/// Generate one synthetic line of the given [`CType`] — the labeled data the
/// rule classifier is measured against (and an ML head could train on).
pub fn gen_typed_line(rng: &mut Rng, t: CType) -> String {
    match t {
        CType::Code => rng.pick_str(CODE_T).to_string(),
        CType::Text => corpus::sentence(rng),
        CType::Json => {
            let k = [
                "name", "status", "cpu", "id", "ok", "path", "size", "tags", "user", "port",
                "enabled",
            ];
            let key = k[rng.below(k.len())];
            let key2 = k[rng.below(k.len())];
            let bn = ["true", "false", "null"];
            match rng.below(6) {
                0 => format!(
                    "{{\"{key}\": \"{}\", \"{key2}\": {}}}",
                    corpus::filename(rng),
                    rng.range(1, 999)
                ),
                1 => format!("  \"{key}\": {},", rng.range(0, 9999)),
                2 => format!(
                    "{{\"arr\": [{}, {}, {}], \"{key}\": {}}}",
                    rng.below(50),
                    rng.below(50),
                    rng.below(50),
                    rng.pick_str(&bn)
                ),
                3 => format!("  \"{key}\": {},", rng.pick_str(&bn)),
                4 => format!(
                    "{{\"{key}\": {{\"n\": {}, \"{key2}\": \"{}\"}}}}",
                    rng.range(1, 99),
                    corpus::filename(rng)
                ),
                _ => format!("  \"{key}\": \"{}\"", corpus::filename(rng)),
            }
        }
        CType::File => {
            let f = corpus::filename(rng);
            let n1 = rng.pick_str(corpus::NOUNS);
            let n2 = rng.pick_str(corpus::NOUNS);
            match rng.below(6) {
                0 => format!(
                    "drwxr-xr-x  {} user staff  {} Jul 10 15:0{} {f}",
                    rng.range(1, 20),
                    rng.range(100, 9999),
                    rng.below(10)
                ),
                1 => format!(
                    "-rw-r--r--  1 user staff  {}K Jul 10 {f}",
                    rng.range(1, 999)
                ),
                2 => format!("/usr/local/{n1}/{n2}/{f}"),
                3 => format!(
                    " {} src/{n1}/{f}",
                    rng.pick_str(&["M", "A", "D", "??", "R "])
                ),
                4 => format!("{}.{}K\t./{n1}/{f}", rng.range(1, 9), rng.below(9)),
                _ => format!("./{n1}/{n2}/{f}"),
            }
        }
        CType::Ui => {
            let b = *rng.pick(crate::symbols::BOX_STYLES);
            match rng.below(7) {
                0 => format!("{}{}{}", b.tl, b.h.repeat(rng.range(10, 40)), b.tr),
                1 => "  user@host ~/project | ctx: 3% used".to_string(),
                2 => format!(
                    "CPU [{}{}] {}%",
                    "█".repeat(rng.range(1, 8)),
                    "░".repeat(rng.range(1, 8)),
                    rng.below(100)
                ),
                3 => format!(
                    "{} {} Output {} {}",
                    b.v,
                    b.h.repeat(2),
                    b.h.repeat(rng.range(6, 20)),
                    b.v
                ),
                4 => "  ^X Exit   ^S Save   ^F Find   ^G Help".to_string(),
                5 => format!(
                    "{} Building {} [{}{}] {}%",
                    rng.pick_str(&["⠋", "⠙", "⠹", "◐"]),
                    corpus::filename(rng),
                    "━".repeat(rng.range(1, 10)),
                    "─".repeat(rng.range(1, 10)),
                    rng.below(100)
                ),
                _ => format!(
                    "{}[1]{} Status {}{}[0]{} Files {}",
                    b.ltee,
                    b.h,
                    b.h.repeat(4),
                    b.ltee,
                    b.h,
                    b.h.repeat(4)
                ),
            }
        }
    }
}
