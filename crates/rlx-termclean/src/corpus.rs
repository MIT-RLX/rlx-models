//! Clean-content generators.
//!
//! These produce the *target* text that later gets rendered into a TUI. The
//! `UNI` bank intentionally sprinkles special UTF-8 (accents, currency, math,
//! arrows, quotes, emoji) into content so the model learns those glyphs are
//! *content to keep* — the mirror image of the chrome glyphs in `symbols`.

use crate::rng::Rng;

pub static ARTICLES: &[&str] = &["the", "a", "this", "each", "every", "its"];
pub static NOUNS: &[&str] = &[
    "system",
    "server",
    "process",
    "request",
    "buffer",
    "kernel",
    "module",
    "packet",
    "session",
    "thread",
    "cache",
    "daemon",
    "socket",
    "payload",
    "cluster",
    "registry",
    "pipeline",
    "token",
    "gateway",
    "volume",
    "index",
    "record",
    "matrix",
    "tensor",
    "vector",
    "channel",
    "runtime",
    "compiler",
    "scheduler",
    "allocator",
    "handler",
    "listener",
    "worker",
    "shard",
];
pub static ADJS: &[&str] = &[
    "fast", "stale", "remote", "nested", "atomic", "hidden", "broken", "virtual", "primary",
    "shared", "dynamic", "corrupt", "idle", "active", "legacy", "secure", "verbose", "partial",
    "opaque", "frozen", "pinned",
];
pub static VERBS: &[&str] = &[
    "restarts",
    "flushes",
    "resolves",
    "allocates",
    "rejects",
    "queues",
    "parses",
    "encodes",
    "streams",
    "validates",
    "compresses",
    "migrates",
    "spawns",
    "throttles",
    "reconnects",
    "buffers",
    "dispatches",
];
pub static CONN: &[&str] = &[
    "while", "because", "although", "whenever", "after", "before", "unless", "until", "since",
];

/// Unicode-heavy *content* tokens. Some contain an internal space (e.g. an
/// emoji + word) — that is fine; splitting on spaces yields content words that
/// reflow reassembles exactly.
pub static UNI: &[&str] = &[
    "café",
    "naïve",
    "résumé",
    "piña",
    "façade",
    "€49.90",
    "£12",
    "¥300",
    "±0.5",
    "3×4",
    "10÷2",
    "≤5",
    "≥7",
    "x≠y",
    "α-decay",
    "β-test",
    "λ=0.3",
    "π≈3.14",
    "µm",
    "°C",
    "№7",
    "™",
    "©2025",
    "→next",
    "✅ done",
    "⚠️ warn",
    "🚀 launch",
    "“quoted”",
    "em—dash",
    "‘tag’",
    "½ cup",
    "²³",
    "Ω",
    "∑x",
    "≈100",
    "•note",
    "☑ ok",
    "★ starred",
];

/// Code line templates — dense with ASCII operators, brackets, and punctuation.
pub static CODE: &[&str] = &[
    "let mut total = 0;",
    "for i in 0..n { total += arr[i]; }",
    "fn parse(s: &str) -> Result<i32, Error> {",
    "    return Ok(s.trim().parse()?);",
    "if x > 0 && y < 10 || z == 42 {",
    "const PI: f64 = 3.14159;",
    "map.insert(key.clone(), value * 2);",
    "println!(\"{}: {}\", name, count);",
    "self.buf[idx] = (b >> 4) & 0x0F;",
    "match opt { Some(v) => v, None => 0 }",
    "let re = Regex::new(r\"^\\d+$\").unwrap();",
    "data.iter().filter(|&&x| x != 0).sum()",
    "#[derive(Clone, Debug)]",
    "assert_eq!(result, expected);",
    "while queue.len() > 0 { pop(); }",
    "pub struct Node { next: Option<Box<Node>> }",
    "use std::collections::HashMap;",
    "let v: Vec<u8> = vec![0; cap];",
    "impl Iterator for Counter {",
    "return a.max(b).min(255);",
];

pub static LEVELS: &[&str] = &["INFO", "WARN", "ERROR", "DEBUG", "TRACE"];
pub static COMPONENTS: &[&str] = &[
    "server", "auth", "db", "cache", "router", "worker", "api", "sync",
];

fn cap(s: &str) -> String {
    let mut ch = s.chars();
    match ch.next() {
        Some(f) => f.to_uppercase().collect::<String>() + ch.as_str(),
        None => String::new(),
    }
}

/// A single sentence with varied structure, optional clause, and an optional
/// unicode content token.
pub fn sentence(rng: &mut Rng) -> String {
    let mut s = cap(rng.pick_str(ARTICLES));
    s.push(' ');
    s.push_str(rng.pick_str(ADJS));
    s.push(' ');
    s.push_str(rng.pick_str(NOUNS));
    s.push(' ');
    s.push_str(rng.pick_str(VERBS));
    s.push(' ');
    s.push_str(rng.pick_str(ARTICLES));
    s.push(' ');
    s.push_str(rng.pick_str(ADJS));
    s.push(' ');
    s.push_str(rng.pick_str(NOUNS));
    if rng.chance(0.5) {
        s.push_str(", ");
        s.push_str(rng.pick_str(CONN));
        s.push(' ');
        s.push_str(rng.pick_str(NOUNS));
        s.push(' ');
        s.push_str(rng.pick_str(VERBS));
    }
    if rng.chance(0.35) {
        s.push(' ');
        s.push_str(rng.pick_str(UNI));
    }
    s.push('.');
    s
}

/// A paragraph of `n` sentences. Returns `(words, text)` where `text` is the
/// canonical single-spaced string (the clean target) and `words` is
/// `text.split(' ')` so wrapping can rejoin them exactly.
pub fn paragraph(rng: &mut Rng, n: usize) -> (Vec<String>, String) {
    let mut s = String::new();
    for i in 0..n.max(1) {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&sentence(rng));
    }
    let words = s.split(' ').map(|w| w.to_string()).collect();
    (words, s)
}

pub fn code_lines(rng: &mut Rng, n: usize) -> Vec<String> {
    (0..n).map(|_| rng.pick_str(CODE).to_string()).collect()
}

pub fn log_lines(rng: &mut Rng, n: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let (h, m, s) = (rng.below(24), rng.below(60), rng.below(60));
        let lvl = rng.pick_str(LEVELS);
        let comp = rng.pick_str(COMPONENTS);
        let id = rng.range(1, 9999);
        let ms = rng.range(1, 900);
        v.push(format!(
            "2025-11-04T{h:02}:{m:02}:{s:02}Z {lvl:<5} {comp}: request {id} handled in {ms}ms"
        ));
    }
    v
}

pub static KV_KEYS: &[&str] = &[
    "hostname", "os", "kernel", "uptime", "cpu", "memory", "disk", "ip", "shell", "user",
    "version", "status", "pid", "port",
];

fn kv_value(rng: &mut Rng, key: &str) -> String {
    match key {
        "cpu" => format!("{}%", rng.below(100)),
        "memory" => format!("{}.{} GiB", rng.range(1, 64), rng.below(10)),
        "disk" => format!("{}G / {}G", rng.range(1, 400), rng.range(400, 900)),
        "pid" => format!("{}", rng.range(100, 99999)),
        "port" => format!("{}", rng.range(1024, 65535)),
        "status" => rng
            .pick_str(&["running", "stopped", "degraded", "ok"])
            .to_string(),
        "ip" => format!(
            "{}.{}.{}.{}",
            rng.below(256),
            rng.below(256),
            rng.below(256),
            rng.below(256)
        ),
        "uptime" => format!("{}d {}h", rng.range(1, 90), rng.below(24)),
        "os" => rng
            .pick_str(&["linux 6.6", "darwin 24.1", "freebsd 14"])
            .to_string(),
        "kernel" => format!("6.{}.{}", rng.range(1, 12), rng.range(0, 90)),
        "version" => format!(
            "{}.{}.{}",
            rng.range(0, 5),
            rng.range(0, 20),
            rng.range(0, 40)
        ),
        _ => format!("{}-{}", rng.pick_str(NOUNS), rng.range(1, 99)),
    }
}

pub fn kv_pairs(rng: &mut Rng, n: usize) -> Vec<(String, String)> {
    let mut idx: Vec<usize> = (0..KV_KEYS.len()).collect();
    rng.shuffle(&mut idx);
    let mut v = Vec::new();
    for &i in idx.iter().take(n.min(KV_KEYS.len())) {
        v.push((KV_KEYS[i].to_string(), kv_value(rng, KV_KEYS[i])));
    }
    v
}

/// Returns `(header, rows)` for one of a few tabular schemas.
pub fn table(rng: &mut Rng) -> (Vec<String>, Vec<Vec<String>>) {
    let cols = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    match rng.below(3) {
        0 => {
            let header = cols(&["PID", "USER", "CPU%", "MEM", "COMMAND"]);
            let n = rng.range(3, 8);
            let rows = (0..n)
                .map(|_| {
                    vec![
                        format!("{}", rng.range(100, 9999)),
                        rng.pick_str(&["root", "www", "dan", "sys", "node"])
                            .to_string(),
                        format!("{}.{}", rng.below(100), rng.below(10)),
                        format!("{}M", rng.range(1, 900)),
                        rng.pick_str(&[
                            "nginx",
                            "postgres",
                            "python app.py",
                            "node index.js",
                            "rustc",
                            "sshd",
                        ])
                        .to_string(),
                    ]
                })
                .collect();
            (header, rows)
        }
        1 => {
            let header = cols(&["NAME", "SIZE", "MODIFIED", "TYPE"]);
            let n = rng.range(3, 8);
            let rows = (0..n)
                .map(|_| {
                    vec![
                        filename(rng),
                        format!("{}K", rng.range(1, 9999)),
                        format!("2025-{:02}-{:02}", rng.range(1, 13), rng.range(1, 28)),
                        rng.pick_str(&["file", "dir", "link", "sock"]).to_string(),
                    ]
                })
                .collect();
            (header, rows)
        }
        _ => {
            let header = cols(&["PACKAGE", "VERSION", "STATUS"]);
            let n = rng.range(3, 8);
            let rows = (0..n)
                .map(|_| {
                    vec![
                        format!("{}-{}", rng.pick_str(NOUNS), rng.pick_str(NOUNS)),
                        format!(
                            "{}.{}.{}",
                            rng.range(0, 5),
                            rng.range(0, 20),
                            rng.range(0, 40)
                        ),
                        rng.pick_str(&["ok", "outdated", "held", "broken"])
                            .to_string(),
                    ]
                })
                .collect();
            (header, rows)
        }
    }
}

pub fn list_items(rng: &mut Rng, n: usize) -> Vec<String> {
    (0..n)
        .map(|_| {
            let mut s = cap(rng.pick_str(ADJS));
            s.push(' ');
            s.push_str(rng.pick_str(NOUNS));
            if rng.chance(0.4) {
                s.push(' ');
                s.push_str(rng.pick_str(VERBS));
            }
            s
        })
        .collect()
}

pub fn label(rng: &mut Rng) -> String {
    let verb = rng.pick_str(&[
        "downloading",
        "building",
        "indexing",
        "compiling",
        "fetching",
        "installing",
        "linking",
        "analyzing",
        "packaging",
    ]);
    format!("{} {}", cap(verb), rng.pick_str(NOUNS))
}

pub fn filename(rng: &mut Rng) -> String {
    format!(
        "{}.{}",
        rng.pick_str(NOUNS),
        rng.pick_str(&["rs", "txt", "toml", "json", "log", "md", "py"])
    )
}
