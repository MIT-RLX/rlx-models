//! Dataset generator: writes JSONL splits + a preview + a manifest into the
//! crate's `data/` directory.
//!
//! Usage:
//!   cargo run -p rlx-termclean --bin rlx-termclean-gen --release
//!   cargo run -p rlx-termclean --bin rlx-termclean-gen --release -- \
//!       --train 20000 --val 2500 --test 2500 --seed 3735928559 --out ./data

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use rlx_termclean::record::{json_escape, write_record};
use rlx_termclean::{Rng, generate};

struct Config {
    out: PathBuf,
    train: usize,
    val: usize,
    test: usize,
    seed: u64,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        out: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data")),
        train: 20_000,
        val: 2_500,
        test: 2_500,
        seed: 0xC0FF_EE00_1234_5678,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let need = |i: usize| -> &str {
            args.get(i + 1).unwrap_or_else(|| {
                eprintln!("missing value for {}", args[i]);
                std::process::exit(2);
            })
        };
        match args[i].as_str() {
            "--out" => {
                cfg.out = PathBuf::from(need(i));
                i += 2;
            }
            "--train" => {
                cfg.train = need(i).parse().expect("--train N");
                i += 2;
            }
            "--val" => {
                cfg.val = need(i).parse().expect("--val N");
                i += 2;
            }
            "--test" => {
                cfg.test = need(i).parse().expect("--test N");
                i += 2;
            }
            "--seed" => {
                cfg.seed = need(i).parse().expect("--seed U64");
                i += 2;
            }
            "-h" | "--help" => {
                println!(
                    "rlx-termclean-gen [--out DIR] [--train N] [--val N] [--test N] [--seed U64]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                i += 1;
            }
        }
    }
    cfg
}

fn main() -> std::io::Result<()> {
    let cfg = parse_args();
    fs::create_dir_all(&cfg.out)?;

    let mut gid = 0u64;
    let mut kind_counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut ct_counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut ansi_count = 0u64;
    let mut total_bytes = 0u64;

    let splits = [
        ("train", cfg.train, cfg.seed ^ 0x1),
        ("val", cfg.val, cfg.seed ^ 0x2),
        ("test", cfg.test, cfg.seed ^ 0x3),
    ];

    for (name, count, sd) in splits {
        let path = cfg.out.join(format!("{name}.jsonl"));
        let mut w = BufWriter::new(File::create(&path)?);
        let mut rng = Rng::new(sd);
        let mut line = String::new();
        for _ in 0..count {
            let s = generate(&mut rng, gid);
            gid += 1;
            *kind_counts.entry(s.kind).or_default() += 1;
            *ct_counts.entry(s.content_type).or_default() += 1;
            if s.ansi {
                ansi_count += 1;
            }
            line.clear();
            write_record(&s, &mut line);
            line.push('\n');
            total_bytes += line.len() as u64;
            w.write_all(line.as_bytes())?;
        }
        w.flush()?;
        println!("wrote {count:>7} samples -> {}", path.display());
    }

    let total = cfg.train + cfg.val + cfg.test;
    write_preview(&cfg)?;
    write_manifest(
        &cfg,
        total,
        total_bytes,
        &kind_counts,
        &ct_counts,
        ansi_count,
    )?;

    println!(
        "\ndone: {total} samples, {:.1} MiB, ANSI in {:.0}% of samples",
        total_bytes as f64 / (1024.0 * 1024.0),
        100.0 * ansi_count as f64 / total as f64
    );
    println!("layout distribution:");
    for (k, v) in &kind_counts {
        println!(
            "  {k:<10} {v:>7}  ({:.1}%)",
            100.0 * *v as f64 / total as f64
        );
    }
    Ok(())
}

/// Write a small human-readable sample of rendered screens for eyeballing.
/// Raw ANSI is kept, so `cat data/preview.txt` renders in color in a terminal.
fn write_preview(cfg: &Config) -> std::io::Result<()> {
    let mut out = String::new();
    let mut rng = Rng::new(cfg.seed ^ 0xBEEF);
    out.push_str("rlx-termclean preview — INPUT is the raw screen (ANSI/box-drawing),\n");
    out.push_str("TARGET is the clean text, TAGS mark each input char C=content X=chrome.\n");
    for id in 0..14u64 {
        let s = generate(&mut rng, id);
        out.push_str(&format!(
            "\n\u{2550}\u{2550}\u{2550}\u{2550} #{id} kind={} ct={} ansi={} style={} \u{2550}\u{2550}\u{2550}\u{2550}\n",
            s.kind, s.content_type, s.ansi, s.style
        ));
        out.push_str("--- INPUT ------------------------------------------------------\n");
        out.push_str(&s.input);
        if !s.input.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("--- TARGET -----------------------------------------------------\n");
        out.push_str(&s.target);
        out.push('\n');
        out.push_str("--- TAGS -------------------------------------------------------\n");
        out.push_str(&s.tags);
        out.push('\n');
    }
    fs::write(cfg.out.join("preview.txt"), out)
}

fn write_manifest(
    cfg: &Config,
    total: usize,
    total_bytes: u64,
    kinds: &BTreeMap<&'static str, u64>,
    cts: &BTreeMap<&'static str, u64>,
    ansi: u64,
) -> std::io::Result<()> {
    use rlx_termclean::symbols::*;

    let mut m = String::new();
    m.push_str("{\n");
    m.push_str("  \"generated_by\": \"rlx-termclean-gen\",\n");
    m.push_str(&format!("  \"seed\": {},\n", cfg.seed));
    m.push_str(&format!(
        "  \"counts\": {{ \"train\": {}, \"val\": {}, \"test\": {}, \"total\": {} }},\n",
        cfg.train, cfg.val, cfg.test, total
    ));
    m.push_str(&format!("  \"total_bytes\": {total_bytes},\n"));
    m.push_str(&format!("  \"ansi_samples\": {ansi},\n"));

    m.push_str("  \"schema\": {\n");
    m.push_str("    \"input\": \"raw rendered terminal screen (chrome + content + ANSI)\",\n");
    m.push_str("    \"target\": \"clean reflowed text\",\n");
    m.push_str("    \"tags\": \"one marker per input char: C=content, X=chrome\",\n");
    m.push_str("    \"fields\": [\"id\",\"kind\",\"content_type\",\"width\",\"ansi\",\"style\",\"input\",\"target\",\"tags\"]\n");
    m.push_str("  },\n");

    let dist = |map: &BTreeMap<&'static str, u64>| {
        let mut s = String::from("{");
        for (i, (k, v)) in map.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let mut ks = String::new();
            json_escape(k, &mut ks);
            s.push_str(&format!("{ks}: {v}"));
        }
        s.push('}');
        s
    };
    m.push_str(&format!("  \"kind_distribution\": {},\n", dist(kinds)));
    m.push_str(&format!(
        "  \"content_type_distribution\": {},\n",
        dist(cts)
    ));

    // Symbol inventory summary (kept in sync with `symbols.rs`).
    let box_names: Vec<String> = BOX_STYLES
        .iter()
        .map(|b| {
            let mut s = String::new();
            json_escape(b.name, &mut s);
            s
        })
        .collect();
    m.push_str("  \"symbol_inventory\": {\n");
    m.push_str(&format!(
        "    \"box_styles\": [{}],\n",
        box_names.join(", ")
    ));
    m.push_str(&format!("    \"box_style_count\": {},\n", BOX_STYLES.len()));
    m.push_str(&format!("    \"bullets\": {},\n", BULLETS.len()));
    m.push_str(&format!("    \"arrows\": {},\n", ARROWS.len()));
    m.push_str(&format!("    \"shades\": {},\n", SHADES.len()));
    m.push_str(&format!(
        "    \"partial_blocks\": {},\n",
        PARTIAL_BLOCKS.len()
    ));
    m.push_str(&format!("    \"spinner_sets\": {},\n", SPINNERS.len()));
    m.push_str(&format!("    \"sgr_params\": {}\n", SGR_PARAMS.len()));
    m.push_str("  }\n");
    m.push_str("}\n");

    fs::write(cfg.out.join("manifest.json"), m)
}
