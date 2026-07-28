//! Walk a directory of GGUFs and report registry / unimplemented status.
fn main() -> anyhow::Result<()> {
    use rlx_models_core::gguf_architecture_from_path;
    use rlx_models_core::model_registry::{
        ensure_builtin_gguf_models, family_for_gguf_arch, hint_for_gguf_arch, runner_for_gguf_arch,
    };
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    ensure_builtin_gguf_models();
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights"));

    let mut files: Vec<PathBuf> = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
                out.push(p);
            }
        }
    }
    walk(&root, &mut files);
    files.sort();

    // Mirror rlx-cli unimplemented keys for classification without depending on rlx-cli.
    let unimplemented: &[&str] = &[
        "phimoe",
        "bonsai",
        "omnicoder",
        "minimax-m2",
        "minimax_m2",
        "minimax",
        "glm4",
        "glm5",
        "chatglm",
        "glm4moe",
        "gpt-oss",
        "gpt_oss",
        "nemotron",
        "nemotron_h",
        "nemotron_h_moe",
        "lfm2moe",
        "qwen3moe",
        "qwen3next",
        "gemma4moe",
        "qwen3_mtp",
        "qwen3-mtp",
        "qwen36_mtp",
        "llada",
        "llada-moe",
        "granite",
        "granitemoe",
        "granitehybrid",
        "deepseek2",
        "deepseek2-ocr",
        "command-r",
        "cohere2",
    ];

    #[derive(Default)]
    struct Bucket {
        n: usize,
        samples: Vec<String>,
    }
    let mut by_arch: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut errors = 0usize;

    for p in &files {
        match gguf_architecture_from_path(p) {
            Ok(arch) => {
                let b = by_arch.entry(arch).or_default();
                b.n += 1;
                if b.samples.len() < 2 {
                    b.samples.push(p.display().to_string());
                }
            }
            Err(e) => {
                errors += 1;
                eprintln!("ERR {}: {e:#}", p.display());
            }
        }
    }

    println!(
        "scanned {} gguf under {} (errors={errors})\n",
        files.len(),
        root.display()
    );
    println!(
        "{:<22} {:>5}  {:<12} {:<10} hint/note",
        "arch", "count", "runner", "status"
    );
    println!("{}", "-".repeat(100));
    for (arch, b) in &by_arch {
        let runner = runner_for_gguf_arch(arch);
        let fam = family_for_gguf_arch(arch);
        let hint = hint_for_gguf_arch(arch).unwrap_or("");
        let status = if runner.is_some() {
            "registered"
        } else if unimplemented.contains(&arch.as_str()) {
            "unimplemented"
        } else if hint.is_empty() {
            "unknown"
        } else {
            "hint-only"
        };
        println!(
            "{:<22} {:>5}  {:<12} {:<10} {}",
            arch,
            b.n,
            runner.unwrap_or("-"),
            status,
            if hint.is_empty() {
                format!("fam={fam:?}")
            } else {
                hint.to_string()
            }
        );
        for s in &b.samples {
            println!("    sample: {s}");
        }
    }
    Ok(())
}
