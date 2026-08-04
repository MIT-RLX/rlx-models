fn main() -> anyhow::Result<()> {
    use rlx_cli::{arch_runner_name, auto_sniff, known_unimplemented_arch};
    use std::fs;
    use std::path::PathBuf;

    let root = PathBuf::from(std::env::args().nth(1).expect("dir"));
    let mut paths: Vec<_> = fs::read_dir(&root)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    paths.sort();
    println!(
        "{:<18} {:<12} {:<14} detail",
        "arch/file", "runner", "class"
    );
    println!("{}", "-".repeat(90));
    for p in paths {
        match auto_sniff(&p) {
            Ok(s) => {
                let arch = match &s.from {
                    rlx_cli::SniffedFrom::GgufArch(a) => a.clone(),
                    other => format!("{other:?}"),
                };
                println!(
                    "{:<18} {:<12} {:<14} path={}",
                    arch,
                    s.runner_name,
                    "registered",
                    p.file_name().unwrap().to_string_lossy()
                );
            }
            Err(e) => {
                let arch = rlx_core::gguf_architecture_from_path(&p).unwrap_or_else(|_| "?".into());
                let runner = arch_runner_name(&arch);
                if let Some(u) = known_unimplemented_arch(&arch) {
                    println!(
                        "{:<18} {:<12} {:<14} {} ({}) — {}",
                        arch,
                        runner.unwrap_or("-"),
                        "unimplemented",
                        u.family,
                        u.milestone,
                        u.note
                    );
                } else {
                    println!(
                        "{:<18} {:<12} {:<14} {}",
                        arch,
                        runner.unwrap_or("-"),
                        "unknown",
                        e.to_string().lines().next().unwrap_or("")
                    );
                }
            }
        }
    }
    Ok(())
}
