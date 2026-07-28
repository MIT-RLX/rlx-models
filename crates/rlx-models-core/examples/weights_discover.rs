//! List or resolve local weight files from LLM app caches.
//!
//! ```text
//! cargo run -p rlx-models-core --example weights_discover --
//! cargo run -p rlx-models-core --example weights_discover -- --query qwen --json
//! cargo run -p rlx-models-core --example weights_discover -- --resolve qwen3-0.6b
//! cargo run -p rlx-models-core --example weights_discover -- --roots
//! ```

use anyhow::{Result, bail};
use rlx_models_core::weights_discover::{
    DiscoverOpts, WeightSourceKind, default_source_roots, resolve_weight_query, scan_weights,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut query: Option<String> = None;
    let mut resolve: Option<String> = None;
    let mut prefer: Option<String> = None;
    let mut sources: Option<Vec<WeightSourceKind>> = None;
    let mut json = false;
    let mut list_roots = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--query" | "-q" => {
                i += 1;
                query = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--query"))?,
                );
                i += 1;
            }
            "--resolve" => {
                i += 1;
                resolve = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--resolve needs a query"))?,
                );
                i += 1;
            }
            "--prefer" | "-p" => {
                i += 1;
                prefer = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--prefer"))?,
                );
                i += 1;
            }
            "--source" | "--sources" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--source needs a list"))?;
                let mut list = Vec::new();
                for part in raw.split(',') {
                    let t = part.trim();
                    if !t.is_empty() {
                        list.push(WeightSourceKind::parse(t)?);
                    }
                }
                sources = Some(list);
                i += 1;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            "--roots" => {
                list_roots = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: weights_discover [--query SUB] [--resolve QUERY] [--prefer Q4_K_M]\n\
                     \n\
                            [--source lmstudio,ollama,hf,…] [--json] [--roots]"
                );
                return Ok(());
            }
            other => bail!("unknown argument `{other}` (try --help)"),
        }
    }

    if list_roots {
        for (kind, path) in default_source_roots() {
            println!("{:<10} {}", kind.as_str(), path.display());
        }
        return Ok(());
    }

    let mut opts = DiscoverOpts::default();
    if let Some(q) = query {
        opts = opts.with_query(q);
    }
    if let Some(p) = prefer {
        opts = opts.with_prefer_quant(p);
    }
    if let Some(s) = sources {
        opts = opts.with_sources(s);
    }

    if let Some(q) = resolve {
        let path = resolve_weight_query(&q, &opts)?;
        if json {
            println!(
                "{{\"query\":{},\"path\":{}}}",
                serde_json::to_string(&q)?,
                serde_json::to_string(&path)?
            );
        } else {
            println!("{}", path.display());
        }
        return Ok(());
    }

    let hits = scan_weights(&opts)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
    } else if hits.is_empty() {
        println!("(no local weights found)");
    } else {
        for w in &hits {
            let srcs: Vec<&str> = w.sources.iter().map(|s| s.as_str()).collect();
            println!(
                "[{}] {}  {}",
                srcs.join("+"),
                w.display_name,
                w.path.display()
            );
        }
        println!("{} weight(s)", hits.len());
    }
    Ok(())
}
