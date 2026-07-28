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

use crate::format::WeightFormat;
use anyhow::{Result, anyhow};
use rlx_core::gguf_config::{gguf_memory_footprint, gguf_runner_hint};
use rlx_core::weight_loader::GgufLoader;
use rlx_core::weight_registry::list_registered_formats;
use rlx_core::weights::{ResolveOpts, gguf_dir_guide};
use rlx_core::weights_discover::{
    DiscoverOpts, DiscoveredWeight, WeightSourceKind, resolve_weight_query, scan_weights,
};
use rlx_gguf::GgufFile;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn estimate_qwen35_footprint(raw: &GgufFile) -> (u64, u64) {
    let fp = gguf_memory_footprint(raw);
    (fp.f32_bytes, fp.packed_file_bytes)
}

pub fn fmt_bytes(b: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let f = b as f64;
    if f >= GB {
        format!("{:.2} GB", f / GB)
    } else if f >= MB {
        format!("{:.1} MB", f / MB)
    } else {
        format!("{b} B")
    }
}

pub fn list_mtp_keys(path: &Path) -> Result<Vec<String>> {
    if WeightFormat::detect(path)? != WeightFormat::Gguf {
        return Ok(vec![]);
    }
    let loader = GgufLoader::from_file(path.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?)?;
    Ok(loader.mtp_keys())
}

struct InspectArgs<'a> {
    path: &'a str,
    prefer: Option<&'a str>,
    list_formats: bool,
    json: bool,
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn parse_inspect_args(args: &[String]) -> Result<InspectArgs<'_>> {
    let mut path = None;
    let mut prefer = None;
    let mut list_formats = false;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prefer" | "-p" => {
                prefer = Some(
                    args.get(i + 1)
                        .ok_or_else(|| anyhow!("--prefer requires a substring (e.g. Q4_K_M)"))?
                        .as_str(),
                );
                i += 2;
            }
            "--list-formats" => {
                list_formats = true;
                i += 1;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            s if s.starts_with('-') => {
                return Err(anyhow!("unknown flag `{s}` (try --help)"));
            }
            s => {
                if path.is_some() {
                    return Err(anyhow!("unexpected argument `{s}`"));
                }
                path = Some(s);
                i += 1;
            }
        }
    }
    if path.is_none() && !list_formats {
        print_usage();
        return Err(anyhow!("missing path (or use --list-formats)"));
    }
    Ok(InspectArgs {
        path: path.unwrap_or(""),
        prefer,
        list_formats,
        json,
    })
}

fn print_usage() {
    eprintln!(
        "usage: rlx-inspect <path> [--prefer Q4_K_M] [--list-formats]\n\
         \n\
                rlx-inspect scan [--source lmstudio,ollama,hf,…] [--query SUB] [--prefer Q4_K_M] [--json] [--sniff]\n\
                rlx-inspect resolve <query> [--prefer Q4_K_M] [--json]\n\
         \n\
         Examples:\n\
           rlx-inspect model.gguf\n\
           rlx-inspect weights/              # lists .gguf files in a directory\n\
           rlx-inspect weights/ --prefer Q4_K_M\n\
           rlx-inspect --list-formats        # show registered weight extensions\n\
           rlx-inspect model.gguf --json     # machine-readable summary\n\
           rlx-inspect scan --query qwen3\n\
           rlx-inspect resolve qwen3-0.6b --prefer Q4_K_M"
    );
}

pub fn run_inspect(args: &[String]) -> Result<()> {
    if let Some(cmd) = args.first().map(String::as_str) {
        match cmd {
            "scan" => return run_scan(&args[1..]),
            "resolve" => return run_resolve(&args[1..]),
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => {}
        }
    }

    let parsed = parse_inspect_args(args)?;
    if parsed.list_formats {
        if parsed.json {
            print!("[");
            for (i, reg) in list_registered_formats().iter().enumerate() {
                if i > 0 {
                    print!(",");
                }
                let exts: Vec<String> = reg.extensions.iter().map(|e| format!("\"{e}\"")).collect();
                print!(
                    "{{\"id\":\"{}\",\"extensions\":[{}]}}",
                    reg.id,
                    exts.join(",")
                );
            }
            println!("]");
        } else {
            println!("registered weight formats:");
            for reg in list_registered_formats() {
                println!("  {} → .{}", reg.id, reg.extensions.join(", ."));
            }
        }
        if parsed.path.is_empty() {
            return Ok(());
        }
    }

    let pb: PathBuf = parsed.path.into();
    if parsed.list_formats && pb.as_os_str().is_empty() {
        return Ok(());
    }

    let fmt = WeightFormat::detect(&pb)?;
    println!("path:   {pb:?}");
    println!("format: {fmt:?}");

    if pb.is_dir() {
        let guide = gguf_dir_guide(&pb)?;
        if !guide.files.is_empty() {
            println!();
            guide.print();
            if let Some(sub) = parsed.prefer {
                let pick = guide.files.iter().position(|p| {
                    p.file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|n| n.contains(sub))
                });
                if let Some(idx) = pick {
                    println!();
                    println!(
                        "resolve:  --prefer {sub} → [{}] {:?}",
                        idx, guide.files[idx]
                    );
                    println!(
                        "rust:     rlx_core::weights::open_map_with(\
                         LoadOpts::map().prefer_substring(\"{sub}\"), path)?"
                    );
                } else {
                    println!();
                    println!("resolve:  no file name contains `{sub}`");
                }
            }
            println!();
        }
    }

    let inspect_path = if pb.is_dir() {
        if let Some(sub) = parsed.prefer {
            let resolved = rlx_core::resolve_weights_file_with_options(
                &pb,
                &ResolveOpts::default().prefer_substring(sub),
            )?;
            println!("picked:   {resolved:?}");
            resolved
        } else if fmt == WeightFormat::Gguf {
            println!(
                "hint:     pass a file path, or --prefer Q4_K_M, or inspect one file from the list above"
            );
            return Ok(());
        } else {
            pb.clone()
        }
    } else {
        pb.clone()
    };

    match fmt {
        WeightFormat::Gguf => inspect_gguf(&inspect_path, parsed.json)?,
        WeightFormat::Safetensors => inspect_safetensors(&inspect_path, parsed.json)?,
    }
    Ok(())
}

fn run_scan(args: &[String]) -> Result<()> {
    let mut sources: Option<Vec<WeightSourceKind>> = None;
    let mut query: Option<String> = None;
    let mut prefer: Option<String> = None;
    let mut json = false;
    let mut sniff = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--source" | "--sources" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--source needs a comma-separated list"))?;
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
            "--query" | "-q" => {
                i += 1;
                query = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--query needs a substring"))?
                        .clone(),
                );
                i += 1;
            }
            "--prefer" | "-p" => {
                i += 1;
                prefer = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--prefer needs a substring"))?
                        .clone(),
                );
                i += 1;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            "--sniff" => {
                sniff = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            s if s.starts_with('-') => {
                return Err(anyhow!("unknown scan flag `{s}`"));
            }
            s => {
                if query.is_some() {
                    return Err(anyhow!("unexpected argument `{s}`"));
                }
                query = Some(s.to_string());
                i += 1;
            }
        }
    }
    let opts = DiscoverOpts {
        sources,
        query,
        prefer_quant: prefer,
        sniff_arch: sniff,
        ..DiscoverOpts::default()
    };
    let hits = scan_weights(&opts)?;
    if json {
        print_discovered_json(&hits);
    } else {
        print_discovered_table(&hits);
    }
    Ok(())
}

fn run_resolve(args: &[String]) -> Result<()> {
    let mut query: Option<String> = None;
    let mut prefer: Option<String> = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prefer" | "-p" => {
                i += 1;
                prefer = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--prefer needs a substring"))?
                        .clone(),
                );
                i += 1;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            s if s.starts_with('-') => {
                return Err(anyhow!("unknown resolve flag `{s}`"));
            }
            s => {
                if query.is_some() {
                    return Err(anyhow!("unexpected argument `{s}`"));
                }
                query = Some(s.to_string());
                i += 1;
            }
        }
    }
    let query = query.ok_or_else(|| anyhow!("resolve needs a query (try --help)"))?;
    let opts = DiscoverOpts {
        prefer_quant: prefer,
        ..DiscoverOpts::default()
    };
    let path = resolve_weight_query(&query, &opts)?;
    if json {
        println!(
            "{{\"query\":\"{}\",\"path\":\"{}\"}}",
            json_escape(&query),
            json_escape(&path.display().to_string())
        );
    } else {
        println!("{}", path.display());
    }
    Ok(())
}

fn print_discovered_table(hits: &[DiscoveredWeight]) {
    if hits.is_empty() {
        println!("(no local weights found)");
        return;
    }
    println!(
        "{:<18}  {:<40}  {:<10}  {:>10}  path",
        "source", "name", "quant", "size"
    );
    for w in hits {
        let srcs: Vec<&str> = w.sources.iter().map(|s| s.as_str()).collect();
        let quant = w.quant_hint.as_deref().unwrap_or("-");
        let size = w
            .size_bytes
            .map(fmt_bytes)
            .unwrap_or_else(|| "-".to_string());
        let name = truncate_display(&w.display_name, 40);
        println!(
            "{:<18}  {:<40}  {:<10}  {:>10}  {}",
            truncate_display(&srcs.join("+"), 18),
            name,
            quant,
            size,
            w.path.display()
        );
    }
    println!("{} weight(s)", hits.len());
}

fn truncate_display(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn print_discovered_json(hits: &[DiscoveredWeight]) {
    print!("[");
    for (i, w) in hits.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        let srcs: Vec<String> = w
            .sources
            .iter()
            .map(|s| format!("\"{}\"", s.as_str()))
            .collect();
        let quant = w
            .quant_hint
            .as_ref()
            .map(|q| format!("\"{}\"", json_escape(q)))
            .unwrap_or_else(|| "null".to_string());
        let arch = w
            .arch_hint
            .as_ref()
            .map(|a| format!("\"{}\"", json_escape(a)))
            .unwrap_or_else(|| "null".to_string());
        let size = w
            .size_bytes
            .map(|n| n.to_string())
            .unwrap_or_else(|| "null".to_string());
        print!(
            "{{\"path\":\"{}\",\"sources\":[{}],\"display_name\":\"{}\",\"format\":\"{}\",\
\"size_bytes\":{},\"quant_hint\":{},\"arch_hint\":{}}}",
            json_escape(&w.path.display().to_string()),
            srcs.join(","),
            json_escape(&w.display_name),
            w.format.as_str(),
            size,
            quant,
            arch
        );
    }
    println!("]");
}

fn inspect_gguf(pb: &Path, json: bool) -> Result<()> {
    let raw = GgufFile::from_path(pb)?;
    println!("version:  {}", raw.version);
    println!("tensors:  {}", raw.tensors.len());
    println!("metadata: {} keys", raw.metadata.len());
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let runner = gguf_runner_hint(arch);
    if json {
        let (f32_bytes, packed_bytes) = estimate_qwen35_footprint(&raw);
        let mtp = list_mtp_keys(pb)?;
        println!(
            "{{\"format\":\"gguf\",\"path\":\"{}\",\"arch\":\"{}\",\"runner\":\"{}\",\
             \"tensors\":{},\"f32_bytes\":{},\"packed_bytes\":{},\"mtp_heads\":{}}}",
            json_escape(&pb.display().to_string()),
            json_escape(arch),
            json_escape(runner),
            raw.tensors.len(),
            f32_bytes,
            packed_bytes,
            mtp.len()
        );
        return Ok(());
    }
    println!("arch:     {arch}");
    println!("runner:   {runner}");
    let mamba = raw
        .tensors
        .keys()
        .any(|k| k.starts_with("blk.0.ssm_") || k == "blk.0.attn_qkv.weight");
    match (arch, mamba) {
        ("qwen3", false) | ("qwen36", false) => {
            println!("compat:   ok — `just qwen3 -- --weights {:?} …`", pb);
            println!(
                "rust:     weights::open_with(LoadOpts::loader(), path)?  // runner validates arch"
            );
        }
        ("llama", false) => {
            println!("compat:   ok — `rlx-llama32` / rlx_models::llama32");
            println!("rust:     weights::open_with(LoadOpts::loader(), path)?");
        }
        ("qwen35", true) | ("qwen35moe", true) | (_, true) => {
            println!("compat:   ok — `rlx-qwen35 --packed`");
            println!("rust:     weights::open_with(LoadOpts::loader(), path)?");
        }
        ("bert", _) | ("modern-bert", _) | ("nomic-bert", _) | ("nomic-bert-moe", _) => {
            println!("compat:   ok — `rlx-embed`");
            println!(
                "rust:     gguf_validate_arch(path, EMBED_GGUF_ARCHES)?; weights::open_map(path)?"
            );
        }
        ("flux", _) => {
            println!("compat:   ok — `rlx-flux2` (denoiser GGUF; VAE/TE safetensors)");
            println!(
                "rust:     gguf_validate_arch(path, FLUX_GGUF_ARCHES)?; weights::open_map(path)?"
            );
        }
        ("dinov2", _) => {
            println!("compat:   ok — `rlx-dinov2` (F32 drain; tensor names must match HF/candle)");
            println!("rust:     rlx_core::load_weight_map(path, DINOV2_GGUF_ARCHES)?");
        }
        ("sam3", _) => {
            println!("compat:   ok — `rlx-sam3`");
            println!("rust:     rlx_core::load_weight_map(path, SAM3_GGUF_ARCHES)?");
        }
        ("sam2", _) => {
            println!("compat:   ok — `rlx-sam2` (community GGUF; parity not verified)");
            println!("rust:     rlx_core::load_weight_map(path, SAM2_GGUF_ARCHES)?");
        }
        ("sam", _) | ("mobile-sam", _) => {
            println!("compat:   ok — `rlx-sam` (ViT-H `sam` or MobileSAM `mobile-sam`)");
            println!("rust:     rlx_core::load_weight_map(path, SAM_GGUF_ARCHES)?");
        }
        ("vjepa2", _) | ("vjepa", _) => {
            println!("compat:   ok — `rlx-vjepa2` (experimental; few public GGUF checkpoints)");
            println!("rust:     rlx_core::load_weight_map(path, VJEPA2_GGUF_ARCHES)?");
        }
        ("w2v-bert", _) | ("wav2vec2", _) | ("wav2vec", _) => {
            println!(
                "compat:   ok — `rlx-wav2vec2-bert` (F32 drain; `config.json` beside weights)"
            );
            println!("rust:     rlx_core::load_weight_map(path, W2V_BERT_GGUF_ARCHES)?");
        }
        _ => {
            println!(
                "compat:   unknown — extend via `rlx_core::model_registry::register_gguf_model`"
            );
        }
    }
    let mut by_dt: BTreeMap<String, usize> = BTreeMap::new();
    for t in raw.tensors.values() {
        *by_dt.entry(format!("{:?}", t.dtype)).or_default() += 1;
    }
    println!("dtypes:");
    for (dt, n) in &by_dt {
        println!("  {dt:>6}: {n}");
    }
    let (f32_bytes, packed_bytes) = estimate_qwen35_footprint(&raw);
    println!(
        "footprint: F32-dequant ≈ {} / on-disk packed ≈ {} \
         (LM: use --packed when F32 does not fit)",
        fmt_bytes(f32_bytes),
        fmt_bytes(packed_bytes),
    );
    let mtp = list_mtp_keys(pb)?;
    if mtp.is_empty() {
        println!("mtp:      (none)");
    } else {
        println!("mtp:      {} heads", mtp.len());
        for k in mtp.iter().take(5) {
            println!("    {k}");
        }
    }
    Ok(())
}

fn inspect_safetensors(pb: &Path, json: bool) -> Result<()> {
    let meta = std::fs::metadata(pb)?;
    if json {
        println!(
            "{{\"format\":\"safetensors\",\"path\":\"{}\",\"size_bytes\":{}}}",
            json_escape(&pb.display().to_string()),
            meta.len()
        );
        return Ok(());
    }
    println!("size:     {} bytes", meta.len());
    println!("rust:     rlx_core::weights::open_map(path)?");
    println!("(tensor names: use WeightMap::from_file for a full listing)");
    Ok(())
}
