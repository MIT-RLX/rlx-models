//! Convert ONNX subgraphs → nested native `.rlxp` packs, then assemble an outer
//! `.rlxp` with **no** `.onnx` / `.onnx.data` sidecars.
//!
//! # Layout
//!
//! Each subgraph is itself an RLXPFLAT package:
//! ```text
//! graphs/<name>.rlxp
//!   ├── tensors/          # hot mmap — f32 + i64 initializers
//!   └── sidecars/         # cold zstd
//!       ├── manifest.json
//!       ├── graph.json
//!       ├── if_branches.json   # optional (If subgraphs)
//!       └── scalar_consts.json # optional
//! ```
//!
//! Outer Hub packs embed those nested packs as file sidecars plus tokenizer /
//! frontend assets. Runtime opens each nested [`rlx_pkg::Package`], reads
//! tensors + graph JSON, and lowers via `build_hir_from_parts` (same length
//! specialization as the old ONNX path — without shipping ONNX or safetensors
//! for neural weights).
//!
//! # Typical pack flow
//!
//! ```ignore
//! use rlx_assets::native_pack::{OnnxGraphSpec, pack_native_from_onnx_dir};
//! let specs = vec![OnnxGraphSpec {
//!     name: "text_encoder".into(),
//!     onnx_path: "onnx/text_encoder.onnx".into(),
//! }];
//! pack_native_from_onnx_dir("weights/tts/tiny-tts-rlx", &specs, "tiny-tts.rlxp", "tiny-tts")?;
//! ```
//!
//! # Typical load flow
//!
//! 1. [`load_native_subgraph_rlxp`] — read tensors + IR sidecars.
//! 2. Length-specialize / shape-propagate as needed.
//! 3. [`install_native_subgraph_tls`] — restore If/scalar thread-locals.
//! 4. `rlx_onnx_import::build_hir_from_parts` on the same thread.
//!
//! Enable with Cargo feature `native-pack` (pulls in `rlxp` + `rlx-onnx-import`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::Graph;
use rlx_onnx_import::{
    BundleManifest, BundleNode, prepare_onnx_file, take_if_branches, take_scalar_consts,
};
use rlx_pkg::{ContainerKind, Package, PackedWeight, WriteOptions, write_package};

use crate::pack::{collect_dir_files, write_named_files_rlxp};

/// One ONNX file to bake into `graphs/<name>.rlxp`.
///
/// `name` becomes the nested pack basename (`graphs/{name}.rlxp`). `onnx_path`
/// is pack-time only — Hub outer packs never embed the `.onnx` itself.
#[derive(Debug, Clone)]
pub struct OnnxGraphSpec {
    /// Subgraph id (also used as the HIR / cache component name).
    pub name: String,
    /// Local ONNX path (external `.data` resolved by `prepare_onnx_file`).
    pub onnx_path: PathBuf,
}

/// Loaded native subgraph: hot tensors + graph IR sidecars from a nested `.rlxp`.
///
/// Ready for length specialization and [`install_native_subgraph_tls`] before
/// `build_hir_from_parts`. Weight maps own their buffers (f32 / i64).
#[derive(Debug)]
pub struct NativeSubgraph {
    pub manifest: BundleManifest,
    pub nodes: Vec<BundleNode>,
    pub params: HashMap<String, Vec<f32>>,
    pub i64_params: HashMap<String, Vec<i64>>,
    pub init_shapes: HashMap<String, Vec<usize>>,
    pub if_branches: HashMap<String, (Vec<BundleNode>, Vec<BundleNode>)>,
    pub scalar_consts: HashSet<String>,
}

/// True when a pack TOC / relative path looks like ONNX or ONNX external data.
///
/// Used to refuse embedding pack-time sources into Hub outer packs.
pub fn is_forbidden_onnx_asset(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".onnx")
        || lower.ends_with(".onnx.data")
        || lower.ends_with(".onnx_data")
        || lower.contains(".onnx/")
}

fn resolve_shape(name: &str, nel: usize, init_shapes: &HashMap<String, Vec<usize>>) -> Vec<usize> {
    let shape = init_shapes
        .get(name)
        .cloned()
        .unwrap_or_else(|| vec![nel]);
    if shape.iter().product::<usize>() == nel {
        shape
    } else {
        vec![nel]
    }
}

/// Bake one ONNX model (external `.data` inlined) into a nested subgraph `.rlxp`.
///
/// Writes hot f32/i64 tensors plus `manifest.json` / `graph.json` (and optional
/// If / scalar-const sidecars). Pack-time only — does not run the graph.
pub fn export_onnx_to_subgraph_rlxp(
    onnx_path: &Path,
    out_rlxp: &Path,
    package_name: &str,
) -> Result<()> {
    ensure!(onnx_path.is_file(), "missing ONNX {}", onnx_path.display());
    if let Some(parent) = out_rlxp.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (mut manifest, nodes, params, i64_params, init_shapes) = prepare_onnx_file(onnx_path)
        .with_context(|| format!("prepare {}", onnx_path.display()))?;
    let if_branches = take_if_branches();
    let scalar_consts = take_scalar_consts();

    manifest.source_onnx = onnx_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.onnx")
        .to_string();

    let mut weights = Vec::new();
    for (name, data) in &params {
        let shape = resolve_shape(name, data.len(), &init_shapes);
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        weights.push(PackedWeight::hot(
            name.clone(),
            shape,
            "f32",
            "row_major",
            bytes,
        ));
    }
    for (name, data) in &i64_params {
        let shape = resolve_shape(name, data.len(), &init_shapes);
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        weights.push(PackedWeight::hot(
            name.clone(),
            shape,
            "i64",
            "row_major",
            bytes,
        ));
    }
    weights.sort_by(|a, b| a.name.cmp(&b.name));

    let mut sidecars: Vec<(String, String, Vec<u8>)> = vec![
        (
            "manifest.json".into(),
            "application/json".into(),
            serde_json::to_vec_pretty(&manifest)?,
        ),
        (
            "graph.json".into(),
            "application/json".into(),
            serde_json::to_vec_pretty(&nodes)?,
        ),
    ];
    if !if_branches.is_empty() {
        let mut map = serde_json::Map::new();
        for (name, (then_n, else_n)) in &if_branches {
            map.insert(
                name.clone(),
                serde_json::json!({ "then": then_n, "else": else_n }),
            );
        }
        sidecars.push((
            "if_branches.json".into(),
            "application/json".into(),
            serde_json::to_vec_pretty(&serde_json::Value::Object(map))?,
        ));
    }
    if !scalar_consts.is_empty() {
        let mut names: Vec<String> = scalar_consts.into_iter().collect();
        names.sort();
        sidecars.push((
            "scalar_consts.json".into(),
            "application/json".into(),
            serde_json::to_vec_pretty(&names)?,
        ));
    }

    let opts = WriteOptions {
        name: package_name.to_string(),
        producer: Some("rlx-assets/native-pack".into()),
        features: vec![
            "native_subgraph".into(),
            "graph_json".into(),
            "weight_only".into(),
        ],
        container: ContainerKind::Flat,
        sidecars,
        include_graph: false,
        compress_sidecars: true,
        write_checksums: true,
        ..WriteOptions::default()
    };
    let graph = Graph::new(package_name);
    write_package(out_rlxp, &graph, &weights, &opts)
        .with_context(|| format!("write subgraph {}", out_rlxp.display()))?;
    Ok(())
}

/// Export every [`OnnxGraphSpec`] to `staging/graphs/<name>.rlxp`.
pub fn export_onnx_graphs(specs: &[OnnxGraphSpec], staging: &Path) -> Result<Vec<PathBuf>> {
    let graphs = staging.join("graphs");
    std::fs::create_dir_all(&graphs)?;
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let path = graphs.join(format!("{}.rlxp", spec.name));
        export_onnx_to_subgraph_rlxp(&spec.onnx_path, &path, &spec.name)
            .with_context(|| format!("export graph {}", spec.name))?;
        out.push(path);
    }
    Ok(out)
}

/// Copy non-ONNX assets from `src_dir` into `staging`.
///
/// Skips `graphs/`, `onnx/`, existing `.rlxp` / `.gguf` packs, and any path
/// matching [`is_forbidden_onnx_asset`]. Keeps tokenizer / frontend / config.
pub fn copy_non_onnx_assets(src_dir: &Path, staging: &Path) -> Result<()> {
    fn walk(src: &Path, rel: &Path, staging: &Path) -> Result<()> {
        for ent in std::fs::read_dir(src)? {
            let ent = ent?;
            let name = ent.file_name();
            let name_s = name.to_string_lossy();
            if name_s == ".DS_Store"
                || name_s == "graphs"
                || name_s == "onnx"
                || name_s.ends_with(".rlxp")
                || name_s.ends_with(".rlxpack")
                || name_s.ends_with(".gguf")
            {
                continue;
            }
            let rel_child = rel.join(&name);
            let rel_str = rel_child.to_string_lossy().replace('\\', "/");
            if is_forbidden_onnx_asset(&rel_str) || name_s.ends_with(".onnx") {
                continue;
            }
            let dest = staging.join(&rel_child);
            if ent.file_type()?.is_dir() {
                std::fs::create_dir_all(&dest)?;
                walk(&ent.path(), &rel_child, staging)?;
            } else {
                if let Some(p) = dest.parent() {
                    std::fs::create_dir_all(p)?;
                }
                std::fs::copy(ent.path(), &dest)?;
            }
        }
        Ok(())
    }
    std::fs::create_dir_all(staging)?;
    walk(src_dir, Path::new(""), staging)
}

/// Build an outer native `.rlxp` from ONNX specs + non-ONNX assets under `src_dir`.
///
/// Stages under a temp dir: bake nested `graphs/<name>.rlxp`, copy sidecars
/// (tokenizer / frontend / config), then write one RLXPFLAT file. **Fails** if
/// any TOC entry looks like ONNX — Hub packs must not ship `.onnx`.
pub fn pack_native_from_onnx_dir(
    src_dir: &Path,
    specs: &[OnnxGraphSpec],
    out_rlxp: &Path,
    package_name: &str,
) -> Result<()> {
    let staging = std::env::temp_dir().join(format!(
        "rlx-native-pack-{}-{}",
        std::process::id(),
        package_name
    ));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;

    export_onnx_graphs(specs, &staging)?;
    copy_non_onnx_assets(src_dir, &staging)?;

    let files = collect_dir_files(&staging)?;
    for (name, _) in &files {
        if is_forbidden_onnx_asset(name) {
            bail!(
                "refusing to pack ONNX asset `{name}` into {}",
                out_rlxp.display()
            );
        }
    }
    if let Some(parent) = out_rlxp.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_named_files_rlxp(files, out_rlxp, Some(package_name))?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

fn parse_if_branches(bytes: &[u8]) -> Result<HashMap<String, (Vec<BundleNode>, Vec<BundleNode>)>> {
    let v: serde_json::Value = serde_json::from_slice(bytes)?;
    let mut map = HashMap::new();
    let Some(obj) = v.as_object() else {
        return Ok(map);
    };
    for (name, entry) in obj {
        let then_n: Vec<BundleNode> =
            serde_json::from_value(entry.get("then").cloned().unwrap_or(serde_json::json!([])))?;
        let else_n: Vec<BundleNode> =
            serde_json::from_value(entry.get("else").cloned().unwrap_or(serde_json::json!([])))?;
        map.insert(name.clone(), (then_n, else_n));
    }
    Ok(map)
}

/// Open a nested subgraph `.rlxp` (tensors + graph IR sidecars).
///
/// Does **not** touch thread-locals — call [`install_native_subgraph_tls`]
/// immediately before `build_hir_from_parts` on the same thread.
pub fn load_native_subgraph_rlxp(path: &Path) -> Result<NativeSubgraph> {
    let pack =
        Package::open(path).with_context(|| format!("open subgraph {}", path.display()))?;
    let man = pack.manifest();
    ensure!(
        man.sidecars.iter().any(|s| s.id == "graph.json"),
        "{} missing graph.json sidecar",
        path.display()
    );

    let manifest: BundleManifest = serde_json::from_slice(&pack.sidecar("manifest.json")?)?;
    let nodes: Vec<BundleNode> = serde_json::from_slice(&pack.sidecar("graph.json")?)?;

    let if_branches = if man.sidecars.iter().any(|s| s.id == "if_branches.json") {
        parse_if_branches(&pack.sidecar("if_branches.json")?)?
    } else {
        HashMap::new()
    };
    let scalar_consts: HashSet<String> =
        if man.sidecars.iter().any(|s| s.id == "scalar_consts.json") {
            serde_json::from_slice(&pack.sidecar("scalar_consts.json")?).unwrap_or_default()
        } else {
            HashSet::new()
        };

    let mut params = HashMap::new();
    let mut i64_params = HashMap::new();
    let mut init_shapes = HashMap::new();
    if let Some(idx) = pack.weights_index() {
        for entry in &idx.tensors {
            init_shapes.insert(entry.name.clone(), entry.shape.clone());
            let scheme = entry.scheme.to_ascii_lowercase();
            if scheme == "i64" || scheme == "int64" {
                let bytes = pack.tensor_bytes(&entry.name)?;
                let v: Vec<i64> = bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                i64_params.insert(entry.name.clone(), v);
            } else {
                let v = pack
                    .tensor_f32(&entry.name)
                    .with_context(|| format!("tensor f32 {}", entry.name))?;
                params.insert(entry.name.clone(), v);
            }
        }
    }

    Ok(NativeSubgraph {
        manifest,
        nodes,
        params,
        i64_params,
        init_shapes,
        if_branches,
        scalar_consts,
    })
}

/// Install If-branch / scalar-const thread-locals for `build_hir_from_parts`.
///
/// Call on the **same thread** immediately before lowering. Re-call after shape
/// propagation if that path drained the thread-locals.
pub fn install_native_subgraph_tls(g: &NativeSubgraph) {
    use rlx_onnx_import::{install_if_branches, install_scalar_consts};
    install_if_branches(g.if_branches.clone());
    install_scalar_consts(g.scalar_consts.clone());
}

/// Resolve `graphs/<component>.rlxp` (preferred) or legacy ONNX under `root`.
///
/// Order: nested `.rlxp` → loose `graphs/<name>/manifest.json` (dev) →
/// `onnx/<name>.onnx` → `<name>.onnx`.
pub fn resolve_component_path(root: &Path, component: &str) -> Result<PathBuf> {
    let nested = root.join("graphs").join(format!("{component}.rlxp"));
    if nested.is_file() {
        return Ok(nested);
    }
    // Loose exported dir (dev): graphs/<name>/manifest.json — still supported.
    let dir = root.join("graphs").join(component);
    if dir.join("manifest.json").is_file() {
        return Ok(dir);
    }
    let onnx = root.join("onnx").join(format!("{component}.onnx"));
    if onnx.is_file() {
        return Ok(onnx);
    }
    let flat = root.join(format!("{component}.onnx"));
    if flat.is_file() {
        return Ok(flat);
    }
    bail!(
        "missing graph `{component}` under {} (expected graphs/{component}.rlxp)",
        root.display()
    )
}

/// True when `root` has at least one nested `graphs/*.rlxp`.
pub fn has_native_graphs(root: &Path) -> bool {
    let graphs = root.join("graphs");
    let Ok(rd) = std::fs::read_dir(graphs) else {
        return false;
    };
    rd.filter_map(|e| e.ok()).any(|e| {
        e.path()
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("rlxp"))
    })
}

/// Build [`OnnxGraphSpec`] list from component names under `root`.
///
/// Looks for `root/onnx/<name>.onnx`, else `root/<name>.onnx` (paths need not
/// exist yet — packers check at bake time).
pub fn specs_from_root(root: &Path, names: &[&str]) -> Vec<OnnxGraphSpec> {
    names
        .iter()
        .map(|name| {
            let onnx = root.join("onnx").join(format!("{name}.onnx"));
            let onnx = if onnx.is_file() {
                onnx
            } else {
                root.join(format!("{name}.onnx"))
            };
            OnnxGraphSpec {
                name: (*name).to_string(),
                onnx_path: onnx,
            }
        })
        .collect()
}
