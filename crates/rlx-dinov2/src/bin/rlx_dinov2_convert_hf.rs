// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// HuggingFace `transformers` DINOv2 → Meta-original key layout that
// `rlx-dinov2` expects.
//
// Renames + concatenates Q/K/V along the output dim per layer. Mirrors
// `weights/dinov2/convert_hf_to_meta.py` line-for-line; this Rust
// binary lets the conversion ship with the workspace and run in CI
// without a Python dependency.
//
// Usage:
//     rlx-dinov2-convert-hf <src.safetensors> [<dst.safetensors>]
//
// Default `dst` is `<src-stem>.meta.safetensors` next to the source.

use anyhow::{Context, Result, anyhow, bail};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src: PathBuf = args
        .next()
        .ok_or_else(|| anyhow!("usage: rlx-dinov2-convert-hf <src.safetensors> [<dst>]"))?
        .into();
    let dst: PathBuf = match args.next() {
        Some(p) => p.into(),
        None => {
            let stem = src
                .file_stem()
                .ok_or_else(|| anyhow!("src has no stem"))?
                .to_string_lossy()
                .to_string();
            src.with_file_name(format!("{stem}.meta.safetensors"))
        }
    };
    println!("reading {}", src.display());
    let buf = fs::read(&src).with_context(|| format!("reading {}", src.display()))?;
    let st = SafeTensors::deserialize(&buf).context("parsing source safetensors")?;
    println!("  {} tensors", st.tensors().len());

    // Pre-collect (name → TensorView) for O(1) lookups.
    let mut by_name: HashMap<String, TensorView> = HashMap::new();
    for (name, view) in st.tensors() {
        by_name.insert(name, view);
    }

    // (name, dtype, shape, raw bytes) — preserve insertion order via Vec
    // so the output safetensors header is deterministic.
    let mut out: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    let take = |name: &str,
                by_name: &HashMap<String, TensorView>|
     -> Result<(Dtype, Vec<usize>, Vec<u8>)> {
        let v = by_name
            .get(name)
            .ok_or_else(|| anyhow!("missing source tensor `{name}`"))?;
        Ok((v.dtype(), v.shape().to_vec(), v.data().to_vec()))
    };
    let rename = |dst_name: &str,
                  src_name: &str,
                  out: &mut Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
                  by: &HashMap<String, TensorView>|
     -> Result<()> {
        let (d, s, b) = take(src_name, by)?;
        out.push((dst_name.to_string(), d, s, b));
        Ok(())
    };

    rename("cls_token", "embeddings.cls_token", &mut out, &by_name)?;
    rename(
        "pos_embed",
        "embeddings.position_embeddings",
        &mut out,
        &by_name,
    )?;
    rename(
        "patch_embed.proj.weight",
        "embeddings.patch_embeddings.projection.weight",
        &mut out,
        &by_name,
    )?;
    rename(
        "patch_embed.proj.bias",
        "embeddings.patch_embeddings.projection.bias",
        &mut out,
        &by_name,
    )?;

    // Enumerate transformer layers from the key set.
    let mut layer_ids: Vec<u32> = by_name
        .keys()
        .filter_map(|k| {
            k.strip_prefix("encoder.layer.").and_then(|rest| {
                let dot = rest.find('.')?;
                rest[..dot].parse::<u32>().ok()
            })
        })
        .collect::<std::collections::BTreeSet<u32>>()
        .into_iter()
        .collect();
    layer_ids.sort();
    println!("  {} transformer layers", layer_ids.len());

    for i in layer_ids {
        let p = format!("encoder.layer.{i}");
        let q = format!("blocks.{i}");
        rename(
            &format!("{q}.norm1.weight"),
            &format!("{p}.norm1.weight"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.norm1.bias"),
            &format!("{p}.norm1.bias"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.norm2.weight"),
            &format!("{p}.norm2.weight"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.norm2.bias"),
            &format!("{p}.norm2.bias"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.ls1.gamma"),
            &format!("{p}.layer_scale1.lambda1"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.ls2.gamma"),
            &format!("{p}.layer_scale2.lambda1"),
            &mut out,
            &by_name,
        )?;

        // Q/K/V concatenated along output dim (axis 0 for the weight,
        // single axis for the bias).
        let (qw_d, qw_s, qw_b) = take(&format!("{p}.attention.attention.query.weight"), &by_name)?;
        let (kw_d, kw_s, kw_b) = take(&format!("{p}.attention.attention.key.weight"), &by_name)?;
        let (vw_d, vw_s, vw_b) = take(&format!("{p}.attention.attention.value.weight"), &by_name)?;
        ensure_same_dtype(&[qw_d, kw_d, vw_d])?;
        ensure_same_inner(&qw_s, &kw_s, &vw_s)?;
        let merged_shape = vec![qw_s[0] + kw_s[0] + vw_s[0], qw_s[1]];
        let mut merged = Vec::with_capacity(qw_b.len() + kw_b.len() + vw_b.len());
        merged.extend_from_slice(&qw_b);
        merged.extend_from_slice(&kw_b);
        merged.extend_from_slice(&vw_b);
        out.push((format!("{q}.attn.qkv.weight"), qw_d, merged_shape, merged));

        let (qb_d, qb_s, qb_b) = take(&format!("{p}.attention.attention.query.bias"), &by_name)?;
        let (kb_d, kb_s, kb_b) = take(&format!("{p}.attention.attention.key.bias"), &by_name)?;
        let (vb_d, vb_s, vb_b) = take(&format!("{p}.attention.attention.value.bias"), &by_name)?;
        ensure_same_dtype(&[qb_d, kb_d, vb_d])?;
        let merged_bias_shape = vec![qb_s[0] + kb_s[0] + vb_s[0]];
        let mut merged_bias = Vec::with_capacity(qb_b.len() + kb_b.len() + vb_b.len());
        merged_bias.extend_from_slice(&qb_b);
        merged_bias.extend_from_slice(&kb_b);
        merged_bias.extend_from_slice(&vb_b);
        out.push((
            format!("{q}.attn.qkv.bias"),
            qb_d,
            merged_bias_shape,
            merged_bias,
        ));

        rename(
            &format!("{q}.attn.proj.weight"),
            &format!("{p}.attention.output.dense.weight"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.attn.proj.bias"),
            &format!("{p}.attention.output.dense.bias"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.mlp.fc1.weight"),
            &format!("{p}.mlp.fc1.weight"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.mlp.fc1.bias"),
            &format!("{p}.mlp.fc1.bias"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.mlp.fc2.weight"),
            &format!("{p}.mlp.fc2.weight"),
            &mut out,
            &by_name,
        )?;
        rename(
            &format!("{q}.mlp.fc2.bias"),
            &format!("{p}.mlp.fc2.bias"),
            &mut out,
            &by_name,
        )?;
    }

    rename("norm.weight", "layernorm.weight", &mut out, &by_name)?;
    rename("norm.bias", "layernorm.bias", &mut out, &by_name)?;

    // Serialize. `safetensors::serialize` takes a (name → TensorView) map.
    // Build owned data first, then borrow into views in a second pass.
    let owned: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> =
        out.into_iter().map(|(n, d, s, b)| (n, (d, s, b))).collect();
    let views: Vec<(String, TensorView)> = owned
        .iter()
        .map(|(n, (d, s, b))| {
            let view = TensorView::new(*d, s.clone(), b).expect("valid TensorView");
            (n.clone(), view)
        })
        .collect();
    println!("writing {} ({} tensors)", dst.display(), views.len());
    let serialized = safetensors::serialize(views, None).context("serializing output")?;
    fs::write(&dst, &serialized).with_context(|| format!("writing {}", dst.display()))?;
    let sz = fs::metadata(&dst)?.len() as f64 / (1u64 << 20) as f64;
    println!("size: {sz:.1} MB");
    Ok(())
}

fn ensure_same_dtype(dts: &[Dtype]) -> Result<()> {
    let first = dts.first().copied().unwrap_or(Dtype::F32);
    if !dts.iter().all(|d| *d == first) {
        bail!("dtype mismatch among Q/K/V tensors: {dts:?}");
    }
    Ok(())
}

fn ensure_same_inner(qs: &[usize], ks: &[usize], vs: &[usize]) -> Result<()> {
    if qs.len() != 2 || ks.len() != 2 || vs.len() != 2 {
        bail!("expected 2-D weight tensors for Q/K/V, got {qs:?}, {ks:?}, {vs:?}");
    }
    if qs[1] != ks[1] || qs[1] != vs[1] {
        bail!("Q/K/V input dim mismatch: {qs:?}, {ks:?}, {vs:?}");
    }
    Ok(())
}
