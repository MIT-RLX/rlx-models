// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a SAM v1 checkpoint as an `.rlxp` package rlx runs alone: three
//! graphs — `encode` (the patch embedding, then the ViT encoder and neck),
//! `decode` (the two-way transformer, the upscaling, the hypernetwork MLPs,
//! their product with the upscaled features and the IoU head, symbolic in
//! the prompt tokens) and `mask_embed` (a mask prompt's downscaling) — and
//! a program answering `run` with what [`crate::Sam::forward`] does around
//! them: the image's resize ([`crate::image`]) and the points' and boxes'
//! embeddings ([`crate::points`]), built from this crate's own files.
//!
//! `Model.run({image: u8 [h, w, 3], points: f32 [N, 2], labels: f32 [N]})`
//! (or `boxes: f32 [M, 4]`, and `mask: f32 [256, 256]`; points in the
//! 1024-pixel input's coordinates, as `Sam::forward` takes them) returns
//! `masks` `[k, 256, 256]` logits, `iou` `[k]` and `size` (the resized
//! image's height and width). Option `multimask` (default on): the three
//! masks, else the one.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::export::{Package, program};
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut};
use rlx_ir::{DType, Graph, Op, Shape};
use serde_json::json;

use crate::config::{SAM_EMBED_HW, SAM_IMG_SIZE, SAM_PATCH_SIZE, SamConfig};
use crate::mask_decoder::{HypernetMlp, MaskDecoderWeights, extract_mask_decoder_weights};
use crate::prompt_encoder::{PromptEncoderWeights, extract_prompt_encoder_weights};
use crate::sam::SAM_MASK_IN_CHANS;

/// The prompt tokens: the IoU and mask tokens, then the points' and boxes'.
const TOKENS: u32 = 0;

type Params = HashMap<String, Vec<f32>>;

fn from_hir(hir: HirModule) -> Result<Graph> {
    Graph::from_hir(hir).map_err(|e| anyhow!("{e}"))
}

/// `graph`'s params renamed `prefix` + name, in it and in `params`.
fn namespaced(mut graph: Graph, params: Params, prefix: &str) -> (Graph, Params) {
    for n in graph.nodes_mut() {
        if let Op::Param { name } = &mut n.op {
            *name = format!("{prefix}{name}");
        }
    }
    let params = params
        .into_iter()
        .map(|(k, v)| (format!("{prefix}{k}"), v))
        .collect();
    (graph, params)
}

fn splice(
    into: &mut Graph,
    src: &Graph,
    bind: &[(&str, rlx_ir::NodeId)],
) -> Result<Vec<rlx_ir::NodeId>> {
    let s = rlx_ir::compose::Splice::new().bind_all(bind.iter().map(|(n, id)| (*n, *id)));
    let out = into
        .splice(src, &s)
        .map_err(|e| anyhow!("joining {}: {e:?}", src.name))?;
    Ok(out.outputs)
}

fn name(g: &mut Graph, id: rlx_ir::NodeId, n: &str) {
    g.node_mut(id).name = Some(n.to_string());
}

/// `encode`: `image` `[1, 3, 1024, 1024]` (normalized, padded) → the
/// 16-pixel patches' embedding and positions → the encoder →
/// `embeddings` `[1, E, 64, 64]`.
fn encode_graph(
    cfg: &SamConfig,
    wm: &mut rlx_core::weight_map::WeightMap,
) -> Result<(Graph, Params)> {
    let (encoder, mut params, pre) =
        crate::image_encoder::build_sam_encoder_graph(&cfg.encoder, wm)?;
    let e = pre.embed_dim;
    let hw = pre.hw;
    let p = SAM_PATCH_SIZE;
    let pd = 3 * p * p;
    let f = DType::F32;
    let mut hir = HirModule::new("encode");
    let mut g = HirMut::new(&mut hir);
    let image = g.input("image", Shape::new(&[1, 3, SAM_IMG_SIZE, SAM_IMG_SIZE], f));
    let x = g.reshape_(image, vec![3, hw as i64, p as i64, hw as i64, p as i64]);
    let x = g.transpose_(x, vec![1, 3, 0, 2, 4]);
    let x = g.reshape_(x, vec![(hw * hw) as i64, pd as i64]);
    let w = g.param("patch.weight", Shape::new(&[pd, e], f));
    let b = g.param("patch.bias", Shape::new(&[e], f));
    let x = g.mm(x, w);
    let mut x = g.add(x, b);
    params.insert("patch.weight".into(), pre.patch_proj_w.clone());
    params.insert("patch.bias".into(), pre.patch_proj_b.clone());
    if let Some(pos) = &pre.pos_embed {
        let pos_id = g.param("patch.pos", Shape::new(&[hw * hw, e], f));
        x = g.add(x, pos_id);
        params.insert("patch.pos".into(), pos.clone());
    }
    let x = g.reshape_(x, vec![1, (hw * hw) as i64, e as i64]);
    hir.set_outputs(vec![x]);
    let glue = from_hir(hir)?;

    let mut out = Graph::new("encode");
    let tokens = splice(&mut out, &glue, &[])?;
    let emb = splice(&mut out, &encoder, &[("hidden", tokens[0])])?;
    name(&mut out, emb[0], "embeddings");
    out.set_outputs(vec![emb[0]]);
    Ok((out, params))
}

fn mlp_layers(m: &HypernetMlp) -> Vec<rlx_sam_ir::mlp_relu_ir::MlpLayerSpec> {
    m.layers
        .iter()
        .map(|l| rlx_sam_ir::mlp_relu_ir::MlpLayerSpec {
            w: l.w.clone(),
            b: l.b.clone(),
            in_d: l.in_d,
            out_d: l.out_d,
        })
        .collect()
}

/// `decode` at `q` prompt tokens: `embeddings` and `dense` `[1, E, 64, 64]`,
/// `tokens` `[1, q, E]` → `masks` `[nm, 256, 256]` and `iou` `[nm]`, as
/// [`crate::mask_decoder_forward`] computes them.
fn decode_graph(dec: &MaskDecoderWeights, image_pe: &[f32], q: usize) -> Result<(Graph, Params)> {
    let e = dec.transformer_dim;
    let hw = SAM_EMBED_HW;
    let k = hw * hw;
    let nm = dec.num_mask_tokens;
    let q8 = e / 8;
    let side = 4 * hw;
    let f = DType::F32;
    let mut params = Params::new();
    let mut out = Graph::new("decode");

    // The image side: embeddings plus the dense prompt, as a sequence.
    let mut hir = HirModule::new("decode_in");
    let mut g = HirMut::new(&mut hir);
    let emb = g.input("embeddings", Shape::new(&[1, e, hw, hw], f));
    let dense = g.input("dense", Shape::new(&[1, e, hw, hw], f));
    let tokens = g.input("tokens", Shape::new(&[1, q, e], f));
    let src = g.add(emb, dense);
    let src = g.reshape_(src, vec![1, e as i64, k as i64]);
    let seq = g.transpose_(src, vec![0, 2, 1]);
    hir.set_outputs(vec![seq, tokens]);
    let glue = from_hir(hir)?;
    let ins = splice(&mut out, &glue, &[])?;

    // The two-way transformer, its image PE a weight.
    let spec = crate::transformer_ir::transformer_spec(&dec.transformer);
    let (mut tw, p) =
        rlx_sam_ir::twoway_transformer_ir::build_transformer_graph(&spec, q, k, false)?;
    for n in tw.nodes_mut() {
        if matches!(&n.op, Op::Input { name } if name == "image_pe") {
            n.op = Op::Param {
                name: "image_pe".into(),
            };
        }
    }
    let (tw, p) = namespaced(tw, p, "tw.");
    params.extend(p);
    let pe = rlx_sam_ir::twoway_transformer_ir::TwoWayTransformerCompiled::nchw_to_seq(
        image_pe, e, hw, hw,
    );
    params.insert("tw.image_pe".into(), pe);
    let hs = splice(&mut out, &tw, &[("tokens", ins[1]), ("image_seq", ins[0])])?;

    // The IoU token, the mask tokens, the image side back to NCHW.
    let mut hir = HirModule::new("decode_split");
    let mut g = HirMut::new(&mut hir);
    let queries = g.input("queries", Shape::new(&[1, q, e], f));
    let keys = g.input("keys", Shape::new(&[1, k, e], f));
    let mut outs = Vec::new();
    for i in 0..=nm {
        let t = g.narrow_(queries, 1, i, 1);
        outs.push(g.reshape_(t, vec![1, e as i64]));
    }
    let src = g.transpose_(keys, vec![0, 2, 1]);
    outs.push(g.reshape_(src, vec![1, e as i64, hw as i64, hw as i64]));
    hir.set_outputs(outs);
    let glue = from_hir(hir)?;
    let split = splice(&mut out, &glue, &[("queries", hs[0]), ("keys", hs[1])])?;

    let (up, p) = crate::upscale_ir::build_mask_upscale_graph(dec)?;
    let (up, p) = namespaced(up, p, "up.");
    params.extend(p);
    let up2 = splice(&mut out, &up, &[("src", split[nm + 1])])?;

    let mut hyper = Vec::new();
    for (i, m) in dec.hyper_mlps.iter().enumerate() {
        let (g, p, _, _) = rlx_sam_ir::mlp_relu_ir::build_mlp_graph(&mlp_layers(m), 1)?;
        let (g, p) = namespaced(g, p, &format!("hyper{i}."));
        params.extend(p);
        hyper.push(splice(&mut out, &g, &[("x", split[1 + i])])?[0]);
    }
    let (g, p, _, _) = rlx_sam_ir::mlp_relu_ir::build_mlp_graph(&mlp_layers(&dec.iou_head), 1)?;
    let (g, p) = namespaced(g, p, "iou.");
    params.extend(p);
    let iou = splice(&mut out, &g, &[("x", split[0])])?[0];

    // Each mask: its hypernetwork's row times the upscaled features.
    let mut hir = HirModule::new("decode_out");
    let mut g = HirMut::new(&mut hir);
    let rows: Vec<_> = (0..nm)
        .map(|i| g.input(format!("h{i}"), Shape::new(&[1, q8], f)))
        .collect();
    let feat = g.input("up", Shape::new(&[1, q8, side, side], f));
    let scores = g.input("scores", Shape::new(&[1, nm], f));
    let h = g.concat_(rows, 0);
    let feat = g.reshape_(feat, vec![q8 as i64, (side * side) as i64]);
    let masks = g.mm(h, feat);
    let masks = g.reshape_(masks, vec![nm as i64, side as i64, side as i64]);
    let scores = g.reshape_(scores, vec![nm as i64]);
    hir.set_outputs(vec![masks, scores]);
    let glue = from_hir(hir)?;
    let mut bind: Vec<(String, rlx_ir::NodeId)> = hyper
        .iter()
        .enumerate()
        .map(|(i, id)| (format!("h{i}"), *id))
        .collect();
    bind.push(("up".into(), up2[0]));
    bind.push(("scores".into(), iou));
    let bind: Vec<(&str, rlx_ir::NodeId)> = bind.iter().map(|(n, id)| (n.as_str(), *id)).collect();
    let res = splice(&mut out, &glue, &bind)?;
    name(&mut out, res[0], "masks");
    name(&mut out, res[1], "iou");
    out.set_outputs(res);
    Ok((out, params))
}

/// `decode` symbolic in [`TOKENS`]: built at sizes no other dim has,
/// generalized, and checked against a build at another.
fn symbolic_decode(dec: &MaskDecoderWeights, image_pe: &[f32]) -> Result<(Graph, Params)> {
    let sizes = [7usize, 11, 13, 19];
    let built: Vec<(Graph, Params)> = sizes
        .iter()
        .map(|&q| decode_graph(dec, image_pe, q))
        .collect::<Result<_>>()?;
    let refs: Vec<&Graph> = built.iter().map(|(g, _)| g).collect();
    let mut g = rlx_ir::dynamic::generalize_many(&refs, &[(TOKENS, sizes.to_vec())], None)
        .map_err(|e| anyhow!("the decoder is not symbolic in its prompt tokens: {e}"))?;
    let check = 23;
    let (want, _) = decode_graph(dec, image_pe, check)?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(TOKENS, check);
    let (got, want) = (
        format!("{}", rlx_ir::dynamic::specialize(&g, &bind)),
        format!("{want}"),
    );
    if got != want {
        let (a, b) = got
            .lines()
            .zip(want.lines())
            .find(|(a, b)| a != b)
            .unwrap_or_default();
        bail!(
            "the decoder generalized over four sizes is not the one built at another: `{a}` where it builds `{b}`"
        );
    }
    g.name = "decode".into();
    let (_, params) = built.into_iter().next().expect("four builds");
    Ok((g, params))
}

/// `mask_embed`: a mask prompt `[1, 1, 256, 256]` → `dense` `[1, E, 64, 64]`.
fn mask_graph(pe: &PromptEncoderWeights) -> Result<(Graph, Params)> {
    let w = rlx_sam_ir::mask_prompt_ir::MaskDownscaleWeights {
        mask_in_chans: pe.mask_in_chans,
        embed_dim: pe.embed_dim,
        mask_conv1_w: &pe.mask_conv1_w,
        mask_conv1_b: &pe.mask_conv1_b,
        mask_ln1_g: &pe.mask_ln1_g,
        mask_ln1_b: &pe.mask_ln1_b,
        mask_conv2_w: &pe.mask_conv2_w,
        mask_conv2_b: &pe.mask_conv2_b,
        mask_ln2_g: &pe.mask_ln2_g,
        mask_ln2_b: &pe.mask_ln2_b,
        mask_conv3_w: &pe.mask_conv3_w,
        mask_conv3_b: &pe.mask_conv3_b,
    };
    let side = 4 * SAM_EMBED_HW;
    let (g, p) = rlx_sam_ir::mask_prompt_ir::build_prompt_mask_graph(w, side, side)?;
    let (mut g, p) = namespaced(g, p, "mask.");
    g.name = "mask_embed".into();
    let out = g.outputs[0];
    name(&mut g, out, "dense");
    Ok((g, p))
}

/// The SAM variant whose encoder `wm` holds, by its patch embedding's width.
fn variant(wm: &rlx_core::weight_map::WeightMap) -> Result<SamConfig> {
    let (_, shape) = wm
        .get("image_encoder.patch_embed.proj.weight")
        .context("no image_encoder.patch_embed.proj.weight: not a SAM v1 checkpoint")?;
    Ok(match shape.first() {
        Some(768) => SamConfig::vit_b(),
        Some(1024) => SamConfig::vit_l(),
        Some(1280) => SamConfig::vit_h(),
        other => bail!("a SAM v1 encoder {other:?} wide is none of ViT-B/L/H"),
    })
}

/// Export the SAM v1 checkpoint at `path` (safetensors or GGUF, as
/// [`crate::Sam::from_safetensors_on`] takes it) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let mut wm = rlx_core::load_weight_map(path, rlx_core::SAM_GGUF_ARCHES)?;
    let cfg = variant(&wm)?;
    let (encode, mut params) = encode_graph(&cfg, &mut wm)?;
    let pe = extract_prompt_encoder_weights(&mut wm, cfg.encoder.out_chans, SAM_MASK_IN_CHANS)?;
    let dec = extract_mask_decoder_weights(
        &mut wm,
        cfg.decoder.transformer_dim,
        cfg.decoder.num_mask_tokens,
        cfg.decoder.iou_head_depth,
        cfg.decoder.iou_head_hidden_dim,
        cfg.decoder.transformer_depth,
        cfg.decoder.transformer_num_heads,
        cfg.decoder.transformer_mlp_dim,
    )?;
    let hw = SAM_EMBED_HW;
    let image_pe = crate::points::image_pe(&pe.points(), hw, hw);
    let (decode, p) = symbolic_decode(&dec, &image_pe)?;
    params.extend(p);
    let (mask, p) = mask_graph(&pe)?;
    params.extend(p);
    // The dense prompt without a mask: `no_mask_embed` over the grid.
    let e = pe.embed_dim;
    let no_mask: Vec<f32> = (0..e)
        .flat_map(|c| std::iter::repeat_n(pe.no_mask_embed[c], hw * hw))
        .collect();

    let mut pkg = Package::new("sam", &opts.version, "sam", "mask-generation", "rlx-sam");
    pkg.function(encode);
    pkg.function(decode);
    pkg.function(mask);
    pkg.dims = vec![("tokens".into(), TOKENS)];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), d, &params[n]);
    }
    pkg.hold_f32("no_mask_dense", vec![1, e, hw, hw], &no_mask);

    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    let mut tokens = dec.iou_token.clone();
    tokens.extend_from_slice(&dec.mask_tokens);
    let config = json!({
        "embed_dim": e,
        "num_mask_tokens": dec.num_mask_tokens,
        "pe_gaussian": pe.pe_gaussian,
        "not_a_point_embed": pe.not_a_point_embed,
        "point_embeddings": pe.point_embeddings,
        "output_tokens": tokens,
    });
    program::entry(
        &mut pkg,
        "segment",
        module,
        &["run"],
        "mask-generation",
        config,
    );
    pkg.metadata("image_size", json!(SAM_IMG_SIZE));
    pkg.metadata("encoder_width", json!(cfg.encoder.embed_dim));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family sam`): a SAM checkpoint names no `model_type`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("sam", "mask-generation", export_rlxp));
}
