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

//! Export a Pocket TTS model as an `.rlxp` package rlx runs alone: text and
//! a released voice in, 24 kHz speech out, through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its graphs are [`crate::PocketTts`]'s: `step` (one FlowLM row — the
//! backbone, the EOS head, the sampler head — symbolic in the caches' rows,
//! each layer's new key and value written into its cache at `pos`, its RoPE
//! rows read from a table there) and `decode` (an 8-frame Mimi decoder
//! block, its streaming states in and out, its RoPE rows read at the
//! block's `positions`). The voices are the runner's key / value states,
//! one file each; the text embeddings, `input_linear` and the latent
//! statistics are files the program reads (it computes them as the runner
//! does, on the host). The program (`program/`) prepares and chunks the
//! text (the crate's `text.rs`, the host's tokenizer), runs each chunk's
//! tokens and frames through `step` from the voice's cache (kept in slots),
//! draws the runner's noise (`StdRng`, `Normal`), stops after EOS, and
//! decodes each chunk with a fresh Mimi state.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::graph::symbolic_in;
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use serde_json::json;

use crate::checkpoint::Checkpoint;
use crate::engine::{DEFAULT_DECODER_BLOCK, ModelFiles, PocketTts};
use crate::graph::flow_lm::{
    FlowLmDims, OUT_EOS, OUT_HIDDEN, OUT_KV, OUT_LATENT, build_step, past_k_name, past_v_name,
};
use crate::graph::mimi::build_decoder_block;
use crate::graph::rope_tables;
use crate::spec::ModelSpec;

/// The caches' rows.
const CAP: u32 = 0;
/// RoPE rows the tables hold: the backbone's positions and a chunk's Mimi
/// positions.
const ROPE_ROWS: usize = 8192;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn push(g: &mut Graph, op: Op, inputs: Vec<NodeId>, shape: Shape, name: Option<String>) -> NodeId {
    let id = g.add_node(op, inputs, shape);
    g.node_mut(id).name = name;
    id
}

/// `graph` with its `rope_cos` / `rope_sin` inputs the rows of `table`'s
/// cos / sin params at the input `at` (`f32[n]`), reshaped as the inputs
/// were.
fn rope_from_tables(graph: &Graph, table: &str, at: &str, n: usize) -> Result<Graph> {
    let mut index = None;
    let g = rlx_ir::stateful::rebuild(graph, |out, _, node, _| {
        let Op::Input { name } = &node.op else {
            return Ok(None);
        };
        let which = match name.as_str() {
            "rope_cos" => "cos",
            "rope_sin" => "sin",
            _ => return Ok(None),
        };
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let half = *dims.last().expect("a rank");
        let i = *index.get_or_insert_with(|| {
            push(
                out,
                Op::Input { name: at.into() },
                vec![],
                f32s(&[n]),
                Some(at.into()),
            )
        });
        let t = push(
            out,
            Op::Param {
                name: format!("{table}.{which}"),
            },
            vec![],
            f32s(&[ROPE_ROWS, half]),
            Some(format!("{table}.{which}")),
        );
        let rows = push(
            out,
            Op::Gather { axis: 0 },
            vec![t, i],
            f32s(&[n, half]),
            None,
        );
        Ok(Some(push(
            out,
            Op::Reshape {
                new_shape: dims.iter().map(|&d| d as i64).collect(),
            },
            vec![rows],
            node.shape.clone(),
            None,
        )))
    })
    .map_err(anyhow::Error::msg)?;
    Ok(g)
}

/// The FlowLM step at `cap` rows: [`build_step`]'s graph with its RoPE rows
/// read at `pos` and each layer's new key and value written into its cache
/// there (`past_k{l}.next`, `past_v{l}.next`), after `latent`, `eos`,
/// `hidden`.
fn step(
    ck: &Checkpoint,
    spec: &ModelSpec,
    cap: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let dims = FlowLmDims::new(spec);
    let b = build_step(ck, spec, cap, 1)?;
    let mut g = rope_from_tables(&b.g, "flow.rope", "pos", 1)?;
    g.name = "step".into();
    let pos = g.input_id("pos").context("no pos")?;
    let outs = g.outputs.clone();
    let mut next = Vec::new();
    for (i, n) in ["latent", "eos", "hidden"].iter().enumerate() {
        let o = [OUT_LATENT, OUT_EOS, OUT_HIDDEN][i];
        g.node_mut(outs[o]).name = Some((*n).into());
        next.push(outs[o]);
    }
    for l in 0..dims.num_layers {
        for (j, name) in [past_k_name(l), past_v_name(l)].into_iter().enumerate() {
            let cache = g.input_id(&name).with_context(|| format!("no {name}"))?;
            let shape = g.node(cache).shape.clone();
            let row = outs[OUT_KV + 2 * l + j];
            next.push(push(
                &mut g,
                Op::DynamicUpdateSlice { axis: 1 },
                vec![cache, row, pos],
                shape,
                Some(format!("{name}.next")),
            ));
        }
    }
    g.set_outputs(next);
    Ok((g, b.params))
}

/// The Mimi decoder block of `frames` frames: [`build_decoder_block`]'s
/// graph with its RoPE rows read at `positions`, its states' next values
/// named `state{i}.next`.
fn decode(
    ck: &Checkpoint,
    spec: &ModelSpec,
    frames: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, Vec<Vec<usize>>)> {
    let block = build_decoder_block(ck, spec, frames)?;
    let t = frames * spec.mimi.steps_per_frame();
    let mut g = rope_from_tables(&block.builder.g, "mimi.rope", "positions", t)?;
    g.name = "decode".into();
    let outs = g.outputs.clone();
    g.node_mut(outs[0]).name = Some("audio".into());
    let mut sizes = Vec::new();
    for (i, &o) in outs.iter().enumerate().skip(1) {
        g.node_mut(o).name = Some(format!("state{}.next", i - 1));
        let id = g
            .input_id(&format!("state{}", i - 1))
            .with_context(|| format!("no state{}", i - 1))?;
        sizes.push(
            g.node(id)
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect::<Vec<usize>>(),
        );
    }
    Ok((g, block.builder.params, sizes))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Export the Pocket TTS language directory `dir` (`model.safetensors`,
/// `tokenizer.json`, `embeddings/`; its name the bundled config's, e.g.
/// `english_2026-10`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let language = rlx_ir::env::var("RLX_POCKET_TTS_LANGUAGE").unwrap_or_else(|| {
        dir.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let spec = ModelSpec::builtin(&language)?;
    let files = ModelFiles::resolve(dir, &language, &spec)?;
    let tokenizer = dir.join("tokenizer.json");
    if !tokenizer.is_file() {
        bail!("{} has no tokenizer.json", dir.display());
    }
    let ck = Checkpoint::open(&files.weights)?;
    let dims = FlowLmDims::new(&spec);
    let mut pkg = Package::new(
        &format!("pocket-tts-{language}"),
        &opts.version,
        "pocket-tts",
        "text-to-speech",
        "rlx-pocket-tts",
    );
    pkg.model["license"] = json!("cc-by-4.0");

    let mut held: HashMap<String, Vec<f32>> = HashMap::new();
    let g = symbolic_in(CAP, (37, 53, 20), |c| Ok(step(&ck, &spec, c)?.0))?;
    pkg.function(g);
    held.extend(step(&ck, &spec, 20)?.1);
    let frames = DEFAULT_DECODER_BLOCK;
    let (dg, dp, state_sizes) = decode(&ck, &spec, frames)?;
    pkg.function(dg);
    // The two graphs name their params `p0`, `p1`, … each from 0: keep
    // them apart.
    let dp: HashMap<String, Vec<f32>> = dp
        .into_iter()
        .map(|(k, v)| (format!("mimi.{k}"), v))
        .collect();
    for g in pkg.graphs.iter_mut().filter(|g| g.name == "decode") {
        for node in g.nodes_mut() {
            if let Op::Param { name } = &mut node.op
                && !name.starts_with("mimi.rope")
            {
                *name = format!("mimi.{name}");
            }
        }
    }
    held.extend(dp);
    let mt = spec.mimi.decoder_transformer();
    for (table, hd, period) in [
        ("flow.rope", dims.head_dim, dims.max_period),
        ("mimi.rope", mt.d_model / mt.num_heads, mt.max_period),
    ] {
        let (cos, sin) = rope_tables(0..ROPE_ROWS, hd, period);
        held.insert(format!("{table}.cos"), cos);
        held.insert(format!("{table}.sin"), sin);
    }
    pkg.dims = vec![("cap".into(), CAP)];
    let pdims = pkg.param_dims();
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = pdims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }
    drop(held);

    // What the program computes on the host: the text embeddings (and their
    // projection), `input_linear`, BOS, the latent statistics.
    let embed = ck.tensor("flow_lm.conditioner.embed.weight")?;
    pkg.sidecar(
        "embed.f32",
        "application/octet-stream",
        f32_bytes(&embed.data),
    );
    let proj = ck.opt("flow_lm.conditioner.proj.weight")?;
    if let Some(p) = &proj {
        pkg.sidecar(
            "embed_proj.f32",
            "application/octet-stream",
            f32_bytes(&p.data),
        );
    }
    let input_linear = ck.tensor("flow_lm.input_linear.weight")?;
    pkg.sidecar(
        "input_linear.f32",
        "application/octet-stream",
        f32_bytes(&input_linear.data),
    );
    let ldim = dims.latent_dim;
    let bos = ck.vec("flow_lm.bos_emb", ldim)?;
    let mean = ck.vec("flow_lm.emb_mean", ldim)?;
    let std = ck.vec("flow_lm.emb_std", ldim)?;

    // The released voices, as the runner loads them: each layer's keys,
    // then its values.
    let mut runner = PocketTts::open_with(spec.clone(), files.clone(), rlx_runtime::Device::Cpu)?;
    let mut voices = serde_json::Map::new();
    if let Some(vdir) = &files.voices {
        let mut entries: Vec<_> = std::fs::read_dir(vdir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        entries.sort();
        for p in entries {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            let state = runner.load_voice(p.to_str().context("voice path")?)?;
            let mut bytes = Vec::new();
            for l in 0..state.num_layers() {
                bytes.extend(f32_bytes(&state.k[l]));
                bytes.extend(f32_bytes(&state.v[l]));
            }
            pkg.sidecar(
                &format!("voices/{name}.f32"),
                "application/octet-stream",
                bytes,
            );
            voices.insert(name, json!(state.offset));
        }
    }
    drop(runner);
    pkg.sidecar(
        "tokenizer.json",
        "application/json",
        std::fs::read(&tokenizer)?,
    );

    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )
    .context("build Pocket TTS' program")?;
    let lut = &spec.flow_lm.lookup_table;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "files": "/package",
            "tokenizer": "sidecar:tokenizer.json",
            "voices": voices,
            "voice": voices.keys().next().cloned().unwrap_or_default(),
            "text": {
                "pad_with_spaces_for_short_inputs": spec.pad_with_spaces_for_short_inputs,
                "remove_semicolons": spec.remove_semicolons,
                "append_terminal_punctuation": spec.append_terminal_punctuation,
                "capitalize_first_letter": spec.capitalize_first_letter,
                "replace_characters": spec.replace_characters,
            },
            "layers": dims.num_layers,
            "kv_heads": dims.num_kv_heads,
            "head_dim": dims.head_dim,
            "d_model": dims.d_model,
            "latent_dim": ldim,
            "embed_width": lut.rank.unwrap_or(lut.dim),
            "embed_proj": proj.is_some(),
            "bos": bos,
            "emb_mean": mean,
            "emb_std": std,
            "kv_bucket": 64,
            "min_frames_before_eos": 6,
            "tokens_per_second": 3.0,
            "gen_seconds_padding": 2.0,
            "max_tokens_per_chunk": crate::text::MAX_TOKENS_PER_CHUNK,
            "temperature": spec.default_temperature,
            "eos_threshold": spec.default_eos_threshold,
            "frames_after_eos": spec.model_recommended_frames_after_eos,
            "seed": 0,
            "frame_rate": spec.mimi.frame_rate,
            "decoder_frames": frames,
            "steps_per_frame": spec.mimi.steps_per_frame(),
            "frame_size": spec.mimi.frame_size(),
            "mimi_context": mt.context,
            "states": state_sizes,
            "sample_rate": spec.mimi.sample_rate,
        }),
    );
    pkg.metadata("sample_rate", json!(spec.mimi.sample_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("pocket-tts", "text-to-speech", export_rlxp));
}
