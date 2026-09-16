//! Is a trunk's residual stream identity-dominated? A one-second pre-check.
//!
//! The Jacobian lens rests on `J_l ≈ I + (what the blocks did)` — the transport
//! being a *correction* to a vector that is already mostly right. When that
//! holds, `J` comes out high-rank with growing diagonal energy and the readout
//! works. When it does not, `J` collapses toward a rank-one projector with
//! enormous gain and the readout is meaningless.
//!
//! Which case you are in is visible from `‖h‖` alone, without fitting anything.
//! Measured on the same photo:
//!
//! ```text
//! Qwen3.5   residual grows modestly     J: diag energy 0.16 -> 0.78, eff rank 73 -> 504
//! DINOv3    residual grows 72x          J: diag energy 0.001,        eff rank 1.1
//! ```
//!
//! A fit costs minutes; this costs one forward. Run it first.
//!
//! ```bash
//! cargo run -p rlx-jlens --features siglip2,metal --release --example stream_growth -- \
//!     --device metal --image crates/rlx-locateanything/fixtures/sample.jpg
//! ```

#![cfg(feature = "siglip2")]

use anyhow::{Context, Result, bail};
use rlx_jlens::taps::{layer_exit_taps, residual_stream_from};
use rlx_runtime::{Device, Session};

fn main() -> Result<()> {
    let mut dir = "weights/siglip2-base-224".to_string();
    let mut device = Device::Cpu;
    let mut image: Option<String> = None;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .with_context(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--weights" => {
                dir = need(i)?;
                i += 2;
            }
            "--device" => {
                device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            "--image" => {
                image = Some(need(i)?);
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }

    let cfg = rlx_siglip2::Siglip2Config::base_patch16_224();
    let width = cfg.vision.width;
    let seq = cfg.vision.seq_len();
    let layers = cfg.vision.layers;
    eprintln!("siglip2 vision: {layers} layers, width {width}, {seq} patches, {device:?}");

    let mut wm =
        rlx_core::load_weight_map(&std::path::Path::new(&dir).join("model.safetensors"), &[])?;
    let pre = rlx_siglip2::extract_vision_embed_weights(&mut wm, &cfg)?;
    let pooling = rlx_siglip2::extract_pooling_weights(&mut wm, &cfg, 1)?;
    let built = rlx_siglip2::build_vision_flow(&cfg, &mut wm, 1, pooling)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;

    let hidden = match &image {
        Some(p) => {
            let img = image::open(p)
                .with_context(|| format!("opening {p}"))?
                .to_rgb8();
            let (w, h) = img.dimensions();
            eprintln!("image {p} ({w}x{h})");
            let nchw = rlx_siglip2::siglip_normalize_nchw(
                img.as_raw(),
                h as usize,
                w as usize,
                cfg.vision.image_size,
            );
            rlx_siglip2::assemble_vision_hidden(
                &pre,
                &nchw,
                1,
                cfg.vision.patch_size,
                cfg.vision.image_size,
            )?
        }
        None => bail!("pass --image"),
    };

    // Where the spine ends has to be named, not inferred from the output. A
    // Qwen3 trunk ends in a norm; this one ends in an attention-pooling head
    // that produces a single pooled vector, so walking back from `image_embeds`
    // finds no add-spine at all. Anchor on `post_layernorm` — the last thing
    // applied to the token sequence — and take its input.
    let spine_end = graph
        .nodes()
        .iter()
        .find(|n| {
            matches!(n.op, rlx_ir::Op::LayerNorm { .. })
                && n.inputs.iter().any(|&i| {
                    matches!(&graph.node(i).op, rlx_ir::Op::Param { name }
                        if name.contains("post_layernorm"))
                })
        })
        .map(|n| n.inputs[0])
        .context("no `vision_model.post_layernorm` found; cannot locate the spine end")?;
    let chain = residual_stream_from(&graph, spine_end)?;
    // `1 + joins·layers` is the clean case. SigLIP2 adds position embeddings on
    // the spine *before* layer 0, so there is one extra join at the head of the
    // chain; solve for it rather than assume either shape.
    let adds = chain.len() - 1;
    let joins = adds / layers;
    let offset = adds - joins * layers;
    if joins == 0 {
        bail!(
            "residual chain has {} points for {layers} layers",
            chain.len()
        );
    }
    eprintln!(
        "residual chain: {} points = {offset} leading join(s) + {joins} per layer x {layers}",
        chain.len()
    );
    let all: Vec<usize> = (0..layers).collect();
    let taps: Vec<rlx_ir::NodeId> = all
        .iter()
        .map(|&l| chain[offset + joins * (l + 1)])
        .collect();
    let _ = layer_exit_taps(&chain, &[0], joins);

    let mut g = graph.clone();
    g.set_outputs(taps);
    let mut fwd = Session::new(device).compile(g);
    for (name, data) in &params {
        fwd.set_param(name, data);
    }
    let outs = fwd.run(&[("hidden", &hidden[..])]);

    let mean_norm = |o: &[f32]| -> f32 {
        let n = o.len() / width;
        (0..n)
            .map(|t| {
                o[t * width..(t + 1) * width]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
            })
            .sum::<f32>()
            / n as f32
    };

    println!("\nmean token ‖h‖ leaving each layer");
    println!("{:<8}{:>12}{:>14}", "layer", "‖h‖", "vs layer 0");
    let base = mean_norm(&outs[0]);
    for (slot, &layer) in all.iter().enumerate() {
        let m = mean_norm(&outs[slot]);
        if layer % 2 == 0 || layer + 1 == layers {
            println!("{:<8}{:>12.2}{:>13.1}x", layer, m, m / base);
        }
    }
    let last = mean_norm(&outs[layers - 1]);
    println!(
        "\ngrowth {:.1}x over {layers} layers — DINOv3 was 72x (lens failed), \
         an LM is near-flat (lens works)",
        last / base
    );
    Ok(())
}
