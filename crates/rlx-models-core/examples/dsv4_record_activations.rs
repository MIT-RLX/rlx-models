// Record the ACTIVATION dataflow of a real DeepSeek-V4-Flash decode across SEVERAL
// layers, to see how activation outliers grow with depth. Builds the paged decode
// graph (real backbone), injects opscope matmul stat-taps, then fills `moe_out.{il}`
// with the REAL host MoE (top-6 experts paged per layer) via the causal L+1-pass
// fill — so every layer's activations are exact. Reports per-site sketches + a
// per-layer summary of the residual-stream / query outliers.
//
//   cargo run -q -p rlx-models-core --example dsv4_record_activations --features metal -- [layers] [token]
use anyhow::{Context, Result};
use rlx_ir::DType;
use rlx_ir::quant::QuantScheme;
use rlx_models_core::standard_decoder::{
    CachedExpertSource, DeepseekV4Spec, DsV4RefLoader, ExpertSource, build_deepseek_v4_decode_moe,
    dequant_packed_linear, paged_moe_forward, v4_moe_spec,
};
use rlx_models_core::weight_loader::{MlxLoader, WeightLoader};
use rlx_opscope::{StatConfig, inject_matmul_stats_filtered};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n_layers: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let token: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let ckpt = "/Volumes/FOUR/DeepSeek/DeepSeek-V4-Flash-0731-MXFP4-MLX";
    let cfg: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(format!("{ckpt}/config.json"))?)?;
    let mut spec = DeepseekV4Spec::from_config(&cfg)?;
    spec.n_layers = n_layers.min(spec.n_layers);
    spec.dspark_target_layer_ids = vec![spec.n_layers - 1];
    let (d, n_exp, topk) = (spec.dim, spec.n_routed_experts, spec.n_activated_experts);
    let mk_loader = || -> Box<dyn WeightLoader> {
        Box::new(DsV4RefLoader::new(
            Box::new(MlxLoader::open_lazy(ckpt).expect("open")),
            n_exp,
        ))
    };

    // Build + inject.
    let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
    let (g, params, onames) =
        build_deepseek_v4_decode_moe(&spec, &mut *mk_loader(), 0, 0, &mut packed, true)?;
    // Lean sketch set: min/max/mean/nnz + per-channel outlier only. NO histogram
    // (hist_bins=0), per-position, or adjacency — those add ~30 nodes/tap and, over
    // ~100 whole-model matmul sites, blow the injected graph past a tractable compile.
    let cfg = StatConfig {
        hist_bins: 0,
        hist_normalize: false,
        per_channel: true,
        per_position: false,
        adjacency: false,
        ..StatConfig::default()
    };
    // Tap ONLY the residual stream (HC-gate input, numel hc·d) + the query
    // (numel nh·hd) — ~3 matmuls/layer instead of ~18, so the injected graph
    // compiles at depth (the tap count, not layer count, is what blows up compile).
    let hcd = spec.hc_mult * d;
    let nhhd = spec.n_heads * spec.head_dim;
    let (g2, specs) =
        inject_matmul_stats_filtered(&g, &cfg, &|lhs, _out| lhs == hcd || lhs == nhhd);
    eprintln!(
        "[rec] {}-layer graph {} nodes; {} taps over {} sites",
        spec.n_layers,
        g.nodes().len(),
        specs.len(),
        specs
            .iter()
            .map(|s| s.site.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut c = Session::new(Device::Cpu).compile_with(g2, &opts);
    for (n, dd) in &params {
        c.set_param(n, dd);
    }
    for (n, (b, _, _)) in &packed {
        c.set_param_typed(n, b, DType::U8);
    }

    // Preload router + shared (dequant) for each MoE layer; per-expert source (paged).
    let ds = v4_moe_spec(&spec);
    let mut ld = mk_loader();
    let mut router: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut shared: HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)> = HashMap::new();
    let moe_layers: Vec<usize> = onames
        .iter()
        .filter_map(|n| n.strip_prefix("moe_in.").and_then(|s| s.parse().ok()))
        .collect();
    let sh = |ld: &mut dyn WeightLoader, k: &str| -> Result<Vec<f32>> {
        match ld.take_packed_mlx(k)? {
            Some(p) => dequant_packed_linear(&p),
            None => Ok(ld.take(k)?.0),
        }
    };
    for &il in &moe_layers {
        let (rw, rs) = ld.take(&format!("model.layers.{il}.ffn.gate.weight"))?;
        router.insert(
            il,
            if rs == vec![d, n_exp] {
                let mut m = vec![0f32; n_exp * d];
                for i in 0..d {
                    for e in 0..n_exp {
                        m[e * d + i] = rw[i * n_exp + e];
                    }
                }
                m
            } else {
                rw
            },
        );
        shared.insert(
            il,
            (
                sh(
                    &mut *ld,
                    &format!("model.layers.{il}.ffn.shared_experts.gate_proj.weight"),
                )?,
                sh(
                    &mut *ld,
                    &format!("model.layers.{il}.ffn.shared_experts.up_proj.weight"),
                )?,
                sh(
                    &mut *ld,
                    &format!("model.layers.{il}.ffn.shared_experts.down_proj.weight"),
                )?,
            ),
        );
    }
    struct XSrc<'a> {
        l: &'a mut dyn WeightLoader,
    }
    impl ExpertSource for XSrc<'_> {
        fn fetch(&mut self, il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
            let w = match proj {
                "gate_proj" => "w1",
                "up_proj" => "w3",
                _ => "w2",
            };
            let k = format!("model.layers.{il}.ffn.experts.{e}.{w}.weight");
            dequant_packed_linear(&self.l.take_packed_mlx(&k)?.with_context(|| k.clone())?)
        }
    }
    let mut xld = mk_loader();
    let mut src = CachedExpertSource::new(XSrc { l: &mut *xld }, 512);

    // Causal fill: run the injected graph N+1×, filling one moe_out per pass; the
    // FINAL pass has all moe_out correct ⇒ all activation taps exact.
    let opos = |nm: &str| onames.iter().position(|x| x == nm);
    let mut moe_out: HashMap<usize, Vec<f32>> =
        moe_layers.iter().map(|&il| (il, vec![0f32; d])).collect();
    let run =
        |c: &mut rlx_runtime::CompiledGraph, mo: &HashMap<usize, Vec<f32>>| -> Vec<Vec<f32>> {
            let mut inp: Vec<(String, Vec<f32>)> = vec![("token_id".into(), vec![token as f32])];
            for (&il, v) in mo {
                inp.push((format!("moe_out.{il}"), v.clone()));
            }
            let r: Vec<(&str, &[f32])> = inp
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            c.run(&r)
        };
    for &il in &moe_layers {
        let out = run(&mut c, &moe_out);
        let xf = out[opos(&format!("moe_in.{il}")).unwrap()].clone();
        let mo = paged_moe_forward(&ds, il, &xf, &router[&il], None, None, &mut src, {
            let (sg, su, sd) = &shared[&il];
            Some((sg.as_slice(), su.as_slice(), sd.as_slice()))
        })?;
        moe_out.insert(il, mo);
    }
    let out = run(&mut c, &moe_out);
    eprintln!(
        "[rec] filled {} MoE layers via real host experts (top-{topk})",
        moe_layers.len()
    );

    // Per-site lhs (activation) sketches, in build order (== depth order).
    #[derive(Default, Clone)]
    struct Agg {
        numel: usize,
        nnz: f32,
        min: f32,
        max: f32,
        chan: Vec<f32>,
    }
    let mut order: Vec<String> = Vec::new();
    let mut aggs: HashMap<String, Agg> = HashMap::new();
    for s in &specs {
        if s.role != "lhs" {
            continue;
        }
        if !order.contains(&s.site) {
            order.push(s.site.clone());
        }
        let e = aggs.entry(s.site.clone()).or_default();
        e.numel = s.numel;
        let v = &out[s.out_idx];
        match s.stat {
            "nnz" => e.nnz = v.first().copied().unwrap_or(0.0),
            "min" => e.min = v.first().copied().unwrap_or(0.0),
            "max" => e.max = v.first().copied().unwrap_or(0.0),
            "chan_maxabs" => e.chan = v.clone(),
            _ => {}
        }
    }
    let chan_out = |e: &Agg| -> f32 {
        if e.chan.len() < 2 {
            return 1.0;
        }
        let mut c = e.chan.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c.last().copied().unwrap_or(0.0) / c[c.len() / 2].max(1e-9)
    };
    // Per-layer summary: residual stream (numel hc*d) + query (numel nh*hd) evolution.
    let hcd = spec.hc_mult * d;
    let nhhd = spec.n_heads * spec.head_dim;
    println!(
        "\n══ ACTIVATION evolution with DEPTH — real DSV4 decode, {} layers (token {token}) ══",
        spec.n_layers
    );
    println!(
        "residual-stream (HC-gate input, |max| & per-channel-outlier per occurrence, attn+ffn per layer):"
    );
    let mut ri = 0;
    for site in &order {
        let e = &aggs[site];
        if e.numel != hcd {
            continue;
        }
        let absmax = e.min.abs().max(e.max.abs());
        println!(
            "   [{}] layer{} {:<5} |max| {:>7.3}  chanOut {:>6.1}x",
            ri,
            ri / 2,
            if ri % 2 == 0 { "attn" } else { "ffn" },
            absmax,
            chan_out(e)
        );
        ri += 1;
    }
    println!("query (post norm+rope, |max| & per-channel-outlier per layer):");
    let mut qi = 0;
    for site in &order {
        let e = &aggs[site];
        if e.numel != nhhd {
            continue;
        }
        let absmax = e.min.abs().max(e.max.abs());
        println!(
            "   layer{} |max| {:>8.3}  chanOut {:>6.1}x",
            qi,
            absmax,
            chan_out(e)
        );
        qi += 1;
    }
    // Biggest per-channel outliers overall.
    let mut ranked: Vec<(&String, f32, f32)> = order
        .iter()
        .map(|s| {
            (
                s,
                chan_out(&aggs[s]),
                aggs[s].min.abs().max(aggs[s].max.abs()),
            )
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("top per-channel-outlier activation sites (chanOut = maxabs/median):");
    for (s, co, am) in ranked.iter().take(8) {
        println!(
            "   {:<12} numel {:>5}  chanOut {:>6.1}x  |max| {:>7.3}",
            s, aggs[*s].numel, co, am
        );
    }
    println!("(all activations were 100% dense in the single-layer probe; sparsity omitted here)");
    Ok(())
}
