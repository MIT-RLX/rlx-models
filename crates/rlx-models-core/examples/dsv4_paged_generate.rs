// RLX — versatile ML compiler + runtime. GPLv3.
//! **(yes) Full real-checkpoint PAGED decode** — attention backbone in-graph, MoE
//! paged host-side. Staged so each wall shows early:
//!   --build-only : just build the paged decode graph (loads backbone, surfaces
//!                  builder limits on the real config)
//!   (default)    : build + compile (the heavy 43-layer step)
//! `--layers N` truncates for a quick smoke.
//!
//!   dsv4_paged_generate --ckpt <dir> --layers 6 --build-only

use anyhow::{Context, Result};
use rlx_ir::quant::QuantScheme;
use rlx_models_core::standard_decoder::{
    CachedExpertSource, DeepseekV4Spec, DsV4RefLoader, ExpertSource, PackedExpertSource,
    PagedGroupedMoe, SharedExpertGpu, V4Decoder, build_deepseek_v4_decode_fixed_stage_moe,
    build_deepseek_v4_decode_moe, build_v4_post_stage, dense_swiglu_ffn, dequant_packed_linear,
    hash_route_experts, paged_moe_forward, paged_moe_forward_fused, paged_moe_route, v4_moe_spec,
};
use rlx_models_core::weight_loader::MlxPackedLinear;
use rlx_models_core::weight_loader::{MlxLoader, WeightLoader};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::time::Instant;

fn flag(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}
fn has(a: &[String], k: &str) -> bool {
    a.iter().any(|x| x == k)
}

static MOE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Drop-guard: accumulates each `moe_fn` call's wall time into `MOE_NS`, so the
/// MoE (expert-paging) fraction of decode can be split from the backbone without
/// threading a counter through `step_io_paged`.
struct MoeTimer(std::time::Instant);
impl Drop for MoeTimer {
    fn drop(&mut self) {
        MOE_NS.fetch_add(
            self.0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = flag(&a, "--ckpt")
        .unwrap_or_else(|| "/Volumes/FOUR/DeepSeek/DeepSeek-V4-Flash-0731-MXFP4-MLX".into());
    let cfg: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(format!("{ckpt}/config.json"))?)?;
    let mut spec = DeepseekV4Spec::from_config(&cfg).context("spec from config")?;
    if let Some(nl) = flag(&a, "--layers").and_then(|s| s.parse::<usize>().ok()) {
        spec.n_layers = nl.min(spec.n_layers);
        spec.dspark_target_layer_ids = vec![spec.n_layers.saturating_sub(1)];
    }
    let n_experts = spec.n_routed_experts;
    let mk_loader = || -> Box<dyn WeightLoader> {
        Box::new(DsV4RefLoader::new(
            Box::new(MlxLoader::open_lazy(&ckpt).expect("open ckpt")),
            n_experts,
        ))
    };
    eprintln!(
        "[dsv4-paged-gen] {ckpt}\n  layers={} hidden={} heads={} head_dim={} ql={} olora={} experts={} top_k={} ratios[..6]={:?}",
        spec.n_layers,
        spec.dim,
        spec.n_heads,
        spec.head_dim,
        spec.q_lora_rank,
        spec.o_lora_rank,
        spec.n_routed_experts,
        spec.n_activated_experts,
        &spec.compress_ratios[..spec.compress_ratios.len().min(6)]
    );

    // ── Stage 1: BUILD the paged decode graph (loads backbone via ref-loader) ──
    let t = Instant::now();
    let mut loader = mk_loader();
    let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
    let (g, params, onames) =
        build_deepseek_v4_decode_moe(&spec, &mut *loader, 0, 0, &mut packed, true)?;
    let moe_outs = onames.iter().filter(|n| n.starts_with("moe_in.")).count();
    eprintln!(
        "  BUILD ok in {:.1}s: {} params, {} packed tensors, {} outputs ({} moe_in = MoE layers)",
        t.elapsed().as_secs_f64(),
        params.len(),
        packed.len(),
        onames.len(),
        moe_outs
    );
    // opscope edge-list (`idx op_name in0 in1 …`) for `../rlx opscope-graph`.
    fn dump_graph(g: &rlx_ir::Graph, title: &str) -> String {
        use rlx_ir::Op;
        let op_name = |op: &Op| -> String {
            match op {
                Op::Activation(x) => format!("{x:?}"),
                Op::Binary(b) => format!("{b:?}"),
                Op::Compare(c) => format!("Cmp{c:?}"),
                Op::Input { .. } => "in".into(),
                Op::Param { .. } => "w".into(),
                Op::Constant { .. } => "k".into(),
                other => format!("{:?}", other.kind()),
            }
        };
        let mut s = format!("# {title}, {} nodes\n", g.nodes().len());
        for (i, n) in g.nodes().iter().enumerate() {
            s.push_str(&i.to_string());
            s.push(' ');
            s.push_str(&op_name(&n.op));
            for inp in &n.inputs {
                s.push(' ');
                s.push_str(&inp.0.to_string());
            }
            s.push('\n');
        }
        s
    }
    // `--dump-graph <file>`: the monolithic paged decode graph (non-split).
    if let Some(path) = flag(&a, "--dump-graph") {
        let s = dump_graph(&g, &format!("DSV4-Flash decode: {} layers", spec.n_layers));
        std::fs::write(&path, s)?;
        eprintln!("  dumped {} nodes → {path}", g.nodes().len());
        return Ok(());
    }
    // `--dump-split <prefix>`: the ATTN/POST split graphs for one mid MoE layer →
    // `<prefix>.attn.txt` + `<prefix>.post.txt`, so opscope-graph shows how #2
    // reshaped the dataflow (attn = no 2nd hc_post; post = isolated hc_post).
    if let Some(prefix) = flag(&a, "--dump-split") {
        drop((g, params, packed));
        let il = spec.n_layers / 2; // a mid routed-MoE layer
        let mut ld = mk_loader();
        let mut pk: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
        let (ag, _, _) = build_deepseek_v4_decode_fixed_stage_moe(
            &spec,
            &mut *ld,
            il..il + 1,
            false,
            false,
            spec.window_size.max(1),
            8,
            &mut pk,
            true,
            true,
        )?;
        let (pg, _, _) = build_v4_post_stage(&spec, &mut *mk_loader(), false, false, &mut pk)?;
        let ap = format!("{prefix}.attn.txt");
        let pp = format!("{prefix}.post.txt");
        std::fs::write(
            &ap,
            dump_graph(&ag, &format!("DSV4 ATTN graph (split, layer {il})")),
        )?;
        std::fs::write(
            &pp,
            dump_graph(&pg, "DSV4 POST graph (weight-free hc_post)"),
        )?;
        eprintln!(
            "  dumped attn={} nodes → {ap}\n  dumped post={} nodes → {pp}",
            ag.nodes().len(),
            pg.nodes().len()
        );
        return Ok(());
    }
    if has(&a, "--build-only") {
        return Ok(());
    }

    drop((g, params, packed)); // stage-1 build was just to surface builder limits
    if !has(&a, "--decode") {
        eprintln!("  (pass --decode for the COMPILE-ONCE paged decode via V4Decoder::step_paged)");
        return Ok(());
    }

    // ── Stage 3: PAGED DECODE (attention in-graph, MoE host-side incl. hash layers) ──
    let ds = v4_moe_spec(&spec);
    let gs = cfg["quantization"]["group_size"].as_u64().unwrap_or(32) as usize;
    let scheme = QuantScheme::MlxMxfp4 {
        group_size: gs as u32,
    };
    // Preload per-MoE-layer router (+bias), shared expert, and tid2eid (hash layers).
    let mut ld = mk_loader();
    let mut router: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut shared: HashMap<usize, (Vec<f32>, Vec<f32>, Vec<f32>)> = HashMap::new();
    let mut tid2eid: HashMap<usize, Vec<f32>> = HashMap::new();
    let (n, d, topk) = (spec.n_routed_experts, spec.dim, spec.n_activated_experts);
    for il in spec.first_k_dense_replace..spec.n_layers {
        let (rw, rs) = ld.take(&format!("model.layers.{il}.ffn.gate.weight"))?;
        router.insert(
            il,
            if rs == vec![d, n] {
                let mut m = vec![0f32; n * d];
                for i in 0..d {
                    for e in 0..n {
                        m[e * d + i] = rw[i * n + e];
                    }
                }
                m
            } else {
                rw
            },
        );
        let sh = |ld: &mut dyn WeightLoader, k: &str| -> Result<Vec<f32>> {
            match ld.take_packed_mlx(k)? {
                Some(p) => dequant_packed_linear(&p),
                None => Ok(ld.take(k)?.0),
            }
        };
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
        if il < spec.n_hash_layers {
            if let Ok((t2e, _)) = ld.take(&format!("model.layers.{il}.ffn.gate.tid2eid")) {
                tid2eid.insert(il, t2e);
            }
        }
    }
    eprintln!(
        "  preloaded router/shared for {} MoE layers ({} hash)",
        router.len(),
        tid2eid.len()
    );

    // Paged experts straight off the ref-mapped lazy loader (only active resident).
    // `prewarm` parallel-page-faults a token's active experts before the serial fetch.
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
            dequant_packed_linear(&self.l.take_packed_mlx(&k)?.with_context(|| k.to_string())?)
        }
        fn prewarm(&mut self, experts: &[(usize, usize)]) {
            let keys: Vec<String> = experts
                .iter()
                .flat_map(|&(il, e)| {
                    ["w1", "w3", "w2"]
                        .into_iter()
                        .map(move |w| format!("model.layers.{il}.ffn.experts.{e}.{w}.weight"))
                })
                .collect();
            let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
            self.l.prewarm(&refs);
        }
    }
    impl PackedExpertSource for XSrc<'_> {
        fn fetch_packed(&mut self, il: usize, e: usize, proj: &str) -> Result<MlxPackedLinear> {
            let w = match proj {
                "gate_proj" => "w1",
                "up_proj" => "w3",
                _ => "w2",
            };
            let k = format!("model.layers.{il}.ffn.experts.{e}.{w}.weight");
            self.l.take_packed_mlx(&k)?.with_context(|| k.to_string())
        }
        fn prewarm(&mut self, experts: &[(usize, usize)]) {
            ExpertSource::prewarm(self, experts);
        }
        fn with_packed_borrowed(
            &mut self,
            il: usize,
            e: usize,
            proj: &str,
            sink: &mut dyn FnMut(&[u8], &[u8]) -> bool,
        ) -> Option<bool> {
            let w = match proj {
                "gate_proj" => "w1",
                "up_proj" => "w3",
                _ => "w2",
            };
            let k = format!("model.layers.{il}.ffn.experts.{e}.{w}.weight");
            // Zero-copy: borrow codes+e8m0 scales straight from the mmap shard, let the
            // sink copy them into the device slot, then DONTNEED the pages so they
            // don't accumulate in the cache alongside the arena (would swap). `?`
            // returns None (→ caller owns) when the loader can't borrow.
            let r = {
                let b = self.l.borrow_packed_mlx(&k)?;
                sink(b.w_q, b.scales)
            };
            self.l.dontneed_packed_mlx(&k);
            Some(r)
        }
    }
    let mut xld = mk_loader();
    // Prompt: `--text "..."` tokenizes via the checkpoint's tokenizer.json (full
    // prompt test); else `--prompt id,id,...` takes raw token ids.
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{ckpt}/tokenizer.json")).ok();
    let prompt: Vec<u32> = if let Some(text) = flag(&a, "--text") {
        let tk = tokenizer.as_ref().expect("tokenizer.json");
        let ids = tk
            .encode(text.as_str(), true)
            .expect("encode")
            .get_ids()
            .to_vec();
        eprintln!(
            "  tokenized {:?} → {} tokens {:?}",
            text,
            ids.len(),
            &ids[..ids.len().min(12)]
        );
        ids
    } else {
        flag(&a, "--prompt")
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![0u32, 1, 2])
    };
    let n_new = flag(&a, "--new")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1usize);
    let _ = &scheme;

    // COMPILE-ONCE: build+compile the fixed backbone ONCE (the O(L²) compile is paid
    // a single time, not per token), then step_paged decodes with no recompile.
    let (mw, mc) = (spec.window_size.max(1), 8usize);
    // FUSED host MoE: dequant-matvec directly on the packed MXFP4 routed codes (no
    // [out,in] f32 materialization) + MXFP8 shared. Optional `--dequant` uses the
    // materialize-then-matmul path with a cross-token f32 LRU cache instead.
    // Default = dequant-then-matmul (both passes vectorize → faster than the
    // memory-lean-but-scalar `--fused` matvec, which the FP4-LUT gather de-vectorizes).
    let dequant_mode = !has(&a, "--fused");
    // `--gpu [--device metal|mlx]`: route the MoE through the native GPU grouped
    // kernel (PagedGroupedMoe) instead of the host matvec; shared stays host-side.
    let gpu = has(&a, "--gpu");
    let device = match flag(&a, "--device").as_deref() {
        Some("metal") => Device::Metal,
        Some("mlx") => Device::Mlx,
        _ => Device::Cpu,
    };
    // Expert residency capacity (distinct experts kept resident across tokens).
    // Sized to hold ~2 tokens' working set (n_layers·top_k·2) so a decode token
    // REUSES the previous token's experts instead of re-paging all of them —
    // combined with `PagedGroupedMoe`'s incremental per-slot upload, a stable hot
    // set drives fetch+upload toward zero. Capped for RAM safety (~31 MB/slot host
    // +device). Override with `--acap N`.
    let acap = flag(&a, "--acap")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (spec.n_layers * spec.n_activated_experts * 2).clamp(topk, 640));
    let mut gmoe = gpu.then(|| {
        PagedGroupedMoe::new(
            device,
            acap,
            topk,
            spec.dim,
            spec.moe_intermediate_size,
            gs,
            spec.swiglu_limit,
            scheme,
        )
    });
    if gpu {
        eprintln!("  PagedGroupedMoe residency a_cap={acap} experts (m_cap={topk} rows)");
    }
    // GPU dense shared expert (`--gpu-shared`): puts the last host-side MoE piece
    // on-device. NB it's a tiny per-layer 1-token op whose per-layer f32 weight upload
    // + dispatch overhead usually LOSES to the parallel host matmul — kept opt-in.
    let se_inter = spec.n_shared_experts * spec.moe_intermediate_size;
    let mut gshared = (gpu && has(&a, "--gpu-shared"))
        .then(|| SharedExpertGpu::new(device, spec.dim, se_inter, spec.swiglu_limit));
    // Auto-size the f32 expert cache to hold roughly one token's working set
    // (layers×top_k×3 tensors) so it doesn't THRASH (evict before next-token reuse),
    // but cap it (each tensor ~32 MB f32, anon) to stay RAM-safe. `RLX_MLX_KEEP_WARM=1`
    // additionally keeps the mmap'd shard pages resident across tokens.
    let auto_cap = (spec.n_layers * spec.n_activated_experts * 3).min(384);
    let cache_cap = flag(&a, "--cache")
        .and_then(|s| s.parse().ok())
        .unwrap_or(auto_cap);
    let mut cached = CachedExpertSource::new(XSrc { l: &mut xld }, cache_cap);
    let mut xsrc2 = mk_loader();
    let mut xsrc3 = mk_loader();
    let mut moe_fn = |il: usize, tok: u32, xf: &[f32]| -> Result<Vec<f32>> {
        let _mt = MoeTimer(std::time::Instant::now());
        let hash = tid2eid
            .get(&il)
            .map(|t2e| hash_route_experts(t2e, topk, tok));
        let (sg, su, sd) = &shared[&il];
        if let Some(gm) = gmoe.as_mut() {
            // GPU routed experts (grouped kernel) + GPU dense shared expert.
            let (top, w) = paged_moe_route(&ds, xf, &router[&il], None, hash.as_deref());
            let routes = vec![top.into_iter().zip(w).collect::<Vec<_>>()];
            let mut src = XSrc { l: &mut *xsrc3 };
            let routed = gm.forward(il, xf, 1, &routes, &mut src)?;
            let shv = match gshared.as_mut() {
                Some(gs) => gs.forward(xf, sg, su, sd)?,
                None => dense_swiglu_ffn(xf, sg, su, sd, spec.swiglu_limit),
            };
            Ok(routed.iter().zip(&shv).map(|(a, b)| a + b).collect())
        } else if dequant_mode {
            let routed = paged_moe_forward(
                &ds,
                il,
                xf,
                &router[&il],
                None,
                hash.as_deref(),
                &mut cached,
                None,
            )?;
            let shv = dense_swiglu_ffn(xf, sg, su, sd, spec.swiglu_limit);
            Ok(routed.iter().zip(&shv).map(|(a, b)| a + b).collect())
        } else {
            let mut src = XSrc { l: &mut *xsrc2 };
            paged_moe_forward_fused(
                &ds,
                il,
                xf,
                &router[&il],
                None,
                hash.as_deref(),
                &mut src,
                Some((sg.as_slice(), su.as_slice(), sd.as_slice())),
            )
        }
    };

    // Thread one token through the per-layer stages, filling each layer's MoE via moe_fn.
    fn run_token(
        stages: &mut [V4Decoder],
        token: u32,
        moe_fn: &mut dyn FnMut(usize, u32, &[f32]) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let (mut hidden, mut logits): (Option<Vec<f32>>, Option<Vec<f32>>) = (None, None);
        for st in stages.iter_mut() {
            let (l, h) = st.step_io_paged(token, hidden.as_deref(), &mut *moe_fn)?;
            logits = l;
            hidden = h;
        }
        Ok(logits.expect("last stage logits"))
    }
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            )
            .0 as u32
    };
    let mode = if gpu {
        "GPU grouped kernel"
    } else if dequant_mode {
        "dequant+cache"
    } else {
        "FUSED matvec (packed)"
    };

    // Build ONE paged stage per layer (O(L) linear compiles). `--attn-gpu` runs the
    // attention backbone on the same GPU as the MoE (else CPU backbone + GPU MoE).
    let bb_device = if has(&a, "--attn-gpu") {
        device
    } else {
        Device::Cpu
    };
    // `--attn-split`: run each MoE layer's attention ONCE (attn graph → hc_post inputs)
    // + a tiny post graph, instead of the O(2L) two-pass that recomputes attention.
    let attn_split = has(&a, "--attn-split");
    let t = Instant::now();
    let mut stages: Vec<V4Decoder> = Vec::with_capacity(spec.n_layers);
    for il in 0..spec.n_layers {
        let (fst, lst) = (il == 0, il == spec.n_layers - 1);
        let st = if attn_split {
            V4Decoder::new_stage_paged_split(
                &spec,
                &mut *mk_loader(),
                il..il + 1,
                fst,
                lst,
                mw,
                mc,
                bb_device,
            )?
        } else {
            V4Decoder::new_stage_paged(
                &spec,
                &mut *mk_loader(),
                il..il + 1,
                fst,
                lst,
                mw,
                mc,
                bb_device,
            )?
        };
        stages.push(st);
    }
    let compile_s = t.elapsed().as_secs_f64();
    let sp = if attn_split { " +attn-split" } else { "" };
    eprintln!(
        "  compiled {} per-layer stages in {compile_s:.1}s [{mode}{sp}]",
        spec.n_layers
    );

    // Prefill the prompt (build the KV cache).
    let t = Instant::now();
    let mut logits = vec![0f32; spec.vocab_size];
    if has(&a, "--debug-hidden") {
        // Print each stage's hidden/logits norm for the first prompt token.
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
        let mut hidden: Option<Vec<f32>> = None;
        for (i, st) in stages.iter_mut().enumerate() {
            let (l, h) = st.step_io_paged(prompt[0], hidden.as_deref(), &mut moe_fn)?;
            let out = l.as_ref().or(h.as_ref()).unwrap();
            eprintln!(
                "    stage {i}: out_rms {:.4} finite {}",
                rms(out),
                out.iter().all(|x| x.is_finite())
            );
            hidden = h;
        }
        return Ok(());
    }
    for &tk in &prompt {
        logits = run_token(&mut stages, tk, &mut moe_fn)?;
    }
    let prefill_s = t.elapsed().as_secs_f64();

    // Generate + time each token.
    MOE_NS.store(0, std::sync::atomic::Ordering::Relaxed); // measure gen only
    let mut out = Vec::with_capacity(n_new);
    let mut gen_ms = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let nx = argmax(&logits);
        out.push(nx);
        let t = Instant::now();
        logits = run_token(&mut stages, nx, &mut moe_fn)?;
        gen_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let total_gen_s: f64 = gen_ms.iter().sum::<f64>() / 1e3;
    let gen_tps = if total_gen_s > 0.0 {
        n_new as f64 / total_gen_s
    } else {
        0.0
    };
    eprintln!(
        "\n  ── BENCH ({} layers, {mode}) ──\n  compile(all layers, ONCE) {compile_s:.1}s\n  prefill {} tok in {prefill_s:.1}s ({:.2} tok/s)\n  generate {} tok in {total_gen_s:.1}s ({gen_tps:.2} tok/s, {:.0} ms/tok avg)",
        spec.n_layers,
        prompt.len(),
        prompt.len() as f64 / prefill_s.max(1e-6),
        n_new,
        total_gen_s / (n_new.max(1)) as f64 * 1e3
    );
    let moe_ms = MOE_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let gen_ms_total = total_gen_s * 1e3;
    eprintln!(
        "  SPLIT: MoE(expert-paging) {:.0} ms/tok ({:.0}%)  |  backbone+attn {:.0} ms/tok ({:.0}%)",
        moe_ms / n_new.max(1) as f64,
        100.0 * moe_ms / gen_ms_total.max(1e-6),
        (gen_ms_total - moe_ms) / n_new.max(1) as f64,
        100.0 * (gen_ms_total - moe_ms) / gen_ms_total.max(1e-6),
    );
    if dequant_mode {
        eprintln!(
            "  expert cache: {} hits / {} misses",
            cached.hits, cached.misses
        );
    }
    if gpu {
        let (pw, ft, up, cp, ne, calls) =
            rlx_models_core::standard_decoder::paged_moe_io_profile_take();
        eprintln!(
            "  PagedGroupedMoe IO ({calls} calls, {ne} experts fetched): prewarm {pw:.0}ms  disk-read {ft:.0}ms  prep+slot-upload {up:.0}ms  gpu-run {cp:.0}ms"
        );
        if let Some(gm) = gmoe.as_ref() {
            eprintln!("  total expert uploads: {}", gm.uploads);
        }
    }
    if let Some(tk) = &tokenizer {
        let text = tk.decode(&out, true).unwrap_or_default();
        eprintln!("  OUTPUT TEXT: {text:?}");
    }
    eprintln!("  OUTPUT tokens: {out:?}");
    Ok(())
}
