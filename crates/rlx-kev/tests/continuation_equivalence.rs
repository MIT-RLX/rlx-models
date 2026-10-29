//! **Raw-token continuation equivalence**, with no kev structure involved:
//!
//! ```text
//! prefill(H + N)   ==   state_pass(H) then continuation(past H, N new)
//! ```
//!
//! This is the invariant the prefix cache rests on, tested on plain token ids
//! so a failure cannot be blamed on the encoder, the readout or the pointer
//! head. Sweeping `N` also covers the Gated DeltaNet carry: `N = 1` is the
//! ordinary decode path, `N > 1` is the multi-token carry that otherwise only
//! speculative verify exercises.
//!
//! The bug this pins: the decode flow emits the hidden **before**
//! `output_norm` while the prefill flow's `export_normed_hidden` is the value
//! **after** it. Their RMS barely differs (1.763 vs 1.759), so it reads as
//! noise, while element 0 was -50 instead of -6.3.
//!
//! Gated on `KEV_RUN_DIR` + `KEV_BASE_DIR`.

use anyhow::Result;
use rlx_runtime::{Device, Session};

fn cosv(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
fn continuation_equals_prefill_over_the_same_context() {
    let Some(run) = std::env::var_os("KEV_RUN_DIR") else {
        eprintln!("skipping: set KEV_RUN_DIR and KEV_BASE_DIR");
        return;
    };
    let Some(base) = std::env::var_os("KEV_BASE_DIR") else {
        eprintln!("skipping: set KEV_BASE_DIR");
        return;
    };
    run_check(
        std::path::Path::new(&base).to_path_buf(),
        std::path::Path::new(&run).to_path_buf(),
    )
    .expect("continuation equivalence");
}

fn run_check(base_dir: std::path::PathBuf, run_dir: std::path::PathBuf) -> Result<()> {
    let base = base_dir.to_str().expect("utf8").to_string();
    let adapter_dir = Some(run_dir.to_str().expect("utf8").to_string());
    let base = &base;
    let load = || -> Result<rlx_qwen35::Qwen35Weights> {
        let ad = rlx_kev::LoraAdapter::open(std::path::Path::new(
            adapter_dir.as_deref().expect("adapter dir"),
        ))?;
        Ok(rlx_kev::load_merged(std::path::Path::new(base), ad)?.weights)
    };
    let out_norm = rlx_kev::load_merged(
        std::path::Path::new(base),
        rlx_kev::LoraAdapter::open(std::path::Path::new(
            adapter_dir.as_deref().expect("adapter dir"),
        ))?,
    )?
    .weights
    .output_norm
    .clone();
    let kw = rlx_kev::load_merged(
        std::path::Path::new(base),
        rlx_kev::LoraAdapter::open(std::path::Path::new(
            adapter_dir.as_deref().expect("adapter dir"),
        ))?,
    )?;
    let cfg = kw.cfg.clone();
    let d = cfg.hidden_size;
    let rot_half = cfg.rope_dim_count / 2;
    drop(kw);

    let h: usize = 64;
    let ids_all: Vec<f32> = (0..256u32).map(|i| ((i * 37) % 40000 + 5) as f32).collect();

    // Variant A: our hand-built Bias-mask feeds.  Variant B: rlx-qwen35's own
    // `decode_step_feeds` with the multiplicative mask, for seq == 1 only.
    for (label, n_new, bias) in [
        ("bias   ", 1usize, true),
        ("canon  ", 1, false),
        ("bias   ", 2, true),
        ("bias   ", 64, true),
    ] {
        let total = h + n_new;
        // ---- plain prefill over the whole thing
        let (g, params, _) = rlx_qwen35::build_qwen35_graph_sized_ext(
            &cfg,
            load()?,
            1,
            total,
            false,
            false,
            false,
            false,
            None,
            true,
            false,
            true,
            false,
        )?;
        let mut c = Session::new(Device::Cpu).compile(g);
        for (n, v) in params {
            c.set_param(&n, &v);
        }
        let (cos_t, sin_t) = rlx_qwen35::mrope_prefill_feeds(&cfg, total, None, rot_half);
        let ref_out = c.run(&[
            ("input_ids", &ids_all[..total]),
            ("rope_cos", &cos_t[..]),
            ("rope_sin", &sin_t[..]),
        ]);
        let want = ref_out[0].clone();
        drop(c);

        // ---- state pass over the first H
        let (gs, ps, _) = rlx_qwen35::build_qwen35_graph_sized_ext(
            &cfg,
            load()?,
            1,
            h,
            false,
            false,
            false,
            true,
            None,
            true,
            false,
            true,
            false,
        )?;
        let mut cs = Session::new(Device::Cpu).compile(gs);
        for (n, v) in ps {
            cs.set_param(&n, &v);
        }
        let (cos_h, sin_h) = rlx_qwen35::mrope_prefill_feeds(&cfg, h, None, rot_half);
        let pad = rlx_qwen35::prompt_pad_mask_feeds(&cfg, 1, h, &[h]);
        let zeros = rlx_qwen35::zero_recurrent_inputs(&cfg, 1);
        let last = [(h - 1) as f32];
        let mut feeds: Vec<(&str, &[f32])> = vec![
            ("input_ids", &ids_all[..h]),
            ("rope_cos", &cos_h[..]),
            ("rope_sin", &sin_h[..]),
            ("last_token_idx", &last[..]),
        ];
        for (n, v) in pad.iter().chain(zeros.iter()) {
            feeds.push((n.as_str(), v.as_slice()));
        }
        let souts = cs.run(&feeds);
        let (_t, cache, _m) =
            rlx_qwen35::seed_cache_from_outputs(&cfg, 1, h, &[h], souts, false, true)?;
        drop(cs);

        // ---- continuation over the remaining N
        let opts = rlx_qwen35::Qwen35DecodeOpts {
            batch: 1,
            past_seq: h,
            seq: n_new,
            verify_all: true,
            dynamic_past: false,
            use_custom_mask: true,
            bias_mask: bias,
            force_host_embed: false,
            enable_mtp_head: false,
            fast_mtp: false,
            fast_greedy_lm_head: true,
            profile: None,
            residual_out: false,
            greedy_token: false,
            multi_row: false,
            mtp_kv: false,
            keys_mask: false,
        };
        let (hir, pc, _) =
            rlx_qwen35::build_qwen35_decode_model_flow(&cfg, std::sync::Arc::new(load()?), &opts)?;
        let mut cc = Session::new(Device::Cpu).compile_hir(hir)?;
        for (n, v) in pc {
            cc.set_param(&n, &v);
        }
        let sections: Vec<[usize; 4]> = (0..n_new)
            .map(|j| rlx_qwen35::text_section_pos(h + j))
            .collect();
        let (cos_n, sin_n) =
            rlx_qwen35::mrope_prefill_feeds(&cfg, n_new, Some(&sections), rot_half);
        let owned: Vec<(String, Vec<f32>)> = if bias {
            let mut v = rlx_kev::prefix::cache_feeds(&cache);
            v.push((
                "mask".to_string(),
                rlx_qwen35::build_verify_bias_mask(1, cfg.num_attention_heads, h, n_new, h),
            ));
            v.push(("rope_cos".to_string(), cos_n.clone()));
            v.push(("rope_sin".to_string(), sin_n.clone()));
            v.push(("input_ids".to_string(), ids_all[h..total].to_vec()));
            v
        } else {
            // rlx-qwen35's canonical decode feeds: cache tensors, ids, rope and
            // the multiplicative per-key mask, all built by the crate itself.
            let toks: Vec<u32> = ids_all[h..total].iter().map(|x| *x as u32).collect();
            rlx_qwen35::decode_step_feeds(
                &cfg,
                &cache,
                &toks,
                &cos_n,
                &sin_n,
                Some(h),
                &[0],
                None,
                None,
            )?
        };
        let f2: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let got = cc.run(&f2);
        let rows = got[0].len() / d;

        // The decode flow emits the hidden BEFORE `output_norm` (the final RMS
        // norm lives on the logits path). Apply it here and re-compare.
        let eps = cfg.rms_norm_eps as f32;
        let offset = if cfg.rms_norm_offset { 1.0 } else { 0.0 };
        let mut normed = got[0].clone();
        for t in 0..rows {
            let row = &mut normed[t * d..(t + 1) * d];
            let ms = row.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (i, x) in row.iter_mut().enumerate() {
                *x = *x * inv * (out_norm[i] + offset);
            }
        }
        let mut worst = 1.0f32;
        for j in 0..rows.min(n_new) {
            let a = &want[(h + j) * d..(h + j + 1) * d];
            let b = &normed[j * d..(j + 1) * d];
            worst = worst.min(cosv(a, b));
        }
        assert!(
            worst > 0.9999,
            "{label} n_new={n_new}: worst cosine {worst} vs the plain prefill \
             (a value near 0.53 means the final `output_norm` was not applied)"
        );
        eprintln!("{label} n_new={n_new:3}: worst cosine {worst:.6}");
    }
    Ok(())
}
