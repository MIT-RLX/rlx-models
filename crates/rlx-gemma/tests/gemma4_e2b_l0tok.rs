// RLX — GPLv3. Per-token layer-0 attention bisection (positions>0 bug).
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::config::GemmaConfig;
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::PathBuf;
fn dir() -> Option<PathBuf> {
    let h = std::env::var_os("HOME")?;
    let b = std::path::Path::new(&h).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let s = std::fs::read_dir(&b).ok()?.flatten().next()?.path();
    s.join("config.json").is_file().then_some(s)
}
fn rd(p: &std::path::Path) -> Option<Vec<f32>> {
    let r = std::fs::read(p).ok()?;
    Some(
        r.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    d / (na * nb + 1e-12)
}
#[test]
fn e2b_l0_per_token() {
    let Some(d) = dir() else { return };
    let fx =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_e2b_nosrq");
    let cfg = GemmaConfig::from_file(&d.join("config.json")).unwrap();
    let ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];
    let seq = ids.len();
    let lp = GemmaQatLoader::open(&d).unwrap();
    let ple = lp.compute_per_layer_inputs(&cfg, &ids).unwrap();
    unsafe {
        std::env::set_var("RLX_TAP_L0", "1");
        std::env::set_var("RLX_TAP_LAYER", "0");
    }
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let mut packed = HashMap::new();
    let (g, p) = rlx_gemma::builder::build_gemma_graph_sized_packed_ext(
        &cfg,
        &mut loader,
        1,
        seq,
        false,
        false,
        false,
        &mut packed,
        None,
        None,
    )
    .unwrap();
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, p).unwrap();
    let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = c.run(&[
        ("input_ids", idf.as_slice()),
        ("per_layer_inputs", ple.as_slice()),
    ]);
    let qd = cfg.num_attention_heads * cfg.layer_head_dim(0); // 2048
    // Note: hf fixtures dumped for layer 0 (hf_q pre-rope, hf_attn_preo).
    let kvd = cfg.layer_num_kv_heads(0) * cfg.layer_head_dim(0); // 256
    let hq = rd(&fx.join("l0_q.bin")); // no-SRQ Q (pre-rope)
    let hqr = rd(&fx.join("l0_qrope.bin")); // no-SRQ Q (post-rope)
    let hkr = rd(&fx.join("l0_krope.bin")); // no-SRQ K (post-rope)
    let ha = rd(&fx.join("l0_attn_preo.bin")); // no-SRQ attention output
    for ti in 0..seq {
        if let Some(q) = &hq {
            eprintln!(
                "[l0] tok {ti} Q-prerope(tap3) cos={:.5}",
                cos(&outs[8][ti * qd..(ti + 1) * qd], &q[ti * qd..(ti + 1) * qd])
            );
        }
        if let Some(q) = &hqr {
            eprintln!(
                "[l0] tok {ti} Q-rope(tap6)   cos={:.5}",
                cos(
                    &outs[11][ti * qd..(ti + 1) * qd],
                    &q[ti * qd..(ti + 1) * qd]
                )
            );
        }
        if let Some(k) = &hkr {
            eprintln!(
                "[l0] tok {ti} K-rope(tap7)   cos={:.5}",
                cos(
                    &outs[12][ti * kvd..(ti + 1) * kvd],
                    &k[ti * kvd..(ti + 1) * kvd]
                )
            );
        }
        if let Some(a) = &ha {
            eprintln!(
                "[l0] tok {ti} attn(tap8)      cos={:.5}",
                cos(
                    &outs[13][ti * qd..(ti + 1) * qd],
                    &a[ti * qd..(ti + 1) * qd]
                )
            );
        }
    }
    // In-test reconstruction from rlx's OWN tapped Q-rope(tap6)/K-rope(tap7)/V(tap5):
    // scale=1.0, causal, 1 KV head broadcast to 8. Compare to rlx kernel (tap8)
    // and to HF. If recon==HF but tap8!=HF, the rlx attention KERNEL is the bug.
    let (h, nh, hd) = (
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.layer_head_dim(0),
    );
    let _ = h;
    let qr = &outs[11];
    let kr = &outs[12];
    let vv = &outs[10]; // [s, nh*hd], [s, hd], [s, hd]
    let mut recon = vec![0f32; seq * nh * hd];
    for i in 0..seq {
        for hh in 0..nh {
            let q = &qr[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd];
            // scores over j=0..=i
            let mut sc = vec![0f32; i + 1];
            for j in 0..=i {
                let k = &kr[j * hd..(j + 1) * hd];
                sc[j] = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>(); // scale 1.0
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in sc.iter_mut() {
                *s /= sum;
            }
            for j in 0..=i {
                let v = &vv[j * hd..(j + 1) * hd];
                for d in 0..hd {
                    recon[i * nh * hd + hh * hd + d] += sc[j] * v[d];
                }
            }
        }
    }
    // Second reconstruction with scale = head_dim^-0.5 (the kernel's default
    // when score_scale is None) to test whether the kernel is using the wrong scale.
    let mut recon2 = vec![0f32; seq * nh * hd];
    let alt_scale = (hd as f32).powf(-0.5);
    for i in 0..seq {
        for hh in 0..nh {
            let q = &qr[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd];
            let mut sc = vec![0f32; i + 1];
            for j in 0..=i {
                let k = &kr[j * hd..(j + 1) * hd];
                sc[j] = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * alt_scale;
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            for x in sc.iter_mut() {
                *x = (*x - mx).exp();
                s += *x;
            }
            for x in sc.iter_mut() {
                *x /= s;
            }
            for j in 0..=i {
                let v = &vv[j * hd..(j + 1) * hd];
                for d in 0..hd {
                    recon2[i * nh * hd + hh * hd + d] += sc[j] * v[d];
                }
            }
        }
    }
    for ti in 0..seq {
        let r = &recon[ti * qd..(ti + 1) * qd];
        let r2 = &recon2[ti * qd..(ti + 1) * qd];
        eprintln!(
            "[l0] tok {ti} recon(1.0)-vs-kernel cos={:.5} | recon(d^-.5)-vs-kernel cos={:.5} | recon(1.0)-vs-HF cos={:.5}",
            cos(r, &outs[13][ti * qd..(ti + 1) * qd]),
            cos(r2, &outs[13][ti * qd..(ti + 1) * qd]),
            ha.as_ref()
                .map(|a| cos(r, &a[ti * qd..(ti + 1) * qd]))
                .unwrap_or(0.0)
        );
    }
}
