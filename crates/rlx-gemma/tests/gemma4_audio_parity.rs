// RLX — GPLv3. Audio-tower parity vs HF no-SRQ, staged by component.
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_gemma::gemma4_audio::{
    AudioConfig, audio_block_mask, audio_rel_pos, build_attention_test, build_audio_features,
    build_audio_layer0_debug, build_audio_subsample, build_audio_tower, build_depthwise_test,
};
use rlx_gemma::qat_loader::GemmaQatLoader;
use rlx_runtime::{Device, is_available};
use std::path::{Path, PathBuf};

fn dir() -> Option<PathBuf> {
    let h = std::env::var_os("HOME")?;
    let b = Path::new(&h).join(
        ".cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots",
    );
    let s = std::fs::read_dir(&b).ok()?.flatten().next()?.path();
    s.join("config.json").is_file().then_some(s)
}
fn rd(p: &Path) -> Option<Vec<f32>> {
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
fn audio_subsample_parity() {
    let Some(d) = dir() else {
        eprintln!("[audio sub] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(feats) = rd(&fx.join("feats.bin")) else {
        eprintln!("[audio sub] no fixtures — skip");
        return;
    };
    let hf = rd(&fx.join("subsample.bin")).expect("subsample.bin");

    let cfg = AudioConfig::default();
    let t = feats.len() / cfg.feature_size; // 48
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let (g, params, seq) = build_audio_subsample(&cfg, &mut loader, 1, t).expect("build subsample");
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, params).unwrap();
    let outs = c.run(&[("audio_feats", feats.as_slice())]);
    let out = &outs[0];
    let h = cfg.hidden;
    assert_eq!(out.len(), seq * h, "subsample output len");
    let mut worst = 1.0f64;
    let mut maxdiff = 0.0f32;
    for s in 0..seq {
        let r = &out[s * h..(s + 1) * h];
        let f = &hf[s * h..(s + 1) * h];
        let cv = cos(r, f);
        if cv < worst {
            worst = cv;
        }
        for (a, b) in r.iter().zip(f) {
            maxdiff = maxdiff.max((a - b).abs());
        }
        if s < 3 || cv < 0.999 {
            eprintln!("[audio sub] frame {s} cos={cv:.6}");
        }
    }
    eprintln!("[audio sub] seq {seq}, worst cos = {worst:.6}, maxabs = {maxdiff:.5}");
    assert!(worst > 0.99, "subsample diverges: worst cos {worst}");
}

fn run_tower(
    d: &Path,
    feats: &[f32],
    cfg: &AudioConfig,
    n_layers: usize,
    with_proj: bool,
) -> (Vec<f32>, usize) {
    run_tower_dev(Device::Cpu, d, feats, cfg, n_layers, with_proj)
}

fn run_tower_dev(
    dev: Device,
    d: &Path,
    feats: &[f32],
    cfg: &AudioConfig,
    n_layers: usize,
    with_proj: bool,
) -> (Vec<f32>, usize) {
    let t = feats.len() / cfg.feature_size;
    let mut loader = GemmaQatLoader::open(d).unwrap();
    let (g, params, seq) =
        build_audio_tower(cfg, &mut loader, 1, t, n_layers, with_proj).expect("build tower");
    let rel = audio_rel_pos(cfg);
    let (mask, _nb) = audio_block_mask(cfg, seq);
    let mut c = compile_graph_gemma_prefill_with_params(dev, g, params).unwrap();
    let outs = c.run(&[
        ("audio_feats", feats),
        ("audio_rel_pos", rel.as_slice()),
        ("audio_mask", mask.as_slice()),
    ]);
    (outs[0].clone(), seq)
}

fn report(tag: &str, out: &[f32], hf: &[f32], dim: usize, seq: usize) -> f64 {
    let mut worst = 1.0f64;
    let mut maxdiff = 0.0f32;
    for s in 0..seq {
        let r = &out[s * dim..(s + 1) * dim];
        let f = &hf[s * dim..(s + 1) * dim];
        let cv = cos(r, f);
        if cv < worst {
            worst = cv;
        }
        for (a, b) in r.iter().zip(f) {
            maxdiff = maxdiff.max((a - b).abs());
        }
        if s < 3 || cv < 0.999 {
            eprintln!("[{tag}] frame {s} cos={cv:.6}");
        }
    }
    eprintln!("[{tag}] seq {seq}, worst cos = {worst:.6}, maxabs = {maxdiff:.5}");
    worst
}

#[test]
fn audio_layer0_parity() {
    let Some(d) = dir() else {
        eprintln!("[audio l0] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(feats) = rd(&fx.join("feats.bin")) else {
        eprintln!("[audio l0] no fixtures — skip");
        return;
    };
    let hf = rd(&fx.join("layer0.bin")).expect("layer0.bin");
    let cfg = AudioConfig::default();
    let (out, seq) = run_tower(&d, &feats, &cfg, 1, false);
    let worst = report("audio l0", &out, &hf, cfg.hidden, seq);
    assert!(worst > 0.99, "layer0 diverges: worst cos {worst}");
}

#[test]
fn audio_layer0_bisect() {
    let Some(d) = dir() else {
        eprintln!("[audio bis] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(feats) = rd(&fx.join("feats.bin")) else {
        eprintln!("[audio bis] no fixtures — skip");
        return;
    };
    let cfg = AudioConfig::default();
    let t = feats.len() / cfg.feature_size;
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let (g, params, seq) = build_audio_layer0_debug(&cfg, &mut loader, 1, t).expect("build l0 dbg");
    let rel = audio_rel_pos(&cfg);
    let (mask, _nb) = audio_block_mask(&cfg, seq);
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, params).unwrap();
    let outs = c.run(&[
        ("audio_feats", feats.as_slice()),
        ("audio_rel_pos", rel.as_slice()),
        ("audio_mask", mask.as_slice()),
    ]);
    let h = cfg.hidden;
    for (i, tag) in ["ff1", "attn", "lc", "ff2", "nout", "attn_in"]
        .iter()
        .enumerate()
    {
        let fn_ = match *tag {
            "nout" => "layer0".to_string(),
            "attn_in" => "l0_attn_in".to_string(),
            _ => format!("l0_{tag}"),
        };
        let Some(hf) = rd(&fx.join(format!("{fn_}.bin"))) else {
            eprintln!("  no {fn_}.bin");
            continue;
        };
        let worst = report(&format!("audio {tag}"), &outs[i], &hf, h, seq);
        eprintln!("  >>> {tag} worst cos {worst:.6}");
    }
}

#[test]
fn audio_attention_isolated() {
    let Some(d) = dir() else {
        eprintln!("[att] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(ain) = rd(&fx.join("l0_attn_in.bin")) else {
        eprintln!("[att] no fixtures — skip");
        return;
    };
    let aout = rd(&fx.join("l0_attn_out.bin")).expect("l0_attn_out.bin");
    let cfg = AudioConfig::default();
    let h = cfg.hidden;
    let seq = ain.len() / h;
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let (g, params) = build_attention_test(&cfg, &mut loader, 0, 1, seq).expect("build att");
    let rel = audio_rel_pos(&cfg);
    let (mask, _nb) = audio_block_mask(&cfg, seq);
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, params).unwrap();
    let outs = c.run(&[
        ("attn_in", ain.as_slice()),
        ("audio_rel_pos", rel.as_slice()),
        ("audio_mask", mask.as_slice()),
    ]);
    let worst = report("att iso", &outs[0], &aout, h, seq);
    assert!(
        worst > 0.999,
        "isolated attention diverges: worst cos {worst}"
    );
}

#[test]
fn audio_depthwise_isolated() {
    let Some(d) = dir() else {
        eprintln!("[dw] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(din) = rd(&fx.join("l0_dwin.bin")) else {
        eprintln!("[dw] no fixtures — skip");
        return;
    };
    let dout = rd(&fx.join("l0_dwout.bin")).expect("l0_dwout.bin");
    let cfg = AudioConfig::default();
    let h = cfg.hidden;
    let seq = din.len() / h; // [B=1, h, seq]
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let (g, params) = build_depthwise_test(&cfg, &mut loader, 0, 1, seq).expect("build dw");
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, params).unwrap();
    let outs = c.run(&[("dw_in", din.as_slice())]);
    let out = &outs[0];
    let mut maxdiff = 0.0f32;
    for (a, b) in out.iter().zip(&dout) {
        maxdiff = maxdiff.max((a - b).abs());
    }
    let cv = cos(out, &dout);
    eprintln!("[dw] depthwise conv: cos={cv:.6}, maxabs={maxdiff:.6}");
    eprintln!("[dw] mine[0..3]={:?}", &out[..3]);
    eprintln!("[dw] hf  [0..3]={:?}", &dout[..3]);
    assert!(cv > 0.9999, "depthwise conv diverges: cos {cv}");
}

#[test]
fn audio_full_parity_metal() {
    if !is_available(Device::Metal) {
        eprintln!("[audio full metal] no Metal — skip");
        return;
    }
    let Some(d) = dir() else {
        eprintln!("[audio full metal] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(feats) = rd(&fx.join("feats.bin")) else {
        eprintln!("[audio full metal] no fixtures — skip");
        return;
    };
    let hf = rd(&fx.join("out.bin")).expect("out.bin");
    let cfg = AudioConfig::default();
    let (out, seq) = run_tower_dev(Device::Metal, &d, &feats, &cfg, cfg.layers, true);
    let worst = report("audio full metal", &out, &hf, cfg.out_dims, seq);
    assert!(
        worst > 0.99,
        "audio tower on Metal diverges: worst cos {worst}"
    );
}

#[test]
fn audio_features_parity() {
    let Some(d) = dir() else {
        eprintln!("[audio feats] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(feats) = rd(&fx.join("feats.bin")) else {
        eprintln!("[audio feats] no fixtures — skip");
        return;
    };
    let hf = rd(&fx.join("feat_out.bin")).expect("feat_out.bin");
    let cfg = AudioConfig::default();
    let t = feats.len() / cfg.feature_size;
    let mut loader = GemmaQatLoader::open(&d).unwrap();
    let (g, params, seq) =
        build_audio_features(&cfg, &mut loader, 1, t).expect("build audio features");
    let rel = audio_rel_pos(&cfg);
    let (mask, _nb) = audio_block_mask(&cfg, seq);
    let mut c = compile_graph_gemma_prefill_with_params(Device::Cpu, g, params).unwrap();
    let outs = c.run(&[
        ("audio_feats", feats.as_slice()),
        ("audio_rel_pos", rel.as_slice()),
        ("audio_mask", mask.as_slice()),
    ]);
    let dim = hf.len() / seq;
    let worst = report("audio feats", &outs[0], &hf, dim, seq);
    assert!(worst > 0.99, "audio features diverge: worst cos {worst}");
}

#[test]
fn audio_full_parity() {
    let Some(d) = dir() else {
        eprintln!("[audio full] no ckpt — skip");
        return;
    };
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gemma4_audio");
    let Some(feats) = rd(&fx.join("feats.bin")) else {
        eprintln!("[audio full] no fixtures — skip");
        return;
    };
    let hf = rd(&fx.join("out.bin")).expect("out.bin");
    let cfg = AudioConfig::default();
    let (out, seq) = run_tower(&d, &feats, &cfg, cfg.layers, true);
    let worst = report("audio full", &out, &hf, cfg.out_dims, seq);
    assert!(worst > 0.99, "full audio tower diverges: worst cos {worst}");
}
