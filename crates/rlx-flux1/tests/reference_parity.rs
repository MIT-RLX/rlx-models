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

//! Parity against `scripts/flux1_reference.py` dumps (diffusers/transformers).
//!
//! Gated on `RLX_FLUX1_REF` (dump root holding `text/`, `dit_lora/`, `vae/`)
//! and `RLX_FLUX1_DIR` (weights, as `just fetch-flux1-dev` lays them out);
//! `RLX_FLUX1_DEVICE` picks the backend (default cpu). Skips otherwise.

use rlx_flux1::config::{ClipTextConfig, T5Config};
use rlx_flux1::{DitGeometry, DitSource, Flux1Config, Flux1Dit, Flux1Paths, LoraSet, TensorFile};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::str::FromStr;

fn dirs(sub: &str) -> Option<(PathBuf, Flux1Paths, Device)> {
    let r = rlx_ir::env::var("RLX_FLUX1_REF")?;
    let w = rlx_ir::env::var("RLX_FLUX1_DIR")?;
    let refdir = Path::new(&r).join(sub);
    if !refdir.join("meta.json").is_file() {
        eprintln!("skip: no {}", refdir.display());
        return None;
    }
    let dev = rlx_ir::env::var("RLX_FLUX1_DEVICE").unwrap_or_else(|| "cpu".into());
    Some((
        refdir,
        Flux1Paths::discover(Path::new(&w)).unwrap(),
        Device::from_str(&dev).unwrap(),
    ))
}

fn f32s(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn u32s(p: &Path) -> Vec<u32> {
    f32s(p).into_iter().map(f32::to_bits).collect()
}

fn meta(d: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(d.join("meta.json")).unwrap()).unwrap()
}

/// `(max |a-b| / max |b|, ‖a-b‖ / ‖b‖)`.
fn err(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut d2, mut b2, mut dmax, mut bmax) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x as f64 - y as f64).abs();
        d2 += d * d;
        b2 += (y as f64) * (y as f64);
        dmax = dmax.max(d);
        bmax = bmax.max((y as f64).abs());
    }
    (dmax / bmax, (d2 / b2).sqrt())
}

#[test]
fn text_encoders_match_transformers() {
    let Some((d, paths, dev)) = dirs("text") else {
        return;
    };
    let m = meta(&d);
    let t5_ids = u32s(&d.join("t5_ids.bin"));
    let clip_ids = u32s(&d.join("clip_ids.bin"));

    // Tokenizers: our ids vs transformers'.
    let tok = tokenizers::Tokenizer::from_file(paths.t5.join("tokenizer.json")).unwrap();
    let prompt = m["prompt"].as_str().unwrap();
    assert_eq!(
        rlx_flux1::t5::t5_ids(&tok, prompt, t5_ids.len()).unwrap(),
        t5_ids,
        "T5 ids"
    );
    let ctok = tokenizers::Tokenizer::from_file(paths.clip.join("tokenizer.json")).unwrap();
    assert_eq!(
        rlx_flux1::clip::clip_ids(&ctok, prompt, 77, 49407).unwrap(),
        clip_ids,
        "CLIP ids"
    );

    let ccfg = ClipTextConfig::from_json(
        &std::fs::read_to_string(paths.clip.join("config.json")).unwrap(),
    )
    .unwrap();
    let mut clip = rlx_flux1::clip::ClipTextEncoder::compile(&paths.clip, ccfg, dev).unwrap();
    let pooled = clip.pooled(&clip_ids).unwrap();
    let e = err(&pooled, &f32s(&d.join("pooled.bin")));
    eprintln!("CLIP pooled: max {:.2e} rel {:.2e}", e.0, e.1);
    assert!(e.1 < 1e-4, "CLIP pooled {e:?}");
    drop(clip);

    let tcfg: T5Config =
        serde_json::from_str(&std::fs::read_to_string(paths.t5.join("config.json")).unwrap())
            .unwrap();
    let t5_quant = rlx_ir::env::var("RLX_FLUX1_T5_QUANT")
        .map(|v| rlx_flux1::weights::WeightQuant::parse(&v).unwrap())
        .unwrap_or(None);
    let mut t5 =
        rlx_flux1::t5::T5Encoder::compile_quant(&paths.t5, tcfg, t5_ids.len(), dev, 6, t5_quant)
            .unwrap();
    let hidden = t5.encode(&t5_ids).unwrap();
    let e = err(&hidden, &f32s(&d.join("t5_hidden.bin")));
    eprintln!("T5 hidden: max {:.2e} rel {:.2e}", e.0, e.1);
    assert!(e.1 < 1e-3, "T5 hidden {e:?}");
}

#[test]
fn truncated_denoiser_with_lora_matches_diffusers() {
    let Some((d, paths, dev)) = dirs("dit_lora") else {
        return;
    };
    let m = meta(&d);
    let file = TensorFile::open(&paths.transformer).unwrap();
    let mut cfg = Flux1Config::from_shapes(|k| file.shape(k)).unwrap();
    cfg.depth = m["doubles"].as_u64().unwrap() as usize;
    cfg.depth_single = m["singles"].as_u64().unwrap() as usize;
    let mut lora = LoraSet::default();
    if let Some(l) = m["lora"].as_str() {
        let modules: Vec<String> = file
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_suffix(".weight").map(str::to_string))
            .collect();
        let lp = if Path::new(l).is_file() {
            PathBuf::from(l)
        } else {
            paths
                .transformer
                .parent()
                .unwrap()
                .join("lora")
                .join(Path::new(l).file_name().unwrap())
        };
        lora.add_file(
            &lp,
            m["lora_scale"].as_f64().unwrap() as f32,
            &cfg,
            &modules,
        )
        .unwrap();
    }
    let geo = DitGeometry {
        txt_len: m["txt_len"].as_u64().unwrap() as usize,
        grid_h: m["grid_h"].as_u64().unwrap() as usize,
        grid_w: m["grid_w"].as_u64().unwrap() as usize,
    };
    let mut src = DitSource {
        file: &file,
        lora: &lora,
        cfg: &cfg,
        quant: rlx_ir::env::var("RLX_FLUX1_WEIGHT_QUANT")
            .map(|v| rlx_flux1::weights::WeightQuant::parse(&v).unwrap())
            .unwrap_or(None),
    };
    // One block per graph exercises the chunk hand-off too.
    let mut dit = Flux1Dit::compile(&cfg, geo, &mut src, dev, 1, 1).unwrap();
    let out = dit
        .forward(
            &f32s(&d.join("img.bin")),
            &f32s(&d.join("txt.bin")),
            &f32s(&d.join("pooled.bin")),
            m["t"].as_f64().unwrap() as f32,
            m["guidance"].as_f64().unwrap() as f32,
        )
        .unwrap();
    let e = err(&out, &f32s(&d.join("out.bin")));
    eprintln!(
        "denoiser ({}+{} blocks, LoRA {}): max {:.2e} rel {:.2e}",
        cfg.depth,
        cfg.depth_single,
        !lora.is_empty(),
        e.0,
        e.1
    );
    assert!(e.1 < 1e-4, "denoiser {e:?}");
}

#[test]
fn vae_decoder_matches_diffusers() {
    let Some((d, paths, dev)) = dirs("vae") else {
        return;
    };
    let m = meta(&d);
    let shape: Vec<usize> = serde_json::from_value(m["latents"].clone()).unwrap();
    let (cfg, w) = rlx_flux1::vae::load_flux1_vae(&paths.vae).unwrap();
    let mut dec =
        rlx_flux1::vae::Flux1VaeDecoder::compile(&cfg, &w, shape[1], shape[2], dev).unwrap();
    let rgb = dec.decode(&f32s(&d.join("latents.bin"))).unwrap();
    let e = err(&rgb, &f32s(&d.join("rgb.bin")));
    eprintln!("VAE: max {:.2e} rel {:.2e}", e.0, e.1);
    assert!(e.1 < 1e-4, "VAE {e:?}");
}

/// The web export (`rlx_flux1::web`): a denoiser built from the recorded GGUF,
/// read back from bytes as a browser does, gives exactly the output of the one
/// built from the checkpoint — Q8_0 projections with the LoRA merged in.
#[test]
fn web_export_replays_bit_identically() {
    let Some((d, paths, dev)) = dirs("dit_lora") else {
        return;
    };
    let m = meta(&d);
    let file = TensorFile::open(&paths.transformer).unwrap();
    let mut cfg = Flux1Config::from_shapes(|k| file.shape(k)).unwrap();
    cfg.depth = m["doubles"].as_u64().unwrap() as usize;
    cfg.depth_single = m["singles"].as_u64().unwrap() as usize;
    let modules: Vec<String> = file
        .keys()
        .into_iter()
        .filter_map(|k| k.strip_suffix(".weight").map(str::to_string))
        .collect();
    let mut lora = LoraSet::default();
    let l = m["lora"].as_str().unwrap();
    lora.add_file(
        Path::new(l),
        m["lora_scale"].as_f64().unwrap() as f32,
        &cfg,
        &modules,
    )
    .unwrap();
    let geo = DitGeometry {
        txt_len: m["txt_len"].as_u64().unwrap() as usize,
        grid_h: m["grid_h"].as_u64().unwrap() as usize,
        grid_w: m["grid_w"].as_u64().unwrap() as usize,
    };
    let mut src = DitSource {
        file: &file,
        lora: &lora,
        cfg: &cfg,
        quant: Some(rlx_flux1::weights::WeightQuant::Q8_0),
    };
    let (img, txt, pooled) = (
        f32s(&d.join("img.bin")),
        f32s(&d.join("txt.bin")),
        f32s(&d.join("pooled.bin")),
    );
    let (t, g) = (
        m["t"].as_f64().unwrap() as f32,
        m["guidance"].as_f64().unwrap() as f32,
    );
    let mut rec = rlx_flux1::web::Recorder::new(&mut src);
    let want = Flux1Dit::compile(&cfg, geo, &mut rec, dev, 1, 1)
        .unwrap()
        .forward(&img, &txt, &pooled, t, g)
        .unwrap();
    let path = std::env::temp_dir().join("rlx_flux1_web_export_test.gguf");
    let n = rec.tensors.len();
    rlx_flux1::web::write_gguf(
        &path,
        std::mem::take(&mut rec.tensors).into_values(),
        rlx_flux1::web::config_meta(&cfg),
    )
    .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let gguf = rlx_gguf::GgufFile::from_bytes(bytes.clone()).unwrap();
    assert_eq!(rlx_flux1::web::config_from_meta(&gguf).unwrap(), cfg);
    let mut replay = rlx_flux1::web::Replay::new(gguf);
    let got = Flux1Dit::compile(&cfg, geo, &mut replay, dev, 1, 1)
        .unwrap()
        .forward(&img, &txt, &pooled, t, g)
        .unwrap();
    eprintln!(
        "web export: {n} tensors; replayed output {} values",
        got.len()
    );
    assert_eq!(
        want, got,
        "replayed denoiser differs from the checkpoint-built one"
    );

    // Split across 3 stages, each built from only its own tensors, fetched
    // as byte ranges of the file — what each browser worker does.
    let count = 3;
    let header = rlx_gguf::GgufFile::header_from_bytes(&bytes).unwrap();
    let mut x: Vec<f32> = Vec::new();
    let mut vec_c: Vec<f32> = Vec::new();
    for i in 0..count {
        let parts = rlx_flux1::web::stage_parts(&cfg, i, count).unwrap();
        let keep = |name: &str| rlx_flux1::web::stage_keeps(&parts, name);
        let chunks: Vec<(u64, Vec<u8>)> = header
            .tensor_ranges(keep, 1 << 20)
            .unwrap()
            .into_iter()
            .map(|r| (r.start, bytes[r.start as usize..r.end as usize].to_vec()))
            .collect();
        let file = rlx_gguf::GgufFile::header_from_bytes(&bytes)
            .unwrap()
            .with_tensor_chunks(keep, &chunks)
            .unwrap();
        let mut replay = rlx_flux1::web::Replay::new(file);
        let mut stage = rlx_flux1::web::Flux1DitStage::compile(
            &cfg,
            geo,
            &mut replay,
            dev,
            parts.clone(),
            i + 1,
        )
        .unwrap();
        x = if stage.is_first() {
            vec_c = stage.vec(t, g, &pooled).unwrap();
            stage.forward(&img, &txt, &vec_c).unwrap()
        } else {
            stage.forward(&x, &[], &vec_c).unwrap()
        };
        eprintln!("stage {i}/{count}: {parts:?}");
    }
    assert_eq!(want, x, "3-stage pipeline differs from the whole denoiser");
}

/// The web T5 (`rlx_flux1::web::export_t5`): Q8_0, split into 2 stages each
/// built from its own byte ranges, encodes as the native Q8_0 encoder does.
#[test]
fn web_t5_export_matches_encoder() {
    let Some((d, paths, dev)) = dirs("text") else {
        return;
    };
    let t5_ids = u32s(&d.join("t5_ids.bin"));
    let len = t5_ids.len();
    let q = rlx_flux1::weights::WeightQuant::Q8_0;
    let path = std::env::temp_dir().join("rlx_flux1_web_t5_test.gguf");
    let n = rlx_flux1::web::export_t5(&paths.t5, q, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let header = rlx_gguf::GgufFile::header_from_bytes(&bytes).unwrap();
    let (tcfg, embed_key) = rlx_flux1::web::t5_config_from_meta(&header).unwrap();
    let count = 2;
    let mut x: Vec<f32> = Vec::new();
    for i in 0..count {
        let layers = rlx_flux1::web::t5_stage_layers(&tcfg, i, count).unwrap();
        let keep = |name: &str| rlx_flux1::web::t5_stage_keeps(&tcfg, &embed_key, &layers, name);
        let chunks: Vec<(u64, Vec<u8>)> = header
            .tensor_ranges(keep, 1 << 20)
            .unwrap()
            .into_iter()
            .map(|r| (r.start, bytes[r.start as usize..r.end as usize].to_vec()))
            .collect();
        let file = rlx_gguf::GgufFile::header_from_bytes(&bytes)
            .unwrap()
            .with_tensor_chunks(keep, &chunks)
            .unwrap();
        let mut stage = rlx_flux1::web::T5WebStage::compile(file, len, i, count, dev).unwrap();
        if stage.is_first() {
            x = stage.embed(&t5_ids).unwrap();
        }
        x = stage.forward(&x).unwrap();
    }
    drop(bytes);
    let mut t5 =
        rlx_flux1::t5::T5Encoder::compile_quant(&paths.t5, tcfg, len, dev, 6, Some(q)).unwrap();
    let want = t5.encode(&t5_ids).unwrap();
    let e = err(&x, &want);
    eprintln!(
        "web T5 ({n} tensors, {count} stages) vs Q8_0 encoder: max {:.2e} rel {:.2e}",
        e.0, e.1
    );
    assert!(e.1 < 1e-5, "web T5 {e:?}");
}

/// The web CLIP + VAE file (`rlx_flux1::web::export_aux`) gives the native
/// pooled embedding and VAE decode.
#[test]
fn web_aux_export_matches_native() {
    let Some((d, paths, dev)) = dirs("text") else {
        return;
    };
    let clip_ids = u32s(&d.join("clip_ids.bin"));
    let path = std::env::temp_dir().join("rlx_flux1_web_aux_test.gguf");
    let n = rlx_flux1::web::export_aux(&paths.clip, &paths.vae, &path).unwrap();
    let file = rlx_gguf::GgufFile::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    let (lh, lw) = (20, 20);
    let mut aux = rlx_flux1::web::AuxStage::compile(file, lh, lw, dev).unwrap();
    let got = aux.pooled(&clip_ids).unwrap();
    let ccfg = ClipTextConfig::from_json(
        &std::fs::read_to_string(paths.clip.join("config.json")).unwrap(),
    )
    .unwrap();
    let want = rlx_flux1::clip::ClipTextEncoder::compile(&paths.clip, ccfg, dev)
        .unwrap()
        .pooled(&clip_ids)
        .unwrap();
    assert_eq!(got, want, "web CLIP pooled differs");
    let latents: Vec<f32> = (0..16 * lh * lw)
        .map(|i| ((i as f32) * 0.37).sin())
        .collect();
    let got = aux.decode_rgb8(&latents).unwrap();
    let (vcfg, vw) = rlx_flux1::vae::load_flux1_vae(&paths.vae).unwrap();
    let rgb = rlx_flux1::vae::Flux1VaeDecoder::compile(&vcfg, &vw, lh, lw, dev)
        .unwrap()
        .decode(&latents)
        .unwrap();
    let want = rlx_flux1::vae::rgb_to_u8(&rgb, lh * 8, lw * 8);
    assert_eq!(got, want, "web VAE decode differs");
    eprintln!("web aux: {n} tensors; CLIP pooled and VAE decode identical");
}
