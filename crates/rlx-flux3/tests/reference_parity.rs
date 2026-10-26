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

//! Parity against Black Forest Labs' reference code, from the dumps of
//! `scripts/flux3_reference.py` (`RLX_FLUX3_REF`), on the package in
//! `RLX_FLUX3_DIR`, on `RLX_FLUX3_DEVICE` (default cpu). Skips without them.

use rlx_flux3::sampler::timesteps;
use rlx_flux3::vae::{AeDecoder, AeEncoder, load_ae};
use rlx_flux3::{
    Flux3Options, Flux3Paths, Flux3Pipeline, Flux3Sample, PackageConfig, ReferenceTokens,
    TextEncoder, TextTokens,
};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::str::FromStr;

fn dirs(sub: &str) -> Option<(PathBuf, Flux3Paths, Device)> {
    let r = rlx_ir::env::var("RLX_FLUX3_REF")?;
    let w = rlx_ir::env::var("RLX_FLUX3_DIR")?;
    let refdir = Path::new(&r).join(sub);
    if !refdir.join("meta.json").is_file() {
        eprintln!("skip: no {}", refdir.display());
        return None;
    }
    let dev = rlx_ir::env::var("RLX_FLUX3_DEVICE").unwrap_or_else(|| "cpu".into());
    Some((
        refdir,
        Flux3Paths::discover(Path::new(&w)).unwrap(),
        Device::from_str(&dev).unwrap(),
    ))
}

fn f32s(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
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

/// `(max |a−b| / max |b|, ‖a−b‖ / ‖b‖)`.
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

fn text_tokens(d: &Path, name: &str, width: usize) -> TextTokens {
    let tokens = f32s(&d.join(format!("{name}_tokens.bin")));
    let len = tokens.len() / width;
    let real = u32s(&d.join(format!("{name}_ids.bin"))).len();
    TextTokens { tokens, len, real }
}

#[test]
fn text_encoder_matches_reference() {
    let Some((d, paths, dev)) = dirs("text") else {
        return;
    };
    let m = meta(&d);
    let pkg = PackageConfig::load(&paths.root).unwrap();
    let mut te = TextEncoder::open(
        &paths.text_encoder,
        pkg.text_layers(),
        pkg.pad_multiple(),
        4096,
        dev,
        None,
    )
    .unwrap();
    let width = te.context_dim();
    let caption = m["caption"].as_str().unwrap();
    for (name, prompt) in [("uncond", ""), ("cond", caption)] {
        let want_ids = u32s(&d.join(format!("{name}_ids.bin")));
        let ids = te.token_ids(prompt).unwrap();
        assert_eq!(ids, want_ids, "{name}: token ids");
        let want = text_tokens(&d, name, width);
        let got = te.encode_ids(&ids).unwrap();
        assert_eq!(
            (got.len, got.real),
            (want.len, want.real),
            "{name}: lengths"
        );
        let (mx, rms) = err(&got.tokens, &want.tokens);
        eprintln!(
            "text {name}: len {} (real {}): max-rel {mx:.2e} rms-rel {rms:.2e}",
            got.len, got.real
        );
        // The padding rows separately: they are the subtle part.
        if got.len > got.real {
            let (pmx, prms) = err(
                &got.tokens[got.real * width..],
                &want.tokens[got.real * width..],
            );
            eprintln!("text {name}: pad rows max-rel {pmx:.2e} rms-rel {prms:.2e}");
            assert!(prms < 5e-3, "{name}: pad rows rms-rel {prms:.2e}");
        }
        assert!(rms < 5e-3, "{name}: text tokens rms-rel {rms:.2e}");
    }
}

fn open(paths: Flux3Paths, dev: Device) -> Flux3Pipeline {
    Flux3Pipeline::open(
        paths,
        Flux3Options {
            device: dev,
            ..Flux3Options::default()
        },
    )
    .unwrap()
}

#[test]
fn dit_and_decoder_match_reference() {
    let Some((d, paths, dev)) = dirs("t2i") else {
        return;
    };
    let td = d.parent().unwrap().join("text");
    let m = meta(&d);
    let mut pipe = open(paths, dev);
    let width = pipe.cfg.context_dim;
    let texts = [
        text_tokens(&td, "uncond", width),
        text_tokens(&td, "cond", width),
    ];
    let size = m["size"].as_u64().unwrap() as usize;
    let steps = m["steps"].as_u64().unwrap() as usize;
    let alpha = m["alpha"].as_f64().unwrap() as f32;
    let guidance = m["guidance"].as_f64().unwrap() as f32;
    let noise = f32s(&d.join("noise.bin"));

    // Our own noise draw is torch's.
    let ours = rlx_flux3::sampler::initial_noise(noise.len(), m["seed"].as_u64().unwrap());
    let (nmx, _) = err(&ours, &noise);
    eprintln!("noise: max-rel {nmx:.2e}");
    assert!(nmx < 1e-6, "torch.randn reproduction");
    let sched = timesteps(steps, alpha);
    let want_sched: Vec<f32> = m["schedule"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(sched, want_sched, "schedule");

    let g = size / 16;
    let mut prep = pipe
        .prepare(&texts, &ReferenceTokens::default(), g, g)
        .unwrap();
    let v = pipe.velocity(&mut prep, &noise, sched[0], None).unwrap();
    let want_v = f32s(&d.join("velocity0.bin"));
    let half = v.len() / 2;
    let (umx, urms) = err(&v[..half], &want_v[..half]);
    let (cmx, crms) = err(&v[half..], &want_v[half..]);
    eprintln!(
        "velocity0 uncond max-rel {umx:.2e} rms-rel {urms:.2e}; cond max-rel {cmx:.2e} rms-rel {crms:.2e}"
    );
    assert!(urms < 2e-3 && crms < 2e-3, "first-step velocity");
    drop(prep);

    let sample = Flux3Sample {
        width: Some(size),
        height: Some(size),
        steps: Some(steps),
        guidance: Some(guidance),
        alpha: Some(alpha),
        noise: Some(noise),
        ..Flux3Sample::default()
    };
    let lat = pipe
        .denoise(&texts, &ReferenceTokens::default(), &sample)
        .unwrap();
    let want_lat = f32s(&d.join("latents.bin"));
    let (lmx, lrms) = err(&lat, &want_lat);
    eprintln!("latents after {steps} steps: max-rel {lmx:.2e} rms-rel {lrms:.2e}");
    assert!(lrms < 1e-2, "latents");
    pipe.release_dit();

    // Decoder on the reference's own latents.
    let (acfg, aw) = load_ae(&pipe.paths.ae).unwrap();
    let mut dec = AeDecoder::compile(&acfg, &aw, g, g, dev).unwrap();
    let rgb = dec.decode(&acfg, &aw, &want_lat).unwrap();
    let want_rgb = f32s(&d.join("rgb.bin"));
    let (rmx, rrms) = err(&rgb, &want_rgb);
    eprintln!("decoder: max-rel {rmx:.2e} rms-rel {rrms:.2e}");
    assert!(rrms < 1e-3, "decoder");
}

#[test]
fn reference_conditioning_matches_reference() {
    let Some((d, paths, dev)) = dirs("edit") else {
        return;
    };
    let td = d.parent().unwrap().join("text");
    let m = meta(&d);
    let size = m["size"].as_u64().unwrap() as usize;
    let g = size / 16;
    let rgb8 = std::fs::read(d.join("ref_rgb8.bin")).unwrap();
    let (acfg, aw) = load_ae(&paths.ae).unwrap();
    let mut enc = AeEncoder::compile(&acfg, &aw, size, size, dev).unwrap();
    let tokens = enc.encode_rgb8(&acfg, &aw, &rgb8).unwrap();
    let want_tok = f32s(&d.join("ref_tokens.bin"));
    let (emx, erms) = err(&tokens, &want_tok);
    eprintln!("encoder: max-rel {emx:.2e} rms-rel {erms:.2e}");
    assert!(erms < 1e-3, "encoder");
    drop(enc);

    let t2i = d.parent().unwrap().join("t2i");
    let tm = meta(&t2i);
    let mut pipe = open(paths, dev);
    let width = pipe.cfg.context_dim;
    let texts = [
        text_tokens(&td, "uncond", width),
        text_tokens(&td, "cond", width),
    ];
    let cond = ReferenceTokens {
        tokens: want_tok
            .iter()
            .map(|&v| rlx_flux3::embed::round_bf16(v))
            .collect(),
        ids: rlx_flux3::rope::latent_ids(g, g, pipe.cfg.image_rope_scale, -1.0),
    };
    let steps = tm["steps"].as_u64().unwrap() as usize;
    let alpha = tm["alpha"].as_f64().unwrap() as f32;
    let sched = timesteps(steps, alpha);
    let noise = f32s(&t2i.join("noise.bin"));
    let mut prep = pipe.prepare(&texts, &cond, g, g).unwrap();
    let v = pipe.velocity(&mut prep, &noise, sched[0], None).unwrap();
    let want_v = f32s(&d.join("velocity0.bin"));
    let (mx, rms) = err(&v, &want_v);
    eprintln!("edit velocity0: max-rel {mx:.2e} rms-rel {rms:.2e}");
    assert!(rms < 2e-3, "edit velocity");
}

#[test]
fn flow_map_matches_reference() {
    let Some((d, _, dev)) = dirs("flow_map") else {
        return;
    };
    let Some(fm) = rlx_ir::env::var("RLX_FLUX3_FLOW_MAP_DIR") else {
        eprintln!("skip: RLX_FLUX3_FLOW_MAP_DIR not set");
        return;
    };
    let td = d.parent().unwrap().join("text");
    let m = meta(&d);
    let mut pipe = open(Flux3Paths::discover(Path::new(&fm)).unwrap(), dev);
    assert!(pipe.cfg.flow_map, "{fm} is not the distilled checkpoint");
    let texts = [text_tokens(&td, "cond", pipe.cfg.context_dim)];
    let size = m["size"].as_u64().unwrap() as usize;
    let steps = m["steps"].as_u64().unwrap() as usize;
    let alpha = m["alpha"].as_f64().unwrap() as f32;
    let g = size / 16;
    let noise = rlx_flux3::sampler::initial_noise(
        g * g * pipe.cfg.in_channels,
        m["seed"].as_u64().unwrap(),
    );
    let sched = timesteps(steps, alpha);
    let mut prep = pipe
        .prepare(&texts, &ReferenceTokens::default(), g, g)
        .unwrap();
    let v = pipe
        .velocity(&mut prep, &noise, sched[0], Some(sched[1]))
        .unwrap();
    let (mx, rms) = err(&v, &f32s(&d.join("velocity0.bin")));
    eprintln!("flow-map velocity0: max-rel {mx:.2e} rms-rel {rms:.2e}");
    assert!(rms < 2e-3, "flow-map velocity");
    drop(prep);
    let sample = Flux3Sample {
        width: Some(size),
        height: Some(size),
        steps: Some(steps),
        alpha: Some(alpha),
        noise: Some(noise),
        ..Flux3Sample::default()
    };
    let lat = pipe
        .denoise(&texts, &ReferenceTokens::default(), &sample)
        .unwrap();
    let (lmx, lrms) = err(&lat, &f32s(&d.join("latents.bin")));
    eprintln!("flow-map latents after {steps} steps: max-rel {lmx:.2e} rms-rel {lrms:.2e}");
    assert!(lrms < 1e-2, "flow-map latents");
}
