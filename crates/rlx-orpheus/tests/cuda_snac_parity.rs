//! Native RLX CUDA SNAC decoder vs host eager parity.

#![cfg(feature = "cuda")]

use rlx_orpheus::{SnacBackend, SnacLoadOptions, decode_orpheus_codes};
use rlx_runtime::{Device, is_available};

fn snac_path() -> Option<std::path::PathBuf> {
    std::env::var("ORPHEUS_SNAC_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_file())
}

fn golden_frame_codes() -> Option<Vec<i32>> {
    let ref_dir = std::env::var("ORPHEUS_SNAC_REF_DIR")
        .ok()
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_dir())
        .or_else(|| {
            let beside = snac_path()?.parent()?.to_path_buf();
            beside.join("ref_codes.json").is_file().then_some(beside)
        })
        .or_else(|| {
            let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/tts/snac_24khz");
            p.join("ref_codes.json").is_file().then_some(p)
        })?;
    let codes_json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(ref_dir.join("ref_codes.json")).ok()?).ok()?;
    let n = codes_json["codes_0"][0].as_array()?.len();
    let c0: Vec<i32> = codes_json["codes_0"][0]
        .as_array()?
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let c1: Vec<i32> = codes_json["codes_1"][0]
        .as_array()?
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let c2: Vec<i32> = codes_json["codes_2"][0]
        .as_array()?
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let mut frame = Vec::with_capacity(n * 7);
    for j in 0..n {
        frame.push(c0[j]);
        frame.push(c1[2 * j]);
        frame.push(c2[4 * j]);
        frame.push(c2[4 * j + 1]);
        frame.push(c1[2 * j + 1]);
        frame.push(c2[4 * j + 2]);
        frame.push(c2[4 * j + 3]);
    }
    Some(frame)
}

#[test]
fn cuda_snac_matches_eager() {
    if !is_available(Device::Cuda) {
        eprintln!("skip cuda_snac_matches_eager: CUDA unavailable");
        return;
    }
    let Some(path) = snac_path() else {
        eprintln!("skip cuda_snac_matches_eager: set ORPHEUS_SNAC_PATH");
        return;
    };
    let Some(codes) = golden_frame_codes() else {
        eprintln!("skip cuda_snac_matches_eager: set ORPHEUS_SNAC_REF_DIR with ref_codes.json");
        return;
    };

    let eager = SnacBackend::open(
        &path,
        SnacLoadOptions {
            exec: rlx_orpheus::SnacExec::CpuEager,
        },
    )
    .expect("eager");
    let cuda = SnacBackend::open(
        &path,
        SnacLoadOptions {
            exec: rlx_orpheus::SnacExec::Cuda,
        },
    )
    .expect("cuda");
    let ref_pcm = decode_orpheus_codes(&eager, &codes).expect("eager decode");
    let rlx_pcm = decode_orpheus_codes(&cuda, &codes).expect("cuda decode");
    assert_eq!(ref_pcm.len(), rlx_pcm.len(), "length mismatch");
    let maxdiff = ref_pcm
        .iter()
        .zip(&rlx_pcm)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("cuda-vs-eager SNAC maxdiff = {maxdiff:.3e}");
    assert!(
        maxdiff < 0.08,
        "native CUDA SNAC diverged from eager (maxdiff={maxdiff:.3e})"
    );
    let peak = ref_pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(peak > 0.01, "eager peak too low ({peak})");
}
