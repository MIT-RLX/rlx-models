// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Backend quick check for PP-OCRv6 (env-gated weights).

use rlx_ppocrv6::{PpOcrV6Runner, Tier};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn model_dir(tier: &str) -> Option<PathBuf> {
    let p = workspace_root().join(format!(".cache/ppocrv6/{tier}"));
    let det = p.join("det/model.safetensors").is_file()
        || p.join(format!("det/ppocrv6_{tier}_det.safetensors")).is_file();
    let rec = p.join("rec/model.safetensors").is_file()
        || p.join(format!("rec/ppocrv6_{tier}_rec.safetensors")).is_file();
    if det && rec {
        Some(p)
    } else {
        None
    }
}


#[test]
fn tiny_cpu_dry_build() {
    let Some(dir) = model_dir("tiny") else {
        eprintln!("skip: .cache/ppocrv6/tiny missing — run just fetch-ppocrv6-tiny");
        return;
    };
    let runner = PpOcrV6Runner::builder()
        .tier(Tier::Tiny)
        .model_dir(dir)
        .device(Device::Cpu)
        .build()
        .expect("build tiny runner");
    assert_eq!(runner.engine().device(), Device::Cpu);
}

#[test]
fn tiny_cpu_e2e_hello() {
    let Some(dir) = model_dir("tiny") else {
        eprintln!("skip: .cache/ppocrv6/tiny missing");
        return;
    };
    let img = workspace_root().join(".cache/ppocrv6/hello.png");
    if !img.is_file() {
        eprintln!("skip: hello.png missing");
        return;
    }
    let runner = PpOcrV6Runner::builder()
        .tier(Tier::Tiny)
        .model_dir(dir)
        .device(Device::Cpu)
        .build()
        .expect("build");
    let out = runner.predict_path(&img).expect("ocr");
    assert!(
        out.text.to_ascii_lowercase().contains("hello"),
        "unexpected OCR text: {:?}",
        out.text
    );
}

#[test]
fn small_cpu_dry_build() {
    let Some(dir) = model_dir("small") else {
        eprintln!("skip: .cache/ppocrv6/small missing — run just fetch-ppocrv6-small");
        return;
    };
    let runner = PpOcrV6Runner::builder()
        .tier(Tier::Small)
        .model_dir(dir)
        .device(Device::Cpu)
        .build()
        .expect("build small runner");
    assert_eq!(runner.engine().device(), Device::Cpu);
}

#[test]
fn lcnet_configs() {
    let d = rlx_ppocrv6::backbone::LcNetV4Cfg::detection(Tier::Tiny);
    assert_eq!(d.stem, 16);
    let r = rlx_ppocrv6::backbone::LcNetV4Cfg::recognition(Tier::Small);
    assert!(r.asymmetric_stride);
    assert_eq!(r.stages[3].width, 384);
}
