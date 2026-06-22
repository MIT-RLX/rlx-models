use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var_os("CARGO_FEATURE_NATIVE_CODEC").is_none() {
        return;
    }

    let vendor = PathBuf::from("vendor");
    if !vendor.join("src/tsac_codec.c").is_file() {
        fetch_vendor(&vendor);
    }

    let src = vendor.join("src");
    let mut build = cc::Build::new();
    build
        .std("c11")
        .include(vendor.join("include"))
        .include(&src)
        .include(src.join("vulkan"))
        .warnings(false)
        .flag_if_supported("-Wno-incompatible-function-pointer-types")
        .flag_if_supported("-O3")
        .flag_if_supported("-ffast-math");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match target_arch.as_str() {
        "aarch64" => {
            build.file(src.join("arch/arm/cpu_arm.c"));
            build.flag_if_supported("-march=armv8-a+simd");
        }
        "riscv64" => {
            build.file(src.join("arch/riscv/cpu_riscv.c"));
        }
        "x86_64" => {
            build.flag_if_supported("-mavx2").flag_if_supported("-mfma");
        }
        _ => {}
    }

    for name in [
        "tsac_codec.c",
        "tsac_transformer.c",
        "tsac_normal_decode.c",
        "dac_model.c",
        "txc_format.c",
        "model_loader.c",
        "cpu_decoder.c",
        "range_coder.c",
    ] {
        build.file(src.join(name));
    }

    if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        build.define("USE_VULKAN", None);
        build.file(src.join("vulkan/vulkan_arch.c"));
    } else {
        build.file(src.join("vulkan_stubs.c"));
    }

    build.file(src.join("cuda_stubs.c"));
    build.file(src.join("hip_stubs.c"));
    build.file(src.join("llvm_stubs.c"));

    build.compile("tsac_ng");
    println!("cargo:rerun-if-changed=vendor/");
}

fn fetch_vendor(vendor: &PathBuf) {
    let status = Command::new("curl")
        .args([
            "-fsSL",
            "https://github.com/Hope2333/tsac-ng/archive/refs/heads/master.tar.gz",
            "-o",
            "/tmp/tsac-ng-vendor.tar.gz",
        ])
        .status()
        .expect("curl for tsac-ng vendor");
    assert!(status.success(), "failed to download tsac-ng sources");
    std::fs::create_dir_all(vendor).ok();
    let status = Command::new("tar")
        .args([
            "xzf",
            "/tmp/tsac-ng-vendor.tar.gz",
            "-C",
            vendor.to_str().unwrap(),
            "--strip-components=1",
        ])
        .status()
        .expect("tar tsac-ng vendor");
    assert!(status.success(), "failed to extract tsac-ng vendor");
}
