// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Link to the macOS Accelerate framework when the `accelerate` feature is on.
//! Accelerate bundles a BLAS implementation tuned for macOS arm64 — Pocket
//! TTS's hot path is `linear()`, so this gates the only platform-specific code.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `cfg!(target_os = ...)` inside a build script describes the HOST running
    // the script, not the target being compiled for — so cross-compiling from
    // a Mac emitted this framework link for Windows and Linux too, and the
    // build died with "library kind `framework` is only supported on macOS
    // targets" (seen via `cargo xwin check --target x86_64-pc-windows-msvc`;
    // a native Windows runner never reproduced it, since there the host cfg is
    // false by accident).  CARGO_CFG_TARGET_OS is the target.  `cfg!(feature)`
    // is fine here: features *are* passed through to build scripts.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if cfg!(feature = "accelerate") && (target_os == "macos" || target_os == "ios") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
