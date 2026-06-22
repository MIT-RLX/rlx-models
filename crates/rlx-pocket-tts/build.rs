// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Link to the macOS Accelerate framework when the `accelerate` feature is on.
//! Accelerate bundles a BLAS implementation tuned for Apple silicon — Pocket
//! TTS's hot path is `linear()`, so this gates the only platform-specific code.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if cfg!(feature = "accelerate") && (cfg!(target_os = "macos") || cfg!(target_os = "ios")) {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
