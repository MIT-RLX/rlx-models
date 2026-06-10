// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Mirrors `rlx-runtime` so `#[cfg(rlx_mlx_host)]` gates match upstream cost models.

use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo::rustc-check-cfg=cfg(rlx_mlx_host)");
    if matches!(target_os.as_str(), "macos" | "linux" | "windows") {
        println!("cargo:rustc-cfg=rlx_mlx_host");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
