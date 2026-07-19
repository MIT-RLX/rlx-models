// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    if let Err(e) = rlx_gepard::cli::run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
