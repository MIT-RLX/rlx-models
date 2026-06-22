// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use rlx_orpheus::cli;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = cli::run(&args) {
        eprintln!("rlx-orpheus: {e:#}");
        std::process::exit(1);
    }
}
