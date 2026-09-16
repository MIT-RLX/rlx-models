// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use std::process::ExitCode;

fn main() -> ExitCode {
    rlx_core::weights::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match rlx_moonshine::cli::run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rlx-moonshine: {e:#}");
            ExitCode::FAILURE
        }
    }
}
