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

use rlx_gemma::{cli, multimodal_cli};
use std::process::ExitCode;

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (multimodal, args): (bool, Vec<String>) = if raw.iter().any(|a| a == "--multimodal") {
        (
            true,
            raw.into_iter().filter(|a| a != "--multimodal").collect(),
        )
    } else {
        (false, raw)
    };
    let result = if multimodal {
        multimodal_cli::run(&args)
    } else {
        cli::run(&args)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rlx-gemma: {e:#}");
            ExitCode::FAILURE
        }
    }
}
