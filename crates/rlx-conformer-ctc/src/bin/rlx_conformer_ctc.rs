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

//! `rlx-conformer-ctc` binary — see crate docs / README for usage.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = rlx_conformer_ctc::cli::run(&args) {
        eprintln!("rlx-conformer-ctc: {e:#}");
        std::process::exit(1);
    }
}
