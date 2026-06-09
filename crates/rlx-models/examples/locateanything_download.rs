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

//! Download [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) into the Hugging Face cache.
//!
//! ```bash
//! just fetch-locateanything
//! # or
//! cargo run -p rlx-locateanything --features hf-download --release -- --download
//! ```

use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    let quiet = std::env::args().any(|a| a == "--quiet");
    let dir = rlx_locateanything::fetch_default()?;
    if quiet {
        writeln!(io::stdout(), "{}", dir.display())?;
    } else {
        eprintln!("\nLocateAnything snapshot:\n  {}", dir.display());
        eprintln!(
            "\nOptional:\n  export RLX_LOCATEANYTHING_DIR={}",
            dir.display()
        );
        eprintln!("\nRust: LocateAnythingSession::open_default()? — uses HF cache automatically.");
        eprintln!("\nNext: just fetch-locateanything-tokenizer  (processor prompt)");
    }
    Ok(())
}
