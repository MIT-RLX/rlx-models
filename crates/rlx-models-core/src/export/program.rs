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

//! A model's host logic as an RLX program (`rlx_program`): a WebAssembly
//! module a package carries beside its graphs, for what a graph says badly
//! — a phonemizer, a search, a decode loop. [`build`] compiles a program
//! crate for `wasm32-unknown-unknown`; [`entry`] adds it to a package as an
//! entry of kind `program`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use super::Package;

/// Where programs build: beside the workspace's other targets, apart from
/// them (another target triple).
fn target_dir(manifest: &Path) -> PathBuf {
    if let Ok(dir) = std::env::var("RLX_PROGRAM_TARGET_DIR") {
        return PathBuf::from(dir);
    }
    // The workspace root: the nearest ancestor with a `target` directory.
    manifest
        .ancestors()
        .find(|d| d.join("target").is_dir())
        .map(|d| d.join("target/rlx-programs"))
        .unwrap_or_else(|| std::env::temp_dir().join("rlx-programs"))
}

/// Build the program crate at `manifest` (its `Cargo.toml`; a `cdylib`
/// over `rlx-program`) for `wasm32-unknown-unknown`, release, with `cargo`
/// (`$CARGO`, else on the path): the module's bytes.
pub fn build(manifest: &Path) -> Result<Vec<u8>> {
    build_for(manifest, "wasm32-unknown-unknown")
}

/// [`build`] for `target`: `wasm32-wasip1` for a program whose code reads
/// files with `std::fs` (the package's, mounted at `/package`).
pub fn build_for(manifest: &Path, target: &str) -> Result<Vec<u8>> {
    // One build at a time in a process: two exports (tests run in parallel)
    // building one program would read its module while the other writes it.
    static BUILD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    let manifest = manifest
        .canonicalize()
        .with_context(|| format!("{}", manifest.display()))?;
    let text = std::fs::read_to_string(&manifest)?;
    let name = text
        .lines()
        .find_map(|l| {
            let l = l.trim();
            l.strip_prefix("name")
                .and_then(|r| r.trim_start().strip_prefix('='))
                .map(|r| r.trim().trim_matches('"').to_string())
        })
        .with_context(|| format!("{}: no package name", manifest.display()))?;
    let dir = target_dir(&manifest);
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(cargo)
        .args(["build", "--quiet", "--release", "--target", target])
        .arg("--manifest-path")
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", &dir)
        .status()
        .context("running cargo to build the program")?;
    if !status.success() {
        bail!("building the program {} failed", manifest.display());
    }
    let wasm = dir
        .join(target)
        .join("release")
        .join(format!("{}.wasm", name.replace('-', "_")));
    std::fs::read(&wasm).with_context(|| format!("{}", wasm.display()))
}

/// Add the program `module` to `pkg` as the entry `name`: the tasks it
/// answers (`run`, `transcribe`, `synthesize`, `embed`, `generate`, `chat`,
/// `score`), what the model is (`task`, a Hugging Face task name), and the
/// `config` its `init` gets.
pub fn entry(
    pkg: &mut Package,
    name: &str,
    module: Vec<u8>,
    tasks: &[&str],
    task: &str,
    config: Value,
) {
    let id = format!("{name}.wasm");
    let module = pkg.sidecar(&id, "application/wasm", module);
    pkg.entry(
        name,
        json!({
            "kind": "program",
            "module": module,
            "tasks": tasks,
            "task": task,
            "config": config,
        }),
    );
}

/// Add greedy CTC transcription to `pkg` as the program entry
/// `transcribe` (the shared `programs/ctc` crate): `config` names the
/// package's `features` function and its `encoders` (`[{frames,
/// function}]`, each returning the argmax id of every encoder frame last),
/// and gives `sample_rate`, `pad_value`, `subsampling` (`[[kernel, stride,
/// pad], …]`), `blank`, `pieces`, `space` and `strip_specials`.
pub fn ctc(pkg: &mut Package, config: Value) -> Result<()> {
    let module = build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("programs/ctc/Cargo.toml"))?;
    entry(
        pkg,
        "transcribe",
        module,
        &["transcribe"],
        "automatic-speech-recognition",
        config,
    );
    Ok(())
}
