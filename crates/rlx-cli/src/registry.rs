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

use anyhow::{Result, bail};
use std::sync::{Mutex, OnceLock};

/// One CLI entry per model family. Each per-crate `rlx-<family>` binary
/// calls its own `run` directly; the optional `rlx-run` multiplexer
/// registers many `ModelRunner` implementations.
pub trait ModelRunner: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, args: &[String]) -> Result<()>;
}

type RegistryInner = Vec<Box<dyn ModelRunner>>;

fn registry() -> &'static Mutex<RegistryInner> {
    static R: OnceLock<Mutex<RegistryInner>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn register_runner(runner: Box<dyn ModelRunner>) {
    let mut g = registry().lock().expect("runner registry poisoned");
    let name = runner.name();
    if let Some(idx) = g.iter().position(|r| r.name() == name) {
        g[idx] = runner;
    } else {
        g.push(runner);
    }
}

pub fn registered_runners() -> Vec<(&'static str, &'static str)> {
    let g = registry().lock().expect("runner registry poisoned");
    g.iter().map(|r| (r.name(), r.description())).collect()
}

pub fn run_registered(name: &str, args: &[String]) -> Result<Option<()>> {
    let g = registry().lock().expect("runner registry poisoned");
    for runner in g.iter() {
        if runner.name() == name {
            return runner.run(args).map(Some);
        }
    }
    Ok(None)
}

pub fn dispatch(args: &[String]) -> Result<()> {
    let Some(sub) = args.first() else {
        eprintln!("{}", dispatch_help());
        return Ok(());
    };
    match sub.as_str() {
        "help" | "--help" | "-h" => {
            println!("{}", dispatch_help());
            return Ok(());
        }
        _ => {}
    }
    match run_registered(sub, &args[1..])? {
        Some(()) => Ok(()),
        None => {
            eprintln!("{}", dispatch_help());
            bail!("unknown subcommand: {sub}");
        }
    }
}

pub fn dispatch_help() -> String {
    let mut s = String::from(
        "rlx-run — multi-model launcher (or use per-model binaries: rlx-qwen3, rlx-flux2, …)\n\
         USAGE:\n  rlx-run <subcommand> [flags]\n\nSUBCOMMANDS:\n",
    );
    let mut any = false;
    for (name, desc) in registered_runners() {
        s.push_str(&format!("  {name:<14} {desc}\n"));
        any = true;
    }
    if !any {
        s.push_str("  (no runners registered)\n");
    }
    s.push_str("  help           print this help\n");
    s
}

/// Register a model CLI under the multiplexer.
pub fn register_cli(
    name: &'static str,
    description: &'static str,
    run: fn(&[String]) -> Result<()>,
) {
    struct R {
        name: &'static str,
        description: &'static str,
        run: fn(&[String]) -> Result<()>,
    }
    impl ModelRunner for R {
        fn name(&self) -> &'static str {
            self.name
        }
        fn description(&self) -> &'static str {
            self.description
        }
        fn run(&self, args: &[String]) -> Result<()> {
            (self.run)(args)
        }
    }
    register_runner(Box::new(R {
        name,
        description,
        run,
    }));
}
