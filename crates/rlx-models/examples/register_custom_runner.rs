// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Demonstrates plugging custom subcommands into `rlx-cli`'s dispatch
// registry from a downstream crate (no changes to `rlx-models` required).
//
//   1. Implement `ModelRunner` for your model.
//   2. `register_runner(Box::new(YourRunner))` early in `main`.
//   3. `dispatch(&argv)` — routes by first argv token.
//
// Running:
//   cargo run -p rlx-models --example register_custom_runner -- echo hello
//   cargo run -p rlx-models --example register_custom_runner -- adder 3 4
//
// A production binary would depend on `rlx-cli` directly, register its
// own runners, and optionally call `register_cli` for built-ins (see
// `rlx-models/src/bin/rlx_run.rs`).

use anyhow::{Context, Result};
use rlx_cli::{ModelRunner, dispatch, dispatch_help, register_runner};

/// `echo <args...>` — prints its arguments back. Useful as a
/// no-deps sanity test of the dispatch path.
struct EchoRunner;
impl ModelRunner for EchoRunner {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn description(&self) -> &'static str {
        "Print the supplied arguments back to stdout"
    }
    fn run(&self, args: &[String]) -> Result<()> {
        println!("echo: {}", args.join(" "));
        Ok(())
    }
}

/// `adder <a> <b>` — parses two integers and prints the sum.
/// Shows how a real runner does its own arg parsing inside `run`.
struct AdderRunner;
impl ModelRunner for AdderRunner {
    fn name(&self) -> &'static str {
        "adder"
    }
    fn description(&self) -> &'static str {
        "Add two integers (demonstrates per-runner arg parsing)"
    }
    fn run(&self, args: &[String]) -> Result<()> {
        let a: i64 = args
            .first()
            .context("adder: missing first integer")?
            .parse()
            .context("adder: first arg not an integer")?;
        let b: i64 = args
            .get(1)
            .context("adder: missing second integer")?
            .parse()
            .context("adder: second arg not an integer")?;
        println!("{a} + {b} = {}", a + b);
        Ok(())
    }
}

fn main() -> Result<()> {
    register_runner(Box::new(EchoRunner));
    register_runner(Box::new(AdderRunner));

    eprintln!("[register_custom_runner] registered subcommands:");
    eprintln!("{}", dispatch_help());

    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!(
            "[register_custom_runner] no subcommand given; try `echo hi` or `adder 3 4` or `help`"
        );
        return Ok(());
    }
    dispatch(&argv)
}
